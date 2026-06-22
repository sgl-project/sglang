"""MLX auxiliary-state snapshots for unified radix cache.

Hybrid MLX models may include non-softmax-attention layers whose native
``mlx-lm`` cache state cannot be reconstructed from the attention KV pool.
The global scheduler exposes that state through its existing MAMBA component
contract, so this MLX adapter keeps those scheduler-facing field names while
storing model-agnostic native cache snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import mlx.core as mx
import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import TreeComponent

_CACHE_ATTRS = ("offset", "lengths", "left_padding")
_MISSING = object()


def _clone_tree(value: Any) -> Any:
    if isinstance(value, mx.array):
        return mx.array(value)
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    return value


def _arrays_in_tree(value: Any) -> list[mx.array]:
    arrays: list[mx.array] = []

    def collect(item: Any) -> None:
        if isinstance(item, mx.array):
            arrays.append(item)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)

    collect(value)
    return arrays


@dataclass
class _CacheSnapshot:
    state: Any
    meta_state: Any = _MISSING
    attrs: dict[str, Any] | None = None


def _snapshot_cache(cache: Any) -> _CacheSnapshot:
    state = _clone_tree(getattr(cache, "state", ()))
    meta_state = (
        _clone_tree(cache.meta_state) if hasattr(cache, "meta_state") else _MISSING
    )
    attrs = {
        name: _clone_tree(getattr(cache, name))
        for name in _CACHE_ATTRS
        if hasattr(cache, name)
    }
    arrays = _arrays_in_tree((state, meta_state, attrs))
    if arrays:
        mx.eval(*arrays)
    return _CacheSnapshot(state=state, meta_state=meta_state, attrs=attrs)


def _restore_cache(cache: Any, snapshot: _CacheSnapshot) -> None:
    cache.state = _clone_tree(snapshot.state)
    if snapshot.meta_state is not _MISSING and hasattr(cache, "meta_state"):
        cache.meta_state = _clone_tree(snapshot.meta_state)
    for name, value in (snapshot.attrs or {}).items():
        setattr(cache, name, _clone_tree(value))


class MlxAuxiliaryStatePool:
    """Index-addressable snapshots of native MLX auxiliary cache state."""

    def __init__(self, size: int, device: str):
        self.size = size
        self.device = device
        self.mamba_cache = None
        self.mem_usage = 0
        self._snapshots: dict[int, dict[int, _CacheSnapshot]] = {}
        self.clear()

    def _tensor(self, indices: Any) -> torch.Tensor:
        return torch.as_tensor(indices, dtype=torch.int64, device=self.device).view(-1)

    def _index(self, index: Any) -> int:
        flat = self._tensor(index)
        assert flat.numel() == 1
        return int(flat.item())

    def available_size(self) -> int:
        return int(self.free_slots.numel())

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > self.available_size():
            return None
        slots = self.free_slots[:need_size].clone()
        self.free_slots = self.free_slots[need_size:]
        for slot in slots.tolist():
            self._snapshots.pop(int(slot), None)
        return slots

    def free(self, indices: Any) -> None:
        if indices is None:
            return
        indices = self._tensor(indices)
        if indices.numel() == 0:
            return
        for slot in indices.tolist():
            self._snapshots.pop(int(slot), None)
        self.free_slots = torch.cat([self.free_slots, indices])

    def clear(self) -> None:
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self._snapshots.clear()

    def copy_from(self, src: Any, dst: Any) -> None:
        src_indices = self._tensor(src)
        dst_indices = self._tensor(dst)
        assert src_indices.numel() == dst_indices.numel()
        for src_idx, dst_idx in zip(src_indices.tolist(), dst_indices.tolist()):
            snapshot = self._snapshots.get(int(src_idx))
            if snapshot is None:
                self._snapshots.pop(int(dst_idx), None)
            else:
                self._snapshots[int(dst_idx)] = {
                    layer_idx: _CacheSnapshot(
                        state=_clone_tree(cache_snapshot.state),
                        meta_state=_clone_tree(cache_snapshot.meta_state),
                        attrs=_clone_tree(cache_snapshot.attrs),
                    )
                    for layer_idx, cache_snapshot in snapshot.items()
                }

    def fork_from(self, src: Any) -> Optional[torch.Tensor]:
        src_indices = self._tensor(src)
        dst = self.alloc(src_indices.numel())
        if dst is None:
            return None
        self.copy_from(src_indices, dst)
        return dst

    def store_cache(
        self,
        index: Any,
        cache: list[Any],
        layer_indices: Iterable[int],
    ) -> None:
        self._snapshots[self._index(index)] = {
            layer_idx: _snapshot_cache(cache[layer_idx]) for layer_idx in layer_indices
        }

    def restore_cache(
        self,
        index: Any,
        cache: list[Any],
        layer_indices: Iterable[int] | None = None,
    ) -> bool:
        snapshot = self._snapshots.get(self._index(index))
        if snapshot is None:
            return False
        selected_layers = set(layer_indices) if layer_indices is not None else None
        for layer_idx, cache_snapshot in snapshot.items():
            if selected_layers is not None and layer_idx not in selected_layers:
                continue
            _restore_cache(cache[layer_idx], cache_snapshot)
        return True

    def has_snapshot(self, index: Any) -> bool:
        return self._index(index) in self._snapshots


class MlxAuxiliaryStateReqToTokenPool(ReqToTokenPool):
    """Req-to-token pool with MLX auxiliary-state slot bookkeeping."""

    def __init__(
        self,
        *,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        auxiliary_state_size: int,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )
        self.mamba_pool = MlxAuxiliaryStatePool(
            size=auxiliary_state_size,
            device=device,
        )
        # The unified radix base MAMBA component still reads ``mamba_pool``.
        # Keep the MLX-owned name beside it so local code can avoid model-
        # specific terminology.
        self.auxiliary_state_pool = self.mamba_pool
        self.enable_mamba_extra_buffer = False
        self.req_index_to_auxiliary_state_index_mapping = torch.zeros(
            self._alloc_size, dtype=torch.int32, device=device
        )

    def alloc(self, reqs):
        select_index = super().alloc(reqs)
        if select_index is None:
            return None

        from sglang.srt.managers.schedule_batch import ReqMambaInfo

        auxiliary_state_indices = []
        for req in reqs:
            if req.mamba is not None:
                mid = req.mamba.mamba_pool_idx
            else:
                allocated = self.auxiliary_state_pool.alloc(1)
                assert allocated is not None, "Not enough MLX auxiliary state slots"
                mid = allocated[0]
                req.mamba = ReqMambaInfo(
                    mamba_pool_idx=mid,
                    mamba_ping_pong_track_buffer=None,
                    mamba_next_track_idx=None,
                    mamba_last_track_seqlen=None,
                    mamba_branching_seqlen=None,
                )
            auxiliary_state_indices.append(mid.to(dtype=torch.int32))
        self.req_index_to_auxiliary_state_index_mapping[select_index] = torch.stack(
            auxiliary_state_indices
        )
        return select_index

    def get_auxiliary_state_indices(self, req_indices) -> torch.Tensor:
        return self.req_index_to_auxiliary_state_index_mapping[req_indices]

    def get_mamba_indices(self, req_indices) -> torch.Tensor:
        return self.get_auxiliary_state_indices(req_indices)

    def get_mamba_ping_pong_other_idx(self, mamba_next_track_idx: int) -> int:
        return 0

    def free_mamba_cache(self, req, mamba_ping_pong_track_buffer_to_keep=None):
        if req.mamba is None:
            return
        self.auxiliary_state_pool.free(req.mamba.mamba_pool_idx.unsqueeze(0))
        track_buffer = req.mamba.mamba_ping_pong_track_buffer
        if track_buffer is not None:
            if mamba_ping_pong_track_buffer_to_keep is None:
                self.auxiliary_state_pool.free(track_buffer)
        req.mamba = None

    def free_auxiliary_state_cache(self, req, track_buffer_to_keep=None):
        self.free_mamba_cache(
            req,
            mamba_ping_pong_track_buffer_to_keep=track_buffer_to_keep,
        )

    def free(self, req):
        super().free(req)

    def clear(self):
        super().clear()
        self.auxiliary_state_pool.clear()
        self.req_index_to_auxiliary_state_index_mapping.zero_()


class MlxAuxiliaryStateComponent(MambaComponent):
    """Unified radix component for MLX native auxiliary-state snapshots."""

    def __init__(self, cache, params):
        if params.enable_mamba_extra_buffer:
            raise NotImplementedError(
                "MLX auxiliary-state radix cache does not support "
                "enable_mamba_extra_buffer yet."
            )
        pool = getattr(cache.req_to_token_pool, "auxiliary_state_pool", None)
        if not isinstance(pool, MlxAuxiliaryStatePool):
            raise TypeError(
                "MlxAuxiliaryStateComponent requires MlxAuxiliaryStatePool, "
                f"got {type(pool)}"
            )
        TreeComponent.__init__(self, cache, params)
        self.enable_mamba_extra_buffer = False
        self._mamba_pool_host = None

    @staticmethod
    def _tracked_value(req) -> tuple[object | None, bool]:
        track_buffer = req.mamba.mamba_ping_pong_track_buffer
        track_len = req.mamba.mamba_last_track_seqlen
        if track_buffer is not None and track_len is not None:
            return track_buffer[0].unsqueeze(-1).clone(), True
        return req.mamba.mamba_pool_idx.unsqueeze(-1).clone(), False

    def prepare_for_caching_req(
        self,
        req,
        insert_params,
        token_ids_len: int,
        is_finished: bool,
    ) -> int | None:
        cache_len = req.mamba.mamba_last_track_seqlen
        auxiliary_value, uses_track_slot = self._tracked_value(req)
        setattr(insert_params, "mlx_auxiliary_state_uses_track_slot", uses_track_slot)

        if auxiliary_value is None:
            return 0 if is_finished else None

        if cache_len is None:
            cache_len = token_ids_len
        if is_finished:
            insert_params.mamba_value = auxiliary_value
        else:
            source_value = auxiliary_value
            forked_value = self.cache.req_to_token_pool.auxiliary_state_pool.fork_from(
                source_value
            )
            if forked_value is None:
                self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
                forked_value = (
                    self.cache.req_to_token_pool.auxiliary_state_pool.fork_from(
                        source_value
                    )
                )
                assert forked_value is not None, "Can not alloc MLX auxiliary cache"
            insert_params.mamba_value = forked_value
        return cache_len

    def cleanup_after_caching_req(
        self,
        req,
        is_finished: bool,
        insert_result: InsertResult | None = None,
        insert_params=None,
    ) -> None:
        if not is_finished:
            if (
                insert_params is not None
                and insert_params.mamba_value is not None
                and (insert_result is None or insert_result.mamba_exist)
            ):
                self.cache.req_to_token_pool.auxiliary_state_pool.free(
                    insert_params.mamba_value
                )
            if req.mamba is not None:
                if bool(
                    getattr(insert_params, "mlx_auxiliary_state_uses_track_slot", False)
                ):
                    track_buffer = req.mamba.mamba_ping_pong_track_buffer
                    if track_buffer is not None:
                        self.cache.req_to_token_pool.auxiliary_state_pool.free(
                            track_buffer
                        )
                    req.mamba.mamba_ping_pong_track_buffer = None
                    req.mamba.mamba_next_track_idx = None
                req.mamba.mamba_last_track_seqlen = None
            return

        auxiliary_value_exists = (
            insert_result.mamba_exist if insert_result is not None else True
        )
        uses_track_slot = bool(
            getattr(insert_params, "mlx_auxiliary_state_uses_track_slot", False)
        )
        if uses_track_slot:
            keep_track_slot = not auxiliary_value_exists
            self.cache.req_to_token_pool.free_auxiliary_state_cache(
                req,
                track_buffer_to_keep=0 if keep_track_slot else None,
            )
        elif auxiliary_value_exists:
            self.cache.req_to_token_pool.free_auxiliary_state_cache(req)
        else:
            # The radix tree now owns the live auxiliary-state slot.
            track_buffer = (
                req.mamba.mamba_ping_pong_track_buffer
                if req.mamba is not None
                else None
            )
            if track_buffer is not None:
                self.cache.req_to_token_pool.auxiliary_state_pool.free(track_buffer)
            req.mamba = None
