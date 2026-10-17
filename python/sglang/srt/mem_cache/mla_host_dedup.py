"""Deduplicate MLA/DSA host cache across attention-TP ranks."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, List, Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_device_module, is_cuda

logger = logging.getLogger(__name__)
device_module = get_device_module()


# These backends tolerate buffer-less host pools on non-source ranks.
_DEDUP_COMPATIBLE_STORAGE = frozenset({None, "", "file"})


def storage_supports_host_dedup(storage_backend: Optional[str]) -> bool:
    """Whether MLA/DSA host-memory dedup can engage with this storage backend."""
    return storage_backend in _DEDUP_COMPATIBLE_STORAGE


def mla_dedup_rank_and_size() -> tuple[int, int]:
    """Attn-TP rank/size when DP attention is enabled, model-TP otherwise."""
    parallel = get_parallel()
    if is_dp_attention_enabled():
        return parallel.attn_tp_rank, parallel.attn_tp_size
    return parallel.tp_rank, parallel.tp_size


def mla_host_dedup_eligible(kv_cache, storage_backend: Optional[str]) -> bool:
    """Rank-independent gate. CUDA only; FP4 excluded (its per-rank scale
    buffer is not covered by the broadcast)."""
    return (
        isinstance(kv_cache, MLATokenToKVPool)
        and not isinstance(kv_cache, MLATokenToKVPoolFP4)
        and is_cuda()
        and storage_supports_host_dedup(storage_backend)
    )


class MLAHostDedupBroadcaster:
    """Layerwise MLA/DSA broadcast over a dedicated NCCL group.

    With ``rotate_owners``, layer slot ``s`` (each KV layer, then each layer that
    holds index keys) is owned and broadcast by dedup rank ``s % size``;
    otherwise every layer comes from the source rank.
    """

    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        group: torch.distributed.ProcessGroup,
        src_global_rank: int,
        rotate_owners: bool = False,
    ):
        self.device_pool = device_pool
        self.group = group
        self.src_global_rank = src_global_rank
        self.dedup_rank, dedup_size = mla_dedup_rank_and_size()
        self.is_src = self.dedup_rank == 0
        self.owner_size = dedup_size if rotate_owners else None
        self.layer_num = device_pool.layer_num
        self.device = device_pool.device
        self.chunk_tokens = envs.SGLANG_MLA_DEDUP_CHUNK_TOKENS.get()
        if self.chunk_tokens <= 0:
            raise ValueError(
                "SGLANG_MLA_DEDUP_CHUNK_TOKENS must be positive, "
                f"got {self.chunk_tokens}."
            )
        self.kv_staging = torch.empty(
            self.layer_num * self.chunk_tokens * device_pool.kv_cache_dim,
            dtype=device_pool.kv_buffer[0].dtype,
            device=self.device,
        )
        self.idx_bufs = None
        self.idx_elem = None
        self.idx_staging = None
        if isinstance(device_pool, DSATokenToKVPool):
            self.idx_bufs = device_pool.index_k_with_scale_buffer
            self.idx_elem = math.prod(self.idx_bufs[0].shape[1:]) or 1
            # Index rows are pages: cover the same tokens as the KV staging.
            idx_rows = max(
                1, self.layer_num * self.chunk_tokens // device_pool.page_size
            )
            self.idx_staging = torch.empty(
                idx_rows * self.idx_elem,
                dtype=self.idx_bufs[0].dtype,
                device=self.device,
            )
            # Shared-topk layers hold a 0-row placeholder and no index keys.
            live = [i for i, buf in enumerate(self.idx_bufs) if buf.shape[0] > 0]
            self.idx_ordinal = {layer: i for i, layer in enumerate(live)}
        logger.info(
            "MLA host-dedup broadcast chunk configured: base_tokens=%d, "
            "effective_layer_tokens=%d",
            self.chunk_tokens,
            self.layer_num * self.chunk_tokens,
        )

    @classmethod
    def build(
        cls,
        device_pool,
        tp_group: torch.distributed.ProcessGroup,
        attn_tp_group: Optional[torch.distributed.ProcessGroup],
        **kwargs,
    ) -> MLAHostDedupBroadcaster:
        """Build and initialize the NCCL group before host-pool allocation."""
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        base_group = tp_group
        if is_dp_attention_enabled() and attn_tp_group is not None:
            base_group = attn_tp_group
        group_ranks = torch.distributed.get_process_group_ranks(base_group)
        group = create_custom_parallel_group(
            group_ranks=list(group_ranks), backend="nccl"
        )
        broadcaster = cls(device_pool, group, src_global_rank=group_ranks[0], **kwargs)
        broadcaster._warmup_group()
        return broadcaster

    def _warmup_group(self) -> None:
        """Initialize the NCCL communicator before serving."""
        warmup = self.kv_staging[:1]
        if self.is_src:
            warmup.zero_()
        torch.distributed.broadcast(warmup, src=self.src_global_rank, group=self.group)
        torch.cuda.synchronize(self.device)
        logger.info("MLA host-dedup NCCL broadcast group warmup completed")

    def prepare_broadcast(
        self, device_indices: torch.Tensor, load_stream
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare reusable KV/indexer indices for one layerwise load."""
        indices = device_indices
        if not indices.is_cuda:
            indices = indices.to(self.device)
        if indices.is_cuda:
            indices.record_stream(load_stream)

        page_idx = None
        if self.idx_bufs is not None:
            page_size = self.device_pool.page_size
            if page_size > 1:
                if indices.numel() % page_size != 0:
                    raise ValueError(
                        "DSA dedup broadcast expects page-aligned device indices: "
                        f"got {indices.numel()} indices for page_size={page_size}."
                    )
                # Preserve logical page order across rank-local allocations.
                page_idx = indices[::page_size] // page_size
            else:
                page_idx = indices
            if page_idx.is_cuda:
                page_idx.record_stream(load_stream)
        return indices, page_idx

    def broadcast_loaded_layer(
        self,
        layer_id: int,
        prepared: tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> None:
        """Broadcast one loaded KV layer and its optional DSA indexer layer."""
        indices, page_idx = prepared
        self._bcast_layer(
            self.device_pool.kv_buffer,
            self.kv_staging,
            indices,
            self.device_pool.kv_cache_dim,
            layer_id,
            owner=self._owner(layer_id),
        )
        if self.idx_bufs is not None:
            assert page_idx is not None
            ordinal = self.idx_ordinal.get(layer_id)
            if ordinal is None:
                return
            self._bcast_layer(
                self.idx_bufs,
                self.idx_staging,
                page_idx,
                self.idx_elem,
                layer_id,
                owner=self._owner(self.layer_num + ordinal),
            )

    def _owner(self, slot: int) -> Optional[int]:
        return None if self.owner_size is None else slot % self.owner_size

    def _bcast_layer(
        self,
        buf_list,
        staging,
        target,
        elem,
        layer_id: int,
        owner: Optional[int] = None,
    ) -> None:
        """Broadcast one layer in chunks using the shared staging buffer.

        ``owner`` is the dedup rank holding this layer; None means the source rank.
        """
        is_src = self.is_src if owner is None else owner == self.dedup_rank
        root = {"src": self.src_global_rank} if owner is None else {"group_src": owner}
        n = target.shape[0]
        rows_per_chunk = staging.numel() // elem
        assert rows_per_chunk > 0
        layer_buf = buf_list[layer_id]
        row_shape = layer_buf.shape[1:]

        for start in range(0, n, rows_per_chunk):
            cur = min(rows_per_chunk, n - start)
            idx = target[start : start + cur]
            chunk = staging[: cur * elem]
            chunk_rows = chunk.view(cur, *row_shape)
            if is_src:
                torch.index_select(layer_buf, 0, idx, out=chunk_rows)
            torch.distributed.broadcast(chunk, group=self.group, **root)
            if not is_src:
                layer_buf.index_copy_(0, idx, chunk_rows)

    def destroy(self) -> None:
        if self.group is None:
            return
        try:
            torch.distributed.destroy_process_group(self.group)
        except Exception:
            pass
        self.group = None


class _PendingLoad(msgspec.Struct):
    """A load-back whose layers the forward broadcasts as it reaches them."""

    device_indices: torch.Tensor
    finish_event: Any
    prepared: Optional[tuple[torch.Tensor, Optional[torch.Tensor]]] = None
    next_layer: int = 0


@dataclass
class MLAHostDedupContext:
    """All state owned by the optional MLA host-dedup path."""

    broadcaster: MLAHostDedupBroadcaster
    prefetch_hits_sync_groups: Optional[List[torch.distributed.ProcessGroup]]
    prefetch_completion_sync_groups: Optional[List[torch.distributed.ProcessGroup]]
    producer_stream: Optional[object] = None
    last_write_finish_event: Optional[object] = None
    # Load-backs by HiCache layer-done counter index, for broadcast_ready_layers.
    pending_loads: dict[int, _PendingLoad] = field(default_factory=dict)

    @property
    def is_src(self) -> bool:
        return self.broadcaster.is_src

    @property
    def is_dummy_rank(self) -> bool:
        return self.broadcaster.owner_size is None and not self.is_src

    def start_load(
        self, *, index: int, device_indices: torch.Tensor, finish_event: Any
    ) -> None:
        """Queue load-back ``index``; its ack ``finish_event`` is recorded again
        after the last broadcast, so waiting on the load waits on the broadcasts."""
        self.pending_loads[index] = _PendingLoad(device_indices, finish_event)

    def broadcast_ready_layers(self, index: int, threshold: int) -> None:
        """Layer-wait hook: broadcast load ``index``'s layers up to ``threshold``.

        Runs on the forward stream right after its wait for that layer's H2D copy,
        so each layer is broadcast once, in order, at the same point of the
        forward on every rank, and later layers keep loading meanwhile.
        """
        load = self.pending_loads.get(index)
        if load is None:
            return
        broadcaster = self.broadcaster
        if load.prepared is None:
            load.prepared = broadcaster.prepare_broadcast(
                load.device_indices, device_module.current_stream()
            )
        last = min(threshold, broadcaster.layer_num - 1)
        for layer_id in range(load.next_layer, last + 1):
            broadcaster.broadcast_loaded_layer(layer_id, load.prepared)
        load.next_layer = max(load.next_layer, last + 1)
        if load.next_layer == broadcaster.layer_num:
            load.finish_event.record()
            del self.pending_loads[index]

    def destroy(self) -> None:
        self.broadcaster.destroy()
        groups = (self.prefetch_hits_sync_groups or []) + (
            self.prefetch_completion_sync_groups or []
        )
        for group in groups:
            try:
                torch.distributed.destroy_process_group(group)
            except Exception:
                pass
        self.prefetch_hits_sync_groups = None
        self.prefetch_completion_sync_groups = None


def maybe_create_mla_host_dedup_context(
    kv_cache,
    tp_group: torch.distributed.ProcessGroup,
    attn_cp_group: Optional[torch.distributed.ProcessGroup],
    attn_tp_group: Optional[torch.distributed.ProcessGroup],
    storage_backend: Optional[str],
    enabled: bool = False,
    rotate_owners: bool = False,
) -> Optional[MLAHostDedupContext]:
    """Create dedup state before host allocation, or preserve the original path.

    ``rotate_owners``: each layer's host copy lives on one rank in turn, instead
    of all on the source rank (see MLAHostDedupBroadcaster).
    """
    if not enabled:
        return None
    if not mla_host_dedup_eligible(kv_cache, storage_backend):
        return None
    size = mla_dedup_rank_and_size()[1]
    if size <= 1:
        return None
    if rotate_owners:
        layer_nums = [kv_cache.layer_num]
        if isinstance(kv_cache, DSATokenToKVPool):
            idx_bufs = kv_cache.index_k_with_scale_buffer
            layer_nums.append(sum(buf.shape[0] > 0 for buf in idx_bufs))
        if min(layer_nums) < size:
            logger.warning(
                "MLA host dedup inactive: %s layers < %d ranks", layer_nums, size
            )
            return None

    broadcaster = MLAHostDedupBroadcaster.build(
        kv_cache, tp_group, attn_tp_group, rotate_owners=rotate_owners
    )
    prefetch_hits_sync_groups = None
    prefetch_completion_sync_groups = None
    if storage_backend is not None:
        prefetch_hits_sync_groups = _prebuild_prefetch_sync_groups(
            tp_group, attn_cp_group, attn_tp_group
        )
        prefetch_completion_sync_groups = _prebuild_prefetch_sync_groups(
            tp_group, attn_cp_group, attn_tp_group
        )
    return MLAHostDedupContext(
        broadcaster,
        prefetch_hits_sync_groups,
        prefetch_completion_sync_groups,
    )


def _prebuild_prefetch_sync_groups(
    tp_group: torch.distributed.ProcessGroup,
    attn_cp_group: Optional[torch.distributed.ProcessGroup],
    attn_tp_group: Optional[torch.distributed.ProcessGroup],
) -> List[torch.distributed.ProcessGroup]:
    """Prebuild one set of HiCache storage synchronization groups."""
    from sglang.srt.distributed.parallel_state import create_custom_parallel_group

    groups: List[torch.distributed.ProcessGroup] = []
    seen_rank_sets = set()
    if attn_cp_group is not None or attn_tp_group is not None:
        base_groups = [attn_cp_group, attn_tp_group]
    else:
        base_groups = [tp_group]
    for group in base_groups:
        if group is None or torch.distributed.get_world_size(group=group) == 1:
            continue
        ranks = tuple(torch.distributed.get_process_group_ranks(group))
        if ranks in seen_rank_sets:
            continue
        seen_rank_sets.add(ranks)
        groups.append(
            create_custom_parallel_group(group_ranks=list(ranks), backend="gloo")
        )
    return groups
