"""HiSparse V2 request ownership over the DSA device pool and HiCache.

Active prefixes retain GPU indexes addressed by their Device or Host KV pages.
The tree pins the host copy before publishing device sentinels on demotion.
Decode tails stay request-owned; temporary swap slots are never inserted in the tree.
"""

from __future__ import annotations

import logging

import msgspec
import torch

from sglang.srt.managers.hisparse_v2_cache import HiSparseV2Cache
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

logger = logging.getLogger(__name__)


class _RequestState(msgspec.Struct, kw_only=True):
    prefix_len: int
    temp: torch.Tensor
    nodes: list
    host_locks: dict = {}


class _PrefillReservation(msgspec.Struct, frozen=True):
    temp: torch.Tensor
    prefix_len: int


class HiSparseV2Coordinator:
    def __init__(self, *, cache: HiSparseV2Cache, allocator):
        self.cache = cache
        self.caches = (cache,)
        self.allocator = allocator
        self.tree_cache = None
        self.forward_stream = None
        self.requests: dict[int, _RequestState] = {}
        self.pending: dict[str, _PrefillReservation] = {}
        self._admit_count = 0

    def attach_tree(self, tree_cache, *, draft_caches=()) -> None:
        from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

        if not isinstance(tree_cache.tree_core, UnifiedTreeCore):
            raise ValueError("HiSparse V2 requires the Python unified tree core")
        if len(draft_caches) > 1:
            raise ValueError("HiSparse V2 supports one MTP draft cache")
        self.tree_cache = tree_cache
        host_pool = tree_cache.host_pool_group.get_pool(PoolName.KV)
        self.cache.attach_host_pool(host_pool)
        for draft_cache in draft_caches:
            if draft_cache is None or (
                draft_cache.pool.size < self.cache.pool.size
                or draft_cache.page_size != self.cache.page_size
                or draft_cache.reserved_buffer_tokens
                != self.cache.reserved_buffer_tokens
                or draft_cache.top_k != self.cache.top_k
            ):
                raise ValueError("HiSparse V2 MTP requires compatible DSA cache pools")
            draft_buffers = host_pool.device_kv_buffers[host_pool.target_layer_num :]
            if draft_buffers:
                if len(draft_buffers) != len(draft_cache.pool.kv_buffer) or any(
                    packed.data_ptr() != device.data_ptr()
                    for packed, device in zip(draft_buffers, draft_cache.pool.kv_buffer)
                ):
                    raise ValueError(
                        "HiSparse V2 draft does not match HiCache's packed pool"
                    )
                draft_cache.attach_host_pool(
                    host_pool, layer_offset=host_pool.target_layer_num
                )
            else:
                draft_cache.attach_host_pool(
                    tree_cache.host_pool_group.get_pool(PoolName.DRAFT)
                )
        self.caches = (self.cache, *draft_caches)
        tree_cache.hisparse_v2_coordinator = self
        tree_cache.tree_core.hisparse_v2_coordinator = self

    def checkpoint_up_to(self, req, up_to: int) -> int:
        """How far a running request's checkpoint may publish into the tree."""
        return up_to

    def _wait_for_forward(self) -> None:
        if self.forward_stream is not None:
            torch.cuda.current_stream().wait_stream(self.forward_stream)

    def _prefix_nodes(self, req) -> list:
        node = self.tree_cache.tree_core.node_by_id(req.last_node)
        nodes = []
        while node.parent is not None:
            nodes.append(node)
            node = node.parent
        return nodes

    def temp_tokens_required(self, req, *, max_new_tokens: int) -> int:
        if req.rid in self.pending:
            return 0
        prefix_len = len(req.full_untruncated_fill_ids)
        if prefix_len < self.cache.reserved_buffer_tokens or (
            prefix_len
            + self.cache.reserved_buffer_tokens
            + max_new_tokens
            + self.cache.page_size
            >= self.cache.pool.size
        ):
            return 0
        return self.cache.reserved_buffer_tokens

    def reserve_prefill(self, req, *, max_new_tokens: int) -> bool:
        needed = self.temp_tokens_required(req, max_new_tokens=max_new_tokens)
        if not needed:
            return True
        prefix_len = (
            len(req.full_untruncated_fill_ids)
            // self.cache.page_size
            * self.cache.page_size
        )
        pending_tokens = sum(p.prefix_len for p in self.pending.values())
        claimed = {
            node.id: len(node.key)
            for state in self.requests.values()
            for node in state.nodes
        }
        # Keep room for decode growth and fragmentation while allowing each
        # active prefix one home across the device and host tiers.
        capacity = self.cache.host_pool.size + int(self.cache.pool.size * 0.95)
        used = sum(claimed.values()) + pending_tokens + self.reserved_temp_tokens()
        if prefix_len + needed > capacity - used:
            return False
        from sglang.srt.mem_cache.common import evict_from_tree_cache

        evict_from_tree_cache(self.tree_cache, needed)
        temp = self.allocator.alloc(needed)
        if temp is None:
            return False
        self.pending[req.rid] = _PrefillReservation(temp, prefix_len)
        return True

    def admit_request(self, req) -> bool:
        req_id = req.kv.req_pool_idx
        if req_id in self.requests:
            raise RuntimeError("HiSparse V2 request admitted twice")
        prefix_len = req.kv.cache_protected_len
        reservation = self.pending.pop(req.rid, None)
        if prefix_len < self.cache.reserved_buffer_tokens:
            if reservation is not None:
                self.allocator.free(reservation.temp)
            return False
        if prefix_len % self.cache.page_size:
            raise RuntimeError("HiSparse V2 tree prefix is not page-aligned")
        self._wait_for_forward()
        temp = (
            reservation.temp
            if reservation is not None
            else self.allocator.alloc(self.cache.reserved_buffer_tokens)
        )
        if temp is None:
            logger.debug(
                "HiSparse V2 keeps request %s resident: no swap slots", req.rid
            )
            return False
        for cache in self.caches:
            cache.assign_temp_locations(req_id, temp)
            cache.protected_lens[req_id] = prefix_len
        self.requests[req_id] = _RequestState(
            prefix_len=prefix_len, temp=temp, nodes=self._prefix_nodes(req)
        )
        req.hisparse_v2_active = True
        self.tree_cache.unlock(req.lock)
        req.lock = None
        self._admit_count += 1
        if self._admit_count == 1 or self._admit_count % 64 == 0:
            logger.info(
                "HiSparse V2: admitted request #%d (%d prefix tokens, %d active)",
                self._admit_count,
                prefix_len,
                len(self.requests),
            )
        return True

    def node_is_active(self, node) -> bool:
        return any(
            any(prefix_node.id == node.id for prefix_node in state.nodes)
            for state in self.requests.values()
        )

    def on_node_split(self, *, new_parent, child) -> None:
        for state in self.requests.values():
            if any(node.id == child.id for node in state.nodes):
                state.nodes.append(new_parent)
        # FullComponent moves the receipt boundary on split, so existing host
        # receipts already cover both fragments and need no new acquisition.

    def on_node_evicted(self, node) -> None:
        if not self.requests:
            return
        component = node.component_data[ComponentType.FULL]
        if component.value is None:
            return
        affected = [
            (req_id, state)
            for req_id, state in self.requests.items()
            if any(prefix_node.id == node.id for prefix_node in state.nodes)
        ]
        if not affected:
            return
        if component.host_value is None:
            raise RuntimeError(
                "HiSparse V2 cannot discard an active prefix without backup"
            )
        self._wait_for_forward()
        device_rows, order = component.value.to(self.cache.device).long().sort()
        host_rows = component.host_value.to(self.cache.device).long()[order]
        for _, state in affected:
            if node.id not in state.host_locks:
                state.host_locks[node.id] = self.tree_cache.inc_host_lock_ref(
                    node.id
                ).to_dec_params()
        for cache in self.caches:
            cache.preserve_index(device_rows, host_rows)
        for req_id, state in affected:
            row = self.cache.req_to_token[req_id, : state.prefix_len]
            lookup = torch.searchsorted(device_rows, row.long()).clamp(
                max=device_rows.numel() - 1
            )
            positions = ((row >= 0) & (device_rows[lookup] == row)).nonzero().flatten()
            if not positions.numel():
                continue
            # Publish both maps before the allocator can hand these rows to another request.
            for cache in self.caches:
                cache.host_locs[req_id, positions] = host_rows[lookup[positions]]
            row[positions] = -1

    def request_finished(self, req) -> None:
        reservation = self.pending.pop(req.rid, None)
        if reservation is not None:
            self.allocator.free(reservation.temp)
        state = self.requests.pop(req.kv.req_pool_idx, None)
        if state is None:
            return
        self._wait_for_forward()
        for node_id, receipt in state.host_locks.items():
            self.tree_cache.dec_host_lock_ref(node_id, receipt)
        for cache in self.caches:
            cache.release(req.kv.req_pool_idx)
        self.allocator.free(state.temp)
        req.hisparse_v2_active = False

    def reserved_temp_tokens(self) -> int:
        return (
            len(self.requests) + len(self.pending)
        ) * self.cache.reserved_buffer_tokens

    def reset(self) -> None:
        if self.requests or self.pending:
            raise RuntimeError(
                "HiSparse V2 cache reset requires all requests to finish"
            )

    def destroy(self) -> None:
        self._wait_for_forward()
        for cache in self.caches:
            cache.host_bindings.zero_()
        torch.cuda.current_stream().synchronize()


def build_hisparse_v2_cache(
    *, req_to_token_pool, token_to_kv_pool, model_config, device
):
    from sglang.srt.configs.model_config import get_dsa_index_kpool
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.mem_cache.sparsity.hisparse_v2_config import (
        parse_hisparse_v2_config,
        resolve_hisparse_v2_query_steps,
    )
    from sglang.srt.runtime_context import get_spec

    if (
        not isinstance(token_to_kv_pool, DSATokenToKVPool)
        or get_dsa_index_kpool(model_config.hf_text_config) != 1
    ):
        raise ValueError("HiSparse V2 requires unpooled DSA target and draft KV pools")
    config = parse_hisparse_v2_config(
        model_top_k=model_config.hf_text_config.index_topk
    )
    spec = get_spec()
    num_query_steps = resolve_hisparse_v2_query_steps(
        algorithm=spec.speculative_algorithm,
        num_draft_tokens=spec.speculative_num_draft_tokens,
        eagle_topk=spec.speculative_eagle_topk,
        adaptive=spec.speculative_adaptive,
        multi_layer=spec.enable_multi_layer_eagle,
    )
    return HiSparseV2Cache(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        top_k=config.top_k,
        buffer_tokens=config.buffer_tokens,
        block_size=config.block_size,
        device=device,
        num_query_steps=num_query_steps,
    )
