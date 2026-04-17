from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
    TreeComponent,
    get_and_increase_time_counter,
)
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


class MambaComponent(TreeComponent):
    component_type = ComponentType.MAMBA

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

        assert isinstance(
            cache.req_to_token_pool, HybridReqToTokenPool
        ), f"MambaComponent requires HybridReqToTokenPool, got {type(cache.req_to_token_pool)}"
        if not params.enable_mamba_extra_buffer:
            assert (
                cache.page_size == 1
            ), f"MambaComponent requires page_size=1 when mamba_extra_buffer is disabled, got {cache.page_size}"
        super().__init__(cache, params)
        self.enable_mamba_extra_buffer = params.enable_mamba_extra_buffer

    def create_match_validator(self) -> Callable[[UnifiedTreeNode], bool]:
        ct = self.component_type
        return lambda node: node.component_data[ct].value is not None

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        cow_mamba = params.cow_mamba
        req = params.req
        last_node = result.last_device_node

        if len(value_chunks) > best_value_len:
            chunk_size = get_global_server_args().mamba_cache_chunk_size
            aligned_seqlen = (
                sum(len(v) for v in value_chunks) // chunk_size
            ) * chunk_size
            branching_seqlen = aligned_seqlen if aligned_seqlen > 0 else None
        else:
            branching_seqlen = None

        mamba_value = last_node.component_data[self.component_type].value
        if cow_mamba and mamba_value is not None:
            assert req is not None
            if req.mamba_pool_idx is None:
                dst_index = self.cache.req_to_token_pool.mamba_pool.alloc(1)
                if dst_index is None:
                    self.cache.inc_lock_ref(last_node)
                    self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
                    dst_index = self.cache.req_to_token_pool.mamba_pool.alloc(1)
                    self.cache.dec_lock_ref(last_node)
                    assert dst_index is not None, "Can not alloc mamba cache"
                self.cache.req_to_token_pool.mamba_pool.copy_from(
                    mamba_value, dst_index
                )
                req.mamba_pool_idx = dst_index[0]
            else:
                dst_index = req.mamba_pool_idx.unsqueeze(0)
                self.cache.req_to_token_pool.mamba_pool.copy_from(
                    mamba_value, dst_index
                )

        return result._replace(mamba_branching_seqlen=branching_seqlen)

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
    ) -> None:
        assert params.mamba_value is not None
        if is_new_leaf:
            node.component_data[self.component_type].value = params.mamba_value
            self.cache.lru_lists[self.component_type].insert_mru(node)
            self.cache.component_evictable_size_[self.component_type] += len(
                params.mamba_value
            )
            return
        if node.component_data[self.component_type].value is None:
            node.component_data[self.component_type].value = params.mamba_value
            self.cache.lru_lists[self.component_type].insert_mru(node)
            self.cache.component_evictable_size_[self.component_type] += len(
                params.mamba_value
            )
            node.last_access_time = get_and_increase_time_counter()
            return
        self.cache.lru_lists[self.component_type].reset_node_mru(node)
        node.last_access_time = get_and_increase_time_counter()
        result.mamba_exist = True

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        new_parent.component_data[self.component_type].value = None
        new_parent.component_data[self.component_type].lock_ref = 0

    def evict_component(self, node: UnifiedTreeNode, is_leaf: bool) -> int:
        value = node.component_data[self.component_type].value
        self.cache.req_to_token_pool.mamba_pool.free(value)
        freed = len(value)
        self.cache.component_evictable_size_[self.component_type] -= freed
        if not is_leaf:
            node.component_data[self.component_type].value = None
        return freed

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        request = params.mamba_num
        lru = self.cache.lru_lists[self.component_type]
        x = lru.get_lru_no_lock()
        while (
            tracker[self.component_type] < request and x is not None and lru.in_list(x)
        ):
            assert x.component_data[self.component_type].value is not None
            if len(x.children) > 0:
                x_next = lru.get_prev_no_lock(x)
                self.cache._evict_component_and_detach_lru(
                    x, self, is_leaf=False, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)
                x = x_next
            else:
                self.cache._evict_component_and_detach_lru(
                    x, self, is_leaf=True, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)
                x = lru.get_lru_no_lock()

    def acquire_component_lock(
        self, node: UnifiedTreeNode, result: IncLockRefResult
    ) -> IncLockRefResult:
        ct = self.component_type
        cd = node.component_data[ct]
        value = cd.value
        if value is not None:
            if cd.lock_ref == 0:
                vlen = len(value)
                self.cache.component_evictable_size_[ct] -= vlen
                self.cache.component_protected_size_[ct] += vlen
            cd.lock_ref += 1
        return result

    def release_component_lock(
        self, node: UnifiedTreeNode, params: Optional[DecLockRefParams]
    ) -> None:
        ct = self.component_type
        cd = node.component_data[ct]
        value = cd.value
        if value is not None:
            assert cd.lock_ref > 0
            if cd.lock_ref == 1:
                vlen = len(value)
                self.cache.component_evictable_size_[ct] += vlen
                self.cache.component_protected_size_[ct] -= vlen
            cd.lock_ref -= 1

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        cache_len = (
            req.mamba_last_track_seqlen
            if self.enable_mamba_extra_buffer
            else token_ids_len
        )
        if is_finished:
            if cache_len is None:
                cache_len = 0
            if self.enable_mamba_extra_buffer:
                if req.pending_radix_mamba_slot is not None:
                    # The track kernel already wrote the snapshot into the
                    # pending slot; transfer ownership to the tree (zero-copy).
                    mamba_value = req.pending_radix_mamba_slot.clone()
                    req.pending_radix_mamba_slot = None
                    insert_params.mamba_value = mamba_value
                    return cache_len
                # No pending slot available (producer-side alloc did not
                # succeed under mamba pool pressure). Returning 0 causes
                # cache_finished_req to free kv beyond cache_protected_len
                # and issue an empty-key insert, which _insert_helper
                # short-circuits — no tree nodes created.
                return 0
            mamba_value = req.mamba_pool_idx.unsqueeze(-1).clone()
            insert_params.mamba_value = mamba_value
            return cache_len
        else:
            if cache_len is None:
                return 0
            if self.enable_mamba_extra_buffer:
                if req.pending_radix_mamba_slot is None:
                    # No pending slot available; skip this caching round.
                    # Returning 0 takes the effective_cache_len<=0 branch
                    # in unified_radix_cache.cache_unfinished_req: it
                    # preserves prefix_indices and invokes cleanup, which
                    # retries producer-side alloc for the next round.
                    return 0
                # The track kernel already wrote the snapshot into the
                # pending slot; transfer ownership to the tree (zero-copy,
                # in contrast to the fork_from path below).
                mamba_value = req.pending_radix_mamba_slot.clone()
                req.pending_radix_mamba_slot = None
                insert_params.mamba_value = mamba_value
                return cache_len
            # enable_mamba_extra_buffer=False: legacy fork_from path.
            mamba_value = self.cache.req_to_token_pool.get_mamba_indices(
                req.req_pool_idx
            ).unsqueeze(-1)
            mamba_value_forked = self.cache.req_to_token_pool.mamba_pool.fork_from(
                mamba_value
            )
            if mamba_value_forked is None:
                self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
                mamba_value_forked = self.cache.req_to_token_pool.mamba_pool.fork_from(
                    mamba_value
                )
                assert mamba_value_forked is not None, "Can not alloc mamba cache"
            insert_params.mamba_value = mamba_value_forked
            return cache_len

    def cleanup_after_caching_req(
        self,
        req: Req,
        is_finished: bool,
        insert_result: Optional[InsertResult] = None,
        insert_params: Optional[InsertParams] = None,
    ) -> None:
        if is_finished:
            mamba_exist = (
                insert_result.mamba_exist if insert_result is not None else True
            )
            if mamba_exist and self.enable_mamba_extra_buffer:
                # Tree already holds mamba state for this prefix (or the
                # insert was a no-op); the pre-allocated pending slot is
                # no longer needed.
                if req.pending_radix_mamba_slot is not None:
                    self.cache.req_to_token_pool.mamba_pool.free(
                        req.pending_radix_mamba_slot
                    )
                    req.pending_radix_mamba_slot = None
            free_mamba_cache = True if self.enable_mamba_extra_buffer else mamba_exist
            if free_mamba_cache:
                # Frees the working slot and any remaining pending slot.
                self.cache.req_to_token_pool.free_mamba_cache(req)
        else:
            if insert_params.mamba_value is not None and (
                insert_result is None or insert_result.mamba_exist
            ):
                self.cache.req_to_token_pool.mamba_pool.free(insert_params.mamba_value)
            req.mamba_last_track_seqlen = None
            # Producer-side: pre-allocate the next pending slot off the
            # forward hot path so the next track interval can write into it
            # directly. Evict once on failure; leave as None if the pool
            # remains exhausted (tracking will be skipped for that interval).
            if self.enable_mamba_extra_buffer and req.pending_radix_mamba_slot is None:
                pending = self.cache.req_to_token_pool.mamba_pool.alloc(1)
                if pending is None:
                    self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
                    pending = self.cache.req_to_token_pool.mamba_pool.alloc(1)
                if pending is not None:
                    req.pending_radix_mamba_slot = pending
