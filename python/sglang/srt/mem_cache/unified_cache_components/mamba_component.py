from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

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
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    TreeComponent,
    get_and_increase_time_counter,
)
from sglang.srt.runtime_context import get_server_args

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
        self.enable_mamba_extra_buffer_lazy = params.enable_mamba_extra_buffer_lazy
        # HiCache state
        self._mamba_pool_host = None  # set to host mamba pool when HiCache enabled

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        ct = self.component_type
        if match_device_only:
            return lambda node: node.component_data[ct].value is not None

        # HiCache: evicted + backuped (host_value present) is also a valid match
        return lambda node: (
            node.component_data[ct].value is not None
            or node.component_data[ct].host_value is not None
        )

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        cow_mamba = params.cow_mamba
        req = params.req
        last_node = result.best_match_node

        # Full may extend beyond the latest Mamba state. Both lengths use
        # logical matches: device only without HiCache, device + host with it.
        full_hit_len = result.full_kv_hierarchical_hit_length
        mamba_boundary_len = (
            len(result.device_indices) + result.host_hit_length
        )
        chunk_size = get_server_args().mamba_cache_chunk_size
        aligned_seqlen = (full_hit_len // chunk_size) * chunk_size
        branching_seqlen = (
            aligned_seqlen
            if aligned_seqlen > mamba_boundary_len
            else None
        )

        mamba_value = last_node.component_data[self.component_type].value
        if cow_mamba and mamba_value is not None:
            assert req is not None
            if req.mamba_pool_idx is None:
                dst_index = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
                if dst_index is None:
                    # Capture the inc result and thread swa_uuid_for_lock back
                    # into dec. Without it, SWA's release walks past this
                    # request's window boundary all the way to root and
                    # over-decrements SWA locks held by other resident requests
                    # on ancestor nodes.
                    lock_result = self.cache.inc_lock_ref(last_node)
                    self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
                    dst_index = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
                    self.cache.dec_lock_ref(last_node, lock_result.to_dec_params())
                    assert dst_index is not None, "Can not alloc mamba cache"
                req.mamba_pool_idx = dst_index[0]
            req.mamba_cow_src_index = mamba_value
            req.mamba_needs_clear = False

        # HiCache: if mamba was evicted from device but has host backup,
        # ensure mamba_host_hit_length >= 1 so load_back is triggered.
        cd = last_node.component_data[self.component_type]
        if cd.value is None and cd.host_value is not None:
            result = result._replace(
                mamba_host_hit_length=max(result.mamba_host_hit_length, 1)
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
            # move from host LRU to device LRU
            host_lru = self.cache.host_lru_lists[self.component_type]
            if host_lru.in_list(node):
                host_lru.remove_node(node)
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
        ct = self.component_type
        new_parent.component_data[ct].value = None
        new_parent.component_data[ct].lock_ref = 0
        # HiCache: mamba host_value stays on child (mamba = leaf-only data)
        new_parent.component_data[ct].host_value = None
        new_parent.component_data[ct].host_lock_ref = 0

    def evict_component(
        self,
        node: UnifiedTreeNode,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        cd = node.component_data[self.component_type]
        freed = 0
        host_freed = 0

        # Device layer
        if EvictLayer.DEVICE in target and cd.value is not None:
            self.cache.req_to_token_pool.mamba_allocator.free(cd.value)
            freed = len(cd.value)
            self.cache.component_evictable_size_[self.component_type] -= freed
            cd.value = None

        # Host layer
        host_lru = self.cache.host_lru_lists[self.component_type]
        if EvictLayer.HOST in target and cd.host_value is not None:
            host_freed = len(cd.host_value)
            if self._mamba_pool_host is not None:
                self._mamba_pool_host.free(cd.host_value)
            cd.host_value = None
            if host_lru.in_list(node):
                host_lru.remove_node(node)

        # After device tombstone: if only host_value remains, insert into host LRU
        if (
            target is EvictLayer.DEVICE
            and cd.value is None
            and cd.host_value is not None
        ):
            if not host_lru.in_list(node):
                host_lru.insert_mru(node)

        return freed, host_freed

    def drive_eviction(
        self, params: EvictParams, tracker: dict[ComponentType, int]
    ) -> None:
        request = params.mamba_num
        ct = self.component_type
        lru = self.cache.lru_lists[ct]
        x = lru.get_lru_no_lock()
        while tracker[ct] < request and x is not None and lru.in_list(x):
            assert x.component_data[ct].value is not None
            if x in self.cache.evictable_device_leaves:
                # D-leaf: atomic eviction of all components
                x_next = lru.get_prev_no_lock(x)
                self.cache._evict_device_leaf(x, tracker)
                if not lru.in_list(x_next):
                    x_next = lru.get_lru_no_lock()
                x = x_next
            else:
                # Internal: tombstone Mamba + cascade
                x_next = lru.get_prev_no_lock(x)
                self.cache._evict_component_and_detach_lru(
                    x, self, target=EvictLayer.DEVICE, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker)
                x = x_next

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        ct = self.component_type
        if node is self.cache.root_node:
            return result
        cd = node.component_data[ct]
        value = cd.host_value if lock_host else cd.value
        # A node in skip_lock_node_ids was a tombstone when this lock was acquired.
        if value is None:
            result.skip_lock_node_ids.setdefault(ct, set()).add(node.id)
            return result

        if lock_host:
            if cd.host_lock_ref == 0:
                host_lru = self.cache.host_lru_lists[ct]
                if host_lru.in_list(node):
                    host_lru.remove_node(node)
            cd.host_lock_ref += 1
        else:
            if cd.lock_ref == 0:
                vlen = len(value)
                self.cache.component_evictable_size_[ct] -= vlen
                self.cache.component_protected_size_[ct] += vlen
            cd.lock_ref += 1
        return result

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: Optional[DecLockRefParams],
        lock_host: bool = False,
    ) -> None:
        ct = self.component_type
        if node is self.cache.root_node:
            return
        cd = node.component_data[ct]
        skip_lock_node_ids = params.skip_lock_node_ids.get(ct, ()) if params else ()
        if node.id in skip_lock_node_ids:
            return

        value = cd.host_value if lock_host else cd.value
        if lock_host:
            cd.host_lock_ref -= 1
            if cd.host_lock_ref == 0 and cd.value is None and cd.host_value is not None:
                host_lru = self.cache.host_lru_lists[ct]
                if not host_lru.in_list(node):
                    host_lru.insert_mru(node)
            return

        if cd.lock_ref > 0:
            if cd.lock_ref == 1:
                vlen = len(value)
                self.cache.component_evictable_size_[ct] += vlen
                self.cache.component_protected_size_[ct] -= vlen
            cd.lock_ref -= 1

    def _alloc_mamba_slot(self) -> torch.Tensor:
        """Allocate one mamba pool slot, evicting if necessary."""
        slot = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
        if slot is None:
            self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
            slot = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
            assert slot is not None, "Can not alloc mamba cache"
        return slot

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
                keep_idx = self.cache.req_to_token_pool.get_mamba_ping_pong_keep_idx(
                    req
                )
                mamba_value = (
                    req.mamba_ping_pong_track_buffer[keep_idx].unsqueeze(-1).clone()
                )
            else:
                mamba_value = req.mamba_pool_idx.unsqueeze(-1).clone()
            insert_params.mamba_value = mamba_value
            return cache_len
        else:
            if cache_len is None:
                return 0
            # Donate the mamba index to the radix cache instead of copying.
            if self.enable_mamba_extra_buffer:
                new_slot = self._alloc_mamba_slot()
                mamba_value_donated = (
                    self.cache.req_to_token_pool.donate_mamba_ping_pong_slot(
                        req, new_slot
                    )
                )
            else:
                mamba_value_donated = self._alloc_mamba_slot()
                # mamba_pool is a pure PHYSICAL store; translate both slot ids
                # virtual->physical (identity for the non-unified memory pool) first.
                translate = self.cache.req_to_token_pool.translate_mamba_indices
                self.cache.req_to_token_pool.mamba_pool.copy_from(
                    translate(req.mamba_pool_idx.unsqueeze(0)),
                    translate(mamba_value_donated),
                )
            insert_params.mamba_value = mamba_value_donated
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
            if self.enable_mamba_extra_buffer:
                keep_idx = self.cache.req_to_token_pool.get_mamba_ping_pong_keep_idx(
                    req
                )
            else:
                keep_idx = None
            if mamba_exist:
                keep_idx = None
            free_mamba_cache = True if self.enable_mamba_extra_buffer else mamba_exist
            if free_mamba_cache:
                self.cache.req_to_token_pool.free_mamba_cache(
                    req, mamba_ping_pong_track_buffer_to_keep=keep_idx
                )
        else:
            if insert_params.mamba_value is not None and (
                insert_result is None or insert_result.mamba_exist
            ):
                self.cache.req_to_token_pool.mamba_allocator.free(
                    insert_params.mamba_value
                )
            req.mamba_last_track_seqlen = None

    # ---- HiCache Hooks ----

    def build_hicache_transfers(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        *,
        req: Optional[Req] = None,
        token_ids: Optional[Sequence[int]] = None,
        prefetch_tokens: int = 0,
        last_hash: Optional[str] = None,
    ) -> Optional[list[PoolTransfer]]:
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            cd = node.component_data[ct]
            if cd.value is None:
                return None
            return [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    device_indices=cd.value,
                )
            ]

        if phase == CacheTransferPhase.LOAD_BACK:
            transfers: list[PoolTransfer] = []

            cd = node.component_data[ct]
            if cd.value is not None:
                return None

            # restore single node if host_value exists
            if cd.host_value is not None and cd.value is None:
                transfers.append(
                    PoolTransfer(
                        name=PoolName.MAMBA,
                        host_indices=cd.host_value,
                        nodes_to_load=[node],
                    )
                )

            # Per-request mamba CoW (H→D copy into request's device slot)
            cd = node.component_data[ct]
            if req is not None and cd.host_value is not None:
                if req.mamba_pool_idx is None:
                    dst = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
                    if dst is None:
                        self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
                        dst = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
                        assert dst is not None, "Cannot alloc mamba for load_back"
                    req.mamba_pool_idx = dst[0]
                transfers.append(
                    PoolTransfer(
                        name=PoolName.MAMBA,
                        host_indices=cd.host_value,
                        device_indices=req.mamba_pool_idx.unsqueeze(0),
                    )
                )

            return transfers if transfers else None

        if phase == CacheTransferPhase.BACKUP_STORAGE:
            cd = node.component_data[ct]
            if cd.host_value is None or not node.hash_value:
                return None
            return [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    host_indices=cd.host_value,
                    keys=[node.hash_value[-1]],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ]

        if phase == CacheTransferPhase.PREFETCH:
            host_indices = self._mamba_pool_host.alloc(1)
            if host_indices is None:
                self.cache.evict_host(1, ComponentType.MAMBA)
                host_indices = self._mamba_pool_host.alloc(1)
            if host_indices is None:
                return []
            return [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    host_indices=host_indices,
                    keys=["__placeholder__"],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
            ]

        return None

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
        *,
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            if transfers and transfers[0].host_indices is not None:
                cd = node.component_data[ct]
                if cd.host_value is None:
                    cd.host_value = transfers[0].host_indices.clone()

        elif phase == CacheTransferPhase.LOAD_BACK:
            if not transfers:
                return
            transfer = transfers[0]
            if transfer.device_indices is not None:
                cd = node.component_data[ct]
                cd.value = transfer.device_indices.clone()
                count = len(cd.value)
                # Move from host LRU to device LRU
                host_lru = self.cache.host_lru_lists[ct]
                if host_lru.in_list(node):
                    host_lru.remove_node(node)
                self.cache.lru_lists[ct].insert_mru(node)
                self.cache.component_evictable_size_[ct] += count

        elif phase == CacheTransferPhase.PREFETCH:
            if not transfers:
                return
            transfer = transfers[0]
            host_indices = transfer.host_indices
            loaded = (
                pool_storage_result is not None
                and pool_storage_result.extra_pool_hit_pages.get(PoolName.MAMBA, 0) >= 1
            )
            target_node = (
                insert_result.inserted_host_node if insert_result is not None else None
            )
            if (
                host_indices is None
                or target_node is None
                or not loaded
                or target_node.component_data[ct].host_value is not None
            ):
                self.cache.cache_controller.append_host_mem_release(
                    extra_pools=[transfer]
                )
                if insert_result is not None:
                    insert_result.mamba_exist = True
                return

            target_node.component_data[ct].host_value = host_indices.clone()
            if target_node.component_data[ct].value is None:
                host_lru = self.cache.host_lru_lists[ct]
                if not host_lru.in_list(target_node):
                    host_lru.insert_mru(target_node)
            if insert_result is not None:
                insert_result.mamba_exist = False

    def drive_host_eviction(
        self, num_tokens: int, tracker: dict[ComponentType, int]
    ) -> None:
        """Evict mamba host resources.
        Internal nodes: private tombstone (free host mamba only).
        Host leaves: atomic eviction via _evict_host_leaf."""
        ct = self.component_type
        host_lru = self.cache.host_lru_lists[ct]
        x = host_lru.get_lru_no_host_lock()
        while tracker[ct] < num_tokens and x is not None and host_lru.in_list(x):
            x_next = host_lru.get_prev_no_host_lock(x)
            cd = x.component_data[ct]
            if x in self.cache.evictable_host_leaves:
                # Host leaf: atomic eviction (all components host + delete)
                self.cache._evict_host_leaf(x, tracker)
            else:
                # Internal: tombstone Mamba + cascade
                assert cd.host_value is not None
                self.cache._evict_component_and_detach_lru(
                    x, self, target=EvictLayer.HOST, tracker=tracker
                )
                self.cache._cascade_evict(x, self, tracker, target=EvictLayer.HOST)
                self.cache._update_evictable_leaf_sets(x)
            x = x_next
