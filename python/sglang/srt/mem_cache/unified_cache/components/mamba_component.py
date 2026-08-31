from __future__ import annotations

from collections import defaultdict
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
from sglang.srt.mem_cache.unified_cache.cache_action import (
    FreeComponentDeviceSlot,
    FreeComponentHostSlot,
    MambaEvictExcessPathStates,
)
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    LinkerTransferPhase,
    LRURefreshPhase,
    PrepareLoadBackResult,
    PreparePrefetchResult,
    TreeComponent,
    get_and_increase_time_counter,
)
from sglang.srt.runtime_context import (
    get_exec,
    mamba_cache_chunk_size,
    mamba_checkpoint_grid,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )
    from sglang.srt.mem_cache.unified_radix_cache import (
        NodeId,
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


class MambaComponent(TreeComponent):
    component_type = ComponentType.MAMBA

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

        assert isinstance(
            params.req_to_token_pool, HybridReqToTokenPool
        ), f"MambaComponent requires HybridReqToTokenPool, got {type(params.req_to_token_pool)}"
        if not params.enable_mamba_extra_buffer:
            assert (
                params.page_size == 1
            ), f"MambaComponent requires page_size=1 when mamba_extra_buffer is disabled, got {params.page_size}"
        super().__init__(cache, params)
        self.mamba_cache_chunk_size = mamba_cache_chunk_size()
        # params.page_size is the tree page the allocator actually uses, already
        # widened by dcp_size, so it is the one grid a checkpoint depth can land on.
        self.mamba_checkpoint_grid = mamba_checkpoint_grid(params.page_size)
        self.mamba_max_states_per_path = get_exec().mamba.mamba_max_states_per_path
        # HiCache state
        self._mamba_pool_host = None  # set to host mamba pool when HiCache enabled

    def needs_incremental_backup(self, node: UnifiedTreeNode) -> bool:
        data = node.component_data[self.component_type]
        return data.value is not None and data.host_value is None

    def _inc_session_coverage(self, session_id: str, leaf: UnifiedTreeNode) -> None:
        cd = leaf.component_data[self.component_type]
        cd.session_ref += 1
        if cd.session_ref == 1:
            self._refresh_session_partition(leaf)

    def _dec_session_coverage(self, session_id: str, leaf: UnifiedTreeNode) -> None:
        cd = leaf.component_data[self.component_type]
        assert cd.session_ref > 0
        cd.session_ref -= 1
        if cd.session_ref == 0:
            self._refresh_session_partition(leaf)

    def _advance_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        old_ancestor: Optional[UnifiedTreeNode],
    ) -> None:
        self._inc_session_coverage(session_id, leaf)
        if old_ancestor is not None:
            self._dec_session_coverage(session_id, old_ancestor)

    def _recede_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        fallback: Optional[UnifiedTreeNode],
    ) -> None:
        self._dec_session_coverage(session_id, leaf)
        if fallback is not None:
            self._inc_session_coverage(session_id, fallback)

    def refresh_lru(
        self,
        phase: LRURefreshPhase,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
    ) -> None:
        # A match consumes only best_match_node's mamba state (cf. inc_lock_ref,
        # which locks just this node's mamba value), unlike Full whose whole matched
        # path is reused as prefix. Refreshing ancestors would keep a whole session's
        # states adjacent in the mamba LRU and evict cold sessions wholesale, so touch
        # only the used state. New leaf states enter the LRU via
        # commit_insert_component_data, so the insert walk (WALKDOWN) is a no-op here.
        ct = self.component_type
        match phase:
            case LRURefreshPhase.WALKDOWN:
                return
            case LRURefreshPhase.MATCH_END:
                if node.component_data[ct].value is not None:
                    self.tree_core.lru_lists[ct].reset_node_mru(node)
            case LRURefreshPhase.INSERT_END:
                return
            case _:
                raise ValueError(f"Unknown LRURefreshPhase: {phase}")

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

    def finalize_match_result_in_tree_core(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        last_node = result.best_match_node

        mamba_boundary_len = len(result.device_indices) + result.host_hit_length

        # Full KV may extend beyond the latest reusable Mamba state. The branching
        # point is the last Mamba-cache-chunk-aligned position within the Full-KV hit
        # that lies beyond the current Mamba boundary. With HiCache, incremental
        # persistence of a new branching state is currently write-through only;
        # write-back eviction may discard the device-only state.
        aligned_seqlen = (
            result.full_kv_hit_length // self.mamba_checkpoint_grid
        ) * self.mamba_checkpoint_grid
        branching_seqlen = (
            aligned_seqlen if aligned_seqlen > mamba_boundary_len else None
        )

        # HiCache: if mamba was evicted from device but has host backup,
        # ensure mamba_host_hit_length >= 1 so load_back is triggered.
        if self.has_host_value_only(last_node):
            result = result._replace(
                mamba_host_hit_length=max(result.mamba_host_hit_length, 1)
            )

        return result._replace(mamba_branching_seqlen=branching_seqlen)

    def finalize_match_result_in_cache(
        self, params: MatchPrefixParams, result: MatchResult
    ) -> MatchResult:
        # Copy-on-write the matched device mamba state into a per-request slot.
        if not params.cow_mamba:
            return result
        src_index = self.tree_core.get_component_device_value(
            result.best_match_node, self.component_type
        )
        if src_index is None:
            return result
        req = params.req
        assert req is not None
        if not req.kv.holds_mamba:
            dst_index = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
            if dst_index is None:
                # Pin the window via inc/dec_lock_ref so evict's SWA release
                # stops at this request's window boundary instead of walking to
                # root and over-decrementing locks held by other requests.
                lock_result = self.cache.inc_lock_ref(result.best_match_node)
                self.cache.evict_for_alloc(EvictParams(num_tokens=0, mamba_num=1))
                dst_index = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
                self.cache.dec_lock_ref(
                    result.best_match_node, lock_result.to_dec_params()
                )
                assert dst_index is not None, "Can not alloc mamba cache"
            req.kv.mamba_pool_idx = dst_index[0]
        req.kv.mamba_cow_src_index = src_index
        req.kv.mamba_needs_clear = False
        return result

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        assert params.mamba_value is not None
        if is_new_leaf:
            node.component_data[self.component_type].value = params.mamba_value
            self.tree_core.lru_lists[self.component_type].insert_mru(node)
            self.tree_core.component_evictable_size_[self.component_type] += len(
                params.mamba_value
            )
            self._emit_excess_path_states_eviction(node, cache_actions)
            return
        if node.component_data[self.component_type].value is None:
            node.component_data[self.component_type].value = params.mamba_value
            # move from host LRU to device LRU
            host_lru = self.tree_core.host_lru_lists[self.component_type]
            if host_lru.in_list(node):
                host_lru.remove_node(node)
            self.tree_core.lru_lists[self.component_type].insert_mru(node)
            self.tree_core.component_evictable_size_[self.component_type] += len(
                params.mamba_value
            )
            node.last_access_time = get_and_increase_time_counter()
            self._emit_excess_path_states_eviction(node, cache_actions)
            return
        self.tree_core.lru_lists[self.component_type].reset_node_mru(node)
        node.last_access_time = get_and_increase_time_counter()
        result.mamba_exist = True

    def _emit_excess_path_states_eviction(
        self,
        tail: UnifiedTreeNode,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        """Defer the path-cap eviction so it runs after the insert's BackupKV."""
        if self.mamba_max_states_per_path < 0:
            return
        cache_actions.append(MambaEvictExcessPathStates(tail.id))

    def _evict_excess_path_states(
        self,
        tail: UnifiedTreeNode,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Evict shallow eligible device checkpoints beyond the path cap.

        Full KV and any existing host backup are retained. The tail, forks,
        locked nodes (including a pending backup chain's write-through locks),
        and device leaves are preserved, so the cap is a best-effort soft
        limit. Freed slots are collected into the caller's dicts.
        """
        cap = self.mamba_max_states_per_path
        if cap < 0:
            return

        ct = self.component_type
        holders = []
        node = tail
        while node is not None and node is not self.tree_core.root_node:
            if node.component_data[ct].value is not None:
                holders.append(node)
            node = node.parent

        excess = len(holders) - cap
        if excess <= 0:
            return

        tracker = {component: 0 for component in self.cache.tree_components}
        for node in reversed(holders):
            if excess <= 0 or node is tail:
                break
            if node.component_data[ct].lock_ref > 0 or len(node.children) != 1:
                continue
            if node in self.tree_core.evictable_device_leaves:
                continue
            self.tree_core._evict_component_and_detach_lru(
                node,
                self,
                device_frees,
                host_frees,
                target=EvictLayer.DEVICE,
                tracker=tracker,
            )
            self.tree_core._cascade_evict(node, self, tracker, device_frees, host_frees)
            excess -= 1

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        ct = self.component_type
        new_parent.component_data[ct].value = None
        new_parent.component_data[ct].lock_ref = 0
        new_parent.component_data[ct].session_ref = 0
        new_parent.component_data[ct].session_ids = None
        # HiCache: mamba host_value stays on child (mamba = leaf-only data)
        new_parent.component_data[ct].host_value = None
        new_parent.component_data[ct].host_lock_ref = 0

    def evict_component(
        self,
        node: UnifiedTreeNode,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        cd = node.component_data[self.component_type]
        freed = 0
        host_freed = 0

        # Device layer
        if EvictLayer.DEVICE in target and cd.value is not None:
            device_frees[self.component_type].append(cd.value)
            freed = len(cd.value)
            self.tree_core.component_evictable_size_[self.component_type] -= freed
            cd.value = None

        # Host layer
        host_lru = self.tree_core.host_lru_lists[self.component_type]
        if EvictLayer.HOST in target and cd.host_value is not None:
            host_freed = len(cd.host_value)
            host_frees[self.component_type].append(cd.host_value)
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

    def _evict_device_start(self, request_cnt: int) -> None:
        """Begin the device-eviction walk from this component's LRU cursor."""
        self._evict_device_request_cnt = request_cnt
        if self.tree_core.enable_session_radix_cache:
            lru = self.tree_core.lru_lists[self.component_type]
            lru.cursor_begin()
            self._evict_device_cursor = lru.cursor_next()
        else:
            self._evict_device_cursor = self.tree_core.lru_lists[
                self.component_type
            ].get_lru_no_lock()

    def _evict_device_next_node(
        self,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> Optional[NodeId]:
        """Advance one device-eviction step and return a leaf, if selected.

        An internal tombstone is one complete step so the caller can apply its
        pending frees and recheck allocator capacity before the next mutation.
        If the previous node's eviction removed the cursor, the walk resumes
        from the partition sentinel with session refs on, else it restarts at
        the LRU tail.
        """
        ct = self.component_type
        lru = self.tree_core.lru_lists[ct]
        enabled = self.tree_core.enable_session_radix_cache
        if self._evict_device_cursor is not None and not lru.in_list(
            self._evict_device_cursor
        ):
            self._evict_device_cursor = (
                lru.cursor_next() if enabled else lru.get_lru_no_lock()
            )
        if (
            tracker[ct] >= self._evict_device_request_cnt
            or self._evict_device_cursor is None
            or not lru.in_list(self._evict_device_cursor)
        ):
            return None

        x = self._evict_device_cursor
        assert x.component_data[ct].value is not None
        if x in self.tree_core.evictable_device_leaves and (
            not enabled or self._can_evict_leaf_atomically(x)
        ):
            self._evict_device_cursor = (
                lru.cursor_next() if enabled else lru.get_prev_no_lock(x)
            )
            return x.id
        if not enabled:
            x_next = lru.get_prev_no_lock(x)
        self.tree_core._evict_component_and_detach_lru(
            x,
            self,
            target=EvictLayer.DEVICE,
            tracker=tracker,
            device_frees=device_frees,
            host_frees=host_frees,
        )
        self.tree_core._cascade_evict(
            x, self, tracker, device_frees=device_frees, host_frees=host_frees
        )
        self._evict_device_cursor = lru.cursor_next() if enabled else x_next
        return None

    def _evict_device_end(self) -> None:
        """Clear the device-eviction walk cursor state."""
        if self.tree_core.enable_session_radix_cache:
            self.tree_core.lru_lists[self.component_type].cursor_end()
        self._evict_device_cursor = None

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        ct = self.component_type
        if node is self.tree_core.root_node:
            return result
        cd = node.component_data[ct]
        value = cd.host_value if lock_host else cd.value
        # A node in skip_lock_node_ids was a tombstone when this lock was acquired.
        if value is None:
            result.skip_lock_node_ids.setdefault(ct, set()).add(node.id)
            return result

        if lock_host:
            if cd.host_lock_ref == 0:
                host_lru = self.tree_core.host_lru_lists[ct]
                if host_lru.in_list(node):
                    host_lru.remove_node(node)
            cd.host_lock_ref += 1
        else:
            if cd.lock_ref == 0:
                vlen = len(value)
                self.tree_core.component_evictable_size_[ct] -= vlen
                self.tree_core.component_protected_size_[ct] += vlen
            cd.lock_ref += 1
        return result

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: Optional[DecLockRefParams],
        lock_host: bool = False,
    ) -> None:
        ct = self.component_type
        if node is self.tree_core.root_node:
            return
        cd = node.component_data[ct]
        skip_lock_node_ids = params.skip_lock_node_ids.get(ct, ()) if params else ()
        if node.id in skip_lock_node_ids:
            return

        value = cd.host_value if lock_host else cd.value
        if lock_host:
            cd.host_lock_ref -= 1
            if cd.host_lock_ref == 0 and cd.value is None and cd.host_value is not None:
                host_lru = self.tree_core.host_lru_lists[ct]
                if not host_lru.in_list(node):
                    host_lru.insert_mru(node)
            return

        if cd.lock_ref > 0:
            if cd.lock_ref == 1:
                vlen = len(value)
                self.tree_core.component_evictable_size_[ct] += vlen
                self.tree_core.component_protected_size_[ct] -= vlen
            cd.lock_ref -= 1

    def _alloc_mamba_slot(self) -> torch.Tensor:
        """Allocate one mamba pool slot, evicting if necessary."""
        slot = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
        if slot is None:
            self.cache.evict_for_alloc(EvictParams(num_tokens=0, mamba_num=1))
            slot = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
            assert slot is not None, "Can not alloc mamba cache"
        return slot

    @property
    def int8_ckpt_pool(self):
        return getattr(self.cache.req_to_token_pool, "mamba_ckpt_pool", None)

    def _alloc_int8_ckpt_slot(self) -> torch.Tensor:
        slot = self.int8_ckpt_pool.alloc(1)
        if slot is None:
            self.cache.evict(EvictParams(num_tokens=0, mamba_num=1))
            slot = self.int8_ckpt_pool.alloc(1)
            assert slot is not None, "Can not alloc int8 mamba checkpoint slot"
        return slot

    def _commit_int8_checkpoint(self, active_slots: torch.Tensor) -> torch.Tensor:
        ckpt_slot = self._alloc_int8_ckpt_slot()
        self.int8_ckpt_pool.store_from_active(
            self.cache.req_to_token_pool.mamba_pool,
            active_slots.view(-1),
            ckpt_slot,
        )
        return ckpt_slot

    def _free_mamba_value(self, mamba_value: torch.Tensor) -> None:
        if self.int8_ckpt_pool is not None:
            self.int8_ckpt_pool.free(mamba_value)
        else:
            self.cache.req_to_token_pool.mamba_allocator.free(mamba_value)

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        if self.cache.enable_mamba_extra_buffer:
            cache_len = req.kv.mamba_last_track_seqlen
        else:
            cache_len = token_ids_len
            # ReplaySSM (no_buffer): `temporal[slot]` lags the live state by the
            # slot's unflushed ring depth (`write_pos`), so on request finish cap
            # the donate to the last flush boundary (where temporal is current)
            # and reset the cursor, keeping the donated checkpoint consistent with
            # its key length. page_size is asserted == 1, so no realign. Mirrors
            # MambaRadixCache.cache_finished_req.
            if is_finished:
                write_pos_buf = (
                    self.cache.req_to_token_pool.mamba_pool.replayssm_write_pos
                )
                if write_pos_buf is not None:
                    cache_len -= int(write_pos_buf[req.kv.mamba_pool_idx].item())
                    write_pos_buf[req.kv.mamba_pool_idx] = 0

        if is_finished:
            if cache_len is None:
                cache_len = 0
            if self.cache.enable_mamba_extra_buffer:
                keep_idx = self.cache.req_to_token_pool.get_mamba_ping_pong_keep_idx(
                    req
                )
                active_value = (
                    req.kv.mamba_ping_pong_track_buffer[keep_idx].unsqueeze(-1).clone()
                )
            else:
                active_value = req.kv.mamba_pool_idx.unsqueeze(-1).clone()
            if self.int8_ckpt_pool is not None:
                insert_params.mamba_value = self._commit_int8_checkpoint(active_value)
            else:
                insert_params.mamba_value = active_value
            return cache_len
        else:
            if cache_len is None:
                return 0
            # Donate the mamba index to the radix cache instead of copying.
            if self.int8_ckpt_pool is not None:
                if self.cache.enable_mamba_extra_buffer:
                    new_slot = self._alloc_mamba_slot()
                    src_active = (
                        self.cache.req_to_token_pool.donate_mamba_ping_pong_slot(
                            req, new_slot
                        )
                    )
                    mamba_value_donated = self._commit_int8_checkpoint(src_active)
                    self.cache.req_to_token_pool.mamba_allocator.free(src_active)
                else:
                    mamba_value_donated = self._commit_int8_checkpoint(
                        req.kv.mamba_pool_idx.view(-1)
                    )
            elif self.cache.enable_mamba_extra_buffer:
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
                    translate(req.kv.mamba_pool_idx.unsqueeze(0)),
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
            mamba_value_inserted = (
                insert_result is not None and not insert_result.mamba_exist
            )
            pool = self.cache.req_to_token_pool

            if self.int8_ckpt_pool is not None:
                insert_value_unused = (
                    not mamba_value_inserted
                    and insert_params is not None
                    and insert_params.mamba_value is not None
                )
                if insert_value_unused:
                    self._free_mamba_value(insert_params.mamba_value)
                pool.free_mamba_cache(req)
                return

            if self.cache.enable_mamba_extra_buffer:
                keep_idx = (
                    pool.get_mamba_ping_pong_keep_idx(req)
                    if mamba_value_inserted
                    else None
                )
                pool.free_mamba_cache(
                    req, mamba_ping_pong_track_buffer_to_keep=keep_idx
                )
                return

            if not mamba_value_inserted:
                pool.free_mamba_cache(req)
        else:
            if insert_params.mamba_value is not None and (
                insert_result is None or insert_result.mamba_exist
            ):
                self._free_mamba_value(insert_params.mamba_value)
            req.kv.mamba_last_track_seqlen = None

    def build_external_linker_transfer(
        self,
        phase: LinkerTransferPhase,
        node: Optional[UnifiedTreeNode],
        keys: Optional[Sequence[str]],
    ) -> Optional[PoolTransfer]:
        raise AssertionError(
            "MambaComponent does not support external linker mode, will support soon"
        )

    # ---- HiCache Hooks ----

    def prepare_load_back(
        self,
        node_id: NodeId,
        *,
        req: Optional[Req] = None,
    ) -> PrepareLoadBackResult:
        if (
            req is None
            or req.kv.holds_mamba
            or not self.tree_core.component_has_host_value_only(
                node_id, self.component_type
            )
        ):
            return PrepareLoadBackResult()
        dst = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
        if dst is None:
            self.cache.evict_for_alloc(EvictParams(num_tokens=0, mamba_num=1))
            dst = self.cache.req_to_token_pool.mamba_allocator.alloc(1)
            assert dst is not None, "Cannot alloc mamba for load_back"
        req.kv.mamba_pool_idx = dst[0]
        return PrepareLoadBackResult(allocated_mamba_slot=dst)

    def finalize_load_back(
        self, req: Optional[Req], prep: PrepareLoadBackResult, success: bool
    ) -> None:
        # A called-off load-back returns the slot prepare allocated and clears req (the H->D copy never ran).
        if not success and prep.allocated_mamba_slot is not None:
            self.cache.req_to_token_pool.mamba_allocator.free(prep.allocated_mamba_slot)
            req.kv.mamba_pool_idx = None

    def prepare_prefetch(
        self,
        node_id: NodeId,
        *,
        prefetch_tokens: int = 0,
    ) -> PreparePrefetchResult:
        host_indices = self.cache.host_pool_group.alloc(
            1,
            pool=PoolName.MAMBA,
            reclaim=lambda size: self.cache.evict_host(size, ComponentType.MAMBA),
        )
        if host_indices is None:
            return PreparePrefetchResult(alloc_failed=True)
        return PreparePrefetchResult(host_indices=host_indices)

    def build_hicache_transfers(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        *,
        mamba_pool_idx: Optional[torch.Tensor] = None,
        host_indices: Optional[torch.Tensor] = None,
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
                        nodes_to_load=[node.id],
                    )
                )

            # Per-request mamba CoW: H→D copy into the request's device slot pre-allocated on the caller side.
            cd = node.component_data[ct]
            if mamba_pool_idx is not None and cd.host_value is not None:
                transfers.append(
                    PoolTransfer(
                        name=PoolName.MAMBA,
                        host_indices=cd.host_value,
                        device_indices=mamba_pool_idx.unsqueeze(0),
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
            assert host_indices is not None
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
        cache_actions: list[CacheAction | ComponentAction],
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
                host_lru = self.tree_core.host_lru_lists[ct]
                if host_lru.in_list(node):
                    host_lru.remove_node(node)
                self.tree_core.lru_lists[ct].insert_mru(node)
                self.tree_core.component_evictable_size_[ct] += count

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
                self.tree_core.node_by_id(insert_result.inserted_host_node)
                if insert_result is not None
                and insert_result.inserted_host_node is not None
                else None
            )
            if (
                host_indices is None
                or target_node is None
                or not loaded
                or target_node.component_data[ct].host_value is not None
            ):
                cache_actions.append(
                    FreeComponentHostSlot(
                        [host_indices], component_type=ComponentType.MAMBA
                    )
                )
                if insert_result is not None:
                    insert_result.mamba_exist = True
                return

            target_node.component_data[ct].host_value = host_indices.clone()
            if target_node.component_data[ct].value is None:
                host_lru = self.tree_core.host_lru_lists[ct]
                if not host_lru.in_list(target_node):
                    host_lru.insert_mru(target_node)
            if insert_result is not None:
                insert_result.mamba_exist = False

    def drive_host_eviction(
        self,
        num_tokens: int,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Evict mamba host resources.
        Internal nodes: private tombstone (free host mamba only).
        Host leaves: atomic eviction via _evict_host_leaf."""
        ct = self.component_type
        host_lru = self.tree_core.host_lru_lists[ct]
        enabled = self.tree_core.enable_session_radix_cache
        if enabled:
            host_lru.cursor_begin()
            x = host_lru.cursor_next(host_lock=True)
        else:
            x = host_lru.get_lru_no_host_lock()
        while tracker[ct] < num_tokens and x is not None and host_lru.in_list(x):
            if not enabled:
                x_next = host_lru.get_prev_no_host_lock(x)
            cd = x.component_data[ct]
            if x in self.tree_core.evictable_host_leaves and (
                not enabled or self._can_evict_leaf_atomically(x)
            ):
                # Host leaf: atomic eviction (all components host + delete)
                self.tree_core._evict_host_leaf(x, tracker, device_frees, host_frees)
            else:
                # Internal (or a leaf a session still pins): tombstone Mamba + cascade
                assert cd.host_value is not None
                self.tree_core._evict_component_and_detach_lru(
                    x,
                    self,
                    target=EvictLayer.HOST,
                    tracker=tracker,
                    device_frees=device_frees,
                    host_frees=host_frees,
                )
                self.tree_core._cascade_evict(
                    x,
                    self,
                    tracker,
                    device_frees=device_frees,
                    host_frees=host_frees,
                    target=EvictLayer.HOST,
                )
                self.tree_core._update_evictable_leaf_sets(x)
            if enabled:
                x = host_lru.cursor_next(host_lock=True)
            else:
                x = x_next
        if enabled:
            host_lru.cursor_end()

    def free_host_values(self, host_values: list[torch.Tensor]) -> None:
        if self._mamba_pool_host is None:
            return
        for host_value in host_values:
            self.cache.host_pool_group.free(host_value, pool=PoolName.MAMBA)

    def apply_component_action(self, action: ComponentAction) -> None:
        if isinstance(action, MambaEvictExcessPathStates):
            device_frees: dict[ComponentType, list[torch.Tensor]] = defaultdict(list)
            host_frees: dict[ComponentType, list[torch.Tensor]] = defaultdict(list)
            # Drain even if the walk raises so tombstoned slots are not leaked;
            # the walk runs behind the tree-core interface (Rust runs it natively).
            try:
                self.tree_core.evict_excess_path_states(
                    action.tail_node_id, device_frees, host_frees
                )
            finally:
                self.cache._free_values(device_frees, host_frees)
            return
        if isinstance(action, FreeComponentDeviceSlot):
            for indices in action.indices:
                self._free_mamba_value(indices)
            return
        if isinstance(action, FreeComponentHostSlot):
            for host_indices in action.host_indices:
                if host_indices is not None and host_indices.numel() > 0:
                    self.cache.cache_controller.append_host_mem_release(
                        extra_pools=[
                            PoolTransfer(name=PoolName.MAMBA, host_indices=host_indices)
                        ]
                    )
            return
        raise AssertionError(
            f"MambaComponent: unhandled ComponentAction {type(action).__name__}"
        )
