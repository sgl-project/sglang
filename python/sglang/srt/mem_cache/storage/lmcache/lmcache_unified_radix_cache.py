"""Unified radix tree backed by the LMCache multiprocess service.

LMCache owns external storage; retrieved KV is published after reaching GPU.
"""

from __future__ import annotations

import atexit
import logging
from array import array
from typing import TYPE_CHECKING, Any, Optional

import torch
from lmcache.integration.sglang.lmcache_mp_metadata import (
    LMCacheExternalFlow,
    LMCachePendingStore,
)
from lmcache.integration.sglang.unified_lmcache_mp_connector import (
    UnifiedLMCacheMPConnector,
)

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import NodeId, UnifiedRadixCache

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

logger = logging.getLogger(__name__)


class LMCacheUnifiedRadixCache(UnifiedRadixCache):
    """Unified radix tree with device-direct LMCache MP KV/state I/O."""

    def __init__(
        self,
        params: CacheInitParams,
        *,
        model_config: ModelConfig,
        tp_size: int,
        tp_rank: int,
        lmcache_config_file: Optional[str],
        forward_stream: Any,
    ) -> None:
        super().__init__(params)
        self._mamba_component = self._find_mamba_component()
        self.lmcache_connector = UnifiedLMCacheMPConnector(
            config_file=lmcache_config_file,
            model_config=model_config,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=params.tp_cache_group,
            pp_size=params.pp_size,
            pp_rank=params.pp_rank,
            pp_group=params.pp_cache_group,
            page_size=self.page_size,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            tree_components=self.tree_components,
            mamba_component=self._mamba_component,
            sliding_window_size=self._sliding_window_size,
        )
        self._external_flows: dict[str, LMCacheExternalFlow] = {}
        self._forward_stream = forward_stream
        self._pending_stores: list[LMCachePendingStore] = []
        self._pending_store_counts: dict[str, int] = {}
        self._session_finish_requested: set[str] = set()
        self._lmcache_closed = False
        atexit.register(self.shutdown)

    def is_backuped(self, node_id: NodeId) -> bool:
        # LMCache rebuilds lookup keys from tokens, so any L1 node is valid.
        return True

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Report LMCache hits through host fields until GPU slots are ready."""
        requested_key_len = len(params.key)
        req = params.req
        flow = self._external_flows.get(req.rid) if req is not None else None
        if (
            flow is not None
            and flow.total_hit is not None
            and flow.local_hit_tokens is not None
            and flow.load is None
        ):
            # Use the shortest cross-rank L1 prefix for consistent admission.
            params = MatchPrefixParams(
                key=params.key[: flow.local_hit_tokens],
                cow_mamba=params.cow_mamba,
                req=req,
            )
        elif flow is not None and flow.load is not None and params.cow_mamba:
            # Re-arm Mamba COW from the retrieved checkpoint on retry.
            self._arm_external_mamba_cow(flow, req)
            params = MatchPrefixParams(key=params.key, cow_mamba=False, req=req)
        result = super().match_prefix(params)
        if req is None:
            return result
        if flow is None or flow.total_hit is None or flow.cancelled:
            return result

        total_hit = min(flow.total_hit, requested_key_len, len(flow.key))
        local_hit = len(result.device_indices)
        skip = 0
        if flow.load is not None and not flow.prefix_published:
            # Track private slots shadowed by a concurrent tree insertion.
            skip = max(
                min(local_hit, total_hit) - flow.load.local_hit_tokens,
                0,
            )
            flow.loaded_skip_tokens = max(flow.loaded_skip_tokens, skip)
            if flow.load.result:
                self._release_unused_loaded_slots(flow)
        if total_hit <= local_hit:
            return result

        if flow.load is None:
            external_hit = total_hit - local_hit
            return result._replace(
                last_host_node=result.last_device_node,
                best_match_node=result.last_device_node,
                host_hit_length=external_hit,
                swa_host_hit_length=(
                    min(
                        external_hit,
                        self.lmcache_connector.aligned_swa_window_size(),
                    )
                    if self.is_swa_enabled
                    else 0
                ),
                mamba_host_hit_length=(1 if self._mamba_component is not None else 0),
                mamba_branching_seqlen=None,
                full_kv_hit_length=max(result.full_kv_hit_length, total_hit),
            )

        # Reuse an already submitted H2D load on the next admission attempt.
        suffix = flow.load.device_indices[skip : skip + max(total_hit - local_hit, 0)]
        return result._replace(
            device_indices=torch.cat([result.device_indices, suffix]),
            last_host_node=result.last_device_node,
            best_match_node=result.last_device_node,
            host_hit_length=0,
            swa_host_hit_length=0,
            mamba_host_hit_length=0,
            mamba_branching_seqlen=None,
            full_kv_hit_length=max(result.full_kv_hit_length, total_hit),
        )

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node_id: NodeId,
        new_input_tokens: list[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[list[str]] = None,
        matched_prefix_tokens: Optional[list[int]] = None,
        extra_key: Optional[str] = None,
        cache_salt: Optional[str] = None,
    ) -> None:
        del last_hash, prefix_keys
        if req_id in self._external_flows:
            return
        local_tokens = list(matched_prefix_tokens or [])
        token_ids = local_tokens + list(new_input_tokens)
        anchor_extra_key, anchor_cache_salt = self.tree_core.prefetch_anchor_info(
            last_host_node_id
        )
        extra_key = extra_key or anchor_extra_key
        cache_salt = cache_salt or anchor_cache_salt
        key = RadixKey(
            array("q", token_ids),
            extra_key=extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=cache_salt,
        ).page_aligned(self.page_size)
        if len(key) == 0:
            return
        token_ids = key.raw_token_ids()[: len(key)]
        lookup = self.lmcache_connector.submit_lookup(
            req_id,
            token_ids,
            local_hit_tokens=min(len(local_tokens), len(key)),
            cache_salt=self.lmcache_connector.build_cache_salt(cache_salt, extra_key),
        )
        self._external_flows[req_id] = LMCacheExternalFlow(
            key=key,
            lookup=lookup,
        )

    def check_prefetch_progress(self, req_id: str) -> bool:
        flow = self._external_flows.get(req_id)
        if flow is None:
            return True
        if flow.cancelled:
            return False

        if flow.total_hit is None:
            total_hit = self.lmcache_connector.poll_lookup(flow.lookup)
            if total_hit is None:
                return False
            latest = super().match_prefix(MatchPrefixParams(key=flow.key))
            local_hit = torch.tensor(
                [len(latest.device_indices)], dtype=torch.int64, device="cpu"
            )
            self.lmcache_connector.parallel_all_reduce(
                local_hit, torch.distributed.ReduceOp.MIN
            )
            total_hit = min(total_hit, len(flow.key))
            local_hit_tokens = int(local_hit.item())
            release_end = min(
                total_hit,
                local_hit_tokens
                // self.lmcache_connector.chunk_size
                * self.lmcache_connector.chunk_size,
            )
            if release_end > 0:
                # All ranks advance state; only the leader releases remote locks.
                self.lmcache_connector.free_lookup_locks(
                    req_id, start=0, end=release_end
                )
            if total_hit <= local_hit_tokens:
                self._external_flows.pop(req_id, None)
                self.prefetch_loaded_tokens_by_reqid[req_id] = 0
                self.prefetch_loaded_storage_start_by_reqid.pop(req_id, None)
                return True
            flow.total_hit = total_hit
            flow.local_hit_tokens = local_hit_tokens
            self.prefetch_loaded_tokens_by_reqid[req_id] = total_hit - local_hit_tokens
            self.prefetch_loaded_storage_start_by_reqid[req_id] = local_hit_tokens

        return True

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        self.prefetch_loaded_storage_start_by_reqid.pop(req_id, None)
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    def cache_unfinished_req(self, req: Req, chunked: bool = False, **kwargs) -> None:
        self._publish_external_loaded_prefix(req, token_ids_len=len(req.get_fill_ids()))
        super().cache_unfinished_req(req, chunked=chunked, **kwargs)
        self._retire_loaded_flow(req.rid)
        self._submit_store(req, req.get_fill_ids())

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int, **kwargs
    ) -> None:
        if not is_insert:
            self.release_aborted_request(req.rid)
        else:
            self._publish_external_loaded_prefix(req, token_ids_len=kv_len_to_handle)
        super().cache_finished_req(
            req,
            is_insert=is_insert,
            kv_len_to_handle=kv_len_to_handle,
            **kwargs,
        )
        self._retire_loaded_flow(req.rid)
        if is_insert:
            token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
            self._submit_store(req, token_ids)
            self._request_session_finish(req.rid)

    def check_hicache_events(self) -> None:
        """Poll LMCache retrieve/store futures at the scheduler safe point."""
        load_flows = [
            flow
            for flow in self._external_flows.values()
            if flow.load is not None and not flow.load_completed
        ]
        ready_loads = self.lmcache_connector.ready_prefix_count(
            [f.load for f in load_flows]
        )
        for flow in load_flows[:ready_loads]:
            assert flow.load is not None
            success = self.lmcache_connector.complete_load(flow.load)
            flow.load_completed = True
            cancelled = flow.cancelled
            if cancelled or not success:
                self._finish_failed_load(flow)
                if not cancelled:
                    logger.warning(
                        "LMCache retrieve failed after admission for request %s",
                        flow.lookup.request_id,
                    )
            elif flow.retire_requested:
                self._finalize_retired_flow(flow)
            else:
                self._finish_successful_load(flow)

        ready_stores = self.lmcache_connector.ready_prefix_count(
            [pending.operation for pending in self._pending_stores]
        )
        for _ in range(ready_stores):
            pending = self._pending_stores.pop(0)
            rid = pending.operation.request_id
            try:
                self.lmcache_connector.complete_store(pending.operation)
            finally:
                # Always unlock the source node after store completion.
                self.dec_lock_ref(pending.node_id, pending.lock_params)
                remaining = self._pending_store_counts[rid] - 1
                if remaining > 0:
                    self._pending_store_counts[rid] = remaining
                else:
                    self._pending_store_counts.pop(rid)
                self._finish_session_if_store_idle(rid)

    def has_pending_cache_operations(self) -> bool:
        return bool(self._external_flows or self._pending_stores)

    def release_aborted_request(self, rid: str) -> None:
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)
        self.prefetch_loaded_storage_start_by_reqid.pop(rid, None)
        flow = self._external_flows.get(rid)
        if flow is None:
            self._request_session_finish(rid)
            return
        flow.cancelled = True
        # Keep the checkpoint alive until H2D completes.
        if flow.load is not None and flow.mamba_value is not None:
            flow.free_mamba_after_load = True
            if (
                flow.load_req is not None
                and flow.load_req.kv.mamba_cow_src_index is flow.mamba_value
            ):
                flow.load_req.kv.mamba_cow_src_index = None
        if flow.load is None:
            if flow.total_hit is not None:
                # No retrieve will release locks from this completed lookup.
                self._retire_loaded_flow(rid)
            else:
                # END_SESSION follows the in-flight lookup and releases its locks.
                self._external_flows.pop(rid, None)
        elif flow.load_completed:
            self._finish_failed_load(flow)
        else:
            # Complete the cross-rank load only in check_hicache_events().
            flow.retire_requested = True
        self._request_session_finish(rid)

    def init_load_back(self, params: InitLoadBackParams) -> tuple[torch.Tensor, NodeId]:
        req = params.req
        if req is None:
            return (
                self.tree_core.empty_match_result.device_indices,
                params.best_match_node,
            )
        flow = self._external_flows.get(req.rid)
        if flow is None or flow.total_hit is None or flow.load is not None:
            return (
                self.tree_core.empty_match_result.device_indices,
                params.best_match_node,
            )

        device_indices = self._start_external_load(flow, req)
        if device_indices is None:
            req.storage_hit_length = 0
            req.host_hit_length = 0
            req.swa_host_hit_length = 0
            req.mamba_host_hit_length = 0
            self.prefetch_loaded_tokens_by_reqid.pop(req.rid, None)
            self.prefetch_loaded_storage_start_by_reqid.pop(req.rid, None)
            # Retire the unloaded flow to release its lookup locks.
            self._retire_loaded_flow(req.rid)
            return (
                self.tree_core.empty_match_result.device_indices,
                params.best_match_node,
            )
        return device_indices, params.best_match_node

    def ready_to_load_host_cache(self) -> int:
        # H2D is submitted in init_load_back to preserve prefill fallback.
        return -1

    def supports_retraction_backup(self) -> bool:
        # TODO(chunxiaozheng): implement retraction backup
        return False

    def clear_storage_backend(self) -> bool:
        return self.lmcache_connector.clear()

    def reset(self) -> None:
        # The parent constructor may call reset before the connector exists.
        connector = getattr(self, "lmcache_connector", None)
        if connector is not None:
            for flow in list(self._external_flows.values()):
                if flow.load is not None:
                    try:
                        flow.load.future.result(timeout=connector.operation_timeout)
                    except Exception:
                        pass
                    connector.complete_load(flow.load, synchronize=False)
                    flow.load_completed = True
                    if flow.prefix_published:
                        # Free only private slots; the tree owns the published suffix.
                        self._release_unused_loaded_slots(flow)
                    else:
                        self.token_to_kv_pool_allocator.free(
                            flow.load.device_indices[flow.released_skip_tokens :]
                        )
                    if flow.mamba_value is not None:
                        self.req_to_token_pool.mamba_allocator.free(flow.mamba_value)
                        flow.mamba_value = None
                    self._release_flow_anchor(flow)
            for pending in list(self._pending_stores):
                try:
                    pending.operation.future.result(timeout=connector.operation_timeout)
                except Exception:
                    pass
                connector.complete_store(pending.operation, synchronize=False)
                self.dec_lock_ref(pending.node_id, pending.lock_params)
            self._external_flows.clear()
            self._pending_stores.clear()
            self._pending_store_counts.clear()
            self._session_finish_requested.clear()
            self.prefetch_loaded_tokens_by_reqid.clear()
            connector.end_all_sessions()
        super().reset()

    def shutdown(self) -> None:
        if self._lmcache_closed:
            return
        self.reset()
        self.lmcache_connector.close()
        self._lmcache_closed = True

    def release_host_resources(self) -> None:
        self.shutdown()

    def _find_mamba_component(self):
        return next(
            (
                component
                for component in self._components_tuple
                if component.component_type is ComponentType.MAMBA
            ),
            None,
        )

    def _allocate_external_mamba_slot(self) -> Optional[torch.Tensor]:
        if self._mamba_component is None:
            return None
        allocator = self.req_to_token_pool.mamba_allocator
        slot = allocator.alloc(1)
        if slot is None:
            self.evict(EvictParams(num_tokens=0, mamba_num=1))
            slot = allocator.alloc(1)
        if slot is not None:
            physical = self.req_to_token_pool.translate_mamba_indices(slot)
            self.lmcache_connector.reset_mamba_checkpoint_metadata(physical)
        return slot

    def _arm_external_mamba_cow(self, flow: LMCacheExternalFlow, req: Req) -> None:
        """Give a request a mutable state slot and COW from the loaded checkpoint."""
        checkpoint = flow.mamba_value
        if checkpoint is None:
            return
        if req.kv.mamba_pool_idx is None:
            active = self._allocate_external_mamba_slot()
            assert active is not None, "Cannot allocate Mamba request state for LMCache"
            req.kv.mamba_pool_idx = active[0]
            flow.request_mamba_value = active
            flow.allocated_request_mamba_for_load = True
        elif flow.request_mamba_value is None:
            flow.request_mamba_value = req.kv.mamba_pool_idx.reshape(1)
            flow.allocated_request_mamba_for_load = False
        flow.load_req = req
        req.kv.mamba_cow_src_index = checkpoint
        req.kv.mamba_needs_clear = False

    def _allocate_external_slots(self, num_tokens: int) -> Optional[torch.Tensor]:
        allocator = self.token_to_kv_pool_allocator
        if not self.is_swa_enabled:
            if allocator.available_size() < num_tokens:
                self.evict(EvictParams(num_tokens=num_tokens))
            return allocator.alloc(num_tokens)

        # Allocate SWA only for the trailing window; older pages use dummy page 0.
        assert self._sliding_window_size is not None
        swa_tail_tokens = min(
            num_tokens,
            self.lmcache_connector.aligned_swa_window_size(),
        )
        full_allocator = allocator.full_attn_allocator
        swa_allocator = allocator.swa_attn_allocator
        if full_allocator.available_size() < num_tokens:
            self.evict(EvictParams(num_tokens=num_tokens))
        if swa_allocator.available_size() < swa_tail_tokens:
            self.evict(EvictParams(swa_num_tokens=swa_tail_tokens))

        full_indices = full_allocator.alloc(num_tokens)
        if full_indices is None:
            return None
        if swa_tail_tokens == 0:
            return full_indices

        tail_full_indices = full_indices[-swa_tail_tokens:]
        if hasattr(swa_allocator, "alloc_with_virtual"):
            # Unified-memory FULL and SWA share virtual page IDs.
            virtual_pages = torch.unique(tail_full_indices // self.page_size)
            try:
                swa_allocator.alloc_with_virtual(virtual_pages)
            except Exception:
                full_allocator.free(full_indices)
                logger.exception("Failed to allocate unified SWA slots for LMCache")
                return None
        else:
            swa_indices = swa_allocator.alloc(swa_tail_tokens)
            if swa_indices is None:
                full_allocator.free(full_indices)
                return None
            allocator.set_full_to_swa_mapping(tail_full_indices, swa_indices)
        return full_indices

    def _start_external_load(
        self, flow: LMCacheExternalFlow, req: Req
    ) -> Optional[torch.Tensor]:
        assert flow.total_hit is not None
        assert flow.local_hit_tokens is not None
        local_hit = flow.local_hit_tokens
        latest = super().match_prefix(MatchPrefixParams(key=flow.key[:local_hit]))
        total_hit = min(flow.total_hit, len(flow.key))
        if total_hit <= local_hit:
            return self.tree_core.empty_match_result.device_indices

        # Pin the common L1 boundary before allocation can trigger eviction.
        flow.anchor_node = latest.last_device_node
        flow.anchor_lock = self.inc_lock_ref(flow.anchor_node).to_dec_params()
        num_tokens = total_hit - local_hit
        device_indices = self._allocate_external_slots(num_tokens)
        mamba_value = None
        allocated_mamba_for_load = False
        request_mamba_value = None
        allocated_request_mamba_for_load = False
        if device_indices is not None and self._mamba_component is not None:
            # Keep the immutable checkpoint separate from mutable request state.
            mamba_value = self._allocate_external_mamba_slot()
            allocated_mamba_for_load = mamba_value is not None
            if req.kv.mamba_pool_idx is None:
                request_mamba_value = self._allocate_external_mamba_slot()
                allocated_request_mamba_for_load = request_mamba_value is not None
            else:
                request_mamba_value = req.kv.mamba_pool_idx.reshape(1)
        allocation_ok = torch.tensor(
            [
                int(
                    device_indices is not None
                    and (
                        self._mamba_component is None
                        or (mamba_value is not None and request_mamba_value is not None)
                    )
                )
            ],
            dtype=torch.int32,
            device="cpu",
        )
        self.lmcache_connector.parallel_all_reduce(
            allocation_ok, torch.distributed.ReduceOp.MIN
        )
        if not allocation_ok.item():
            if device_indices is not None:
                self.token_to_kv_pool_allocator.free(device_indices)
            if allocated_mamba_for_load and mamba_value is not None:
                self.req_to_token_pool.mamba_allocator.free(mamba_value)
            if allocated_request_mamba_for_load and request_mamba_value is not None:
                self.req_to_token_pool.mamba_allocator.free(request_mamba_value)
            self._release_flow_anchor(flow)
            logger.debug(
                "LMCache retrieve declined for %s: a parallel rank cannot allocate "
                "%d GPU slots",
                flow.lookup.request_id,
                num_tokens,
            )
            return None
        assert device_indices is not None
        flow.mamba_value = mamba_value
        flow.request_mamba_value = request_mamba_value
        flow.allocated_request_mamba_for_load = allocated_request_mamba_for_load
        flow.load_req = req
        flow.free_mamba_after_load = allocated_mamba_for_load
        flow.loaded_skip_tokens = 0
        if allocated_request_mamba_for_load:
            assert request_mamba_value is not None
            req.kv.mamba_pool_idx = request_mamba_value[0]
            req.kv.mamba_needs_clear = False

        try:
            load_start = local_hit // self.lmcache_connector.chunk_size
            load_start *= self.lmcache_connector.chunk_size
            # Submit asynchronously while admission can still fall back to prefill.
            flow.load = self.lmcache_connector.submit_load(
                flow.lookup,
                self.lmcache_connector.device_indices_by_group(
                    device_indices,
                    mamba_value=mamba_value,
                    mamba_transfer_tokens=total_hit - load_start,
                ),
                local_hit_tokens=local_hit,
                owned_device_indices=device_indices,
                producer_stream=self._forward_stream,
            )
            if not self.lmcache_connector.prepare_load_on_stream(
                flow.load, self._forward_stream
            ):
                raise RuntimeError("LMCache server rejected the retrieve request")
            if mamba_value is not None:
                # Forward waits for H2D before deferred Mamba COW runs.
                self._arm_external_mamba_cow(flow, req)
        except Exception:
            self.token_to_kv_pool_allocator.free(device_indices)
            if allocated_mamba_for_load and mamba_value is not None:
                self.req_to_token_pool.mamba_allocator.free(mamba_value)
            if allocated_request_mamba_for_load and request_mamba_value is not None:
                self.req_to_token_pool.mamba_allocator.free(request_mamba_value)
                req.kv.mamba_pool_idx = None
            if req.kv.mamba_cow_src_index is mamba_value:
                req.kv.mamba_cow_src_index = None
            flow.mamba_value = None
            flow.request_mamba_value = None
            flow.allocated_request_mamba_for_load = False
            flow.load_req = None
            flow.free_mamba_after_load = False
            self._release_flow_anchor(flow)
            logger.exception(
                "LMCache retrieve submission failed for %s", flow.lookup.request_id
            )
            flow.load = None
            return None
        return device_indices[flow.loaded_skip_tokens :]

    def _release_flow_anchor(self, flow: LMCacheExternalFlow) -> None:
        assert (flow.anchor_node is None) == (flow.anchor_lock is None), (
            "LMCache flow anchor node and lock must be set together"
        )
        if flow.anchor_node is not None:
            assert flow.anchor_lock is not None
            self.dec_lock_ref(flow.anchor_node, flow.anchor_lock)
        flow.anchor_node = None
        flow.anchor_lock = None

    def _release_unused_loaded_slots(self, flow: LMCacheExternalFlow) -> None:
        """Free retrieved slots shadowed by a longer rank-local L1 prefix."""
        assert flow.load is not None
        release_end = min(flow.loaded_skip_tokens, len(flow.load.device_indices))
        if release_end <= flow.released_skip_tokens:
            return
        self.token_to_kv_pool_allocator.free(
            flow.load.device_indices[flow.released_skip_tokens : release_end]
        )
        flow.released_skip_tokens = release_end

    def _finish_failed_load(self, flow: LMCacheExternalFlow) -> None:
        assert flow.load is not None
        if not flow.prefix_published:
            self.token_to_kv_pool_allocator.free(
                flow.load.device_indices[flow.released_skip_tokens :]
            )
        if flow.free_mamba_after_load and flow.mamba_value is not None:
            self.req_to_token_pool.mamba_allocator.free(flow.mamba_value)
            if (
                flow.load_req is not None
                and flow.load_req.kv.mamba_cow_src_index is flow.mamba_value
            ):
                flow.load_req.kv.mamba_cow_src_index = None
            flow.mamba_value = None
        if (
            not flow.cancelled
            and flow.allocated_request_mamba_for_load
            and flow.request_mamba_value is not None
            and flow.load_req is not None
            and flow.load_req.kv.mamba_pool_idx is not None
            and int(flow.load_req.kv.mamba_pool_idx.item())
            == int(flow.request_mamba_value[0].item())
        ):
            self.req_to_token_pool.mamba_allocator.free(flow.request_mamba_value)
            flow.load_req.kv.mamba_pool_idx = None
        flow.request_mamba_value = None
        flow.allocated_request_mamba_for_load = False
        flow.load_req = None
        flow.free_mamba_after_load = False
        self._release_flow_anchor(flow)
        rid = flow.lookup.request_id
        self._external_flows.pop(rid, None)
        if flow.cancelled:
            self.prefetch_loaded_tokens_by_reqid.pop(rid, None)
        else:
            self.prefetch_loaded_tokens_by_reqid[rid] = 0
        self.prefetch_loaded_storage_start_by_reqid.pop(rid, None)

    def _finish_successful_load(self, flow: LMCacheExternalFlow) -> None:
        """Finish local bookkeeping after LMCache has completed the retrieve."""
        assert flow.load is not None and flow.load_completed and flow.load.result
        self._release_unused_loaded_slots(flow)
        # Keep Mamba state until the cache callback publishes it.

    def _finalize_retired_flow(self, flow: LMCacheExternalFlow) -> None:
        """Release a loaded flow after its cache callback retires it."""
        assert flow.load is not None and flow.load_completed and flow.load.result
        self._finish_successful_load(flow)
        if flow.mamba_value is not None:
            # Free checkpoints from request paths that retire without insertion.
            if (
                flow.load_req is not None
                and flow.load_req.kv.mamba_cow_src_index is flow.mamba_value
            ):
                flow.load_req.kv.mamba_cow_src_index = None
            self.req_to_token_pool.mamba_allocator.free(flow.mamba_value)
            flow.mamba_value = None
            flow.free_mamba_after_load = False
        flow.request_mamba_value = None
        flow.allocated_request_mamba_for_load = False
        flow.load_req = None
        self._release_flow_anchor(flow)
        rid = flow.lookup.request_id
        self._external_flows.pop(rid, None)

    def _retire_loaded_flow(self, rid: str) -> None:
        """Mark a load for retirement without entering a collective."""
        flow = self._external_flows.get(rid)
        if flow is None:
            return
        if flow.load is None:
            if flow.total_hit is None:
                return
            self.lmcache_connector.free_lookup_locks(
                rid,
                start=flow.lookup.lock_start,
                end=flow.total_hit,
            )
            self._external_flows.pop(rid, None)
            return

        flow.retire_requested = True
        if not flow.load_completed:
            return
        if flow.load.result:
            self._finalize_retired_flow(flow)
        else:
            cancelled = flow.cancelled
            self._finish_failed_load(flow)
            if not cancelled:
                logger.warning(
                    "LMCache retrieve failed after admission for request %s", rid
                )

    def _prepare_external_slots_for_insert(self, req: Req) -> None:
        """Restore tree ownership and mark the missing SWA prefix as evicted."""
        flow = self._external_flows.get(req.rid)
        if flow is None or flow.load is None:
            return
        tree_owned_len = flow.load.local_hit_tokens + flow.loaded_skip_tokens
        req.kv.cache_protected_len = min(req.kv.cache_protected_len, tree_owned_len)
        if self.is_swa_enabled:
            external_tokens = len(flow.load.device_indices)
            swa_missing_end = flow.load.local_hit_tokens + max(
                external_tokens - self.lmcache_connector.aligned_swa_window_size(),
                0,
            )
            req.kv.swa_evicted_seqlen = max(req.kv.swa_evicted_seqlen, swa_missing_end)

    def _publish_external_loaded_prefix(self, req: Req, *, token_ids_len: int) -> None:
        """Publish retrieved KV and immutable Mamba state into the device tree."""
        flow = self._external_flows.get(req.rid)
        if (
            flow is None
            or flow.load is None
            or flow.total_hit is None
            or flow.prefix_published
        ):
            return
        # Restore the ownership boundary once before the parent cache callback.
        self._prepare_external_slots_for_insert(req)
        if flow.mamba_value is None:
            # The parent callback will adopt the retrieved Full/SWA slots.
            flow.prefix_published = True
            return

        total_hit = min(flow.total_hit, len(flow.key))
        assert total_hit > 0, "LMCache loaded prefix must contain at least one token"
        if total_hit > token_ids_len:
            raise RuntimeError(
                "LMCache Mamba checkpoint lies beyond the request KV boundary: "
                f"{total_hit}>{token_ids_len} for request {req.rid}"
            )
        prev_prefix_len = min(req.kv.cache_protected_len, total_hit)
        kv_indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :token_ids_len
        ]
        key = flow.key[:total_hit]
        checkpoint = flow.mamba_value
        result = self.insert(
            InsertParams(
                key=key,
                value=kv_indices[:total_hit].to(dtype=torch.int64, copy=True),
                mamba_value=checkpoint,
                prev_prefix_len=prev_prefix_len,
                swa_evicted_seqlen=req.kv.swa_evicted_seqlen,
                chunked=True,
                priority=req.priority or 0,
            )
        )

        # Free our checkpoint if a concurrent insertion already owns this boundary.
        if result.mamba_exist:
            self.req_to_token_pool.mamba_allocator.free(checkpoint)
        flow.mamba_value = None
        flow.free_mamba_after_load = False

        matched = super().match_prefix(
            MatchPrefixParams(key=key, req=req, cow_mamba=False)
        )
        if len(matched.device_indices) < total_hit:
            raise RuntimeError(
                "LMCache loaded prefix was inserted but is not reusable across "
                f"all UnifiedRadixCache components: {len(matched.device_indices)}"
                f"/{total_hit} tokens for request {req.rid}"
            )
        canonical = matched.device_indices[:total_hit]
        self.req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(prev_prefix_len, total_hit)),
            canonical[prev_prefix_len:],
        )

        # Move the request lock to the published boundary.
        if req.last_node is not None:
            self._dec_req_lock(req)
        lock_result = self.inc_lock_ref(matched.last_device_node)
        if total_hit < token_ids_len:
            req.prefix_indices = torch.cat(
                [canonical, kv_indices[total_hit:].to(dtype=torch.int64, copy=True)]
            )
        else:
            req.prefix_indices = canonical
        req.kv.cache_protected_len = total_hit
        req.last_node = matched.last_device_node
        req.lock_receipt = lock_result.to_dec_params()
        req.swa_prefix_lock_released = False
        flow.request_mamba_value = None
        flow.allocated_request_mamba_for_load = False
        flow.load_req = None
        flow.prefix_published = True

    def _submit_store(self, req: Req, token_ids: list[int]) -> None:
        key = RadixKey(
            token_ids,
            req.extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=req.cache_salt,
        ).page_aligned(self.page_size)
        aligned_len = (
            len(key)
            // self.lmcache_connector.chunk_size
            * self.lmcache_connector.chunk_size
        )
        key = key[:aligned_len]
        if len(key) == 0:
            return
        matched = super().match_prefix(MatchPrefixParams(key=key))
        # Store the currently resident SWA prefix; later callbacks may extend it.
        resident_len = torch.tensor(
            [min(len(matched.device_indices), len(key))],
            dtype=torch.int64,
            device="cpu",
        )
        self.lmcache_connector.parallel_all_reduce(
            resident_len, torch.distributed.ReduceOp.MIN
        )
        resident_len = (
            int(resident_len.item())
            // self.lmcache_connector.chunk_size
            * self.lmcache_connector.chunk_size
        )
        if resident_len == 0:
            logger.debug(
                "LMCache store skipped for %s: radix prefix has no complete "
                "LMCache chunk (%d/%d tokens)",
                req.rid,
                len(matched.device_indices),
                len(key),
            )
            return
        if resident_len < len(key):
            logger.debug(
                "LMCache store for %s is limited to the common resident prefix: "
                "%d/%d tokens",
                req.rid,
                resident_len,
                len(key),
            )
            key = key[:resident_len]
            # Re-match so the lock and Mamba state use the shortened boundary.
            matched = super().match_prefix(MatchPrefixParams(key=key))
        lock_params = self.inc_lock_ref(matched.last_device_node).to_dec_params()
        mamba_value = (
            self.tree_core.get_component_device_value(
                matched.best_match_node, ComponentType.MAMBA
            )
            if self._mamba_component is not None
            else None
        )
        mamba_is_resident = torch.tensor(
            [int(self._mamba_component is None or mamba_value is not None)],
            dtype=torch.int32,
            device="cpu",
        )
        self.lmcache_connector.parallel_all_reduce(
            mamba_is_resident, torch.distributed.ReduceOp.MIN
        )
        if not mamba_is_resident.item():
            self.dec_lock_ref(matched.last_device_node, lock_params)
            logger.debug(
                "LMCache store skipped for %s: no Mamba checkpoint at token %d",
                req.rid,
                len(key),
            )
            return
        try:
            operation = self.lmcache_connector.submit_store(
                req.rid,
                key.raw_token_ids()[: len(key)],
                self.lmcache_connector.device_indices_by_group(
                    matched.device_indices[: len(key)], mamba_value=mamba_value
                ),
                cache_salt=self.lmcache_connector.build_cache_salt(
                    req.cache_salt, req.extra_key
                ),
            )
        except Exception:
            self.dec_lock_ref(matched.last_device_node, lock_params)
            logger.exception("LMCache store submission failed for %s", req.rid)
            return
        if operation is None:
            self.dec_lock_ref(matched.last_device_node, lock_params)
            return
        self._pending_stores.append(
            LMCachePendingStore(operation, matched.last_device_node, lock_params)
        )
        self._pending_store_counts[req.rid] = (
            self._pending_store_counts.get(req.rid, 0) + 1
        )

    def _request_session_finish(self, rid: str) -> None:
        """End the LMCache session after this request's stores have completed."""
        self._session_finish_requested.add(rid)
        self._finish_session_if_store_idle(rid)

    def _finish_session_if_store_idle(self, rid: str) -> None:
        if (
            rid not in self._session_finish_requested
            or self._pending_store_counts.get(rid, 0) > 0
        ):
            return
        self._session_finish_requested.remove(rid)
        self.lmcache_connector.finish_request(rid)
