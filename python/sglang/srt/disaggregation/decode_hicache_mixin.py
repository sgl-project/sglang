"""HiCache integration mixins for the decode side of PD disaggregation"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional

import torch
import torch.distributed as dist

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_policy import match_prefix_for_req
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InitLoadBackParams

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@dataclass
class DecodePrefixMatch:
    prefix_indices: torch.Tensor
    l2_host_hit_length: int
    l3_storage_hit_length: int
    last_device_node: Any
    last_host_node: Any = None
    prefetch_registered: bool = False

    @property
    def l1_prefix_len(self) -> int:
        return len(self.prefix_indices)

    @property
    def decode_prefix_len(self) -> int:
        return self.l1_prefix_len + self.l2_host_hit_length + self.l3_storage_hit_length

    @property
    def needs_local_restore(self) -> bool:
        return self.decode_prefix_len > self.l1_prefix_len

    @property
    def restore_token_count(self) -> int:
        """Number of tokens that need L2/L3 load_back to device."""
        return self.decode_prefix_len - self.l1_prefix_len


class HiCacheRestoreResult(Enum):
    """Outcome of one tick of the HiCache local-restore state machine."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class _L2OnlyTreeLoadBackFailed(Exception):
    """Tree-segment load_back failed after the tail op was enqueued."""


_L2_ONLY_RESTORE_SENTINEL = object()


class DecodeHiCachePreallocMixin:
    """HiCache hooks for ``DecodePreallocQueue``: issue prefetch + reserve tokens."""

    def _build_decode_prefix_match(self, req: Req, result: Any) -> DecodePrefixMatch:
        """Convert a ``match_prefix_for_req`` result into ``DecodePrefixMatch``.

        Performs the optional L3 storage hit length query when decode-side
        HiCache is enabled and the last host node is backed up.
        """
        prefix_indices = result.device_indices
        l1_prefix_len = len(prefix_indices)
        l2_host_hit_length = result.host_hit_length

        l3_storage_hit_length = 0
        last_host_node = None
        if self.scheduler.enable_decode_hicache:
            last_host_node = result.last_host_node
            if self.tree_cache.is_backuped(last_host_node) or self.tree_cache.is_root(
                last_host_node
            ):
                matched_len = l1_prefix_len + l2_host_hit_length
                suffix_tokens = req.origin_input_ids[matched_len:]
                last_hash = self.tree_cache.get_last_hash_value(last_host_node)
                prefix_keys = (
                    self.tree_cache.get_prefix_hash_values(last_host_node)
                    if self.tree_cache.hicache_storage_pass_prefix_keys
                    else None
                )
                l3_storage_hit_length = self.tree_cache.query_storage_hit_length(
                    result.last_host_node,
                    suffix_tokens,
                    last_hash,
                    prefix_keys,
                )

        return DecodePrefixMatch(
            prefix_indices=prefix_indices,
            l2_host_hit_length=l2_host_hit_length,
            l3_storage_hit_length=l3_storage_hit_length,
            last_device_node=result.last_device_node,
            last_host_node=(
                result.last_host_node if l3_storage_hit_length > 0 else None
            ),
        )

    def _start_hicache_prefetch(
        self, req: Req, prefix_match: Optional[DecodePrefixMatch]
    ) -> None:
        """Issue L3 storage prefetch after admission succeeds.

        On failure, degrades to L2-only restore by clearing l3 fields.
        """
        if (
            prefix_match is None
            or prefix_match.l3_storage_hit_length <= 0
            or prefix_match.last_host_node is None
        ):
            return
        try:
            matched_len = prefix_match.l1_prefix_len + prefix_match.l2_host_hit_length
            suffix = req.origin_input_ids[
                matched_len : matched_len + prefix_match.l3_storage_hit_length
            ]
            last_hash = self.tree_cache.get_last_hash_value(prefix_match.last_host_node)
            prefix_keys = (
                self.tree_cache.get_prefix_hash_values(prefix_match.last_host_node)
                if self.tree_cache.hicache_storage_pass_prefix_keys
                else None
            )
            self.tree_cache.prefetch_from_storage(
                req.rid,
                prefix_match.last_host_node,
                suffix,
                last_hash,
                prefix_keys,
                extra_key=req.extra_key,
                cache_salt=req.cache_salt,
            )
            prefix_match.prefetch_registered = (
                req.rid in self.tree_cache.ongoing_prefetch
            )
        except Exception as e:
            logger.warning(
                "HiCache L3 prefetch failed for rid=%s: %s; falling back to L2-only LoadingBack",
                req.rid,
                e,
            )
            prefix_match.l3_storage_hit_length = 0
            prefix_match.prefetch_registered = False

    def _hicache_pending_restore_tokens(self) -> int:
        """Total device tokens reserved for pending HiCache L2/L3 load_back."""
        if not self.scheduler.enable_decode_hicache:
            return 0
        return sum(
            (dr.prefix_match.restore_token_count if dr.prefix_match else 0)
            for dr in self.transfer_queue.queue
            if (dr.prefix_match is not None or dr.l2_only_delta_len > 0)
            and dr.hicache_restore_status == HiCacheRestoreResult.PENDING
            and dr.hicache_restored_node is None
        )


class HiCacheRestoreGatedKVReceiver:
    """Wraps a kv_receiver so KVPoll.Success is gated on HiCache restore READY."""

    def __init__(self, decode_req: DecodeRequest):
        self.decode_req = decode_req

    def poll(self) -> KVPoll:
        poll = self.decode_req.kv_receiver.poll()
        if (
            poll == KVPoll.Success
            and self.decode_req.hicache_restore_status == HiCacheRestoreResult.PENDING
        ):
            return KVPoll.Transferring
        return poll


class DecodeHiCacheTransferMixin:
    """HiCache hooks for ``DecodeTransferQueue``: drive restore state machine."""

    def _clean_hicache_prefetch_resources(self, decode_req: DecodeRequest) -> None:
        if (
            decode_req.prefix_match is not None
            and decode_req.prefix_match.prefetch_registered
        ):
            self.tree_cache.release_aborted_request(decode_req.req.rid)
        if (
            decode_req.hicache_restored_node is not None
            and decode_req.hicache_restored_node is not _L2_ONLY_RESTORE_SENTINEL
        ):
            self.tree_cache.dec_lock_ref(decode_req.hicache_restored_node)
        decode_req.hicache_restored_node = None
        self._unpin_l2_only_host_prefix(decode_req)
        self._clean_l2_only_resources(decode_req)

    def _clean_l2_only_resources(self, decode_req: DecodeRequest) -> None:
        """Release L2-Only host staging and tail device slots (idempotent)."""
        host_indices = decode_req.l2_only_host_indices
        if host_indices is not None and len(host_indices) > 0:
            self.tree_cache.cache_controller.mem_pool_host.free(host_indices)
        decode_req.l2_only_host_indices = None

        tail_indices = decode_req.l2_only_tail_device_indices
        if tail_indices is None or len(tail_indices) == 0:
            decode_req.l2_only_tail_device_indices = None
            return
        decode_req.l2_only_tail_device_indices = None
        dma_in_flight = not self.tree_cache.is_load_back_event_done_pure(
            decode_req.hicache_load_consumer_index
        )
        if dma_in_flight:
            self.l2_only_pending_frees.append(
                (decode_req.hicache_load_consumer_index, tail_indices)
            )
        else:
            self._l2_only_device_allocator().free(tail_indices)

    def _shrink_l2_only_kv_lens_for_release(self, decode_req: DecodeRequest) -> bool:
        """Narrow a failed L2-Only request's release range to the protected prefix.

        Returns False when the row was released here instead, so the caller must
        skip ``release_kv_cache``.
        """
        if decode_req.l2_only_delta_len <= 0:
            return True
        req = decode_req.req
        protected = req.kv.cache_protected_len
        if protected > 0:
            if req.kv.kv_committed_len > protected:
                req.kv.kv_committed_len = protected
            if req.kv.kv_allocated_len > req.kv.kv_committed_len:
                req.kv.kv_allocated_len = req.kv.kv_committed_len
            return True
        req.kv.kv_committed_len = 0
        self.tree_cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=0)
        self.tree_cache.req_to_token_pool.free(req)
        req.kv.mark_kv_released()
        return False

    def _l2_only_device_allocator(self):
        """Device allocator owning L2-Only tail slots."""
        cc = self.tree_cache.cache_controller
        return getattr(
            cc.mem_pool_device_allocator,
            "full_attn_allocator",
            cc.mem_pool_device_allocator,
        )

    def _try_hicache_queue_load_back(self, dr: DecodeRequest) -> bool:
        """Queue one L2->L1 load_back op for ``dr``; True iff a DMA was queued.

        On success, ``dr.hicache_restored_node`` and ``hicache_restored_kv_indices``
        are populated, and an inc_lock_ref is held until commit/abort.
        Trivial cases (all-on-device / no needed coverage) auto-flip to READY.
        Failback paths flip to FAILED.
        """
        pm = dr.prefix_match
        cc = self.tree_cache.cache_controller

        demand = self._hicache_restore_demand(dr)
        ledger = getattr(self, "_hicache_restore_ledger", None)
        if ledger is not None:
            if ledger < demand:
                return False
            self._hicache_restore_ledger = ledger - demand

        if dr.l2_only_delta_len > 0:
            receiver = dr.kv_receiver
            if receiver is None or receiver.poll() != KVPoll.Success:
                return False

        if pm is not None and pm.l3_storage_hit_length > 0:
            if not dr.hicache_l3_drained:
                return False

        tail_op = None
        tail_indices = None
        if dr.l2_only_delta_len > 0:
            tail_indices = cc.load(
                dr.l2_only_host_indices,
                node_id=-1,
            )
            if tail_indices is None:
                allocator = self._l2_only_device_allocator()
                needed = len(dr.l2_only_host_indices)
                avail = allocator.available_size()
                if avail < needed and self.tree_cache.evictable_size() > 0:
                    self.tree_cache.evict(EvictParams(num_tokens=needed - avail))
                    tail_indices = cc.load(
                        dr.l2_only_host_indices,
                        node_id=-1,
                    )
                if tail_indices is None:
                    return False
            tail_op = cc.load_queue[-1]
            dr.l2_only_tail_device_indices = tail_indices

        try:
            return self._try_hicache_queue_load_back_tree(dr, pm)
        except _L2OnlyTreeLoadBackFailed:
            if tail_op is not None:
                cc.load_queue.remove(tail_op)
                self._l2_only_device_allocator().free(tail_indices)
                dr.l2_only_tail_device_indices = None
            dr.hicache_restore_status = HiCacheRestoreResult.FAILED
            return False

    def _try_hicache_queue_load_back_tree(
        self, dr: DecodeRequest, pm: Optional[DecodePrefixMatch]
    ) -> bool:
        """Tree-segment half of the queue branch; the caller handles the tail."""
        l2_only = dr.l2_only_delta_len > 0

        if pm is None:
            dr.hicache_restored_kv_indices = dr.l2_only_tail_device_indices[
                : dr.l2_only_delta_len
            ]
            dr.l2_only_tail_used_len = dr.l2_only_delta_len
            dr.hicache_restored_node = _L2_ONLY_RESTORE_SENTINEL
            return True

        # Re-match: req.last_node / prefix_indices updated to current device state.
        rematch = match_prefix_for_req(
            self.tree_cache,
            dr.req,
            dr.req.origin_input_ids,
            cow_mamba=False,
            include_req=True,
        )
        new_indices, restored_node = self.tree_cache.init_load_back(
            InitLoadBackParams(
                best_match_node=rematch.best_match_node,
                host_hit_length=rematch.host_hit_length,
                req=dr.req,
            )
        )
        # Failback: total coverage < required prefix means device alloc likely failed.
        if len(rematch.device_indices) + len(new_indices) < pm.decode_prefix_len:
            logger.warning(
                "HiCache load_back failed for rid=%s: device_indices=%d, "
                "new_indices=%d, expected decode_prefix_len=%d (l1=%d, l2=%d, l3=%d)",
                dr.req.rid,
                len(rematch.device_indices),
                len(new_indices),
                pm.decode_prefix_len,
                pm.l1_prefix_len,
                pm.l2_host_hit_length,
                pm.l3_storage_hit_length,
            )
            if l2_only:
                raise _L2OnlyTreeLoadBackFailed
            dr.hicache_restore_status = HiCacheRestoreResult.FAILED
            return False

        restored_indices = torch.cat(
            [rematch.device_indices[pm.l1_prefix_len :], new_indices]
        )
        dr.hicache_restored_node = restored_node
        self.tree_cache.inc_lock_ref(restored_node)

        if l2_only:
            needed = dr.req.kv.kv_committed_len - pm.l1_prefix_len
            tree_cap = min(
                len(restored_indices),
                pm.decode_prefix_len - pm.l1_prefix_len,
            )
            tail_len = needed - tree_cap
            if tail_len == 0:
                dr.hicache_restored_kv_indices = restored_indices[:needed]
                dr.l2_only_tail_used_len = 0
            else:
                dr.hicache_restored_kv_indices = torch.cat(
                    [
                        restored_indices[:tree_cap],
                        dr.l2_only_tail_device_indices[:tail_len],
                    ]
                )
                dr.l2_only_tail_used_len = tail_len
            return True

        dr.hicache_restored_kv_indices = restored_indices
        if len(new_indices) == 0:
            # Whole prefix already on device; no DMA needed.
            dr.hicache_restore_status = HiCacheRestoreResult.READY
            return False
        return True

    def _sync_hicache_restore_headroom(self) -> None:
        """MIN-align device headroom across ranks and reset the per-tick ledger."""
        if not self._l2_only:
            return
        allocator = self._l2_only_device_allocator()
        headroom = int(allocator.available_size()) + int(
            self.tree_cache.evictable_size()
        )
        headroom_t = torch.tensor([headroom], dtype=torch.int64, device="cpu")
        self.tree_cache._all_reduce_attn_groups(headroom_t, dist.ReduceOp.MIN)
        self._hicache_restore_ledger = int(headroom_t.item())

    def _hicache_restore_demand(self, dr: DecodeRequest) -> int:
        """Worst-case device tokens this request's restore will allocate."""
        demand = 0
        if dr.l2_only_delta_len > 0 and dr.l2_only_host_indices is not None:
            demand += len(dr.l2_only_host_indices)
        pm = dr.prefix_match
        if pm is not None and pm.needs_local_restore:
            demand += pm.restore_token_count
        return demand

    def _process_hicache_local_restores(self, decode_reqs: List[DecodeRequest]) -> None:
        if not hasattr(self.tree_cache, "is_load_back_event_done"):
            return

        self._sync_hicache_restore_headroom()

        for dr in decode_reqs:
            dr_pm = dr.prefix_match
            if (
                dr_pm is not None
                and dr_pm.l3_storage_hit_length > 0
                and dr_pm.prefetch_registered
                and not dr.hicache_l3_drained
            ):
                if self.tree_cache.check_prefetch_progress(dr.req.rid):
                    self.tree_cache.pop_prefetch_loaded_tokens(dr.req.rid)
                    dr.hicache_l3_drained = True
                    self._pin_l2_only_host_prefix(dr)

        # Filter once: keep only PENDING reqs that still need restore work;
        # trivially-done reqs (no prefix_match / nothing to restore) flip to READY.
        active: List[DecodeRequest] = []
        for dr in decode_reqs:
            if dr.hicache_restore_status != HiCacheRestoreResult.PENDING:
                continue
            pm = dr.prefix_match
            if (pm is None or not pm.needs_local_restore) and dr.l2_only_delta_len <= 0:
                dr.hicache_restore_status = HiCacheRestoreResult.READY
                continue
            active.append(dr)

        # Phase A: advance in-flight DMAs to READY.
        l2_only = self._l2_only
        uniform_polling = (
            l2_only or envs.SGLANG_DISAGGREGATION_UNIFORM_HICACHE_POLLING.get()
        )
        if (
            uniform_polling
            and not l2_only
            and hasattr(self.tree_cache, "loading_check")
        ):
            self.tree_cache.loading_check()
        for dr in active:
            if dr.hicache_restored_node is not None and (
                self.tree_cache.is_load_back_event_done(dr.hicache_load_consumer_index)
                if not uniform_polling
                else self.tree_cache.is_load_back_event_done_pure(
                    dr.hicache_load_consumer_index
                )
            ):
                dr.hicache_restore_status = HiCacheRestoreResult.READY

        # Phase B: queue new load_back ops if the next slot is free.
        # The (producer_index + 1) check ensures we never overwrite a still-in-flight slot:
        # if a previous req holds that slot and isn't done, its event won't be signaled.
        counter = self.tree_cache.cache_controller.layer_done_counter
        if not uniform_polling:
            slot_ok = bool(
                self.tree_cache.is_load_back_event_done(
                    (counter.producer_index + 1) % counter.num_counters
                )
            )
        else:
            slot_ok = bool(
                self.tree_cache.is_load_back_event_done_pure(
                    (counter.producer_index + 1) % counter.num_counters
                )
            )
        candidates = [dr for dr in active if dr.hicache_restored_node is None]
        if not l2_only:
            if not (slot_ok and candidates):
                return
        else:
            cand_n = len(candidates)
            gate_t = torch.tensor(
                [1 if (slot_ok and candidates) else 0, cand_n, -cand_n],
                dtype=torch.int64,
                device="cpu",
            )
            self.tree_cache._all_reduce_attn_groups(gate_t, dist.ReduceOp.MIN)
            if int(gate_t[0].item()) == 0:
                return
            if int(gate_t[1].item()) != -int(gate_t[2].item()):
                return
            ready_bits = []
            for dr in candidates:
                local_ok = True
                if dr.l2_only_delta_len > 0:
                    receiver = dr.kv_receiver
                    local_ok = (
                        receiver is not None and receiver.poll() == KVPoll.Success
                    )
                if (
                    local_ok
                    and dr.prefix_match is not None
                    and dr.prefix_match.l3_storage_hit_length > 0
                ):
                    local_ok = dr.hicache_l3_drained
                ready_bits.append(1 if local_ok else 0)
            ready_t = torch.tensor(ready_bits, dtype=torch.int64, device="cpu")
            self.tree_cache._all_reduce_attn_groups(ready_t, dist.ReduceOp.MIN)
            candidates = [dr for dr, ok in zip(candidates, ready_t.tolist()) if ok]
        queued = [dr for dr in candidates if self._try_hicache_queue_load_back(dr)]
        if not queued:
            return

        # Phase C: kick off merged DMA, bind consumer_index for Phase A polling next tick.
        consumer_index = self.tree_cache.ready_to_load_host_cache()
        if consumer_index < 0:
            for dr in queued:
                dr.hicache_restore_status = HiCacheRestoreResult.READY
            return
        for dr in queued:
            dr.hicache_load_consumer_index = consumer_index

    def _drain_l2_only_pending_frees(self) -> None:
        """Free deferred tail device slots once their restore DMA is done."""
        if not self.l2_only_pending_frees:
            return
        allocator = self._l2_only_device_allocator()
        still_pending = []
        for consumer_index, indices in self.l2_only_pending_frees:
            if self.tree_cache.is_load_back_event_done_pure(consumer_index):
                allocator.free(indices)
            else:
                still_pending.append((consumer_index, indices))
        self.l2_only_pending_frees = still_pending

    def _pin_l2_only_host_prefix(self, decode_req: DecodeRequest) -> None:
        if not self._l2_only:
            return
        if decode_req.l2_host_lock_node is not None:
            return
        rematch = match_prefix_for_req(
            self.tree_cache,
            decode_req.req,
            decode_req.req.origin_input_ids,
            cow_mamba=False,
            include_req=True,
        )
        node_id = rematch.best_match_node
        if node_id is None or rematch.host_hit_length <= 0:
            return
        decode_req.l2_host_lock_params = self.tree_cache.inc_host_lock_ref(
            node_id
        ).to_dec_params()
        decode_req.l2_host_lock_node = node_id

    def _unpin_l2_only_host_prefix(self, decode_req: DecodeRequest) -> None:
        """Release the pin taken by ``_pin_l2_only_host_prefix`` (idempotent).

        ``to_dec_params()`` replays the tombstone set captured at acquire time,
        so the params must be passed back rather than releasing by node alone.
        """
        node_id = decode_req.l2_host_lock_node
        if node_id is None:
            return
        decode_req.l2_host_lock_node = None
        params = decode_req.l2_host_lock_params
        decode_req.l2_host_lock_params = None
        self.tree_cache.dec_host_lock_ref(node_id, params)

    def _commit_hicache_local_restore_to_req(self, decode_req: DecodeRequest) -> None:
        self._unpin_l2_only_host_prefix(decode_req)
        prefix_match = decode_req.prefix_match
        if prefix_match is None or not prefix_match.needs_local_restore:
            if decode_req.l2_only_delta_len <= 0:
                return

        if prefix_match is not None:
            self.tree_cache.dec_lock_ref(prefix_match.last_device_node)

        if decode_req.l2_only_delta_len > 0:
            fill_len = decode_req.req.kv.kv_committed_len
            start = prefix_match.l1_prefix_len if prefix_match is not None else 0
            self.tree_cache.req_to_token_pool.write(
                (decode_req.req.kv.req_pool_idx, slice(start, fill_len)),
                decode_req.hicache_restored_kv_indices,
            )
            host_indices = decode_req.l2_only_host_indices
            if host_indices is not None and len(host_indices) > 0:
                self.tree_cache.cache_controller.mem_pool_host.free(host_indices)
            decode_req.l2_only_host_indices = None
            tail_indices = decode_req.l2_only_tail_device_indices
            if tail_indices is not None:
                allocator = self._l2_only_device_allocator()
                used_len = decode_req.l2_only_tail_used_len
                pad_start = (
                    (used_len + allocator.page_size - 1)
                    // allocator.page_size
                    * allocator.page_size
                )
                if pad_start < len(tail_indices):
                    allocator.free(tail_indices[pad_start:])
            decode_req.l2_only_tail_device_indices = None
            if prefix_match is not None:
                tree_len = prefix_match.restore_token_count
                decode_req.req.prefix_indices = torch.cat(
                    [
                        prefix_match.prefix_indices,
                        decode_req.hicache_restored_kv_indices[:tree_len],
                    ]
                )
            if decode_req.hicache_restored_node is not _L2_ONLY_RESTORE_SENTINEL:
                decode_req.req.last_node = decode_req.hicache_restored_node
            decode_req.hicache_restored_node = None
            return

        self.tree_cache.req_to_token_pool.write(
            (
                decode_req.req.kv.req_pool_idx,
                slice(prefix_match.l1_prefix_len, prefix_match.decode_prefix_len),
            ),
            decode_req.hicache_restored_kv_indices,
        )
        decode_req.req.prefix_indices = torch.cat(
            [prefix_match.prefix_indices, decode_req.hicache_restored_kv_indices]
        )
        decode_req.req.last_node = decode_req.hicache_restored_node
