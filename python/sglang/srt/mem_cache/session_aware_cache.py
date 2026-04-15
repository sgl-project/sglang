from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


logger = logging.getLogger(__name__)


class _VirtualNode:
    """Sentinel node for streaming session requests.

    Passed to inc_lock_ref / dec_lock_ref so the wrapper can distinguish
    streaming-session locks (no-op) from real radix-tree locks (forwarded).
    """

    pass


@dataclass
class SessionSlot:
    """Holds KV state between streaming session turns."""

    virtual_node: _VirtualNode = field(default_factory=_VirtualNode)

    # KV pool state (None means no KV is currently held by this slot)
    req_pool_idx: Optional[int] = None
    kv_committed_len: int = 0
    kv_allocated_len: int = 0

    # First req's radix tree node (for dec_lock_ref on session close)
    last_node: Any = None
    cache_protected_len: int = 0
    swa_uuid_for_lock: Optional[str] = None

    # SWA state
    swa_evicted_seqlen: int = 0

    # Mamba states
    mamba_pool_idx: Any = None
    mamba_ping_pong_track_buffer: Any = None
    mamba_next_track_idx: Any = None
    mamba_last_track_seqlen: Any = None
    mamba_branching_seqlen: Any = None

    @property
    def is_holding_kv(self) -> bool:
        """Whether this slot currently holds KV pool resources."""
        return self.req_pool_idx is not None

    def save_from_req(self, req: Req, is_first: bool):
        """Save KV state from a finishing request into this slot."""
        self.req_pool_idx = req.req_pool_idx
        self.kv_committed_len = req.kv_committed_len
        self.kv_allocated_len = req.kv_allocated_len
        self.swa_evicted_seqlen = req.swa_evicted_seqlen

        if is_first:
            self.last_node = req.last_node
            self.cache_protected_len = req.cache_protected_len
            self.swa_uuid_for_lock = req.swa_uuid_for_lock

        self.mamba_pool_idx = req.mamba_pool_idx
        self.mamba_ping_pong_track_buffer = req.mamba_ping_pong_track_buffer
        self.mamba_next_track_idx = req.mamba_next_track_idx
        self.mamba_last_track_seqlen = req.mamba_last_track_seqlen
        self.mamba_branching_seqlen = req.mamba_branching_seqlen

        req.req_pool_idx = None
        req.mamba_pool_idx = None

    def restore_to_req(self, req: Req):
        """Restore KV state from this slot into an incoming request."""
        req.req_pool_idx = self.req_pool_idx
        req.kv_committed_len = self.kv_committed_len
        req.kv_allocated_len = self.kv_allocated_len
        req.swa_evicted_seqlen = self.swa_evicted_seqlen
        req.swa_uuid_for_lock = self.swa_uuid_for_lock

        req.mamba_pool_idx = self.mamba_pool_idx
        req.mamba_ping_pong_track_buffer = self.mamba_ping_pong_track_buffer
        req.mamba_next_track_idx = self.mamba_next_track_idx
        req.mamba_last_track_seqlen = self.mamba_last_track_seqlen
        req.mamba_branching_seqlen = self.mamba_branching_seqlen

        # NOTE: req_pool_idx and mamba_pool_idx are intentionally NOT cleared
        # from the slot. During chunked prefill, a request may be rejected by
        # the scheduler (e.g. budget exhausted) and retried in the next cycle.
        # Each retry calls match_prefix -> restore_to_req again, so the slot
        # must remain intact for idempotent restoration.


def _is_streaming(req: Optional[Req]) -> bool:
    return req is not None and req.session is not None and req.session.streaming


class SessionAwareCache(BasePrefixCache):
    """Decorator around any BasePrefixCache that manages streaming session KV.

    Non-streaming requests are pure pass-through. Streaming requests have their
    KV lifecycle managed by SessionSlot objects, avoiding any invasive changes
    to the scheduling pipeline.
    """

    def __init__(self, inner: BasePrefixCache):
        self.inner = inner
        self.slots: Dict[str, SessionSlot] = {}

    # -- Forward PrefixCacheTrait properties to inner cache --

    @property
    def req_to_token_pool(self):
        return self.inner.req_to_token_pool

    @req_to_token_pool.setter
    def req_to_token_pool(self, value):
        self.inner.req_to_token_pool = value

    @property
    def token_to_kv_pool_allocator(self):
        return self.inner.token_to_kv_pool_allocator

    @token_to_kv_pool_allocator.setter
    def token_to_kv_pool_allocator(self, value):
        self.inner.token_to_kv_pool_allocator = value

    @property
    def page_size(self):
        return self.inner.page_size

    @page_size.setter
    def page_size(self, value):
        self.inner.page_size = value

    @property
    def disable(self):
        return self.inner.disable

    @disable.setter
    def disable(self, value):
        self.inner.disable = value

    @property
    def metrics_collector(self):
        return self.inner.metrics_collector

    @metrics_collector.setter
    def metrics_collector(self, value):
        self.inner.metrics_collector = value

    # -- BasePrefixCache abstract methods --

    def reset(self):
        self.slots.clear()
        self.inner.reset()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        req = params.req
        if not _is_streaming(req):
            return self.inner.match_prefix(params)

        session_id = req.session.session_id
        slot = self.slots.get(session_id)
        if slot is None or slot.req_pool_idx is None:
            return self.inner.match_prefix(params)

        # Pre-aborted req (scheduler-level abort, e.g. input too long):
        # detach from session so cache_finished_req treats it as a normal
        # req. The slot stays intact for the next request.
        if req.to_finish is not None:
            req.session.abort_req()
            req.session = None
            return self.inner.match_prefix(params)

        slot.restore_to_req(req)

        # token_ids = fill_ids[:input_len-1] (1-token logit reserve already
        # applied). min handles retract retry where committed_len can
        # exceed len(token_ids) by 1.
        prefix_len = min(req.kv_committed_len, len(params.key.token_ids))

        # Streaming sessions are append-only (session_controller rollback
        # ensures req_nodes always points to the last successful req).
        assert prefix_len >= slot.cache_protected_len, (
            f"streaming session prefix shrank: {prefix_len=} < "
            f"{slot.cache_protected_len=}"
        )

        # Free orphaned tail: alloc_for_extend will overwrite
        # req_to_token[prefix_len:] with new indices. The range
        # [prefix_len, kv_allocated_len) has stale indices from the
        # previous turn's decode (e.g. alloc-commit gap on retract,
        # or speculative draft tokens).
        self._free_tail(slot, req, prefix_len)

        device_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefix_len
        ].to(dtype=torch.int64)

        return MatchResult(
            device_indices=device_indices,
            last_device_node=slot.virtual_node,
            last_host_node=slot.virtual_node,
            cache_protected_len=slot.cache_protected_len,
        )

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        if not _is_streaming(req):
            return self.inner.cache_finished_req(req, is_insert=is_insert, **kwargs)

        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        session_id = req.session.session_id
        slot = self.slots.get(session_id)
        is_first = slot is None

        # Mid-processing abort only. Pre-aborted reqs have session=None
        # (set in match_prefix) and never reach here.
        # Nuke all KV via release_session, delete slot. Token IDs stay
        # in req_nodes (finish_req was never called -> last successful
        # req). Next request re-prefills from scratch.
        if isinstance(req.finished_reason, FINISH_ABORT):
            if slot is None:
                # First-request mid-processing abort: create ephemeral
                # slot from req state so release_session handles cleanup.
                # Include last_node/cache_protected_len from the req so
                # release_session calls dec_lock_ref on the tree lock.
                slot = SessionSlot(
                    req_pool_idx=req.req_pool_idx,
                    kv_allocated_len=req.kv_allocated_len,
                    last_node=req.last_node,
                    cache_protected_len=req.cache_protected_len,
                    swa_uuid_for_lock=req.swa_uuid_for_lock,
                )
                self.slots[session_id] = slot
            slot.kv_allocated_len = max(slot.kv_allocated_len, req.kv_allocated_len)
            self.release_session(session_id)
            req.req_pool_idx = None
            req.session.abort_req()
            self._mark_kv_freed(req)
            return

        if is_first:
            slot = SessionSlot()
            self.slots[session_id] = slot

        finished_len = (
            req.finished_len if req.finished_len is not None else len(req.output_ids)
        )
        self._trim_overshoot(req, finished_len)

        slot.save_from_req(req, is_first=is_first)

        # Update req_nodes to this successfully finished request.
        req.session.finish_req(req)

        self._mark_kv_freed(req)

    def _free_tail(self, slot: SessionSlot, req: Req, prefix_len: int):
        """match_prefix path: free orphaned KV in [prefix_len, kv_allocated_len)
        before alloc_for_extend overwrites it. The gap appears when spec
        decoding pushes allocated above committed, or when retract retry's
        logit-reserve pulls prefix_len below committed.
        """
        self._free_kv_aligned(slot.req_pool_idx, prefix_len, slot.kv_allocated_len)
        slot.kv_allocated_len = prefix_len
        slot.kv_committed_len = min(slot.kv_committed_len, prefix_len)
        slot.swa_evicted_seqlen = min(slot.swa_evicted_seqlen, prefix_len)
        req.kv_allocated_len = prefix_len
        req.kv_committed_len = min(req.kv_committed_len, prefix_len)
        req.swa_evicted_seqlen = min(req.swa_evicted_seqlen, prefix_len)

    def _trim_overshoot(self, req: Req, finished_len: int):
        """Trim slot KV to finished_len boundary. Spec v2 may overshoot
        max_new_tokens (verify round commits M+1 at a time); next turn's
        input is output_ids[:finished_len], so positions past that must
        be released to avoid token/KV mismatch.
        """
        target = len(req.origin_input_ids) + finished_len
        self._free_kv_aligned(req.req_pool_idx, target, req.kv_allocated_len)
        req.kv_allocated_len = min(req.kv_allocated_len, target)
        req.kv_committed_len = min(req.kv_committed_len, target)
        req.output_ids = req.output_ids[:finished_len]

    def _free_kv_aligned(self, pool_idx: int, target: int, end: int):
        """Free req_to_token[pool_idx, ceil_align(target):end). Page-aligned
        because PagedTokenToKVPoolAllocator.free returns whole pages
        (free_index // page_size), so partial-page free would corrupt pages
        still holding committed tokens. The range [target, ceil_align(target))
        stays attached until release_session frees the whole page.
        """
        if end <= target:
            return
        start = target
        if self.page_size > 1:
            start = ceil_align(start, self.page_size)
        if start < end:
            tail = self.req_to_token_pool.req_to_token[pool_idx, start:end]
            self.token_to_kv_pool_allocator.free(tail)

    @staticmethod
    def _mark_kv_freed(req: Req):
        """Set bookkeeping flags so busy check skips this finished req."""
        if not req.kv_committed_freed:
            req.pop_committed_kv_cache()
        if not req.kv_overallocated_freed:
            req.pop_overallocated_kv_cache()

    def cache_unfinished_req(self, req: Req, **kwargs):
        if _is_streaming(req):
            # in chunked_prefill for streaming, we skip the stash path which triggers radix.
            # only the last chunk in first turn trigger a full prompt radix insert.
            if kwargs.get("chunked", False):
                kv_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, : len(req.fill_ids)
                ]
                req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
                return
            if req.session.session_id in self.slots:
                # Subsequent turns: slot exists, skip inner entirely.
                return
            # First turn (no slot): fall through to inner for lock management,
            # tree insertion, and cache_protected_len updates between chunks.
        self.inner.cache_unfinished_req(req, **kwargs)

    def evict(self, params: EvictParams) -> EvictResult:
        return self.inner.evict(params)

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        if isinstance(node, _VirtualNode):
            return IncLockRefResult()
        return self.inner.inc_lock_ref(node)

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if isinstance(node, _VirtualNode):
            return DecLockRefResult()
        return self.inner.dec_lock_ref(node, params)

    # -- Session lifecycle --

    def release_session(self, session_id: str):
        """Release all KV resources held by a streaming session."""
        slot = self.slots.pop(session_id, None)
        if slot is None:
            return
        protected_len = slot.cache_protected_len
        lock_node = slot.last_node
        tokens_freed = (
            max(0, slot.kv_allocated_len - protected_len) if slot.is_holding_kv else 0
        )
        logger.info(
            "Session KV released: %s (%d tokens freed)", session_id, tokens_freed
        )

        if lock_node is not None:
            if slot.swa_uuid_for_lock is not None:
                self.inner.dec_lock_ref(
                    lock_node,
                    DecLockRefParams(swa_uuid_for_lock=slot.swa_uuid_for_lock),
                )
            else:
                self.inner.dec_lock_ref(lock_node)

        if slot.is_holding_kv:
            start = protected_len
            end = slot.kv_allocated_len
            if start < end:
                kv_indices = self.req_to_token_pool.req_to_token[
                    slot.req_pool_idx, start:end
                ]
                self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free_slots.append(slot.req_pool_idx)

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        """Total KV tokens held by session slots, not tracked by the tree.

        Excludes slots whose KV is currently owned by an owning request —
        those tokens are counted via uncached_size in the busy mem check.
        A slot's pool_idx being in active_pool_idxs indicates a req owns it.
        """
        total = 0
        for slot in self.slots.values():
            in_batch = (
                active_pool_idxs is not None and slot.req_pool_idx in active_pool_idxs
            )
            if slot.is_holding_kv and not in_batch:
                allocated = ceil_align(slot.kv_allocated_len, self.page_size)
                total += allocated - slot.cache_protected_len
        return total

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        """An alias to align the naming style of SWA"""
        return self.session_held_tokens(active_pool_idxs)

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        """Total SWA tokens held by session slots, not tracked by the tree."""
        total = 0
        for slot in self.slots.values():
            in_batch = (
                active_pool_idxs is not None and slot.req_pool_idx in active_pool_idxs
            )
            if slot.is_holding_kv and not in_batch:
                allocated = ceil_align(slot.kv_allocated_len, self.page_size)
                total += allocated - max(
                    slot.cache_protected_len, slot.swa_evicted_seqlen
                )
        return total

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:
        """Number of req pool slots held by session slots."""

        def _owned(s):
            in_batch = (
                active_pool_idxs is not None and s.req_pool_idx in active_pool_idxs
            )
            return s.is_holding_kv and not in_batch

        return sum(_owned(s) for s in self.slots.values())

    # -- Pass-through methods --

    def evictable_size(self):
        return self.inner.evictable_size()

    def full_evictable_size(self):
        return self.inner.full_evictable_size()

    def swa_evictable_size(self):
        return self.inner.swa_evictable_size()

    def protected_size(self):
        return self.inner.protected_size()

    def full_protected_size(self):
        return self.inner.full_protected_size()

    def swa_protected_size(self):
        return self.inner.swa_protected_size()

    def total_size(self):
        return self.inner.total_size()

    def pretty_print(self):
        return self.inner.pretty_print()

    def init_load_back(self, params: InitLoadBackParams):
        return self.inner.init_load_back(params)

    def ready_to_load_host_cache(self):
        return self.inner.ready_to_load_host_cache()

    def flush_write_through_acks(self) -> None:
        return self.inner.flush_write_through_acks()

    def check_hicache_events(self):
        return self.inner.check_hicache_events()

    def take_events(self):
        return self.inner.take_events()

    def supports_swa(self):
        return self.inner.supports_swa()

    def supports_mamba(self):
        return self.inner.supports_mamba()

    def is_chunk_cache(self):
        return self.inner.is_chunk_cache()

    def is_tree_cache(self):
        return self.inner.is_tree_cache()

    def available_and_evictable_str(self):
        return self.inner.available_and_evictable_str()

    def init_metrics_collector(self):
        return self.inner.init_metrics_collector()

    def sanity_check(self):
        # Skip inner sanity check when sessions hold tree locks, because
        # the check asserts all nodes are unlocked during idle.
        if any(s.is_holding_kv for s in self.slots.values()):
            return
        self.inner.sanity_check()

    # Forward attribute access for cache-specific methods (e.g.
    # sliding_window_size, all_values_flatten, etc.)
    def __getattr__(self, name):
        return getattr(self.inner, name)
