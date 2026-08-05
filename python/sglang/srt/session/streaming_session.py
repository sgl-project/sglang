from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol

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
    from sglang.srt.managers.schedule_batch import Req, ReqKvInfo


logger = logging.getLogger(__name__)


class _VirtualNode:
    """Sentinel node for streaming session requests.

    Passed to inc_lock_ref / dec_lock_ref so the cache can distinguish
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
    kv: Optional[ReqKvInfo] = None

    # First req's radix tree node (for dec_lock_ref on session close)
    last_node: Any = None
    cache_protected_len: int = 0
    swa_uuid_for_lock: Optional[str] = None
    # components the first req skipped locking on last_node, so release dec
    # releases only what it took (may share the node with another req).
    skip_lock_node_ids: dict = field(default_factory=dict)

    # Mamba states
    mamba_pool_idx: Any = None
    mamba_ping_pong_track_buffer: Any = None
    mamba_next_track_idx: Any = None
    mamba_last_track_seqlen: Any = None
    mamba_branching_seqlen: Any = None

    @property
    def is_holding_kv(self) -> bool:
        """Whether this slot currently holds KV pool resources."""
        return self.kv is not None

    def save_from_req(self, req: Req, is_first: bool):
        """Save KV state from a finishing request into this slot."""
        self.req_pool_idx = req.req_pool_idx
        self.kv_committed_len = req.kv_committed_len
        self.kv = copy.copy(req.kv)

        if is_first:
            self.last_node = req.last_node
            self.cache_protected_len = req.cache_protected_len
            self.swa_uuid_for_lock = req.swa_uuid_for_lock
            self.skip_lock_node_ids = req.skip_lock_node_ids

        self.mamba_pool_idx = req.mamba_pool_idx
        self.mamba_ping_pong_track_buffer = req.mamba_ping_pong_track_buffer
        self.mamba_next_track_idx = req.mamba_next_track_idx
        self.mamba_last_track_seqlen = req.mamba_last_track_seqlen
        self.mamba_branching_seqlen = req.mamba_branching_seqlen

        # Ownership has transferred to the slot. Null *all* of the req's
        # references so any later alloc()/free path that inspects the req
        # (e.g. the alloc-skip check on `req.mamba_ping_pong_track_buffer
        # is None`, or the retract cleanup) sees no dangling pointers
        # into slot-owned tensors. Without this the alloc path can decide
        # the req still has a ping-pong buffer and skip alloc, causing
        # the slot's tensor to be reused by a new req and leaked when
        # the slot is later freed.
        req.req_pool_idx = None
        req.kv = None
        req.mamba_pool_idx = None
        req.mamba_ping_pong_track_buffer = None
        req.mamba_next_track_idx = None
        req.mamba_last_track_seqlen = None
        req.mamba_branching_seqlen = None

    def restore_to_req(self, req: Req):
        """Restore KV state from this slot into an incoming request."""
        req.req_pool_idx = self.req_pool_idx
        req.kv_committed_len = self.kv_committed_len
        req.kv = copy.copy(self.kv)
        req.swa_uuid_for_lock = self.swa_uuid_for_lock
        req.skip_lock_node_ids = self.skip_lock_node_ids

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


class StreamingSessionLifecycle(Protocol):
    def reset(self) -> None: ...

    def on_request_start(self, req: Any, slot: SessionSlot) -> None: ...

    def on_request_committed(
        self, session_id: str, slot: SessionSlot, req: Any
    ) -> None: ...

    def on_session_released(self, session_id: str) -> None: ...

    def next_prefill_chunk_end(self, req: Any, start: int, end: int) -> int: ...

    def on_prefill_forward_complete(self, req: Any, start: int, end: int) -> None: ...

    def on_decode_token(self, req: Any) -> None: ...

    def held_mamba_slots(self) -> int: ...


class _NoopStreamingSessionLifecycle:
    def reset(self) -> None:
        return None

    def on_request_start(self, req: Any, slot: SessionSlot) -> None:
        return None

    def on_request_committed(
        self, session_id: str, slot: SessionSlot, req: Any
    ) -> None:
        return None

    def on_session_released(self, session_id: str) -> None:
        return None

    def next_prefill_chunk_end(self, req: Any, start: int, end: int) -> int:
        return end

    def on_prefill_forward_complete(self, req: Any, start: int, end: int) -> None:
        return None

    def on_decode_token(self, req: Any) -> None:
        return None

    def held_mamba_slots(self) -> int:
        return 0


class StreamingSession(BasePrefixCache):
    """Adds streaming-session KV save/restore on top of any BasePrefixCache.

    Works both as an external wrapper (``StreamingSession(RadixCache(...))``)
    and in embedded composition (``StreamingSession(inner=self)``). For the
    embedded case, the composing cache must pre-check dispatch conditions
    (``_is_streaming`` / ``find_active_slot`` / ``has_slot``) so the internal
    fall-through to ``self.inner.xxx`` never fires -- otherwise it recurses.
    """

    def __init__(self, inner: BasePrefixCache):
        self.inner = inner
        self.slots: Dict[str, SessionSlot] = {}
        self._session_lifecycle: StreamingSessionLifecycle = (
            _NoopStreamingSessionLifecycle()
        )
        self._has_attached_lifecycle = False

    @property
    def has_attached_lifecycle(self) -> bool:
        return self._has_attached_lifecycle

    def attach_session_lifecycle(
        self, session_control: StreamingSessionLifecycle
    ) -> None:
        if self._has_attached_lifecycle:
            raise RuntimeError("A streaming session control is already attached")
        self._session_lifecycle = session_control
        self._has_attached_lifecycle = True

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

    # -- Condition helpers (used by embedded-mode callers for pre-dispatch) --

    def has_slot(self, session_id: str) -> bool:
        return session_id in self.slots

    def is_retained_boundary(self, req: Any) -> bool:
        if not _is_streaming(req) or req.session.session_id not in self.slots:
            return False
        return any(node.req is req for node in req.session.req_nodes.values())

    def any_holding_kv(self) -> bool:
        return any(s.is_holding_kv for s in self.slots.values())

    # -- Try-handle entries for composition (see class docstring) --

    def try_inc_lock_ref(self, node: Any) -> Optional[IncLockRefResult]:
        """No-op lock if ``node`` is a session-internal sentinel; returns
        None to tell the caller to run its raw tree lock path."""
        if isinstance(node, _VirtualNode):
            return IncLockRefResult()
        return None

    def try_dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> Optional[DecLockRefResult]:
        if isinstance(node, _VirtualNode):
            return DecLockRefResult()
        return None

    def find_active_slot(self, req: Req) -> Optional[SessionSlot]:
        """Returns an active slot for this req, or None.

        Side effect: if req is pre-aborted (to_finish set, e.g. input too
        long), detach it from the session so cache_finished_req treats it
        as a normal req. The slot stays intact for the next request.
        """
        if not _is_streaming(req):
            return None
        slot = self.slots.get(req.session.session_id)
        if slot is None or slot.kv is None:
            return None
        if req.to_finish is not None:
            req.session.abort_req(req)
            req.session = None
            return None
        return slot

    # -- BasePrefixCache abstract methods --

    def reset_state(self) -> None:
        """Clear session-owned state without resetting the composed cache."""
        self._session_lifecycle.reset()
        self.slots.clear()

    def reset(self):
        self.reset_state()
        self.inner.reset()

    # -- Streaming entries: contract with embedded composers (e.g.
    # UnifiedRadixCache) is a uniform "try_handle_*" pattern. Each method
    # executes the streaming body if applicable and signals whether the
    # caller still needs to run its raw path.

    def try_match_prefix(self, params: MatchPrefixParams) -> Optional[MatchResult]:
        """Returns a MatchResult iff the request hits an active session slot;
        otherwise None (caller falls back to its raw match)."""
        slot = self.find_active_slot(params.req)
        if slot is None:
            self._limit_first_prefill_match(params)
            return None

        req = params.req
        self._session_lifecycle.on_request_start(req, slot)
        slot.restore_to_req(req)

        # token_ids = get_fill_ids()[:input_len-1] (1-token logit reserve
        # already applied). min handles retract retry where committed_len
        # can exceed len(token_ids) by 1.
        prefix_len = min(req.kv_committed_len, len(params.key))

        # Streaming sessions are append-only; truncation updates the session's
        # append target before another request can start.
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
            best_match_node=slot.virtual_node,
            cache_protected_len=slot.cache_protected_len,
        )

    def try_cache_finished_req(
        self, req: Req, is_insert: bool = True, **kwargs
    ) -> bool:
        """Handles a streaming-session finish (save slot / mid-abort nuke).
        Returns True if handled; False means caller runs its raw path."""
        if not _is_streaming(req):
            return False

        from sglang.srt.managers.schedule_batch import (
            FINISH_ABORT,
            FINISH_MATCHED_TOKEN,
            StreamingSessionAbortPolicy,
        )

        session = req.session
        session_id = session.session_id
        slot = self.slots.get(session_id)
        is_first = slot is None
        is_abort = isinstance(req.finished_reason, FINISH_ABORT)

        if is_abort and session._inflight_req is not req:
            if not self.detach_queued_request(req):
                raise RuntimeError(
                    "A non-owning streaming request advanced beyond its "
                    f"session boundary: session={session_id}"
                )
            self._release_unretained_multimodal_inputs(req, session)
            return True

        # Cancel keeps forwarded output. The sampled tail has not run a forward,
        # so exclude it from both the token history and KV boundary.
        if (
            is_abort
            and req.streaming_abort_policy
            is StreamingSessionAbortPolicy.COMMIT_FORWARDED
        ):
            if is_first:
                slot = SessionSlot()
                self.slots[session_id] = slot
            finished_len = max(0, len(req.output_ids) - 1)
            self._trim_overshoot(req, finished_len)
            slot.save_from_req(req, is_first=is_first)
            target = len(req.origin_input_ids) + finished_len
            slot.kv_committed_len = min(target, slot.kv.kv_allocated_len)
            self._session_lifecycle.on_request_committed(session_id, slot, req)
            req.session.finish_req(req)
            return True

        # Other mid-processing aborts release the slot. The session still points
        # at its last successful request and can re-prefill on the next turn.
        if is_abort:
            retains_boundary = self.is_retained_boundary(req)
            if not session.abort_req(req):
                raise RuntimeError(
                    f"Request {req.rid} lost ownership of session {session_id}"
                )
            if slot is None:
                # Transfer first-request resources into a temporary slot so the
                # normal release path frees KV, tree locks, and recurrent state.
                slot = SessionSlot(
                    req_pool_idx=req.req_pool_idx,
                    kv=copy.copy(req.kv),
                    last_node=req.last_node,
                    cache_protected_len=req.cache_protected_len,
                    swa_uuid_for_lock=req.swa_uuid_for_lock,
                    skip_lock_node_ids=req.skip_lock_node_ids,
                    mamba_pool_idx=req.mamba_pool_idx,
                    mamba_ping_pong_track_buffer=req.mamba_ping_pong_track_buffer,
                )
                self.slots[session_id] = slot
                # Slot now owns the mamba state — drop the req's refs so
                # the abort fall-through doesn't double-free.
                req.mamba_pool_idx = None
                req.mamba_ping_pong_track_buffer = None
            slot.kv.kv_allocated_len = max(
                slot.kv.kv_allocated_len, req.kv.kv_allocated_len
            )
            self.release_session(session_id)
            req.req_pool_idx = None
            req.kv = None
            req.mamba_pool_idx = None
            req.mamba_ping_pong_track_buffer = None
            req.mamba_next_track_idx = None
            req.mamba_last_track_seqlen = None
            req.mamba_branching_seqlen = None
            if not retains_boundary:
                self._release_unretained_multimodal_inputs(req, session)
                req.session = None
            return True

        is_retraction = not req.finished()
        if is_retraction and is_insert:
            raise RuntimeError(
                "An unfinished streaming request can only be cached for retraction"
            )

        if is_first:
            slot = SessionSlot()
            self.slots[session_id] = slot

        finished_len = (
            req.finished_len if req.finished_len is not None else len(req.output_ids)
        )
        # A matched stop is sampled but never forwarded, so it cannot be part of
        # the next turn's committed context.
        if (
            req.drop_trailing_stop_token
            and finished_len > 0
            and isinstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        ):
            finished_len -= 1
        target = len(req.origin_input_ids) + finished_len
        self._trim_overshoot(req, finished_len)

        slot.save_from_req(req, is_first=is_first)
        # Inherit the authoritative finished length on the slot, not the lagging
        # req clock (under overlap + honest committed the clock lags the in-flight
        # verify by ~1, which would short-change inheritance). Clamp to allocated
        # to keep committed <= allocated for prepare_for_decode.
        slot.kv_committed_len = min(target, slot.kv.kv_allocated_len)

        self._session_lifecycle.on_request_committed(session_id, slot, req)

        if is_retraction:
            req.session.checkpoint_retracted_req(req)
        else:
            req.session.finish_req(req)

        return True

    def try_cache_unfinished_req(
        self, req: Req, chunked: bool = False, **kwargs
    ) -> bool:
        """Handles a streaming-session mid-flight cache op:
          - chunked prefill: snapshot current KV as prefix, skip radix
          - subsequent turn: skip radix (slot already holds KV)
        Returns False for first-turn non-chunked (caller must run raw radix
        insert to set up the initial tree lock)."""
        if not _is_streaming(req):
            return False
        if chunked:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : req.extend_range.end
            ]
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            return True
        if req.session.session_id in self.slots:
            return True
        return False

    # -- BasePrefixCache abstract methods: thin adapters over try_handle_* --

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        result = self.try_match_prefix(params)
        if result is not None:
            return result
        return self.inner.match_prefix(params)

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        if self.try_cache_finished_req(req, is_insert=is_insert, **kwargs):
            return
        self.inner.cache_finished_req(req, is_insert=is_insert, **kwargs)

    def cache_unfinished_req(self, req: Req, **kwargs):
        if self.try_cache_unfinished_req(req, **kwargs):
            return
        self.inner.cache_unfinished_req(req, **kwargs)

    def evict(self, params: EvictParams) -> EvictResult:
        return self.inner.evict(params)

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        result = self.try_inc_lock_ref(node)
        if result is not None:
            return result
        return self.inner.inc_lock_ref(node)

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        result = self.try_dec_lock_ref(node, params)
        if result is not None:
            return result
        return self.inner.dec_lock_ref(node, params)

    # -- Session lifecycle --

    def release_session(self, session_id: str) -> None:
        self._session_lifecycle.on_session_released(session_id)
        slot = self.slots.pop(session_id, None)
        if slot is None:
            return
        protected_len = slot.cache_protected_len
        lock_node = slot.last_node
        tokens_freed = (
            max(0, slot.kv.kv_allocated_len - protected_len)
            if slot.is_holding_kv
            else 0
        )
        logger.info(
            "Session KV released: %s (%d tokens freed)", session_id, tokens_freed
        )

        if lock_node is not None:
            self.inner.dec_lock_ref(
                lock_node,
                DecLockRefParams(
                    swa_uuid_for_lock=slot.swa_uuid_for_lock,
                    skip_lock_node_ids=slot.skip_lock_node_ids,
                ),
            )

        if slot.is_holding_kv:
            start = protected_len
            end = slot.kv.kv_allocated_len
            if start < end:
                kv_indices = self.req_to_token_pool.req_to_token[
                    slot.req_pool_idx, start:end
                ]
                self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free_slots.append(slot.req_pool_idx)

        self._free_slot_mamba(slot)

    def release_radix_session(self, session_id: str) -> None:
        self.inner.release_radix_session(session_id)

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        """Total KV tokens held by session slots, not tracked by the tree.

        Excludes slots whose KV is currently owned by an owning request --
        those tokens are counted via uncached_size in the busy mem check.
        A slot's pool_idx being in active_pool_idxs indicates a req owns it.
        """
        total = 0
        for slot in self.slots.values():
            in_batch = (
                active_pool_idxs is not None and slot.req_pool_idx in active_pool_idxs
            )
            if slot.is_holding_kv and not in_batch:
                allocated = ceil_align(slot.kv.kv_allocated_len, self.page_size)
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
                allocated = ceil_align(slot.kv.kv_allocated_len, self.page_size)
                total += allocated - max(
                    slot.cache_protected_len, slot.kv.swa_evicted_seqlen
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

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:
        """Total mamba_pool entries held by session slots (mamba_pool_idx +
        mamba_ping_pong_track_buffer). Excludes slots whose owning req is
        currently in the batch -- those slots are counted via the normal
        alloc/free paths (same convention as the sibling ``session_held_*``
        accessors).
        """
        total = 0
        for slot in self.slots.values():
            in_batch = (
                active_pool_idxs is not None and slot.req_pool_idx in active_pool_idxs
            )
            if in_batch:
                continue
            if slot.mamba_pool_idx is not None:
                total += slot.mamba_pool_idx.numel()
            if slot.mamba_ping_pong_track_buffer is not None:
                total += slot.mamba_ping_pong_track_buffer.numel()
        total += self._session_lifecycle.held_mamba_slots()
        return total

    def _free_slot_mamba(self, slot: SessionSlot) -> None:
        """Return a session slot's mamba pool state to the allocator."""
        mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        if mamba_allocator is None:
            return
        if slot.mamba_pool_idx is not None:
            mamba_allocator.free(slot.mamba_pool_idx.unsqueeze(0))
            slot.mamba_pool_idx = None
        if slot.mamba_ping_pong_track_buffer is not None:
            mamba_allocator.free(slot.mamba_ping_pong_track_buffer)
            slot.mamba_ping_pong_track_buffer = None

    def record_decode_token(self, req: Any) -> None:
        self._session_lifecycle.on_decode_token(req)

    def next_prefill_chunk_end(self, req: Any, start: int, end: int) -> int:
        if not _is_streaming(req):
            return end
        return self._session_lifecycle.next_prefill_chunk_end(req, start, end)

    def record_prefill_forward_complete(self, req: Any, start: int, end: int) -> None:
        if _is_streaming(req):
            self._session_lifecycle.on_prefill_forward_complete(req, start, end)

    def detach_queued_request(self, req: Any) -> bool:
        """Detach a request that has not advanced beyond its saved session slot."""
        if not _is_streaming(req):
            return False
        slot = self.slots.get(req.session.session_id)
        if req.req_pool_idx is not None:
            if slot is None or req.req_pool_idx != slot.req_pool_idx:
                return False
            if (
                req.kv_committed_len != slot.kv_committed_len
                or req.kv.kv_allocated_len != slot.kv.kv_allocated_len
            ):
                return False

        for name in (
            "mamba_pool_idx",
            "mamba_ping_pong_track_buffer",
        ):
            req_value = getattr(req, name)
            slot_value = None if slot is None else getattr(slot, name)
            if req_value is not None and not self._same_slot_reference(
                req_value, slot_value
            ):
                return False

        session = req.session
        session_id = session.session_id
        owns_session = session.abort_req(req)
        req.session = None
        req.req_pool_idx = None
        req.kv = None
        req.mamba_pool_idx = None
        req.mamba_ping_pong_track_buffer = None
        req.mamba_next_track_idx = None
        req.mamba_last_track_seqlen = None
        req.mamba_branching_seqlen = None
        if owns_session and slot is None:
            self._session_lifecycle.on_session_released(session_id)
        return True

    @staticmethod
    def _same_slot_reference(left: Any, right: Any) -> bool:
        if right is None:
            return False
        if torch.is_tensor(left) and torch.is_tensor(right):
            return bool(torch.equal(left, right))
        return bool(left == right)

    def truncate_kv(self, session_id: str, target: int) -> SessionSlot:
        slot = self.slots.get(session_id)
        if slot is None or not slot.is_holding_kv:
            raise RuntimeError(f"Session {session_id} has no committed KV state")
        if not 0 <= target <= slot.kv_committed_len:
            raise ValueError(
                f"Truncate target {target} is outside " f"[0, {slot.kv_committed_len}]"
            )
        if target < slot.cache_protected_len:
            raise ValueError(
                f"Truncate target {target} is below the protected session prefix "
                f"{slot.cache_protected_len}"
            )
        self._free_kv_aligned(slot.req_pool_idx, target, slot.kv.kv_allocated_len)
        slot.kv.kv_allocated_len = target
        slot.kv_committed_len = target
        slot.kv.swa_evicted_seqlen = min(slot.kv.swa_evicted_seqlen, target)
        return slot

    def get_kv_state(self, session_id: str) -> Optional[dict[str, Any]]:
        slot = self.slots.get(session_id)
        if slot is None:
            return None
        state: dict[str, Any] = {
            "found": True,
            "kv_committed_len": slot.kv_committed_len,
            "kv_allocated_len": (
                slot.kv.kv_allocated_len if slot.kv is not None else 0
            ),
            "is_holding_kv": slot.is_holding_kv,
        }
        return state

    # -- Internal helpers (streaming body bits) --

    def _limit_first_prefill_match(self, params: MatchPrefixParams) -> None:
        if not _is_streaming(params.req) or not self._has_attached_lifecycle:
            return
        current_limit = (
            len(params.key.token_ids)
            if params.key.limit is None
            else min(params.key.limit, len(params.key.token_ids))
        )
        full_end = len(params.req.full_untruncated_fill_ids)
        if not full_end:
            return
        chunk_end = self.next_prefill_chunk_end(params.req, 0, full_end)
        if not 0 < chunk_end <= full_end:
            raise RuntimeError(
                "Streaming lifecycle returned an invalid first prefill "
                f"boundary: chunk_end={chunk_end} end={full_end}"
            )
        match_limit = min(current_limit, chunk_end - 1)
        if match_limit < current_limit:
            params.key.limit = match_limit

    def _free_tail(self, slot: SessionSlot, req: Req, prefix_len: int) -> None:
        """match_prefix path: free orphaned KV in [prefix_len, kv_allocated_len)
        before alloc_for_extend overwrites it. The gap appears when spec
        decoding pushes allocated above committed, or when retract retry's
        logit-reserve pulls prefix_len below committed.
        """
        self._free_kv_aligned(slot.req_pool_idx, prefix_len, slot.kv.kv_allocated_len)
        slot.kv.kv_allocated_len = prefix_len
        slot.kv_committed_len = min(slot.kv_committed_len, prefix_len)
        slot.kv.swa_evicted_seqlen = min(slot.kv.swa_evicted_seqlen, prefix_len)
        req.kv.kv_allocated_len = prefix_len
        req.kv_committed_len = min(req.kv_committed_len, prefix_len)
        req.kv.swa_evicted_seqlen = min(req.kv.swa_evicted_seqlen, prefix_len)

    def _trim_overshoot(self, req: Req, finished_len: int) -> None:
        """Trim slot KV to finished_len boundary. Spec v2 may overshoot
        max_new_tokens (verify round commits M+1 at a time); next turn's
        input is output_ids[:finished_len], so positions past that must
        be released to avoid token/KV mismatch.
        """
        target = len(req.origin_input_ids) + finished_len
        self._free_kv_aligned(req.req_pool_idx, target, req.kv.kv_allocated_len)
        req.kv.kv_allocated_len = min(req.kv.kv_allocated_len, target)
        req.kv_committed_len = min(req.kv_committed_len, target)
        req.kv.swa_evicted_seqlen = min(req.kv.swa_evicted_seqlen, target)
        req.output_ids = req.output_ids[:finished_len]

    def _free_kv_aligned(self, pool_idx: int, target: int, end: int) -> None:
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
    def _release_unretained_multimodal_inputs(req: Req, session: Any) -> None:
        """Release aborted-turn features without clearing the saved boundary."""
        mm_inputs = req.multimodal_inputs
        if mm_inputs is None:
            return

        retained_item_ids = {
            id(item)
            for node in session.req_nodes.values()
            if node.req is not req and node.req.multimodal_inputs is not None
            for item in node.req.multimodal_inputs.mm_items
        }
        releasable = copy.copy(mm_inputs)
        releasable.mm_items = [
            item for item in mm_inputs.mm_items if id(item) not in retained_item_ids
        ]
        releasable.release_features()
        req.multimodal_inputs = None

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

    def check_hicache_events(self):
        return self.inner.check_hicache_events()

    def take_events(self):
        return self.inner.take_events()

    def supports_swa(self):
        return self.inner.supports_swa()

    def supports_mamba(self):
        return self.inner.supports_mamba()

    def supports_streaming_session(self) -> bool:
        return True

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
        if self.any_holding_kv():
            return
        self.inner.sanity_check()

    # Forward attribute access for cache-specific methods (e.g.
    # sliding_window_size, all_values_flatten, etc.)
    def __getattr__(self, name):
        return getattr(self.inner, name)


def get_streaming_session(cache: BasePrefixCache) -> Optional[StreamingSession]:
    """Return a cache's direct or composed streaming-session component."""
    if isinstance(cache, StreamingSession):
        return cache
    session = getattr(cache, "session", None)
    return session if isinstance(session, StreamingSession) else None
