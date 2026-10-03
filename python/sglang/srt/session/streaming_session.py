from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    IncLockRefResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.utils.common import ceil_align, is_npu

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


logger = logging.getLogger(__name__)


class _VirtualNode:
    """Lock target for session-owned KV; locking it is a no-op."""

    pass


@dataclass
class SessionSlot:
    """Holds KV state between streaming session turns."""

    virtual_node: _VirtualNode = field(default_factory=_VirtualNode)

    # KV pool state
    kv: ReqKvInfo = field(default_factory=ReqKvInfo)

    # First req's radix tree node (for dec_lock_ref on session close)
    last_node: Any = None
    # Receipt of the first request's tree lock on last_node.
    lock_receipt: DecLockRefParams = field(default_factory=DecLockRefParams)
    # Whether the first request already released its SWA lock.
    swa_prefix_lock_released: bool = False

    def save_from_req(self, req: Req, is_first: bool):
        """Save KV state from a finishing request into this slot."""
        kv = req.detach_kv()
        if is_first:
            self.last_node = req.last_node
            self.lock_receipt = req.lock_receipt
            self.swa_prefix_lock_released = req.swa_prefix_lock_released
            # The slot takes over the request's KV record.
            self.kv = kv
        else:
            # Later turns run on the slot's record (see restore_to_req).
            assert kv is self.kv

        req.swa_branching_seqlen = None

    def restore_to_req(self, req: Req):
        """Restore KV state from this slot into an incoming request."""
        req.kv = self.kv
        req.lock_receipt = self.lock_receipt
        req.swa_prefix_lock_released = self.swa_prefix_lock_released

        # The slot keeps sharing the record: a rejected chunked request calls
        # match_prefix -> restore_to_req again next cycle.


def _is_streaming(req: Optional[Req]) -> bool:
    return req is not None and req.session is not None and req.session.streaming


class StreamingSession:
    """Streaming-session KV save/restore, owned by ``UnifiedRadixCache``.

    The cache calls the ``try_*`` entries first; each runs the session body
    when it applies and tells the cache whether to run its own path.
    """

    def __init__(self, cache: UnifiedRadixCache):
        self.cache = cache
        self.slots: Dict[str, SessionSlot] = {}

    def any_holding_kv(self) -> bool:
        return any(s.kv.holds_kv for s in self.slots.values())

    # -- Try-handle entries (see class docstring) --

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
        """A pre-aborted req (to_finish set) is detached from the session and
        gets None; the slot stays for the next request."""
        if not _is_streaming(req):
            return None
        slot = self.slots.get(req.session.session_id)
        if slot is None or not slot.kv.holds_kv:
            return None
        if req.to_finish is not None:
            req.session.abort_req()
            req.session = None
            return None
        return slot

    def try_match_prefix(self, params: MatchPrefixParams) -> Optional[MatchResult]:
        """Returns a MatchResult iff the request hits an active session slot;
        otherwise None (caller falls back to its raw match)."""
        slot = self.find_active_slot(params.req)
        if slot is None:
            return None

        req = params.req

        # [NPU] Below one aligned page, drop the slot's KV and fully prefill.
        if is_npu() and self.cache.page_size > 1:
            expected_prefix_len = min(slot.kv.kv_committed_len, len(params.key))
            aligned_prefix_len = (
                expected_prefix_len // self.cache.page_size
            ) * self.cache.page_size
            if (
                aligned_prefix_len < slot.kv.cache_protected_len
                or aligned_prefix_len == 0
            ):
                # Release KV to avoid leak and fallback to full prefill.
                # req remains unassigned, so alloc_for_extend treats it as new.
                self.release_session(req.session.session_id)
                return None

        slot.restore_to_req(req)

        # token_ids = get_fill_ids()[:input_len-1] (1-token logit reserve
        # already applied). min handles retract retry where committed_len
        # can exceed len(token_ids) by 1.
        prefix_len = min(req.kv.kv_committed_len, len(params.key))

        # Streaming sessions are append-only (session_controller rollback
        # ensures req_nodes always points to the last successful req).
        assert prefix_len >= slot.kv.cache_protected_len, (
            f"streaming session prefix shrank: {prefix_len=} < "
            f"{slot.kv.cache_protected_len=}"
        )

        # NPU requires page-aligned KV reuse; a rewind below the SWA eviction
        # cursor must also land on a page boundary -- free_kv_row_segments
        # splits dead/alive at the cursor, and a mid-page cut frees a page twice.
        if self.cache.page_size > 1 and (
            is_npu() or req.kv.max_evicted_seqlen > prefix_len
        ):
            prefix_len = (prefix_len // self.cache.page_size) * self.cache.page_size
            req.kv.kv_committed_len = min(req.kv.kv_committed_len, prefix_len)

        self._free_tail(req.kv, prefix_len)

        device_indices = self.cache.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :prefix_len
        ].to(dtype=torch.int64)

        return MatchResult(
            device_indices=device_indices,
            last_device_node=slot.virtual_node,
            last_host_node=slot.virtual_node,
            best_match_node=slot.virtual_node,
            cache_protected_len=slot.kv.cache_protected_len,
        )

    def try_cache_finished_req(self, req: Req) -> bool:
        """Handles a streaming-session finish (save slot / mid-abort nuke).
        Returns True if handled; False means caller runs its raw path."""
        if not _is_streaming(req):
            return False

        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        session_id = req.session.session_id
        slot = self.slots.get(session_id)
        is_first = slot is None

        # Mid-processing abort: free all session KV and drop the slot; req_nodes
        # still points at the last finished request, so the next turn re-prefills.
        if isinstance(req.finished_reason, FINISH_ABORT):
            kv = req.detach_kv()
            if slot is None:
                # First turn: a throwaway slot lets release_session free the
                # record (mamba refs included) and drop the tree lock.
                slot = SessionSlot(
                    kv=kv,
                    last_node=req.last_node,
                    lock_receipt=req.lock_receipt,
                    swa_prefix_lock_released=req.swa_prefix_lock_released,
                )
                self.slots[session_id] = slot
            else:
                assert kv is slot.kv
            self.release_session(session_id)
            req.session.abort_req()
            return True

        if is_first:
            slot = SessionSlot()
            self.slots[session_id] = slot

        finished_len = (
            req.finished_len if req.finished_len is not None else len(req.output_ids)
        )
        target = len(req.origin_input_ids) + finished_len
        self._trim_overshoot(req, finished_len)

        slot.save_from_req(req, is_first=is_first)
        # Use the finished length, not the req clock (it lags an in-flight verify
        # by ~1 under overlap); clamp so committed <= allocated.
        slot.kv.kv_committed_len = min(target, slot.kv.kv_allocated_len)

        # Update req_nodes to this successfully finished request.
        req.session.finish_req(req)

        return True

    def try_checkpoint(self, req: Req, *, up_to: int, **kwargs) -> bool:
        """A first turn checkpoints into the tree like any request (its
        prompt prefix is tree-owned and the slot inherits that lock); later
        turns run on the slot's KV, so only the chunk cursor is kept."""
        if not _is_streaming(req) or req.session.session_id not in self.slots:
            return False
        kv_indices = self.cache.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :up_to
        ]
        req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
        return True

    # -- Session lifecycle --

    def release_session(self, session_id: str) -> None:
        slot = self.slots.pop(session_id, None)
        if slot is None:
            return
        protected_len = slot.kv.cache_protected_len
        lock_node = slot.last_node
        tokens_freed = (
            max(0, slot.kv.kv_allocated_len - protected_len) if slot.kv.holds_kv else 0
        )
        logger.info(
            "Session KV released: %s (%d tokens freed)", session_id, tokens_freed
        )

        if lock_node is not None:
            # skip_swa is an SWA-cache extension kwarg; a slot can only have
            # early-released when the cache supports SWA locks.
            skip = {"skip_swa": True} if slot.swa_prefix_lock_released else {}
            self.cache.dec_lock_ref(lock_node, slot.lock_receipt, **skip)

        if slot.kv.holds_kv:
            self.cache.free_kv_row(slot.kv, [(protected_len, slot.kv.kv_allocated_len)])
            self.cache.req_to_token_pool.free(slot)

        self._free_slot_mamba(slot)

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        """KV tokens held by idle session slots; a slot whose pool idx is in
        ``active_pool_idxs`` is counted via uncached_size instead."""
        total = 0
        for slot in self.slots.values():
            in_batch = (
                active_pool_idxs is not None
                and slot.kv.req_pool_idx in active_pool_idxs
            )
            if slot.kv.holds_kv and not in_batch:
                allocated = ceil_align(slot.kv.kv_allocated_len, self.cache.page_size)
                total += allocated - slot.kv.cache_protected_len
        return total

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return self.session_held_tokens(active_pool_idxs)

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        """Total SWA tokens held by session slots, not tracked by the tree."""
        total = 0
        for slot in self.slots.values():
            in_batch = (
                active_pool_idxs is not None
                and slot.kv.req_pool_idx in active_pool_idxs
            )
            if slot.kv.holds_kv and not in_batch:
                allocated = ceil_align(slot.kv.kv_allocated_len, self.cache.page_size)
                total += allocated - max(
                    slot.kv.cache_protected_len,
                    slot.kv.get_evicted_seqlen(ComponentType.SWA),
                )
        return total

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:
        """Number of req pool slots held by session slots."""

        def _owned(s):
            in_batch = (
                active_pool_idxs is not None and s.kv.req_pool_idx in active_pool_idxs
            )
            return s.kv.holds_kv and not in_batch

        return sum(_owned(s) for s in self.slots.values())

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:
        """mamba_pool entries held by idle session slots (same exclusion as
        ``session_held_tokens``)."""
        total = 0
        for slot in self.slots.values():
            in_batch = (
                active_pool_idxs is not None
                and slot.kv.req_pool_idx in active_pool_idxs
            )
            if in_batch:
                continue
            if slot.kv.holds_mamba:
                total += slot.kv.mamba_pool_idx.numel()
            if slot.kv.mamba_ping_pong_track_buffer is not None:
                buffer = slot.kv.mamba_ping_pong_track_buffer
                # Lazy tracking leaves unallocated entries marked with -1.
                if self.req_to_token_pool.enable_mamba_extra_buffer_lazy:
                    total += (buffer != -1).sum().item()
                else:
                    total += buffer.numel()
        return total

    def _free_slot_mamba(self, slot: SessionSlot) -> None:
        """Return a session slot's mamba pool state to the allocator."""
        mamba_allocator = getattr(self.cache.req_to_token_pool, "mamba_allocator", None)
        if mamba_allocator is None:
            return
        if slot.kv.holds_mamba:
            mamba_allocator.free(slot.kv.mamba_pool_idx.unsqueeze(0))
            slot.kv.mamba_pool_idx = None
        if slot.kv.mamba_ping_pong_track_buffer is not None:
            buffer = slot.kv.mamba_ping_pong_track_buffer
            if self.req_to_token_pool.enable_mamba_extra_buffer_lazy:
                buffer = buffer[buffer != -1]
            mamba_allocator.free(buffer)
            slot.kv.mamba_ping_pong_track_buffer = None

    # -- Internal helpers (streaming body bits) --

    def _free_tail(self, kv: ReqKvInfo, prefix_len: int) -> None:
        """Free [prefix_len, allocated) before alloc_for_extend overwrites it:
        spec decoding or a retract retry leaves stale indices there."""
        self._free_kv_aligned(kv, prefix_len, kv.kv_allocated_len)
        kv.kv_allocated_len = prefix_len
        kv.kv_committed_len = min(kv.kv_committed_len, prefix_len)
        kv.clamp_evicted_seqlens(prefix_len)

    def _trim_overshoot(self, req: Req, finished_len: int) -> None:
        """Spec v2 can commit past max_new_tokens; the next turn's input is
        output_ids[:finished_len], so release the KV past it."""
        target = len(req.origin_input_ids) + finished_len
        if self.cache.page_size > 1 and req.kv.max_evicted_seqlen > target:
            # Same hazard as the match-path rewind: the cursor must stay
            # page-aligned; the partial page is re-prefilled next turn.
            target = (target // self.cache.page_size) * self.cache.page_size
        self._free_kv_aligned(req.kv, target, req.kv.kv_allocated_len)
        req.kv.kv_allocated_len = min(req.kv.kv_allocated_len, target)
        req.kv.kv_committed_len = min(req.kv.kv_committed_len, target)
        req.kv.clamp_evicted_seqlens(target)
        req.output_ids = req.output_ids[:finished_len]

    def _free_kv_aligned(self, kv: ReqKvInfo, target: int, end: int) -> None:
        """Free [ceil_align(target), end): paged free returns whole pages, so
        the partial page stays until release_session."""
        if end <= target:
            return
        start = target
        if self.cache.page_size > 1:
            start = ceil_align(start, self.cache.page_size)
        self.cache.free_kv_row(kv, [(start, end)])
