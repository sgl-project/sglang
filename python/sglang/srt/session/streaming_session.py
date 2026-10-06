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
    """A streaming session's KV record and the tree lock on its prefix. The
    session owns both from its first turn's row allocation until it closes or
    a turn aborts; every turn runs on the record as a borrower."""

    virtual_node: _VirtualNode = field(default_factory=_VirtualNode)

    # KV pool state
    kv: ReqKvInfo = field(default_factory=ReqKvInfo)

    # Tree lock on the session's tree-owned prefix, and its receipt.
    last_node: Any = None
    lock_receipt: DecLockRefParams = field(default_factory=DecLockRefParams)
    # Whether the SWA part of that lock was released early.
    swa_prefix_lock_released: bool = False
    # Until the first turn finishes or is retracted, its checkpoints publish
    # the prompt into the tree and move the lock onto the deepest published node.
    publishes_prompt: bool = False


def _is_streaming(req: Optional[Req]) -> bool:
    return req is not None and req.session is not None and req.session.streaming


def _move_tree_lock(src: Any, dst: Any) -> None:
    """Hand the tree lock ``src`` holds (a request or a slot) to ``dst``."""
    dst.last_node = src.last_node
    dst.lock_receipt = src.lock_receipt
    dst.swa_prefix_lock_released = src.swa_prefix_lock_released


class StreamingSession:
    """Streaming-session KV records, owned by ``UnifiedRadixCache``.

    A session takes its first turn's record and tree lock at row allocation
    (``take``); every turn then borrows them. The cache calls the ``try_*``
    entries first; each runs the session body when it applies and tells the
    cache whether to run its own path.
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

        # Lend the record; the slot keeps the tree lock. A rejected chunked
        # request matches again next cycle and borrows the same record.
        req.kv = slot.kv

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
        """Keeps a finished or retracted turn's record in the session slot.
        An aborted turn gets the record and the tree lock back and is released
        like any request (returns False); the session re-prefills from its
        last finished request next turn."""
        if not _is_streaming(req):
            return False

        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        slot = self.borrowed_slot(req)
        assert slot is not None, f"streaming {req.rid=} does not run on its slot"
        if isinstance(req.finished_reason, FINISH_ABORT):
            # Hand the record and the tree lock back; the caller releases them.
            del self.slots[req.session.session_id]
            _move_tree_lock(slot, req)
            req.session.abort_req()
            return False

        finished_len = (
            req.finished_len if req.finished_len is not None else len(req.output_ids)
        )
        target = len(req.origin_input_ids) + finished_len
        self._trim_overshoot(req, finished_len)

        req.detach_kv()
        req.swa_branching_seqlen = None
        slot.publishes_prompt = False
        # Use the finished length, not the req clock (it lags an in-flight verify
        # by ~1 under overlap); clamp so committed <= allocated.
        slot.kv.kv_committed_len = min(target, slot.kv.kv_allocated_len)

        # Update req_nodes to this successfully finished request.
        req.session.finish_req(req)

        return True

    def try_checkpoint(self, req: Req, *, up_to: int, **kwargs) -> bool:
        """A turn on the slot's record publishes nothing of its own, so only
        the chunk cursor is kept. The exception is the first prompt: the slot
        publishes it for other requests to share, and its lock follows the
        insert."""
        slot = self.borrowed_slot(req)
        if slot is None:
            return False
        if slot.publishes_prompt:
            _move_tree_lock(slot, req)
            self.cache.checkpoint_into_tree(req, up_to=up_to, **kwargs)
            self._lock_to_slot(req, slot)
            return True
        kv_indices = self.cache.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :up_to
        ]
        req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
        return True

    # -- Record ownership --

    def take(self, req: Req) -> None:
        """A streaming turn's first row allocation: the session takes the
        request's record and the tree lock it took at admission, and the
        request borrows them from here on."""
        if not _is_streaming(req) or self.borrowed_slot(req) is not None:
            return
        session_id = req.session.session_id
        assert session_id not in self.slots, f"{session_id=} already has a slot"
        slot = SessionSlot(kv=req.kv, publishes_prompt=True)
        self._lock_to_slot(req, slot)
        self.slots[session_id] = slot

    def borrowed_slot(self, req: Req) -> Optional[SessionSlot]:
        """The slot whose record the request runs on, if any."""
        if not _is_streaming(req):
            return None
        slot = self.slots.get(req.session.session_id)
        return slot if slot is not None and slot.kv is req.kv else None

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

    def _free_slot_mamba(self, slot: SessionSlot) -> None:
        """Return a session slot's mamba pool state to the allocator."""
        mamba_allocator = getattr(self.cache.req_to_token_pool, "mamba_allocator", None)
        if mamba_allocator is None:
            return
        if slot.kv.holds_mamba:
            mamba_allocator.free(slot.kv.mamba_pool_idx.unsqueeze(0))
            slot.kv.mamba_pool_idx = None
        if slot.kv.mamba_ping_pong_track_buffer is not None:
            indices = slot.kv.mamba_ping_pong_track_buffer
            mamba_allocator.free(indices[indices != -1])
            slot.kv.mamba_ping_pong_track_buffer = None

    # -- Internal helpers (streaming body bits) --

    def _lock_to_slot(self, req: Req, slot: SessionSlot) -> None:
        """Move the request's tree lock to the slot; the request is left on the
        slot's virtual node, where locking is a no-op."""
        _move_tree_lock(req, slot)
        req.last_node = slot.virtual_node
        req.lock_receipt = DecLockRefParams()
        req.swa_prefix_lock_released = False

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
