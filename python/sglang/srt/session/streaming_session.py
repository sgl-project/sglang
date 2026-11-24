from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.base_prefix_cache import (
    MatchPrefixParams,
    MatchResult,
    TreeLock,
)
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_disagg, get_parallel, get_schedule, get_spec
from sglang.srt.utils.common import ceil_align
from sglang.srt.utils.device import is_npu

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.models.unlimited_ocr_v2_support.document_reuse import DocReuse


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

    # Tree lock on the session's tree-owned prefix.
    lock: Optional[TreeLock] = None
    # Until the first turn finishes or is retracted, its checkpoints publish
    # the prompt into the tree and move the lock onto the deepest published node.
    publishes_prompt: bool = False
    # Number of consumed positions certified by a completed one-token turn.
    # A rewind invalidates this until a fresh slot computes the prompt again.
    single_token_commit_len: Optional[int] = None


def is_virtual_node(node: Any) -> bool:
    """A slot's stand-in node: locking it is a no-op."""
    return isinstance(node, _VirtualNode)


def _is_streaming(req: Optional[Req]) -> bool:
    return req is not None and req.session is not None and req.session.streaming


class StreamingSession:
    """Streaming-session KV records, owned by ``UnifiedRadixCache``.

    A session takes its first turn's record and tree lock at row allocation
    (``take``); every turn then borrows them. The cache calls the ``try_*``
    entries first; each runs the session body when it applies and tells the
    cache whether to run its own path.
    """

    # Set by UnifiedRadixCache.enable_uocr_doc_reuse for Unlimited-OCR v2.
    uocr_doc_reuse: Optional[DocReuse] = None

    def __init__(self, cache: UnifiedRadixCache):
        self.cache = cache
        self.slots: Dict[str, SessionSlot] = {}

    def any_holding_kv(self) -> bool:
        return any(s.kv.holds_kv for s in self.slots.values())

    # -- Try-handle entries (see class docstring) --

    def find_active_slot(self, req: Req) -> Optional[SessionSlot]:
        """A pre-aborted req (to_finish set) is detached from the session and
        gets None; the slot stays for the next request."""
        if not _is_streaming(req):
            return None
        slot = self.slots.get(req.session.session_id)
        if slot is None or not slot.kv.holds_kv:
            return None
        if req.to_finish is not None:
            req.release_mm_inputs_on_abort()
            req.session.abort_req()
            req.session = None
            return None
        return slot

    def will_lend_record(self, req: Optional[Req]) -> bool:
        """Whether req has a slot record to borrow. Unlike find_active_slot, it
        leaves a pre-aborted req attached."""
        if not _is_streaming(req) or req.to_finish is not None:
            return False
        slot = self.slots.get(req.session.session_id)
        return slot is not None and slot.kv.holds_kv

    def try_match_prefix(self, params: MatchPrefixParams) -> Optional[MatchResult]:
        """Returns a MatchResult iff the request hits an active session slot;
        otherwise None (caller falls back to its raw match)."""
        if self.uocr_doc_reuse is not None:
            reuse = self.uocr_doc_reuse.try_match_prefix(params)
            if reuse is not None:
                return reuse
        slot = self.find_active_slot(params.req)
        if slot is None:
            return None

        req = params.req
        align_npu = (
            is_npu()
            and self.cache.page_size > 1
            and not self._can_append_one_npu_token(params, slot)
        )

        # token_ids = get_fill_ids()[:input_len-1] (1-token logit reserve
        # already applied). min handles retract retry where committed_len
        # can exceed len(token_ids) by 1.
        prefix_len = min(slot.kv.kv_committed_len, len(params.key))

        # A rewind below the SWA eviction cursor must free whole pages.
        if self.cache.page_size > 1 and (
            align_npu or slot.kv.max_evicted_seqlen > prefix_len
        ):
            prefix_len = (prefix_len // self.cache.page_size) * self.cache.page_size

        missing_window_history = (
            prefix_len < slot.kv.kv_committed_len
            and slot.kv.max_evicted_seqlen > 0
            and any(
                slot.kv.get_evicted_seqlen(sub_pool) > max(0, prefix_len - window)
                for sub_pool, window in self.cache.window_sub_pools()
            )
        )

        # Replaying a tail also needs each query's left window, which may have
        # been evicted even when the replay position itself is still retained.
        # The session owns one recurrent state at its committed length, not a
        # state for each KV position. Rewinding KV cannot rewind that state.
        # Drop the borrowed record and use the normal tree match, which restores
        # a compatible checkpoint or starts a fresh prefill on a cache miss.
        if (
            missing_window_history
            or (slot.kv.holds_mamba and prefix_len < slot.kv.kv_committed_len)
            or (
                align_npu
                and (prefix_len < slot.kv.cache_protected_len or prefix_len == 0)
            )
        ):
            if req.kv is slot.kv:
                req.detach_kv()
            self.release_session(req.session.session_id)
            return None

        # Lend only after deciding whether the retained state is reusable.
        # The row stays parked until take admits the next turn.
        req.kv = slot.kv

        # Streaming sessions are append-only (session_controller rollback
        # ensures turns always points to the last successful req).
        assert prefix_len >= slot.kv.cache_protected_len, (
            f"streaming session prefix shrank: {prefix_len=} < "
            f"{slot.kv.cache_protected_len=}"
        )

        if slot.single_token_commit_len != prefix_len:
            slot.single_token_commit_len = None
        self._free_tail(req.kv, prefix_len)

        return MatchResult(
            device_prefix_len=prefix_len,
            last_device_node=slot.virtual_node,
            last_host_node=slot.virtual_node,
            best_match_node=slot.virtual_node,
            cache_protected_len=slot.kv.cache_protected_len,
        )

    def _can_append_one_npu_token(
        self, params: MatchPrefixParams, slot: SessionSlot
    ) -> bool:
        req = params.req
        kv = slot.kv
        prefix_len = len(params.key)
        # Append exactly one token to the last completed, unrewound prefix.
        if not (
            slot.single_token_commit_len == prefix_len > 0
            and prefix_len == kv.kv_committed_len == kv.kv_allocated_len
            and prefix_len + 1 == len(req.full_untruncated_fill_ids)
            and req.sampling_params.max_new_tokens == 1
            and req.retraction_count == 0
        ):
            return False

        # Reuse only private KV and the recurrent state at that same position.
        if not (
            self.cache.disable
            and kv.cache_protected_len == kv.max_evicted_seqlen == 0
            and kv.holds_mamba == self.cache.is_mamba_enabled
            and (
                kv.holds_mamba
                or self.cache.tree_components
                in ((ComponentType.FULL,), (ComponentType.FULL, ComponentType.SWA))
            )
            and kv.mamba_ping_pong_track_buffer is None
            and not kv.mamba_needs_clear
            and kv.mamba_cow_src_index is None
            and kv.retraction_backup is None
            and not self.cache.enable_mamba_extra_buffer
            and self.cache.req_to_token_pool._aux_cache is None
        ):
            return False

        from sglang.srt.hardware_backend.npu.allocator_npu import (
            NPUPagedTokenToKVPoolAllocator,
        )
        from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
        from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        allocator = self.cache.token_to_kv_pool_allocator
        if type(allocator) is NPUPagedTokenToKVPoolAllocator:
            if not kv.holds_mamba and (
                self.cache.tree_components != (ComponentType.FULL,)
                or type(allocator.get_kvcache()) is not NPUMHATokenToKVPool
            ):
                return False
        elif type(allocator) is SWATokenToKVPoolAllocator:
            pool = allocator.get_kvcache()
            if (
                kv.holds_mamba
                or self.cache.tree_components != (ComponentType.FULL, ComponentType.SWA)
                or allocator._swa_req_ring
                or type(pool) is not SWAKVPool
                or allocator.full_to_swa_index_mapping
                is not pool.full_to_swa_index_mapping
            ):
                return False
            # Both mappings append at the same page offset; no window page has
            # been evicted, so neither last location can name a released page.
            for child, child_pool in (
                (allocator.full_attn_allocator, pool.full_kv_pool),
                (allocator.swa_attn_allocator, pool.swa_kv_pool),
            ):
                if (
                    type(child) is not NPUPagedTokenToKVPoolAllocator
                    or child.page_size != allocator.page_size
                    or allocator.page_size != self.cache.page_size
                    or type(child_pool) is not NPUMHATokenToKVPool
                    or child.get_kvcache() is not child_pool
                ):
                    return False
        else:
            return False

        return (
            get_parallel().attn_dcp_size == 1
            and get_disagg().disaggregation_mode == "null"
            and get_schedule().disable_overlap_schedule
            and get_spec().speculative_algorithm is None
        )

    def try_cache_finished_req(self, req: Req) -> bool:
        """Keeps a finished or retracted turn's record in the session slot.
        An aborted turn gets the record and the tree lock back and is released
        like any request (returns False); the session re-prefills from its
        last finished request next turn."""
        if (
            self.uocr_doc_reuse is not None
            and self.uocr_doc_reuse.on_cache_finished_req(req)
        ):
            return True
        if not _is_streaming(req):
            return False

        from sglang.srt.managers.req_types import FINISH_ABORT

        slot = self.borrowed_slot(req)
        assert slot is not None, f"streaming {req.rid=} does not run on its slot"
        if isinstance(req.finished_reason, FINISH_ABORT):
            # Hand the record and the tree lock back; the caller releases them.
            del self.slots[req.session.session_id]
            req.lock, slot.lock = slot.lock, None
            req.release_mm_inputs_on_abort()
            req.session.abort_req()
            return False

        finished_len = (
            req.finished_len if req.finished_len is not None else len(req.output_ids)
        )
        consumed_len = len(req.origin_input_ids)
        if is_npu() and self.cache.page_size > 1 and self.cache.disable:
            kv = slot.kv
            # A one-token turn samples its output from the last prompt row.
            # The sampled token is consumed by the next turn, so it has no KV yet.
            if (
                (slot.publishes_prompt or slot.single_token_commit_len is not None)
                and req.finished()
                and req.sampling_params.max_new_tokens == 1
                and finished_len == len(req.output_ids) == 1
                and req.retraction_count == 0
                and consumed_len == kv.kv_allocated_len == kv.kv_committed_len
            ):
                slot.single_token_commit_len = consumed_len
            else:
                slot.single_token_commit_len = None

        target = len(req.origin_input_ids) + finished_len
        self._trim_overshoot(req, finished_len)

        req.detach_kv()
        req.swa_branching_seqlen = None
        slot.publishes_prompt = False
        # Use the finished length, not the req clock (it lags an in-flight verify
        # by ~1 under overlap); clamp so committed <= allocated.
        slot.kv.commit_finished(target)
        # The slot keeps its row until the next turn, with nothing running.
        self._storage().park_row(slot)

        # Update turns to this successfully finished request.
        req.session.finish_req(req)

        return True

    def checkpoints_into_tree(self, req: Req) -> bool:
        """Whether a checkpoint of ``req`` inserts into the tree: always outside
        a streaming turn, and in a turn only while its slot publishes the first
        prompt (`try_checkpoint`). Reads only; callers may ask ahead."""
        slot = self.borrowed_slot(req)
        return slot is None or slot.publishes_prompt

    def try_checkpoint(self, req: Req, *, up_to: int, **kwargs) -> bool:
        # A turn publishes nothing of its own, except the first prompt: the slot
        # publishes it for other requests to share, and its lock follows the insert.
        slot = self.borrowed_slot(req)
        if slot is None:
            return False
        if self.checkpoints_into_tree(req):
            req.lock, slot.lock = slot.lock, None
            self.cache.checkpoint_into_tree(req, up_to=up_to, **kwargs)
            self._lock_to_slot(req, slot)
        return True

    # -- Record ownership --

    def take(self, req: Req) -> None:
        """A streaming turn's row allocation: the first turn's record and tree
        lock move to a new slot it borrows; a later turn unparks its slot's row."""
        if not _is_streaming(req):
            return
        slot = self.borrowed_slot(req)
        if slot is not None:
            self._storage().unpark_row(slot)
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
        tokens_freed = (
            max(0, slot.kv.kv_allocated_len - protected_len) if slot.kv.holds_kv else 0
        )
        logger.info(
            "Session KV released: %s (%d tokens freed)", session_id, tokens_freed
        )

        self.cache.unlock(slot.lock)

        if slot.kv.holds_kv:
            self.cache.free_kv_row(slot.kv, [(protected_len, slot.kv.kv_allocated_len)])
            kv_storage_of(
                self.cache.req_to_token_pool, self.cache.token_to_kv_pool_allocator
            ).release_row(slot)

        self._free_slot_mamba(slot)

    def _free_slot_mamba(self, slot: SessionSlot) -> None:
        """Return a session slot's mamba pool state to the allocator."""
        mamba_allocator = self.cache.token_to_kv_pool_allocator.state_slots
        if mamba_allocator is None:
            return
        if slot.kv.holds_mamba:
            mamba_allocator.free(slot.kv.mamba_pool_idx.unsqueeze(0))
            slot.kv.mamba_pool_idx = None
        if slot.kv.mamba_ping_pong_track_buffer is not None:
            buffer = slot.kv.mamba_ping_pong_track_buffer
            for position, empty in enumerate(slot.kv.mamba_ping_pong_empty):
                if not empty:
                    mamba_allocator.free(buffer[position : position + 1])
            slot.kv.mamba_ping_pong_track_buffer = None
            slot.kv.mamba_ping_pong_empty = None

    # -- Internal helpers (streaming body bits) --

    def _storage(self):
        return kv_storage_of(
            self.cache.req_to_token_pool, self.cache.token_to_kv_pool_allocator
        )

    def _lock_to_slot(self, req: Req, slot: SessionSlot) -> None:
        slot.lock, req.lock = req.lock, None
        req.last_node = slot.virtual_node

    def _free_tail(self, kv: ReqKvInfo, prefix_len: int) -> None:
        """Free [prefix_len, allocated) before alloc_for_extend overwrites it:
        spec decoding or a retract retry leaves stale indices there."""
        self._free_kv_aligned(kv, prefix_len, kv.kv_allocated_len)
        kv.truncate(prefix_len)

    def _trim_overshoot(self, req: Req, finished_len: int) -> None:
        """Spec v2 can commit past max_new_tokens; the next turn's input is
        output_ids[:finished_len], so release the KV past it."""
        target = len(req.origin_input_ids) + finished_len
        if self.cache.page_size > 1 and req.kv.max_evicted_seqlen > target:
            # Same hazard as the match-path rewind: the cursor must stay
            # page-aligned; the partial page is re-prefilled next turn.
            target = (target // self.cache.page_size) * self.cache.page_size
        self._free_kv_aligned(req.kv, target, req.kv.kv_allocated_len)
        req.kv.truncate(target)
        req.output_ids = req.output_ids[:finished_len]

    def _free_kv_aligned(self, kv: ReqKvInfo, target: int, end: int) -> None:
        """Free [ceil_align(target), end): paged free returns whole pages, so
        the partial page stays until release_session."""
        start = target
        if self.cache.page_size > 1:
            start = ceil_align(start, self.cache.page_size)
        if end <= start:
            return
        self.cache.free_kv_row(kv, [(start, end)])
