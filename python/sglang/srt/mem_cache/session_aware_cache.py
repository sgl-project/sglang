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

    # True while the slot's KV has been restored to an active request.
    # Prevents double-counting in token accounting (the request's tokens
    # are already tracked via uncached_size in the busy mem check).
    is_active: bool = False

    @property
    def is_holding_kv(self) -> bool:
        """Whether this slot currently holds KV pool resources."""
        return self.req_pool_idx is not None

    def save_from_req(self, req: Req, is_first: bool):
        """Save KV state from a finishing request into this slot."""
        self.is_active = False
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

        self.is_active = True

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

        # If the request is destined for abort (e.g. input too long),
        # do NOT restore the slot's KV state.  set_finish_with_abort
        # truncates origin_input_ids to [0], so alloc_for_extend would
        # overwrite the slot's req_to_token row with a 1-token prefix,
        # destroying the session's accumulated KV mapping.  By skipping
        # restore, the request gets a fresh pool slot from alloc_for_extend
        # and the session slot remains untouched.
        if req.to_finish is not None:
            return self.inner.match_prefix(params)

        slot.restore_to_req(req)

        # logprob_start_len is already forced to -1 for streaming sessions
        # (in Req.init_next_round_input), so the prefix key is not truncated
        # and we can directly reuse the committed KV length.
        prefix_len = min(req.kv_committed_len, max(len(params.key.token_ids) - 1, 0))
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

        # When an aborted streaming-session request was scheduled (e.g.
        # input too long), match_prefix skipped restore_to_req so the
        # request got a fresh pool slot from alloc_for_extend.  Don't
        # overwrite the session slot -- free the transient KV and pool slot.
        if not is_first and isinstance(req.finished_reason, FINISH_ABORT):
            if req.req_pool_idx is not None:
                # Free all KV pages allocated for this aborted request.
                end = req.kv_allocated_len
                if end > 0:
                    kv_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, :end
                    ]
                self.token_to_kv_pool_allocator.free(kv_indices)
                self.req_to_token_pool.free_slots.append(req.req_pool_idx)
                req.req_pool_idx = None
            return

        if is_first:
            slot = SessionSlot()
            self.slots[session_id] = slot

        # If the session's KV is shrinking (e.g. client sent a shorter
        # prompt after an abort), free the orphaned tail pages before
        # save_from_req overwrites the slot's committed length.
        # Never free tree-protected tokens — those are managed by the tree.
        if (
            not is_first
            and slot.is_holding_kv
            and req.kv_committed_len < slot.kv_committed_len
        ):
            old_end = slot.kv_allocated_len
            new_end = req.kv_committed_len
            if self.page_size > 1:
                new_end = ceil_align(new_end, self.page_size)
            new_end = max(new_end, slot.cache_protected_len)
            if new_end < old_end:
                kv_indices = self.req_to_token_pool.req_to_token[
                    slot.req_pool_idx, new_end:old_end
                ]
                self.token_to_kv_pool_allocator.free(kv_indices)
            slot.cache_protected_len = min(
                slot.cache_protected_len, req.kv_committed_len
            )

        slot.save_from_req(req, is_first=is_first)

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

    def _resolve_release_state(
        self, slot: SessionSlot, req: Optional[Req]
    ) -> tuple[int, Any]:
        """Resolve the currently tree-owned prefix for a session slot.

        A long-lived session can outlive radix-tree splits caused by unrelated
        traffic. In that case, the saved `last_node` may no longer represent the
        full protected prefix even though the slot's req_to_token row still
        contains tree-owned indices at the front. Re-match the current request
        text, then intersect the returned tree indices with the slot's row so
        release uses the prefix that is still actually backed by the tree.
        """
        protected_len = slot.cache_protected_len
        lock_node = slot.last_node

        # TODO: re-match logic disabled — match_prefix has side effects
        # (splits) that disturb tree accounting. Directly using
        # slot.last_node + cache_protected_len is safe after split analysis.
        return protected_len, lock_node

        if (
            req is None
            or not slot.is_holding_kv
            or slot.req_pool_idx is None
            or protected_len <= 0
        ):
            return protected_len, lock_node

        from sglang.srt.mem_cache.radix_cache import RadixKey

        token_ids = (req.origin_input_ids + req.output_ids)[: slot.kv_committed_len]
        if not token_ids:
            return 0, None

        match = self.inner.match_prefix(
            MatchPrefixParams(
                key=RadixKey(token_ids=token_ids, extra_key=req.extra_key),
                req=None,
            )
        )
        if len(match.device_indices) == 0:
            return 0, None

        max_protected_len = min(len(match.device_indices), protected_len)
        row_indices = self.req_to_token_pool.req_to_token[
            slot.req_pool_idx, :max_protected_len
        ].to(dtype=torch.int64)
        match_indices = match.device_indices[:max_protected_len]
        mismatches = (match_indices != row_indices).nonzero(as_tuple=False)
        if mismatches.numel() == 0 and max_protected_len == len(match.device_indices):
            common_len = max_protected_len
            return common_len, match.last_device_node

        common_len = (
            int(mismatches[0].item()) if mismatches.numel() > 0 else max_protected_len
        )
        if self.page_size > 1:
            common_len = (common_len // self.page_size) * self.page_size
        if common_len <= 0:
            return 0, None

        rematch = self.inner.match_prefix(
            MatchPrefixParams(
                key=RadixKey(token_ids=token_ids[:common_len], extra_key=req.extra_key),
                req=None,
            )
        )
        return len(rematch.device_indices), rematch.last_device_node

    def release_session(self, session_id: str, req: Optional[Req] = None):
        """Release all KV resources held by a streaming session."""
        slot = self.slots.pop(session_id, None)
        if slot is None:
            return
        protected_len, lock_node = self._resolve_release_state(slot, req)
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

    def session_held_tokens(self) -> int:
        """Total KV tokens held by session slots, not tracked by the tree.

        Excludes active slots whose tokens are already counted as part of
        the running request's uncached_size in the busy mem check.
        """
        total = 0
        for slot in self.slots.values():
            if slot.is_holding_kv and not slot.is_active:
                allocated = ceil_align(slot.kv_allocated_len, self.page_size)
                total += allocated - slot.cache_protected_len
        return total

    def session_held_full_tokens(self) -> int:
        """An alias to align the naming style of SWA"""
        return self.session_held_tokens()

    def session_held_swa_tokens(self) -> int:
        """Total SWA tokens held by session slots, not tracked by the tree."""
        total = 0
        for slot in self.slots.values():
            if slot.is_holding_kv and not slot.is_active:
                allocated = ceil_align(slot.kv_allocated_len, self.page_size)
                total += allocated - max(
                    slot.cache_protected_len, slot.swa_evicted_seqlen
                )
        return total

    def session_held_req_count(self) -> int:
        """Number of req pool slots held by session slots."""
        return sum(s.is_holding_kv and not s.is_active for s in self.slots.values())

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
