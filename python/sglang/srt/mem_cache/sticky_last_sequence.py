from __future__ import annotations

"""One-slot token-id continuation on top of ChunkCache.

Used when radix is off (SM70 CSA2 state is not in the tree) but a single
full-history chat should not re-prefill from position 0 every HTTP turn.

Exact continuation (``new_ids`` starts with the whole pinned sequence) reuses
the live CSA2 image. A shorter shared prefix hits only when that length was
recorded at a quiescent stop. Unfinished stops use ``extend_range.end`` — the
seq len the worker snapshotted — not ``len(origin + output)`` after the
prefill step appends the sampled token. Request end uses the finished token
length. An aborted chunked prefill keeps that last snap as the pin, so a
retry of the same prompt extends the tail instead of clearing the image.
The worker restores the SWA ring and ratio-2 pending for that stop; see
``sm70_csa2_boundary``. Any other shape is a miss: drop the pin and
full-prefill, which also clears the boundary store. When
``SGLANG_DSV41_CSA2_SESSION_DIR`` is set, the miss first spills that
conversation and may resume a different saved one at a recorded stop.
No ``session_id``. Do not wrap when streaming-session is on.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch

from sglang.srt.layers.attention.dsv4.sm70_csa2_boundary import (
    BOUNDARY_KEEP,
    csa2_image_len,
    evict_recent,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.managers.utils import is_health_check_generate_req
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    CacheRequestHandle,
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
    """Sentinel for inc/dec_lock_ref so ChunkCache's no-op lock stays no-op."""


@dataclass
class _Slot:
    virtual_node: _VirtualNode = field(default_factory=_VirtualNode)
    kv: ReqKvInfo = field(default_factory=ReqKvInfo)

    def save_from_req(self, req: Req, is_first: bool) -> None:
        kv = req.detach_kv()
        if is_first:
            self.kv = kv
        else:
            assert kv is self.kv

    def restore_to_req(self, req: Req) -> None:
        req.kv = self.kv


def _finished_token_ids(req: Req) -> list[int]:
    out = req.output_ids_through_stop
    return list(req.origin_input_ids) + list(out)


def _longest_cut(
    cuts: Sequence[int], new_ids: Sequence[int], last_ids: tuple[int, ...]
) -> int:
    """Longest recorded stop that is a proper prefix of both sequences."""
    best = 0
    limit = min(len(new_ids), len(last_ids))
    for length in cuts:
        if length <= 0 or length >= len(last_ids) or length > limit:
            continue
        if length > best and tuple(new_ids[:length]) == last_ids[:length]:
            best = length
    return best


def _cuts_still_on(
    cuts: Sequence[int],
    ids: Sequence[int],
    previous: Optional[tuple[int, ...]],
    limit: int,
) -> list[int]:
    """Cuts that are stops of ``ids``.

    A cut past ``len(previous)`` was stashed by the request now pinning, on
    top of the image ``previous`` described, so it needs no comparison.
    """
    return [
        length
        for length in cuts
        if 0 < length <= limit
        and (
            previous is None
            or length > len(previous)
            or tuple(ids[:length]) == previous[:length]
        )
    ]


def _imaged_len(origin_len: int, length: int) -> int:
    """How many of the finished tokens the CSA2 image holds.

    The last token gets its image only from a forward that reads it. The
    overlap loop runs one more verify step after the stop and writes it;
    without that loop the image ends one token short. Pinning that token
    would make the next turn resume past the image, so the pin stops at the
    image and the next turn reads the token again.
    """
    image = csa2_image_len()
    if image is None or not origin_len <= image < length:
        return length
    return image


def _is_exact_continuation(new_ids: Sequence[int], last_ids: tuple[int, ...]) -> bool:
    n = len(last_ids)
    if n == 0 or len(new_ids) < n:
        return False
    return tuple(new_ids[:n]) == last_ids


class StickyLastSequenceCache(BasePrefixCache):
    """Pin the last finished request's KV; reuse only on exact continuation."""

    def __init__(self, inner: BasePrefixCache):
        self.inner = inner
        self._slot: Optional[_Slot] = None
        self._last_ids: Optional[tuple[int, ...]] = None
        self._last_extra_key: Optional[str] = None
        self._last_cache_salt: Optional[str] = None
        # Lengths the worker has (or will have) a CSA2 ring/pending snap for.
        self._cuts: list[int] = []

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

    def supports_prefix_sharing(self) -> bool:
        return self.inner.supports_prefix_sharing()

    def session_records(self) -> dict[str, ReqKvInfo]:
        """The pin, so pool accounting counts its row as held while idle and
        a request running on it as not owning that row."""
        if self._slot is None or not self._slot.kv.holds_kv:
            return {}
        return {"sticky-last-seq": self._slot.kv}

    def reset(self) -> None:
        self._drop_slot("reset")
        self.inner.reset()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        hit = self._try_hit(params)
        if hit is not None:
            return hit
        return self.inner.match_prefix(params)

    def _try_hit(self, params: MatchPrefixParams) -> Optional[MatchResult]:
        # /health and /health_generate use a 1-token dummy that is never an
        # exact continuation of a real chat. Dropping the pin on that miss
        # forces the next Claude Code turn to full-prefill (and can OOM).
        if is_health_check_generate_req(params.req):
            return None
        key = params.key
        new_ids = key.raw_token_ids()
        if self._slot is None or self._last_ids is None:
            return self._match_saved_session(params, new_ids)
        if not self._slot.kv.holds_kv:
            self._clear_pin()
            return self._match_saved_session(params, new_ids)

        extra_ok = key.extra_key == self._last_extra_key
        salt_ok = (key.cache_salt or None) == self._last_cache_salt
        exact = extra_ok and salt_ok and _is_exact_continuation(new_ids, self._last_ids)
        prefix_len = len(self._last_ids) if exact else 0
        if not exact and extra_ok and salt_ok:
            prefix_len = _longest_cut(self._cuts, new_ids, self._last_ids)
        if prefix_len <= 0:
            saved = self._match_saved_session(params, new_ids)
            if saved is None:
                logger.info(
                    "sticky last-seq event=miss pinned=%d new=%d prefix=0 extend=%d cuts=%s",
                    len(self._last_ids),
                    len(new_ids),
                    len(new_ids),
                    self._cuts,
                )
                self._drop_slot("miss")
                return None
            return saved

        req = params.req
        if req is None:
            self._drop_slot("miss-no-req")
            return None

        slot = self._slot
        slot.restore_to_req(req)
        self._free_tail(req.kv, prefix_len)
        pinned_len = len(self._last_ids)
        if prefix_len < pinned_len:
            # The tail past the cut is freed here and dropped by the worker
            # restore, so the pin now describes only the shared prefix.
            self._cuts = [c for c in self._cuts if c <= prefix_len]
            self._last_ids = self._last_ids[:prefix_len]

        device_indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :prefix_len
        ].to(dtype=torch.int64)

        event = "exact" if exact else "prefix"
        logger.info(
            "sticky last-seq event=%s pinned=%d new=%d prefix=%d extend=%d cuts=%s",
            event,
            pinned_len,
            len(new_ids),
            prefix_len,
            max(0, len(new_ids) - prefix_len),
            self._cuts,
        )
        return MatchResult(
            device_indices=device_indices,
            last_device_node=slot.virtual_node,
            last_host_node=slot.virtual_node,
            best_match_node=slot.virtual_node,
            cache_protected_len=0,
        )

    def claim_kv_row(self, req: Req) -> bool:
        """Keep this request's kv row for the next turn.

        ``release_kv_cache`` frees the row unless this returns True after the
        record has been detached onto the pin.
        """
        if is_health_check_generate_req(req):
            return False
        if isinstance(req.finished_reason, FINISH_ABORT):
            # A chunked prefill that already snapshotted a prefix must stay
            # pinned. Otherwise the request frees the row: forget a pin it
            # was running on, or drop one it was not.
            if self._pin_aborted_prefix(req):
                return True
            if self._slot is not None and req.kv is self._slot.kv:
                self._clear_pin()
            else:
                self._drop_slot("abort")
            return False
        self._pin_finished(req)
        return True

    def _pin_finished(self, req: Req) -> None:
        ids = _finished_token_ids(req)
        finished_len = (
            req.finished_len if req.finished_len is not None else len(req.output_ids)
        )
        self._trim_overshoot(req, finished_len)
        ids = list(req.origin_input_ids) + list(req.output_ids[:finished_len])
        imaged = _imaged_len(len(req.origin_input_ids), len(ids))
        if imaged < len(ids):
            logger.info(
                "sticky last-seq pin stops at the CSA2 image: %d of %d tokens",
                imaged,
                len(ids),
            )
            ids = ids[:imaged]

        is_first = self._slot is None
        if is_first:
            self._slot = _Slot()
        self._slot.save_from_req(req, is_first=is_first)
        self._slot.kv.kv_committed_len = min(len(ids), self._slot.kv.kv_allocated_len)
        self._slot.kv.cache_protected_len = 0
        self._note_cut(ids)
        self._last_ids = tuple(ids)
        self._last_extra_key = getattr(req, "extra_key", None)
        self._last_cache_salt = getattr(req, "cache_salt", None) or None
        logger.info(
            "sticky last-seq pin %d tokens cuts=%s", len(self._last_ids), self._cuts
        )

    def checkpoint(self, req: Req, *, up_to: int, **kwargs):
        ids = list(req.origin_input_ids) + list(req.output_ids)
        if ids and not req.finished():
            # Prefill appends the sampled token before this runs. That token
            # is not in the CSA2 image yet; the snap is ``up_to``, which
            # checkpoint_kv_cache passes as extend_range.end.
            self._note_cut(ids, stop=up_to)
            logger.info(
                "sticky last-seq stop %d tokens cuts=%s",
                self._cuts[-1] if self._cuts else 0,
                self._cuts,
            )
        self.inner.checkpoint(req, up_to=up_to, **kwargs)

    def insert(self, *args, **kwargs):
        return self.inner.insert(*args, **kwargs)

    def evict(self, params: EvictParams) -> EvictResult:
        return self.inner.evict(params)

    def evict_for_alloc(self, params: EvictParams) -> EvictResult:
        return self.inner.evict_for_alloc(params)

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

    def protected_size(self):
        return self.inner.protected_size()

    def evictable_size(self):
        return self.inner.evictable_size()

    def full_evictable_size(self):
        return self.inner.full_evictable_size()

    def swa_evictable_size(self):
        return self.inner.swa_evictable_size()

    def full_protected_size(self):
        return self.inner.full_protected_size()

    def swa_protected_size(self):
        return self.inner.swa_protected_size()

    def total_size(self):
        return self.inner.total_size()

    def pretty_print(self):
        n = 0 if self._last_ids is None else len(self._last_ids)
        return f"StickyLastSequence(pinned={n})\n{self.inner.pretty_print()}"

    def init_load_back(self, params: InitLoadBackParams):
        return self.inner.init_load_back(params)

    def pop_prefetch_loaded_span(
        self, handle: CacheRequestHandle
    ) -> tuple[int, Optional[int]]:
        return self.inner.pop_prefetch_loaded_span(handle)

    def finish_storage_prefetch_admission(
        self, handle: CacheRequestHandle, fulfilled_tokens: int, reason: Optional[str]
    ) -> None:
        self.inner.finish_storage_prefetch_admission(handle, fulfilled_tokens, reason)

    def discard_storage_prefetch_accounting(self, handle: CacheRequestHandle) -> None:
        self.inner.discard_storage_prefetch_accounting(handle)

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

    def available_and_evictable_str(self):
        return self.inner.available_and_evictable_str()

    def init_metrics_collector(self):
        return self.inner.init_metrics_collector()

    def sanity_check(self):
        if self._slot is not None and self._slot.kv.holds_kv:
            return
        self.inner.sanity_check()

    def _match_saved_session(self, params: MatchPrefixParams, new_ids):
        from sglang.srt.layers.attention.dsv4.sm70_csa2_session import (
            image_ready,
            list_sessions,
            load_meta,
            remember,
            request_spill,
            session_key,
            sessions_enabled,
        )

        key = params.key

        if not sessions_enabled():
            return None
        current = None
        if self._last_ids:
            current = session_key(
                self._last_ids, self._last_extra_key, self._last_cache_salt
            )
            remember(
                current,
                self._last_ids,
                self._cuts,
                self._last_extra_key,
                self._last_cache_salt,
            )
            request_spill(current)
        best_key = None
        best_len = 0
        for record in list_sessions():
            rec_key = record.get("key")
            if not rec_key or rec_key == current or not image_ready(rec_key):
                continue
            if (record.get("extra_key") or None) != (key.extra_key or None):
                continue
            if (record.get("cache_salt") or None) != (key.cache_salt or None):
                continue
            ids = tuple(int(token) for token in record.get("ids") or [])
            cuts = [int(cut) for cut in record.get("cuts") or []]
            if _is_exact_continuation(new_ids, ids):
                length = len(ids)
            else:
                length = _longest_cut(cuts, new_ids, ids)
            if length > best_len:
                best_len = length
                best_key = rec_key
        if best_key is None:
            return None
        adopted = self._adopt_saved(best_key, best_len, params)
        if adopted is None:
            return None
        meta = load_meta(best_key)
        if meta is None:
            return None
        return adopted

    def _adopt_saved(self, saved_key: str, prefix_len: int, params: MatchPrefixParams):
        from sglang.srt.layers.attention.dsv4.sm70_csa2_session import (
            load_meta,
            request_load,
        )

        req = params.req
        meta = load_meta(saved_key)
        if req is None or meta is None or prefix_len <= 0:
            return None
        if not self._ensure_slot(prefix_len):
            return None
        request_load(saved_key)
        self._slot.restore_to_req(req)
        self._free_tail(req.kv, prefix_len)
        saved_ids = tuple(int(token) for token in meta.get("ids") or [])
        self._last_ids = saved_ids[:prefix_len]
        self._cuts = [
            int(cut) for cut in meta.get("cuts") or [] if 0 < int(cut) <= prefix_len
        ]
        self._last_extra_key = meta.get("extra_key")
        self._last_cache_salt = meta.get("cache_salt") or None
        device_indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :prefix_len
        ].to(dtype=torch.int64)
        logger.info(
            "sticky last-seq event=session pinned=%d new=%d prefix=%d extend=%d cuts=%s",
            len(saved_ids),
            len(params.key.raw_token_ids()),
            prefix_len,
            max(0, len(params.key.raw_token_ids()) - prefix_len),
            self._cuts,
        )
        return MatchResult(
            device_indices=device_indices,
            last_device_node=self._slot.virtual_node,
            last_host_node=self._slot.virtual_node,
            best_match_node=self._slot.virtual_node,
            cache_protected_len=0,
        )

    def _ensure_slot(self, prefix_len: int) -> bool:
        slot = self._slot
        if slot is not None and slot.kv.holds_kv:
            if slot.kv.kv_allocated_len >= prefix_len:
                return True
            if self.page_size != 1:
                return False
            need = prefix_len - int(slot.kv.kv_allocated_len)
            indices = self.token_to_kv_pool_allocator.alloc(need)
            if indices is None:
                return False
            row = self.req_to_token_pool.req_to_token[slot.kv.req_pool_idx]
            if int(row.shape[0]) < prefix_len:
                return False
            flat = indices.reshape(-1)
            start = int(slot.kv.kv_allocated_len)
            row[start:prefix_len] = flat[:need].to(dtype=row.dtype, device=row.device)
            slot.kv.kv_allocated_len = prefix_len
            return True
        return self._alloc_slot(prefix_len)

    def _alloc_slot(self, prefix_len: int) -> bool:
        if self.page_size != 1 or prefix_len <= 0:
            return False
        pool = self.req_to_token_pool
        alloc_rows = getattr(pool, "alloc_rows", None)
        free_rows = getattr(pool, "free_rows", None)
        if alloc_rows is None:
            return False
        rows = alloc_rows(1)
        if not rows:
            return False
        idx = int(rows[0])
        indices = self.token_to_kv_pool_allocator.alloc(prefix_len)
        if indices is None:
            if free_rows is not None:
                free_rows([idx])
            return False
        row = pool.req_to_token[idx]
        flat = indices.reshape(-1)
        if int(row.shape[0]) < prefix_len or int(flat.shape[0]) < prefix_len:
            if free_rows is not None:
                free_rows([idx])
            return False
        row[:prefix_len] = flat[:prefix_len].to(dtype=row.dtype, device=row.device)
        self._slot = _Slot()
        self._slot.kv = ReqKvInfo(
            req_pool_idx=idx,
            kv_committed_len=prefix_len,
            kv_allocated_len=prefix_len,
            cache_protected_len=0,
        )
        return True

    def _clear_pin(self) -> None:
        self._slot = None
        self._last_ids = None
        self._last_extra_key = None
        self._last_cache_salt = None
        self._cuts = []

    def _pin_aborted_prefix(self, req: Req) -> bool:
        """Keep the last completed chunk when a prefill is aborted.

        ``_last_ids`` is only that prefix. Pinning the full prompt would make
        the next exact hit resume past the CSA2 image.
        """
        ids = list(req.origin_input_ids) + list(req.output_ids)
        pin_len = self._aborted_prefix_len(req, ids)
        if pin_len <= 0:
            return False
        if self._slot is not None and req.kv is not self._slot.kv:
            self._release_slot_kv()
        self._free_tail(req.kv, pin_len)
        req.kv.cache_protected_len = 0
        if self._slot is None:
            self._slot = _Slot()
            self._slot.save_from_req(req, is_first=True)
        else:
            self._slot.save_from_req(req, is_first=False)
        prefix = tuple(ids[:pin_len])
        previous = self._last_ids
        self._last_ids = prefix
        self._last_extra_key = getattr(req, "extra_key", None)
        self._last_cache_salt = getattr(req, "cache_salt", None) or None
        self._cuts = _cuts_still_on(self._cuts, ids, previous, pin_len)
        if pin_len not in self._cuts:
            self._cuts.append(pin_len)
        self._cuts = evict_recent(self._cuts, BOUNDARY_KEEP)
        logger.info(
            "sticky last-seq pin %d tokens cuts=%s (abort)",
            pin_len,
            self._cuts,
        )
        return True

    def _aborted_prefix_len(self, req: Req, ids: Sequence[int]) -> int:
        if not ids or not req.kv.holds_kv:
            return 0
        limit = min(len(ids), req.kv.kv_allocated_len)
        best = 0
        for length in self._cuts:
            if 0 < length <= limit and length > best:
                best = length
        # The chunk currently on ``chunked_req`` is snapshotted in the worker
        # before the next schedule, which is where this abort runs. Its cut
        # is not in ``_cuts`` yet: stash happens later in the same step.
        extend_range = getattr(req, "extend_range", None)
        if extend_range is not None:
            stop = int(extend_range.end)
            if 0 < stop <= limit and stop > best:
                best = stop
        return best

    def _release_slot_kv(self) -> None:
        slot = self._slot
        self._slot = None
        if slot is None or not slot.kv.holds_kv:
            return
        self.free_kv_row(slot.kv, [(0, slot.kv.kv_allocated_len)])
        self.req_to_token_pool.free(slot)

    def _note_cut(self, ids: Sequence[int], stop: Optional[int] = None) -> None:
        self._cuts = _cuts_still_on(self._cuts, ids, self._last_ids, len(ids))
        length = len(ids) if stop is None else int(stop)
        if length <= 0 or length > len(ids):
            return
        self._cuts = [cut for cut in self._cuts if cut != length]
        self._cuts.append(length)
        self._cuts = evict_recent(self._cuts, BOUNDARY_KEEP)

    def _drop_slot(self, reason: str) -> None:
        slot = self._slot
        self._clear_pin()
        if slot is None or not slot.kv.holds_kv:
            return
        self.free_kv_row(slot.kv, [(0, slot.kv.kv_allocated_len)])
        self.req_to_token_pool.free(slot)
        logger.info("sticky last-seq drop (%s)", reason)

    def _free_tail(self, kv: ReqKvInfo, prefix_len: int) -> None:
        self._free_kv_aligned(kv, prefix_len, kv.kv_allocated_len)
        kv.kv_allocated_len = prefix_len
        kv.kv_committed_len = min(kv.kv_committed_len, prefix_len)
        kv.clamp_evicted_seqlens(prefix_len)

    def _trim_overshoot(self, req: Req, finished_len: int) -> None:
        target = len(req.origin_input_ids) + finished_len
        if self.page_size > 1 and req.kv.max_evicted_seqlen > target:
            target = (target // self.page_size) * self.page_size
        self._free_kv_aligned(req.kv, target, req.kv.kv_allocated_len)
        req.kv.kv_allocated_len = min(req.kv.kv_allocated_len, target)
        req.kv.kv_committed_len = min(req.kv.kv_committed_len, target)
        req.kv.clamp_evicted_seqlens(target)
        req.output_ids = req.output_ids[:finished_len]

    def _free_kv_aligned(self, kv: ReqKvInfo, target: int, end: int) -> None:
        if end <= target:
            return
        start = target
        if self.page_size > 1:
            start = ceil_align(start, self.page_size)
        self.free_kv_row(kv, [(start, end)])

    def __getattr__(self, name):
        return getattr(self.inner, name)
