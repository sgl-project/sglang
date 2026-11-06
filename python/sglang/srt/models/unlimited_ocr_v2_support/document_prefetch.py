"""Overlap a split document's next window with the one decoding.

The serving layer sends window ``i+1`` while window ``i`` still decodes. The
scheduler parks it here instead of the waiting queue, encodes its pages on a
side stream (on MPS, in slices between forwards), and admits it only once
window ``i`` has finished (its KV is retained, or it could not be). The window
boundary then costs the follow-up's LM prefill alone; tokenisation and vision
ran under the previous decode. On MPS the follow-up, once its pages are
encoded, is queued early to run that prefill too (``DocReuse.hold``).

The follow-up's token budget is unknown when it is sent, so it carries the
whole remaining budget and is capped here by what window ``i`` actually used.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.managers import mm_schedule
from sglang.srt.managers.mm_schedule import (
    PrefetchedEmbeddingFunc,
    _move_items_to_device,
)
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
    ReqList,
    move,
)
from sglang.srt.models.unlimited_ocr_v2_support.document_reuse import (
    forced_len,
    req_state,
)
from sglang.srt.models.unlimited_ocr_v2_support.window_params import KEY_KEEP

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.req_types import MultimodalDataItem
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

# custom_params key: rid of the window this one must wait for.
KEY_AFTER = "uocr_doc_after"
# custom_params key: this window's share of the document budget before the
# predecessor's completion tokens are subtracted.
KEY_BUDGET = "uocr_doc_budget"


def _after(req) -> Optional[str]:
    """Rid of the window ``req`` must wait for, or None."""
    cp = req.sampling_params.custom_params
    if not isinstance(cp, dict):
        return None
    return cp.get(KEY_AFTER)


def _is_predecessor(req) -> bool:
    """Whether a later window of the same document follows ``req``."""
    cp = req.sampling_params.custom_params
    return isinstance(cp, dict) and bool(cp.get(KEY_KEEP))


def _apply_budget(req, used: int) -> None:
    """Cap a follow-up's budget by what its predecessor actually generated."""
    cp = req.sampling_params.custom_params
    budget = cp.get(KEY_BUDGET)
    if budget is None:
        return
    remaining = max(1, int(budget) - int(used))
    sp = req.sampling_params
    if sp.max_new_tokens is None or sp.max_new_tokens > remaining:
        sp.max_new_tokens = remaining


def _token_count(item) -> int:
    """Placeholder tokens an item spans in its prompt."""
    return sum(end - start + 1 for start, end in item.offsets)


class EarlyEmbeddings(PrefetchedEmbeddingFunc):
    """The image encoder, plus parked windows' pages it encoded ahead of prefill.

    Embeddings are keyed by the full item identity and kept until their window finishes:
    chunked prefill embeds an image once per chunk it overlaps.
    """

    # MPS: one encoder slice after every second forward. A slice after every
    # forward slows decode to ~35 tok/s while a window is encoded, this to ~52.
    _STEP_STRIDE = 2

    def __init__(
        self,
        *,
        encode: Callable[[List[MultimodalDataItem]], torch.Tensor],
        encode_steps: Optional[
            Callable[[List[MultimodalDataItem]], Iterator[None]]
        ] = None,
        step_context: Optional[Callable[[], Any]] = None,
    ):
        self._encode = encode
        # MPS: the same encoder as a generator that runs one slice per step.
        self._encode_steps = encode_steps
        self._step_context = step_context or nullcontext
        self._by_key: Dict[bytes, torch.Tensor] = {}
        self._owned: Dict[str, List[bytes]] = {}
        self._bytes = 0
        self._stream: Optional[torch.cuda.Stream] = None
        # MPS: (rid, items, steps) of windows whose pages are being encoded.
        self._stepped: List[Tuple[str, List[MultimodalDataItem], Iterator[None]]] = []
        self._step_tick = 0
        # MPS: windows beyond the first that hold a page (see ``_share``).
        self._sharers: Dict[bytes, int] = {}

    def __call__(self, items):
        return self._encode(items)

    def has_prefetched(self) -> bool:
        return bool(self._by_key) or bool(self._stepped)

    def encoding(self, rid: str) -> bool:
        """Whether window ``rid``'s pages are still being encoded."""
        return any(job[0] == rid for job in self._stepped)

    def prefetched(
        self, item_key: Optional[bytes], token_count: int
    ) -> Optional[torch.Tensor]:
        if self._stepped and item_key is not None and item_key not in self._by_key:
            # A prefill reached a page whose encode is still in slices.
            self._finish_job_of(item_key)
        emb = self._by_key.get(item_key) if item_key is not None else None
        if emb is None or emb.shape[0] != token_count:
            return None
        return emb

    def encode_ahead(
        self,
        *,
        rid: str,
        items: List[MultimodalDataItem],
        device: torch.device,
        forward_stream: Optional[torch.cuda.Stream],
        max_bytes: int,
    ) -> None:
        """Encode a parked window's images on a side stream, under the current decode.

        MPS has no side stream; there ``step`` encodes them a slice at a time.

        Features stay on the item: a retract or a recompute fallback still has
        them, and a miss here costs only time.
        """
        if torch.device(device).type == "mps":
            self._queue_steps(rid=rid, items=items, device=device, max_bytes=max_bytes)
            return
        items = [
            it
            for it in items
            if it.hash is not None and it.cache_key not in self._by_key
        ]
        if not items or any(it.feature is None or not it.offsets for it in items):
            return
        if self._bytes >= max_bytes:
            return
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        try:
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream), torch.no_grad():
                # Same device move as prefill; the moved features stay on the item
                # for a recompute fallback.
                _move_items_to_device(items, device, self._encode)
                embs = self._encode(items)
                if isinstance(embs, list):
                    embs = [e.reshape(-1, e.shape[-1]) for e in embs]
                else:
                    embs = torch.split(
                        embs.reshape(-1, embs.shape[-1]),
                        [_token_count(it) for it in items],
                    )
            # Later forwards on the default and forward streams read these.
            torch.cuda.current_stream().wait_stream(self._stream)
            forward_stream.wait_stream(self._stream)
        except Exception:
            logger.exception(
                "[Unlimited-OCR v2] early vision encode failed; encoding at prefill"
            )
            return
        owned = []
        for it, emb in zip(items, embs):
            if emb.shape[0] != _token_count(it):
                continue
            emb.record_stream(forward_stream)
            self._by_key[it.cache_key] = emb
            self._bytes += emb.numel() * emb.element_size()
            owned.append(it.cache_key)
        self._owned[rid] = owned

    def release(self, rid: str) -> None:
        """Free a window's early embeddings once it no longer needs them."""
        self._stepped = [job for job in self._stepped if job[0] != rid]
        for h in self._owned.pop(rid, ()):
            sharers = self._sharers.pop(h, 0)
            if sharers:
                # MPS: a window that shares this page still needs it.
                if sharers > 1:
                    self._sharers[h] = sharers - 1
                continue
            emb = self._by_key.pop(h, None)
            if emb is not None:
                self._bytes -= emb.numel() * emb.element_size()

    def clear(self) -> None:
        """Drop every early embedding."""
        self._stepped.clear()
        self._sharers.clear()
        self._by_key.clear()
        self._owned.clear()
        self._bytes = 0

    # -- MPS: one command queue, so the encode runs in slices between forwards --

    def _queue_steps(self, *, rid, items, device, max_bytes) -> None:
        missing = []
        for it in items:
            if it.hash is None:
                continue
            if it.cache_key in self._by_key:
                self._share(rid, it.cache_key)
            elif not self._in_embedding_cache(it):
                missing.append(it)
        if not missing or any(it.feature is None or not it.offsets for it in missing):
            return
        if self._encode_steps is None or self._bytes >= max_bytes:
            return
        try:
            _move_items_to_device(missing, device, self._encode)
            self._stepped.append((rid, missing, self._encode_steps(missing)))
        except Exception:
            logger.exception(
                "[Unlimited-OCR v2] early vision encode failed; encoding at prefill"
            )

    def _share(self, rid: str, key: bytes) -> None:
        """Hold a page another window encoded early, until both are done with it.

        A window's lookahead page is the next window's first page. Its prefill
        reads that page after the previous window has finished and released it.
        """
        owned = self._owned.setdefault(rid, [])
        if key not in owned:
            owned.append(key)
            self._sharers[key] = self._sharers.get(key, 0) + 1

    @staticmethod
    def _in_embedding_cache(item: MultimodalDataItem) -> bool:
        """Whether a prefill embedded the page already (such as the previous
        window's lookahead page when that window was not parked)."""
        cache = mm_schedule.embedding_cache
        cached = cache.get_single(item.cache_key) if cache is not None else None
        if cached is None:
            return False
        emb = cached.embedding
        return emb.reshape(-1, emb.shape[-1]).shape[0] == _token_count(item)

    def on_forward_step(self) -> None:
        """Run one slice of the oldest encode after every ``_STEP_STRIDE`` forwards."""
        if not self._stepped:
            return
        self._step_tick += 1
        if self._step_tick % self._STEP_STRIDE == 0:
            self._advance(self._stepped[0])

    def _finish_job_of(self, key: bytes) -> None:
        """Run the encode that holds ``key`` to its end; the others stay queued."""
        job = next(
            (j for j in self._stepped if any(it.cache_key == key for it in j[1])),
            None,
        )
        while job is not None and self._advance(job):
            pass

    def _advance(self, job) -> bool:
        """Run one slice of ``job``; False once it has ended (or failed)."""
        rid, items, steps = job
        try:
            with self._step_context():
                next(steps)
            return True
        except StopIteration as done:
            embs = done.value
        except Exception:
            embs = None
            logger.exception(
                "[Unlimited-OCR v2] early vision encode failed; encoding at prefill"
            )
        self._stepped = [j for j in self._stepped if j is not job]
        if embs is None:
            return False
        if isinstance(embs, list):
            embs = [e.reshape(-1, e.shape[-1]) for e in embs]
        else:
            embs = torch.split(
                embs.reshape(-1, embs.shape[-1]), [_token_count(it) for it in items]
            )
        owned = self._owned.setdefault(rid, [])
        for it, emb in zip(items, embs):
            if emb.shape[0] != _token_count(it):
                continue
            if it.cache_key in self._by_key:
                # Another window's encode finished the same page first.
                self._share(rid, it.cache_key)
                continue
            self._by_key[it.cache_key] = emb
            self._bytes += emb.numel() * emb.element_size()
            owned.append(it.cache_key)
        return False


class DocPrefetch:
    """One scheduler's parked follow-up windows and their predecessors' usage."""

    def __init__(
        self,
        *,
        early: Optional[EarlyEmbeddings],
        device: torch.device,
        forward_stream: Optional[torch.cuda.Stream],
        tp_size: int,
        abort_windows: Callable[..., None],
    ):
        self._abort_windows = abort_windows
        # None when the model encodes nothing ahead (no early embeddings).
        self._early = early
        self._device = device
        self._forward_stream = forward_stream
        # Vision runs on every TP rank in lockstep; it is not encoded out of step.
        self._encode_ahead = early is not None and tp_size == 1
        # Follow-ups waiting for their predecessor.
        self._parked = IndexedReqList(
            ListKind.UOCR_PARKED, abort=lambda r: self._abort(self._parked, r)
        )
        # Completion tokens of finished predecessors whose follow-up has not arrived.
        self._finished: Dict[str, int] = {}
        # Follow-ups whose predecessor finished, for the scheduler to queue.
        self._released = IndexedReqList(
            ListKind.UOCR_RELEASED, abort=lambda r: self._abort(self._released, r)
        )
        # Predecessor windows seen decoding, by rid, until they finish.
        self._live: Dict[str, Req] = {}
        # Follow-ups queued early (see ``DocReuse.hold``), by rid, until their
        # predecessor finishes.
        self._early_out: Dict[str, Req] = {}

    def park(self, req: Req) -> bool:
        """Hold a follow-up until its predecessor finishes; True if it was parked."""
        after = _after(req)
        if after is None or req.finished():
            return False
        st = req_state(req)
        if st.released:
            return False
        st.released = True
        if after in self._finished:
            _apply_budget(req, self._finished.pop(after))
            return False
        if not envs.SGLANG_ENABLE_UOCR_DOC_PREFETCH.get():
            return False
        self._parked.join([req])
        mm = req.multimodal_inputs
        if self._encode_ahead and mm is not None and mm.mm_items:
            self._early.encode_ahead(
                rid=req.rid,
                items=mm.mm_items,
                device=self._device,
                forward_stream=self._forward_stream,
                max_bytes=envs.SGLANG_UOCR_DOC_PREFETCH_MAX_MB.get() * 1024 * 1024,
            )
        return True

    def on_window_finished(self, req: Req) -> List[Req]:
        """Mark every follow-up of a finished ``req`` released, budget capped.

        Called from the KV-release path after the window's KV was retained; the
        scheduler queues released windows with ``take_released``. A retracted
        window comes through the same hook unfinished and releases nothing.
        Returns the follow-ups already queued early, budget capped as well, for
        the caller to release from their hold.
        """
        if _after(req) is None and not _is_predecessor(req):
            return []
        if not req.finished():
            return []
        self._live.pop(req.rid, None)
        self._early_out.pop(req.rid, None)
        self.release_early(req.rid)
        out = len(req.output_ids) if req.finished_len is None else req.finished_len
        used = max(0, out - forced_len(req))
        early = [r for r in self._early_out.values() if _after(r) == req.rid]
        for follow in early:
            del self._early_out[follow.rid]
            _apply_budget(follow, used)
        follows = [r for r in self._parked if _after(r) == req.rid]
        if not follows:
            if _is_predecessor(req) and not early:
                self._finished[req.rid] = used
            return early
        for follow in follows:
            _apply_budget(follow, used)
        move(follows, self._parked, self._released)
        return early

    def take_released(self) -> List[Req]:
        """Follow-ups released since the last call, in release order."""
        return self._released.take()

    def note_running(self, batch) -> None:
        """Record the predecessors a decode step just ran."""
        if batch.forward_mode is None or not batch.forward_mode.is_decode():
            return
        for req in batch.reqs:
            if _is_predecessor(req) and not req.finished():
                self._live[req.rid] = req

    def early_candidates(self) -> List[Tuple[Req, Req]]:
        """Parked follow-ups with a decoding predecessor and their pages encoded."""
        found = []
        for req in self._parked:
            pred = self._live.get(_after(req))
            if pred is None or pred.finished() or not pred.kv.holds_kv:
                continue
            if req_state(req).early_failed:
                continue
            if self._early is not None and self._early.encoding(req.rid):
                continue
            found.append((req, pred))
        return found

    def start_early(self, req: Req) -> None:
        """Take a parked follow-up out to be queued before its predecessor ends."""
        self._parked.take(lambda r: r is req)
        self._early_out[req.rid] = req

    def repark_early(self, req: Req) -> bool:
        """Park again an early follow-up whose predecessor still runs; True if so."""
        if self._early_out.pop(req.rid, None) is None:
            return False
        self._parked.join([req])
        return True

    def forget_early(self, rid: str) -> None:
        self._early_out.pop(rid, None)
        self._live.pop(rid, None)

    def request_lists(self) -> Dict[ListKind, ReqList]:
        return {
            ListKind.UOCR_PARKED: self._parked,
            ListKind.UOCR_RELEASED: self._released,
        }

    def _abort(self, windows: ReqList, recv_req: AbortReq) -> None:
        aborted = windows.take(lambda r: recv_req.matches(r.rid))
        for r in aborted:
            self.release_early(r.rid)
        self._abort_windows(aborted, finished_reason=recv_req.finished_reason)

    def forget_doc(self, doc_id: str) -> None:
        """Drop finished-predecessor records of a document that is done."""
        for k in [k for k in self._finished if k.startswith(doc_id)]:
            self._finished.pop(k)

    def held_count(self) -> int:
        """Follow-up windows held off the waiting queue."""
        return len(self._parked) + len(self._released)

    def step(self) -> None:
        """After a forward: on MPS, advance the parked windows' encode."""
        if self._early is not None:
            self._early.on_forward_step()

    def release_early(self, rid: str) -> None:
        """Free a window's early embeddings once it no longer needs them."""
        if self._early is not None:
            self._early.release(rid)

    def reset(self) -> List[Req]:
        """Forget everything; returns the held follow-ups for the caller to abort."""
        held = [*self._parked, *self._released]
        self._parked.clear()
        self._released.clear()
        self._finished.clear()
        self._live.clear()
        self._early_out.clear()
        if self._early is not None:
            self._early.clear()
        return held
