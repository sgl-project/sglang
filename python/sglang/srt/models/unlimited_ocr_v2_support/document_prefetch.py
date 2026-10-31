"""Overlap a split document's next window with the one decoding.

The serving layer sends window ``i+1`` while window ``i`` still decodes. The
scheduler parks it here instead of the waiting queue, encodes its pages on a
side stream, and admits it only once window ``i`` has finished (its KV is
retained, or it could not be). The window boundary then costs the follow-up's
LM prefill alone; tokenisation and vision ran under the previous decode.

The follow-up's token budget is unknown when it is sent, so it carries the
whole remaining budget and is capped here by what window ``i`` actually used.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.mm_schedule import (
    PrefetchedEmbeddingFunc,
    _move_items_to_device,
)
from sglang.srt.models.unlimited_ocr_v2_support.document_reuse import (
    forced_len,
    req_state,
)
from sglang.srt.models.unlimited_ocr_v2_support.window_params import KEY_KEEP

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import MultimodalDataItem, Req

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

    Embeddings are keyed by item hash and kept until their window finishes:
    chunked prefill embeds an image once per chunk it overlaps.
    """

    def __init__(self, *, encode: Callable[[List[MultimodalDataItem]], torch.Tensor]):
        self._encode = encode
        self._by_hash: Dict[int, torch.Tensor] = {}
        self._owned: Dict[str, List[int]] = {}
        self._bytes = 0
        self._stream: Optional[torch.cuda.Stream] = None

    def __call__(self, items):
        return self._encode(items)

    def has_prefetched(self) -> bool:
        return bool(self._by_hash)

    def prefetched(
        self, item_hash: Optional[int], token_count: int
    ) -> Optional[torch.Tensor]:
        emb = self._by_hash.get(item_hash) if item_hash is not None else None
        if emb is None or emb.shape[0] != token_count:
            return None
        return emb

    def encode_ahead(
        self,
        *,
        rid: str,
        items: List[MultimodalDataItem],
        device: torch.device,
        forward_stream: torch.cuda.Stream,
        max_bytes: int,
    ) -> None:
        """Encode a parked window's images on a side stream, under the current decode.

        Features stay on the item: a retract or a recompute fallback still has
        them, and a miss here costs only time.
        """
        items = [
            it for it in items if it.hash is not None and it.hash not in self._by_hash
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
            self._by_hash[it.hash] = emb
            self._bytes += emb.numel() * emb.element_size()
            owned.append(it.hash)
        self._owned[rid] = owned

    def release(self, rid: str) -> None:
        """Free a window's early embeddings once it no longer needs them."""
        for h in self._owned.pop(rid, ()):
            emb = self._by_hash.pop(h, None)
            if emb is not None:
                self._bytes -= emb.numel() * emb.element_size()

    def clear(self) -> None:
        """Drop every early embedding."""
        self._by_hash.clear()
        self._owned.clear()
        self._bytes = 0


class DocPrefetch:
    """One scheduler's parked follow-up windows and their predecessors' usage."""

    def __init__(
        self,
        *,
        early: Optional[EarlyEmbeddings],
        device: torch.device,
        forward_stream: Optional[torch.cuda.Stream],
        tp_size: int,
    ):
        # None when the model encodes nothing ahead (no early embeddings).
        self._early = early
        self._device = device
        self._forward_stream = forward_stream
        # Vision runs on every TP rank in lockstep; it is not encoded out of step.
        self._encode_ahead = early is not None and tp_size == 1
        # Parked follow-ups, keyed by the rid they wait for.
        self._parked: Dict[str, Req] = {}
        # Completion tokens of finished predecessors whose follow-up has not arrived.
        self._finished: Dict[str, int] = {}
        # Follow-ups whose predecessor finished, for the scheduler to queue.
        self._released: List[Req] = []

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
        self._parked[after] = req
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

    def on_window_finished(self, req: Req) -> None:
        """Mark the follow-up of a finished ``req`` released, budget capped.

        Called from the KV-release path after the window's KV was retained; the
        scheduler queues released windows with ``take_released``. A retracted
        window comes through the same hook unfinished and releases nothing.
        """
        if _after(req) is None and not _is_predecessor(req):
            return
        if not req.finished():
            return
        self.release_early(req.rid)
        out = len(req.output_ids) if req.finished_len is None else req.finished_len
        used = max(0, out - forced_len(req))
        follow = self._parked.pop(req.rid, None)
        if follow is None:
            if _is_predecessor(req):
                self._finished[req.rid] = used
            return
        _apply_budget(follow, used)
        self._released.append(follow)

    def take_released(self) -> List[Req]:
        """Follow-ups released since the last call, in release order."""
        out, self._released = self._released, []
        return out

    def drop(self, rid_prefix: str) -> List[Req]:
        """Remove the parked or released follow-ups an abort names."""
        hit = [k for k, r in self._parked.items() if r.rid.startswith(rid_prefix)]
        out = [self._parked.pop(k) for k in hit]
        kept = []
        for r in self._released:
            (out if r.rid.startswith(rid_prefix) else kept).append(r)
        self._released = kept
        for r in out:
            self.release_early(r.rid)
        return out

    def forget_doc(self, doc_id: str) -> None:
        """Drop finished-predecessor records of a document that is done."""
        for k in [k for k in self._finished if k.startswith(doc_id)]:
            self._finished.pop(k)

    def held_count(self) -> int:
        """Follow-up windows held off the waiting queue."""
        return len(self._parked) + len(self._released)

    def release_early(self, rid: str) -> None:
        """Free a window's early embeddings once it no longer needs them."""
        if self._early is not None:
            self._early.release(rid)

    def reset(self) -> List[Req]:
        """Forget everything; returns the held follow-ups for the caller to abort."""
        held = list(self._parked.values()) + self._released
        self._parked.clear()
        self._released = []
        self._finished.clear()
        if self._early is not None:
            self._early.clear()
        return held
