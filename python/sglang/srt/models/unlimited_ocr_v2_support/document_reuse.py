"""Copy-splice KV reuse between Unlimited-OCR document windows.

A follow-up window copies the previous window's overlap-page KV and OCR text
instead of recomputing them (approximate; validated by OCR accuracy). One
``DocReuse`` holds a scheduler's state; each request carries its own on
``Req.uocr_doc``.
"""

from __future__ import annotations

import logging
from array import array
from collections import deque
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import msgspec
import torch

from sglang.srt.models.unlimited_ocr_v2_support.window_params import (
    KEY_DOC_ID,
    KEY_KEEP,
    KEY_LOOKAHEAD,
    KEY_OVERLAP,
    KEY_PAGE_BASE,
    KEY_REUSE,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.unlimited_ocr_v2_backend import (
        UnlimitedOCRV2AttnBackend,
    )
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.schedule_policy import PrefillAdder
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)

__all__ = [
    "KEY_DOC_ID",
    "KEY_KEEP",
    "KEY_LOOKAHEAD",
    "KEY_OVERLAP",
    "KEY_PAGE_BASE",
    "KEY_REUSE",
    "META_FORCED_TOKENS",
    "DocReuse",
    "UocrReqState",
    "window_reused",
]

# Multimodal placeholders are rewritten above this value (see page_state.py).
MM_PAD_SHIFT_VALUE = 1_000_000

# meta_info key: tokens spliced into this window from its predecessor. Non-zero
# means the window reused KV, so its answer starts at its own first page.
META_FORCED_TOKENS = "uocr_doc_forced_tokens"


def window_reused(meta_info: Optional[dict]) -> bool:
    """Whether the scheduler spliced a predecessor window's KV into this one.

    The count is reported once per streamed chunk, so any non-zero entry counts.
    """
    return bool(any((meta_info or {}).get(META_FORCED_TOKENS) or ()))


# ---------------------------------------------------------------------------
# Pure token-layout helpers (CPU, unit-tested)
# ---------------------------------------------------------------------------
def is_image_token(token: int, image_token_id: int) -> bool:
    """Whether ``token`` is an image placeholder (raw or pad-shifted)."""
    return token >= MM_PAD_SHIFT_VALUE or token == image_token_id


class PromptPages(msgspec.Struct, frozen=True):
    """Token positions of a window prompt: prefix, per-page spans, instruction."""

    prefix_len: int
    page_starts: Tuple[int, ...]
    page_ends: Tuple[int, ...]
    instr_start: int


def parse_prompt(
    ids: Sequence[int], page_token_id: int, image_token_id: int
) -> Optional[PromptPages]:
    """Parse ``prefix <PAGE><image>... <PAGE><image>... instruction``."""
    n = len(ids)
    pos = 0
    while pos < n and ids[pos] != page_token_id:
        if is_image_token(ids[pos], image_token_id):
            return None
        pos += 1
    if pos >= n:
        return None
    prefix_len = pos
    starts: List[int] = []
    ends: List[int] = []
    while pos < n and ids[pos] == page_token_id:
        starts.append(pos)
        pos += 1
        img_start = pos
        while pos < n and is_image_token(ids[pos], image_token_id):
            pos += 1
        if pos == img_start:
            return None
        ends.append(pos)
    # The instruction must not contain more images or prompt markers.
    for t in ids[pos:]:
        if is_image_token(t, image_token_id):
            return None
    return PromptPages(prefix_len, tuple(starts), tuple(ends), pos)


# The dense local window every query reads: sparse_local_block *
# sparse_block_size (== sliding_window_size).
TAIL_TOKENS = 128

# Pages the rewritten prompt carries as context only: the image tail. The
# window does not transcribe it, so decode resumes past it.
CONTEXT_PAGES = 1


class TailCopyPlan(msgspec.Struct, frozen=True):
    """Image tail to copy, and the rewritten prompt that matches it.

    The cached-prefix contract is positional (``prefix_indices[i]`` holds the KV
    of ``fill_ids[i]``), so copying just the tail means rewriting the prompt to
    ``[<PAGE> | image tail | new pages]``: ``kept_from`` is where the new prompt
    resumes, ``offset_delta`` how far its items move. The prompt has no BOS: no
    recomputed query's local window or page-bucketed top-k reaches row 0, only
    page -1 queries could route to it.

    The instruction is dropped too; the document's first window prefilled it
    once. Its prefill sample is replaced by the spliced text, and decode reads
    the text tail and same-page image blocks, so the only loss is the last
    block of the last page. Without an instruction the model does not stop on
    its own, so the window ends on the marker past its last page
    (``UocrReqState.end_on_marker``, see ``_page_guard``).
    """

    src_positions: Tuple[int, ...]
    new_ids: List[int]
    copy_len: int
    kept_from: int
    offset_delta: int
    old_pages: int
    new_pages: int
    # Leading pages of the new window whose KV is copied from the old window's
    # lookahead instead of being prefilled again.
    copied_pages: int = 0
    # The copied page's multimodal item straddles ``kept_from`` on purpose: a
    # one-page window keeps its last image token to prefill (see below).
    allow_straddle: bool = False


def plan_tail_copy(
    old_ids: Sequence[int],
    new_ids: Sequence[int],
    overlap: int,
    page_token_id: int,
    image_token_id: int,
    page_size: int,
    old_lookahead: int = 0,
) -> Optional[TailCopyPlan]:
    """Plan a tail-only reuse of the old window's last own page.

    The tail starts on a KV page (``E - TAIL_TOKENS - E % page_size``, ``E``
    being the row position past the last image token), which is the granularity
    retention frees at (see ``tail_ranges``). ``old_lookahead`` trailing pages
    of the old prompt were only read, so the tail comes from the page before.

    The old window's lookahead page, when it is this window's first page, is
    copied whole after the tail. Its image KV depends on the page itself and
    the 128 tokens before it, which are that very tail, so it equals what this
    window would prefill; vision skips the page too.
    """
    old = parse_prompt(old_ids, page_token_id, image_token_id)
    new = parse_prompt(new_ids, page_token_id, image_token_id)
    if old is None or new is None:
        return None
    p_old, p_new = len(old.page_starts) - old_lookahead, len(new.page_starts)
    if p_old < 1 or overlap < 0 or overlap >= p_new or overlap > p_old:
        return None
    end = old.page_ends[p_old - 1]
    start = end - TAIL_TOKENS - end % page_size
    # The tail's ids come from the old window, so the new one need not carry
    # that page: `overlap` only says how many leading pages the rewrite drops.
    if start <= old.page_starts[p_old - 1]:
        return None
    new_end = new.page_ends[overlap - 1] if overlap > 0 else new.prefix_len
    if overlap > 0 and list(old_ids[start:end]) != list(
        new_ids[new_end - (end - start) : new_end]
    ):
        return None
    # The tail opens its page with a marker whose KV comes from the old
    # window's marker of that page.
    marker_src = old.page_starts[p_old - 1]
    copy_end, kept_from, copied, straddle = end, new_end, 0, False
    if old_lookahead > 0 and overlap == 0:
        la_start, la_end = old.page_starts[p_old], old.page_ends[p_old]
        if la_start == end and list(old_ids[la_start:la_end]) == list(
            new_ids[new.page_starts[0] : new.page_ends[0]]
        ):
            copy_end, kept_from, copied = la_end, new.page_ends[0], 1
            if p_new == 1:
                # The extend needs a token to run: leave the page's last image
                # token to prefill. Its item still gets encoded, one token used.
                copy_end, kept_from, straddle = (
                    la_end - 1,
                    new.page_ends[0] - 1,
                    True,
                )
    head = [page_token_id] + list(old_ids[start:copy_end])
    src = (marker_src,) + tuple(range(start, copy_end))
    return TailCopyPlan(
        src_positions=src,
        new_ids=head + list(new_ids[kept_from : new.instr_start]),
        copy_len=len(head),
        kept_from=kept_from,
        offset_delta=len(head) - kept_from,
        old_pages=p_old,
        new_pages=p_new,
        copied_pages=copied,
        allow_straddle=straddle,
    )


def text_tail(
    output_ids: Sequence[int], n: int, page_token_id: int
) -> Optional[Tuple[int, List[int]]]:
    """Return ``(start, tokens)``: up to ``n`` trailing tokens of the last page.

    Cut at the last marker so the slice carries none: a copied marker would be
    replayed against the rewritten prompt's smaller page count and land in the
    wrong page bucket.
    """
    if not output_ids or n <= 0:
        return None
    last_marker = -1
    for i in range(len(output_ids) - 1, -1, -1):
        if output_ids[i] == page_token_id:
            last_marker = i
            break
    start = max(last_marker + 1, len(output_ids) - n)
    return start, list(output_ids[start:])


def tail_ranges(
    input_ids: Sequence[int],
    output_ids: Sequence[int],
    text_ok: bool,
    page_token_id: int,
    image_token_id: int,
    page_size: int,
    allocated: int,
    lookahead: int = 0,
) -> Optional[List[Tuple[int, int]]]:
    """Row ranges a follow-up window reads: last own marker, image tail, text tail.

    Widened to whole KV pages (the allocation unit) and clipped to ``allocated``.
    Of the ``lookahead`` trailing pages only the first is kept (the next window
    copies it whole). None means the layout is not reusable.
    """
    pages = parse_prompt(input_ids, page_token_id, image_token_id)
    if pages is None or len(pages.page_starts) <= lookahead:
        return None
    last = len(pages.page_starts) - lookahead - 1
    end = pages.page_ends[last]
    spans = [
        (pages.page_starts[last], pages.page_starts[last] + 1),
        (max(0, end - TAIL_TOKENS - end % page_size), end),
    ]
    if lookahead > 0:
        spans.append((pages.page_starts[last + 1], pages.page_ends[last + 1]))
    if text_ok:
        tail = text_tail(output_ids, TAIL_TOKENS, page_token_id)
        if tail is not None:
            start = len(input_ids) + tail[0]
            spans.append((start, start + len(tail[1])))
    merged: List[Tuple[int, int]] = []
    for lo, hi in sorted(spans):
        lo = lo // page_size * page_size
        hi = min(-(-hi // page_size) * page_size, allocated)
        if lo >= hi:
            continue
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def replay_page_isolation(
    ids: Sequence[int], page_token_id: int, image_token_id: int, per_page: int
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Per-slot (img_page, marker_cum, img_run, prompt_pages), as page_state.step."""
    img_page, marker_cum, img_run, prompt_pages = [], [], [], []
    mc = ir = pp = 0
    for t in ids:
        is_img = is_image_token(t, image_token_id)
        if t == page_token_id:
            mc += 1
        if is_img:
            ir += 1
            pp = max(pp, mc)
            img_page.append(mc - 1 if mc > 0 else (ir - 1) // per_page)
        else:
            img_page.append(-1)
        marker_cum.append(mc)
        img_run.append(ir)
        prompt_pages.append(pp)
    return img_page, marker_cum, img_run, prompt_pages


def replay_coords(
    ids: Sequence[int],
    gen_start: int,
    page_token_id: int,
    image_token_id: int,
    max_pages: int,
) -> Tuple[List[int], Tuple[int, int, List[int]]]:
    """Page bucket per token and the final carry, as PageCoordState.step."""
    pages: List[int] = []
    cnt = [0] * max_pages
    mc = pp = 0
    for i, t in enumerate(ids):
        is_img = is_image_token(t, image_token_id)
        if t == page_token_id:
            mc += 1
        if is_img:
            pp = max(pp, mc)
        in_prefill = i < gen_start and (is_img or mc > pp)
        resp = min(max(mc - pp, 0), pp)
        page = mc if in_prefill else resp
        page = min(max(page, 0), max_pages - 1)
        cnt[page] += 1
        pages.append(page)
    return pages, (mc, pp, cnt)


def continuation_carry(
    new_ids: Sequence[int],
    page_token_id: int,
    image_token_id: int,
    per_page: int,
    max_pages: int,
):
    """State a reuse window's first decode step continues from.

    Response text of document page ``k`` shares its page bucket with that
    page's prompt images, so decode resumes ``CONTEXT_PAGES`` pages in: the
    image tail is spliced, not decoded. Returns the RoPE carry
    ``(marker_cum, prompt_pages, cnt)`` and the isolation carry
    ``(marker_cum, img_run, prompt_pages)``.
    """
    _, (mc, pp, cnt) = replay_coords(
        new_ids, len(new_ids), page_token_id, image_token_id, max_pages
    )
    iso = replay_page_isolation(new_ids, page_token_id, image_token_id, per_page)
    return (mc + CONTEXT_PAGES, pp, cnt), (pp + CONTEXT_PAGES, iso[2][-1], pp)


# ---------------------------------------------------------------------------
# GPU-side edits (run by the backend before a forward)
# ---------------------------------------------------------------------------
def _write_page_state(backend, slots, values) -> None:
    img_page, marker_cum, img_run, prompt_pages = (
        torch.tensor(v, dtype=torch.int32, device=slots.device) for v in values
    )
    backend.write_page_isolation(
        slots=slots,
        img_page=img_page,
        marker_cum=marker_cum,
        img_run=img_run,
        prompt_pages=prompt_pages,
    )


def _seed_continuation(backend, slots, plan) -> None:
    """Carry the first decode step's page when no text could be spliced.

    The isolation carry is read off the slot before the step, so the prompt's
    own tail slot stands in for the text that was not copied.
    """
    _write_page_state(backend, slots, tuple([v] for v in (-1, *plan.iso)))
    marker, pp, cnt = plan.coord
    backend.seed_coord_row(
        first_block=plan.first_block, marker=marker, prompt_pages=pp, counts=cnt
    )


def _apply_segment(backend, *, dst, src, values, kbar_pages, kbar_toks, coord) -> None:
    """Copy one segment of KV and rebuild every model state attached to it.

    No page-axis re-rotation: RoPE page coordinates use global page indices
    (see ``KEY_PAGE_BASE``), so a copied page keeps its encoding.
    """
    if dst.numel() > 0:
        backend.kv_pool.move_kv_cache(dst, src)
        _write_page_state(backend, dst, values)
        backend.refresh_page_summaries(
            pages=kbar_pages,
            toks=torch.tensor(kbar_toks, dtype=torch.int64, device=dst.device),
        )
    if coord is not None:
        first_block, marker, pp, cnt = coord
        backend.seed_coord_row(
            first_block=first_block, marker=marker, prompt_pages=pp, counts=cnt
        )


# ---------------------------------------------------------------------------
# Scheduler-side records
# ---------------------------------------------------------------------------
class DocEntry(msgspec.Struct):
    """A finished window retained until its follow-up window consumes it.

    Mutable (``kept`` is set after trimming); ``DocReuse.sources`` keys entries by id().
    """

    holder: Any  # RetainedWindow owning the finished window's KV row
    input_ids: List[int]
    output_ids: List[int]
    # Document page the next window must start at: with no page in common, the
    # only proof that the copied tail is the page before it.
    next_page: int = 0
    # Row ranges still holding KV; None means the whole row [0, allocated).
    kept: Optional[List[Tuple[int, int]]] = None
    # Trailing pages of ``input_ids`` the window only read (KEY_LOOKAHEAD).
    lookahead: int = 0


class TextPlan(msgspec.Struct, frozen=True):
    """Overlap-page OCR text to splice in after a follow-up window's prefill."""

    forced: List[int]
    src_slots: torch.Tensor
    # Page-isolation carry for the first decode step, in the rewritten prompt's
    # frame: ``(marker_cum, img_run, prompt_pages)``.
    iso: Tuple[int, int, int]
    # RoPE coordinate carry for the same step: ``(marker_cum, prompt_pages, cnt)``.
    coord: Tuple[int, int, List[int]]
    first_block: torch.Tensor


class PromptSnapshot(msgspec.Struct, frozen=True):
    """A window's prompt as it was sent, before the tail-copy rewrite."""

    input_ids: Any
    input_ids_unpadded: Any
    fill_ids: Any
    items: List[Any]
    offsets: List[List[Tuple[int, int]]]


class UocrReqState(msgspec.Struct):
    """One request's document-reuse state, kept on ``Req.uocr_doc``; mutable."""

    # (match result, retraction count it was computed at); None until matched.
    match: Optional[Tuple[Any, int]] = None
    # The retained window this request consumes.
    source: Optional[DocEntry] = None
    # Prefix slots the matcher allocated; until alloc_for_extend registers them
    # in the request's row, this is their only record.
    pending: Optional[torch.Tensor] = None
    # The prompt as sent, kept until the request finishes so a retract restores it.
    orig_prompt: Optional[PromptSnapshot] = None
    text_plan: Optional[TextPlan] = None
    # RoPE page base of a rewritten prompt; None means the custom_params one.
    page_base: Optional[int] = None
    # Leading prompt pages the window carries but does not transcribe.
    context_pages: int = 0
    # The prompt has no instruction: the window ends on the marker past its pages.
    end_on_marker: bool = False
    # Queued ahead of new documents (see ``DocReuse.queue_position``).
    front: bool = False
    # Output tokens spliced in from the previous window, not generated.
    forced_len: int = 0
    # The prefill step's sample is replaced by the spliced text.
    skip_prefill_append: bool = False
    # First KV blocks whose coordinate rows were seeded with a pinned stamp.
    pinned_blocks: List[Any] = []
    # Prefetch: the window has been through ``DocPrefetch.park``.
    released: bool = False


def req_state(req: Req) -> UocrReqState:
    """The request's reuse state, created on first write."""
    if req.uocr_doc is None:
        req.uocr_doc = UocrReqState()
    return req.uocr_doc


def forced_len(req: Req) -> int:
    """Output tokens copied from the previous window rather than generated."""
    return 0 if req.uocr_doc is None else req.uocr_doc.forced_len


def _page_base(req) -> int:
    """RoPE page offset of this window: window-local page id + base == global.

    A recomputed window numbers its own pages 1..n, so its base is ``lo``. A
    rewritten prompt numbers the tail and the new pages 1, 2, 3..., so its base
    is ``emit_from - CONTEXT_PAGES``.
    """
    st = req.uocr_doc
    if st is not None and st.page_base is not None:
        return int(st.page_base)
    cp = req.sampling_params.custom_params
    if not isinstance(cp, dict):
        return 0
    return int(cp.get(KEY_PAGE_BASE) or 0)


def _context_pages(req) -> int:
    return 0 if req.uocr_doc is None else req.uocr_doc.context_pages


def _doc_params(req) -> Optional[dict]:
    cp = req.sampling_params.custom_params
    if not isinstance(cp, dict) or not cp.get(KEY_DOC_ID):
        return None
    return cp


def _stop_ids(req) -> set:
    ids = set(req.eos_token_ids or ())
    tok = req.tokenizer
    # Tokenizer backends are duck-typed; not every one exposes eos_token_id.
    eos = getattr(tok, "eos_token_id", None) if tok is not None else None
    if isinstance(eos, int):
        ids.add(eos)
    stop = req.sampling_params.stop_token_ids
    if stop:
        ids.update(stop)
    return ids


def _kept_ranges(entry: DocEntry) -> List[Tuple[int, int]]:
    if entry.kept is not None:
        return entry.kept
    return [(0, entry.holder.kv.kv_allocated_len)]


def prefix_device_indices(req) -> Optional[torch.Tensor]:
    """KV slots of a matched follow-up's prefix, or None if it has none.

    The match reports only the prefix length; ``alloc_for_extend`` asks the
    cache for the slots of a request that holds no row yet, and the copied
    tail lives in the matcher's pending allocation, not on a tree path.
    """
    st = req.uocr_doc
    if st is None or st.pending is None or req.kv.holds_kv:
        return None
    alloc = st.pending
    assert req.prefix_len <= alloc.numel(), (req.rid, req.prefix_len, alloc.numel())
    return alloc[: req.prefix_len]


def _cut_at_extra_marker(out: List[int], page_token_id: int, pages: int) -> List[int]:
    """Drop what a marker-ended window wrote from the marker past its last page.

    Keeping that marker would leave ``text_tail`` nothing to splice after it.
    The guard sees outputs one step late under overlap scheduling, so another
    token, even another marker, can follow before its forced EOS.
    """
    seen = 0
    for i, t in enumerate(out):
        if t == page_token_id:
            seen += 1
            if seen > pages:
                return out[:i]
    return out


def next_page_after(
    input_ids: Sequence[int],
    first_page: int,
    context_pages: int,
    page_token_id: int,
    image_token_id: int,
    lookahead: int = 0,
) -> Optional[int]:
    """Document page index that follows the window described by ``input_ids``.

    A reused window's prompt opens with ``context_pages`` pages it does not
    transcribe and ends with ``lookahead`` pages it only reads, so its own
    pages are the markers between those.
    """
    pages = parse_prompt(input_ids, page_token_id, image_token_id)
    if pages is None:
        return None
    return first_page + len(pages.page_starts) - context_pages - lookahead


def hide_forced_text(req) -> None:
    """Skip spliced tokens; the first detokenize header carries their context."""
    req.send_token_offset = len(req.output_ids)


def _rewrite_prompt(req, plan: TailCopyPlan) -> bool:
    """Replace the window's prompt with the tail-copy layout; True on success.

    The cached prefix is positional and vision only skips multimodal items whose
    whole span is inside it, so the discarded pages' items must go and the rest
    must move by ``plan.offset_delta``. Returns False when an item straddles the
    boundary, which would leave a half-prefilled image.
    """
    mm = req.multimodal_inputs
    items = mm.mm_items if mm is not None else []
    if any(not item.offsets for item in items):
        return False
    straddling = [
        item
        for item in items
        if any(start < plan.kept_from <= end for start, end in item.offsets)
    ]
    if straddling and not (
        plan.allow_straddle
        and len(straddling) == 1
        and straddling[0].offsets[-1][1] == plan.kept_from
    ):
        # A half-prefilled image: vision would skip it, attention would not.
        # The one allowed case prefills only the image's last token, and the
        # chunked embedding path slices exactly that token out of the item.
        return False
    # A dropped or retracted window recomputes the prompt it was sent, so
    # snapshot before the offsets move and the items are culled.
    req_state(req).orig_prompt = PromptSnapshot(
        input_ids=req.origin_input_ids,
        input_ids_unpadded=req.origin_input_ids_unpadded,
        fill_ids=req.full_untruncated_fill_ids,
        items=items,
        offsets=[list(item.offsets) for item in items],
    )
    kept = []
    for item in items:
        if item not in straddling and all(
            end < plan.kept_from for _, end in item.offsets
        ):
            continue
        item.offsets = [
            (start + plan.offset_delta, end + plan.offset_delta)
            for start, end in item.offsets
        ]
        kept.append(item)
    if mm is not None:
        mm.mm_items = kept
    aliased = req.origin_input_ids_unpadded is req.origin_input_ids
    ids = array("q", plan.new_ids)
    req.origin_input_ids = ids
    req.origin_input_ids_unpadded = ids if aliased else array("q", plan.new_ids)
    req.full_untruncated_fill_ids = array("q", plan.new_ids)
    return True


def _put_back_prompt(req, saved: PromptSnapshot) -> None:
    req.origin_input_ids = saved.input_ids
    req.origin_input_ids_unpadded = saved.input_ids_unpadded
    for item, off in zip(saved.items, saved.offsets):
        item.offsets = off
    if req.multimodal_inputs is not None:
        req.multimodal_inputs.mm_items = saved.items


def _restore_prompt(req) -> None:
    """Undo `_rewrite_prompt` for a window that never ran, to recompute it."""
    st = req_state(req)
    saved = st.orig_prompt
    st.orig_prompt = None
    st.page_base = None
    # The restored prompt transcribes every page it carries.
    st.context_pages = 0
    st.end_on_marker = False
    if saved is None:
        return
    _put_back_prompt(req, saved)
    req.full_untruncated_fill_ids = saved.fill_ids


def _mark_rewritten(req, *, alloc: torch.Tensor, page_base: int) -> None:
    """Record the per-request state a rewritten tail-copy prompt runs with."""
    st = req_state(req)
    # The rewritten prompt numbers its pages 1 (tail), 2.. (new), so the global
    # page of the tail is this window's first owned page minus one.
    st.page_base = page_base - CONTEXT_PAGES
    # Set before the first forward: the page guard reads it when it samples
    # the prefill step's token, which is before `_splice_text` runs.
    st.context_pages = CONTEXT_PAGES
    # The rewritten prompt has no instruction, so the page guard ends the
    # window on the marker past its last page instead of waiting for EOS.
    st.end_on_marker = True
    st.pending = alloc


# ---------------------------------------------------------------------------
# Per-scheduler state
# ---------------------------------------------------------------------------
class DocReuse:
    """One scheduler's document-window reuse: retained KV, queued GPU edits."""

    def __init__(
        self,
        *,
        cache: UnifiedRadixCache,
        backend: Optional[UnlimitedOCRV2AttnBackend],
        on_window_finished: Callable[[Req], None],
    ):
        self._cache = cache
        # None when the attention backend is not Unlimited-OCR v2's: no reuse.
        self._backend = backend
        self._on_window_finished = on_window_finished
        # Finished windows retained for their follow-up, by doc id, oldest first.
        self.docs: Dict[str, DocEntry] = {}
        # Entries consumed by a running follow-up window, freed when it is done.
        self.sources: Dict[int, DocEntry] = {}
        # GPU-side edits for the next forward, tagged with the rid they serve;
        # None marks an edit no drop may cancel.
        self._ops: Deque[Tuple[Optional[str], Callable[[Any], None]]] = deque()
        # req_pool_idx -> token the next decode step must read instead.
        self._input_overrides: Dict[int, int] = {}

    # -- Queued GPU work --

    def _enqueue(self, owner: Optional[str], op: Callable[[Any], None]) -> None:
        self._ops.append((owner, op))

    def _cancel(self, owner: str) -> None:
        self._ops = deque((o, op) for o, op in self._ops if o != owner)

    @torch.no_grad()
    def run_pending(self, backend) -> None:
        """Run queued KV/state edits in order; the backend calls this before a forward."""
        # Popped one at a time so an edit that raises leaves the rest queued.
        while self._ops:
            _, op = self._ops.popleft()
            op(backend)

    def apply_input_overrides(
        self,
        *,
        output_tokens_buf: Optional[torch.Tensor],
        relay: Optional[torch.Tensor],
    ) -> None:
        """Replace relayed decode inputs before the next forward reads them."""
        if not self._input_overrides:
            return
        items = list(self._input_overrides.items())
        self._input_overrides.clear()
        rows = torch.tensor([r for r, _ in items], dtype=torch.int64)
        toks = torch.tensor([t for _, t in items], dtype=torch.int64)
        # Prefill-only paths have no future map.
        for target in (output_tokens_buf, relay):
            if target is None:
                continue
            target[rows.to(target.device)] = toks.to(
                device=target.device, dtype=target.dtype
            )

    def note_prompt_lens(self, batch: ScheduleBatch) -> None:
        """Tell the backend each extending request's generation start and page base.

        Page state numbers prompt and generated tokens differently, and an extend
        is not always all prompt: a retracted request re-prefills its prompt plus
        the output it had generated.
        """
        b = self._backend
        mode = batch.forward_mode
        if b is None or mode is None or not mode.is_extend() or not batch.reqs:
            return
        rows = [r.kv.req_pool_idx for r in batch.reqs]
        if any(row is None for row in rows):
            return
        lens = [len(r.origin_input_ids) for r in batch.reqs]
        bases = [_page_base(r) for r in batch.reqs]
        dev = b.device
        b.note_prompt_lens(
            torch.tensor(rows, dtype=torch.int64).to(dev, non_blocking=True),
            torch.tensor(lens, dtype=torch.int64).to(dev, non_blocking=True),
            torch.tensor(bases, dtype=torch.int64).to(dev, non_blocking=True),
        )

    def _unpin(self, req) -> None:
        """Let LRU reuse coordinate rows a request seeded but will not step."""
        st = req.uocr_doc
        if st is None or not st.pinned_blocks:
            return
        blocks = torch.cat(st.pinned_blocks)
        st.pinned_blocks = []
        self._enqueue(None, lambda be, blocks=blocks: be.unpin_coord_rows(blocks))

    # -- Accounting --

    def prefix_device_indices(self, req: Req) -> Optional[torch.Tensor]:
        """KV slots of a matched follow-up's prefix, or None if it has none."""
        return prefix_device_indices(req)

    def _entries(self):
        return chain(self.docs.values(), self.sources.values())

    def held_slots(self) -> List[Any]:
        """Holder slots whose KV rows are owned by document reuse (for accounting)."""
        return [e.holder for e in self._entries() if e.holder.kv.holds_kv]

    def held_tokens(self) -> int:
        """KV tokens (whole pages) that retained windows still hold."""
        return sum(
            self._entry_tokens(e) for e in self._entries() if e.holder.kv.holds_kv
        )

    def has_sources(self) -> bool:
        """Whether a matched or running follow-up still holds a source window."""
        return bool(self.sources)

    def _entry_tokens(self, entry: DocEntry) -> int:
        ps = self._cache.page_size
        return sum(-(-hi // ps) * ps - lo // ps * ps for lo, hi in _kept_ranges(entry))

    # -- Release --

    def _free_entry(self, entry: DocEntry) -> None:
        holder = entry.holder
        if not holder.mark_released() or not holder.kv.holds_kv:
            return
        self._cache.free_kv_row(holder.kv, _kept_ranges(entry))
        self._cache.req_to_token_pool.free(holder)

    def _release_source(self, req) -> None:
        st = req.uocr_doc
        if st is None or st.source is None:
            return
        source, st.source = st.source, None
        self.sources.pop(id(source), None)
        self._free_entry(source)

    def _free_pending(self, req) -> None:
        """Free prefix slots the matcher allocated that no KV row owns yet.

        Matching runs in `init_next_round_input`, before admission, so a window
        the adder then rejects has already copied its prefix into fresh slots.
        Those slots only enter `req_to_token` in `alloc_for_extend`.
        """
        st = req.uocr_doc
        if st is None:
            return
        alloc, st.pending = st.pending, None
        if alloc is None or req.kv.holds_kv:
            return
        self._cache.token_to_kv_pool_allocator.free(alloc)
        logger.info(
            "[Unlimited-OCR v2] %s: freed %d unadmitted prefix slots",
            req.rid,
            alloc.numel(),
        )

    def on_req_dropped(self, req: Req) -> None:
        """Release reuse state for a request dropped before it ever ran a batch."""
        # Its queued copies write slots freed below and read the source window.
        self._cancel(req.rid)
        self._free_pending(req)
        self._release_source(req)
        self._unpin(req)

    def release_doc(self, doc_id: str) -> None:
        """Free a document's retained window once no follow-up will consume it."""
        entry = self.docs.pop(doc_id, None)
        if entry is None:
            return
        self._free_entry(entry)
        logger.info("[Unlimited-OCR v2] %s: released retained window", doc_id)

    def reset(self) -> None:
        """Forget every window and queued edit; the caller clears the KV pools."""
        # Freeing would double-free slots the pool clear already returned.
        self.docs.clear()
        self.sources.clear()
        self._ops.clear()
        self._input_overrides.clear()
        if self._backend is not None:
            self._enqueue(None, lambda be: be.unpin_coord_rows(None))

    def _drop_reuse(self, req) -> None:
        """Turn a matched-but-unadmitted window back into a plain recompute."""
        self.on_req_dropped(req)
        _restore_prompt(req)
        st = req_state(req)
        st.text_plan = None
        st.match = (None, req.retraction_count)
        req.prefix_len = 0
        req.kv.protect_prefix(0)

    def on_retracted(self, req: Req) -> None:
        """Give a retracted rewritten window its original prompt to recompute."""
        st = req.uocr_doc
        if st is None or st.orig_prompt is None:
            return
        _put_back_prompt(req, st.orig_prompt)
        # Rebuilt from the restored prompt plus the outputs on the next round.
        req.full_untruncated_fill_ids = array("q")
        st.orig_prompt = None
        st.text_plan = None
        st.page_base = None
        st.end_on_marker = False
        cp = _doc_params(req)
        # The outputs already open on the window's own first page (the spliced
        # marker), so the overlap pages stay untranscribed.
        st.context_pages = int(cp.get(KEY_OVERLAP) or 0) if cp is not None else 0
        req.page_guard_state = None

    # -- Waiting queue and admission --

    def queue_position(self, queue: List[Req], req: Req) -> Optional[int]:
        """Waiting-queue index for a follow-up window whose source is retained.

        A retained window holds a full window of KV until its follow-up runs, so
        follow-ups go ahead of new documents (in arrival order among themselves).
        None means append as usual.
        """
        cp = _doc_params(req)
        if cp is None or not cp.get(KEY_REUSE) or cp[KEY_DOC_ID] not in self.docs:
            return None
        req_state(req).front = True
        pos = 0
        while pos < len(queue) and (
            queue[pos].uocr_doc is not None and queue[pos].uocr_doc.front
        ):
            pos += 1
        return pos

    def free_for_decode(self, batch: ScheduleBatch) -> None:
        """Evict retained windows (oldest first) until the next decode step fits.

        Running requests come first: otherwise the decode batch retracts, and
        with retained windows filling the pool even its last request is aborted.
        """
        freed = 0
        while self.docs and not batch.check_decode_mem():
            entry = self.docs.pop(next(iter(self.docs)))
            freed += self._entry_tokens(entry)
            self._free_entry(entry)
        if freed:
            logger.info(
                "[Unlimited-OCR v2] decode pressure: freed %d tokens of retained windows",
                freed,
            )

    def free_for_admission(self, waiting: List[Req]) -> bool:
        """Free a retained window when retained windows hold every request row.

        A retained window keeps its finished request's ``req_to_token`` row until
        its follow-up has copied the KV tail, and the follow-up needs a row of its
        own to do so. With as many in-flight documents as rows no follow-up is
        admitted, so none is ever released. Freeing the newest retained window of
        another document lets the waiting follow-up in; that document's next
        window recomputes instead of reusing.
        """
        docs = self.docs
        if not docs:
            return False
        head = None
        for req in waiting:
            cp = _doc_params(req)
            if cp is not None and cp.get(KEY_REUSE) and cp[KEY_DOC_ID] in docs:
                head = cp[KEY_DOC_ID]
                break
        if head is None:
            return False
        victim = next((doc_id for doc_id in reversed(docs) if doc_id != head), None)
        if victim is None:
            return False
        self._free_entry(docs.pop(victim))
        logger.info(
            "[Unlimited-OCR v2] request-row pressure: freed retained window %s for %s",
            victim,
            head,
        )
        return True

    def _admission_need(self, req) -> int:
        from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS

        max_new = max(req.sampling_params.max_new_tokens - len(req.output_ids), 0)
        return (
            len(req.full_untruncated_fill_ids)
            - req.prefix_len
            + min(max_new, CLIP_MAX_NEW_TOKENS)
            + self._cache.page_size
        )

    def defer_new_document(
        self, *, req: Req, adder: PrefillAdder, nothing_running: bool
    ) -> bool:
        """Whether a new document must wait so retained windows' follow-ups fit.

        Never defers with nothing running, so a lost follow-up cannot stall admission.
        """
        if nothing_running:
            return False
        cp = _doc_params(req)
        if cp is not None and cp.get(KEY_REUSE):
            return False
        expected = sum(self._entry_tokens(e) for e in self.docs.values())
        if not expected:
            return False
        return self._admission_need(req) + expected > adder.rem_total_tokens

    def on_no_token(
        self, *, req: Req, adder: PrefillAdder, nothing_running: bool
    ) -> bool:
        """Evict retained windows so a rejected request can be admitted; True if freed.

        Retained windows are not tree-evictable, so only this frees them under
        pressure. While anything runs, only a follow-up evicts, and only if that
        admits it; with nothing running any request evicts, and a follow-up with
        nothing left to evict drops its reuse and recomputes.
        """
        deficit = (
            self._admission_need(req)
            - int(adder.rem_total_tokens)
            + self._cache.page_size
        )
        st = req.uocr_doc
        has_reuse = st is not None and (st.pending is not None or st.source is not None)
        # A follow-up that lost its reuse (e.g. no room to copy its prefix) still
        # continues an in-flight document; it outranks retained windows.
        cp = _doc_params(req)
        is_follow_up = has_reuse or (cp is not None and bool(cp.get(KEY_REUSE)))
        docs = self.docs
        freeable = sum(self._entry_tokens(e) for e in docs.values())
        if not nothing_running and (not is_follow_up or freeable < deficit):
            # Evicting windows that in-flight documents are about to continue from
            # cannot admit this one anyway while running requests hold the memory.
            return False
        freed = 0
        while docs and freed < deficit:
            entry = docs.pop(next(iter(docs)))
            freed += self._entry_tokens(entry)
            self._free_entry(entry)
        if freed:
            logger.info(
                "[Unlimited-OCR v2] memory pressure: freed %d tokens of retained windows for %s",
                freed,
                req.rid,
            )
            return True
        if has_reuse and nothing_running and not adder.can_run_list:
            logger.info(
                "[Unlimited-OCR v2] %s: pool too small for reuse; recomputing", req.rid
            )
            self._drop_reuse(req)
            return True
        return False

    # -- Finished windows --

    def on_cache_finished_req(self, req: Req) -> bool:
        """Keep a finished window's KV for the next window; True if handled."""
        retained = self._retain_finished(req)
        # After retention: the released follow-up's match reads the kept window.
        self._on_window_finished(req)
        return retained

    def _retain_finished(self, req) -> bool:
        from sglang.srt.managers.schedule_batch import (
            FINISH_MATCHED_STR,
            FINISH_MATCHED_TOKEN,
        )
        from sglang.srt.models.unlimited_ocr_v2_support.retained_window import (
            RetainedWindow,
        )

        cp = _doc_params(req)
        if cp is None:
            return False
        self._release_source(req)
        self._unpin(req)
        if req.finished() and req.uocr_doc is not None:
            req.uocr_doc.orig_prompt = None
        text_ok = isinstance(
            req.finished_reason, (FINISH_MATCHED_STR, FINISH_MATCHED_TOKEN)
        )
        # Retraction comes through here unfinished (keeping it retains a half
        # window), and a budget-stopped window ends mid page with no usable text.
        if (
            not cp.get(KEY_KEEP)
            or not req.finished()
            or not text_ok
            or not req.kv.holds_kv
            or self._backend is None
        ):
            return False

        page_id = self._backend.page_state.page_token_id
        image_id = self._backend.page_state.image_token_id
        lookahead = int(cp.get(KEY_LOOKAHEAD) or 0)
        overlap = int(cp.get(KEY_OVERLAP) or 0)
        next_page = next_page_after(
            req.origin_input_ids,
            int(cp.get(KEY_PAGE_BASE) or 0) + overlap,
            # A rewritten prompt carries CONTEXT_PAGES; an untouched one carries the
            # window's own overlap pages. Either way they are not this window's.
            _context_pages(req) or overlap,
            page_id,
            image_id,
            lookahead=lookahead,
        )
        if next_page is None:
            # Decided before the row is detached: returning False hands the row
            # to release_kv_cache, which must still find it on the request.
            return False

        cache = self._cache
        committed = req.kv.kv_committed_len
        holder = RetainedWindow()
        holder.kv = req.detach_kv()
        # release_kv_cache stops once the row is claimed, so the request's tree
        # lock (taken on the match's root anchor at admission) is dropped here.
        cache.unlock(req.lock)
        req.lock = None
        if holder.kv.kv_allocated_len > committed:
            cache.session._free_kv_aligned(
                holder.kv, committed, holder.kv.kv_allocated_len
            )
            holder.kv.truncate(committed)
        out = req.output_ids
        if req.finished_len is not None:
            out = out[: req.finished_len]
        out = list(out)
        stop = _stop_ids(req)
        while out and out[-1] in stop:
            out.pop()
        if lookahead or (req.uocr_doc is not None and req.uocr_doc.end_on_marker):
            own_pages = (
                sum(1 for t in req.origin_input_ids if t == page_id)
                - _context_pages(req)
                - lookahead
            )
            out = _cut_at_extra_marker(out, page_id, own_pages)
        # Only outputs whose KV exists can be copied: positions < committed.
        out = out[: max(0, committed - len(req.origin_input_ids))]
        entry = DocEntry(
            holder=holder,
            input_ids=list(req.origin_input_ids),
            output_ids=out,
            next_page=next_page,
            lookahead=lookahead,
        )
        self._trim_to_tails(entry)
        old = self.docs.pop(cp[KEY_DOC_ID], None)
        if old is not None:
            self._free_entry(old)
        self.docs[cp[KEY_DOC_ID]] = entry
        return True

    def _trim_to_tails(self, entry: DocEntry) -> None:
        """Free the KV the follow-up window will never read.

        Only the overlap page's marker, the image tail and the text tail are
        copied into the next window (see ``tail_ranges``); the rest goes back to
        the pool now instead of when the follow-up consumes the window.
        """
        kv = entry.holder.kv
        kept = tail_ranges(
            entry.input_ids,
            entry.output_ids,
            True,
            self._backend.page_state.page_token_id,
            self._backend.page_state.image_token_id,
            self._cache.page_size,
            kv.kv_allocated_len,
            lookahead=entry.lookahead,
        )
        if kept is None:
            return
        gaps, pos = [], 0
        for lo, hi in kept:
            if lo > pos:
                gaps.append((pos, lo))
            pos = hi
        if pos < kv.kv_allocated_len:
            gaps.append((pos, kv.kv_allocated_len))
        if gaps:
            self._cache.free_kv_row(kv, gaps)
        entry.kept = kept
        logger.info(
            "[Unlimited-OCR v2] retained %d of %d window tokens (marker + tails)",
            self._entry_tokens(entry),
            kv.kv_allocated_len,
        )

    # -- Matching and the prefix copy --

    def try_match_prefix(self, params):
        """Copy the overlap pages' KV into a new prefix for a follow-up window."""
        req = params.req
        if req is None:
            return None
        st = req.uocr_doc
        if st is not None and st.match is not None:
            result, retractions = st.match
            if result is None or retractions != req.retraction_count:
                return None
            return result
        cp = _doc_params(req)
        if cp is None or not cp.get(KEY_REUSE):
            return None
        st = req_state(req)
        st.match = (None, req.retraction_count)
        if req.retraction_count or req.output_ids:
            return None
        entry = self.docs.pop(cp[KEY_DOC_ID], None)
        if entry is None or self._backend is None:
            return None
        st.source = entry
        self.sources[id(entry)] = entry
        overlap = int(cp.get(KEY_OVERLAP) or 0)
        try:
            result = self._build_reuse(
                req,
                entry,
                overlap=overlap,
                page_base=int(cp.get(KEY_PAGE_BASE) or 0) + overlap,
            )
        except Exception:
            logger.exception(
                "[Unlimited-OCR v2] reuse planning failed; recomputing window"
            )
            result = None
        if result is None:
            # Nothing consumes the source entry or the slots the planner may have
            # allocated before it gave up.
            self.on_req_dropped(req)
        st.match = (result, req.retraction_count)
        return result

    def _build_reuse(self, req, entry: DocEntry, *, overlap: int, page_base: int):
        from sglang.srt.mem_cache.base_prefix_cache import MatchResult

        b = self._backend
        cache = self._cache
        page_id = b.page_state.page_token_id
        image_id = b.page_state.image_token_id
        bs = b.block_size
        if page_id is None or cache.page_size != bs:
            return None
        if entry.next_page != page_base:
            # A window in between was aborted: the tail belongs to another page,
            # and nothing in a contiguous prompt would reveal that.
            logger.info(
                "[Unlimited-OCR v2] %s: retained window ends at page %d, "
                "this one starts at %d; recomputing",
                req.rid,
                entry.next_page,
                page_base,
            )
            return None

        plan = plan_tail_copy(
            entry.input_ids,
            list(req.origin_input_ids),
            overlap,
            page_id,
            image_id,
            bs,
            old_lookahead=entry.lookahead,
        )
        if plan is None:
            logger.info(
                "[Unlimited-OCR v2] %s: prompt layout mismatch; recomputing", req.rid
            )
            return None

        text = text_tail(entry.output_ids, TAIL_TOKENS, page_id)
        if text is None or len(text[1]) + 2 >= (
            req.sampling_params.max_new_tokens or 0
        ):
            # Only the spliced marker makes decode skip the context page the
            # rewritten prompt carries, so without text there is nothing to reuse.
            logger.info(
                "[Unlimited-OCR v2] %s: no output to continue from; recomputing",
                req.rid,
            )
            return None
        # The copied KV can still predict EOS where that window stopped; the
        # forced marker moves decode off it.
        forced = text[1] + [page_id]

        alloc = self._alloc_prefix_slots(plan.copy_len, bs)
        if alloc is None:
            return None
        if not _rewrite_prompt(req, plan):
            logger.info(
                "[Unlimited-OCR v2] %s: multimodal item straddles the prefix; recomputing",
                req.rid,
            )
            cache.token_to_kv_pool_allocator.free(alloc)
            return None
        _mark_rewritten(req, alloc=alloc, page_base=page_base)
        req_state(req).text_plan = self._queue_prefix_copy(
            req,
            entry=entry,
            plan=plan,
            dst=alloc[: plan.copy_len],
            text=text,
            forced=forced,
        )
        logger.info(
            "[Unlimited-OCR v2] %s: reuse tail copy=%d forced_text=%d copied_pages=%d "
            "(old_pages=%d new_pages=%d overlap=%d)",
            req.rid,
            plan.copy_len,
            len(forced),
            plan.copied_pages,
            plan.old_pages,
            plan.new_pages,
            overlap,
        )
        # The prefix KV is not on a tree path: `prefix_device_indices` hands
        # alloc_for_extend the pending slots, and with cache_protected_len 0 the
        # request's row owns (and frees) them from then on. The root anchor is
        # what a miss reports, so locking it pins nothing.
        root = cache.root_node_handle(req.extra_key)
        return MatchResult(
            device_prefix_len=plan.copy_len,
            last_device_node=root,
            last_host_node=root,
            best_match_node=root,
            cache_protected_len=0,
        )

    def _alloc_prefix_slots(self, copy_len: int, bs: int) -> Optional[torch.Tensor]:
        """Allocate whole pages for the copied prefix, evicting retained windows."""
        allocator = self._cache.token_to_kv_pool_allocator
        alloc_len = -(-copy_len // bs) * bs
        alloc = allocator.alloc(alloc_len)
        while alloc is None and self.docs:
            # This window's source is already out of `docs`; other retained
            # windows are worth less than the follow-up that frees its own source.
            self._free_entry(self.docs.pop(next(iter(self.docs))))
            alloc = allocator.alloc(alloc_len)
        return alloc

    def _queue_prefix_copy(
        self,
        req,
        *,
        entry: DocEntry,
        plan: TailCopyPlan,
        dst: torch.Tensor,
        text: Tuple[int, List[int]],
        forced: List[int],
    ) -> TextPlan:
        """Queue the prefix KV copy and return the text splice for after prefill."""
        b = self._backend
        page_id = b.page_state.page_token_id
        image_id = b.page_state.image_token_id
        per_page = b.page_state.per_page_image_tokens
        bs = b.block_size
        max_pages = b.coord_state.max_pages
        new_ids = plan.new_ids
        copy_len = plan.copy_len

        old_row = self._cache.req_to_token_pool.req_to_token[
            entry.holder.kv.req_pool_idx
        ]
        src = old_row[
            torch.tensor(plan.src_positions, dtype=torch.int64, device=old_row.device)
        ].to(torch.int64)

        _, prefix_carry = replay_coords(
            new_ids[:copy_len], copy_len, page_id, image_id, max_pages
        )
        iso = replay_page_isolation(new_ids, page_id, image_id, per_page)
        gen_coord, gen_iso = continuation_carry(
            new_ids, page_id, image_id, per_page, max_pages
        )

        first_block = dst[:1] // bs
        n_blocks = -(-copy_len // bs)
        req_state(req).pinned_blocks.append(first_block)
        self._enqueue(
            req.rid,
            lambda be, dst=dst, src=src: _apply_segment(
                be,
                dst=dst,
                src=src,
                values=tuple(v[:copy_len] for v in iso),
                kbar_pages=dst[::bs] // bs,
                kbar_toks=[min(bs, copy_len - j * bs) for j in range(n_blocks)],
                coord=(first_block, prefix_carry[0], prefix_carry[1], prefix_carry[2]),
            ),
        )

        start = len(entry.input_ids) + text[0]
        n_copy_text = len(forced) - 1
        text_src = old_row[start : start + n_copy_text].to(torch.int64).clone()
        return TextPlan(
            forced=forced,
            src_slots=text_src,
            iso=gen_iso,
            coord=gen_coord,
            first_block=first_block,
        )

    # -- After the follow-up's prefill --

    def after_prefill_forward(
        self, batch: ScheduleBatch, *, chunked_req: Optional[Req]
    ) -> None:
        """Splice the overlap OCR text into requests whose prefill just ran."""
        if not batch.forward_mode.is_extend():
            return
        # These requests ran a batch, so alloc_for_extend has registered the slots
        # the matcher allocated in their rows; freeing them is the row's job now.
        for req in batch.reqs:
            if req.uocr_doc is not None:
                req.uocr_doc.pending = None
        if batch.seq_lens_cpu is None:
            return
        deltas = None
        for i, req in enumerate(batch.reqs):
            st = req.uocr_doc
            plan = st.text_plan if st is not None else None
            if plan is None or req is chunked_req or req.finished():
                continue
            st.text_plan = None
            seq_len = int(batch.seq_lens_cpu[i])
            if seq_len != len(req.origin_input_ids):
                continue
            added = self._splice_text(req, seq_len, plan)
            if deltas is None:
                deltas = torch.zeros_like(batch.seq_lens_cpu)
            deltas[i] = added
        if deltas is None or not bool(deltas.any()):
            return
        dev = deltas.to(batch.seq_lens.device, non_blocking=True)
        batch.seq_lens_cpu = batch.seq_lens_cpu + deltas
        batch.seq_lens = batch.seq_lens + dev.to(batch.seq_lens.dtype)
        if batch.orig_seq_lens is not None:
            batch.orig_seq_lens = batch.orig_seq_lens + dev.to(
                batch.orig_seq_lens.dtype
            )
        batch.seq_lens_sum = None

    def _splice_text(self, req, seq_len: int, plan: TextPlan) -> int:
        bs = self._backend.block_size
        row = req.kv.req_pool_idx
        req_to_token = self._cache.req_to_token_pool.req_to_token
        n = len(plan.src_slots)
        dst = None
        if n > 0:
            allocator = self._cache.token_to_kv_pool_allocator
            device = req_to_token.device
            last_loc = req_to_token[row, seq_len - 1 : seq_len].to(torch.int64)
            prefix_cpu = torch.tensor([seq_len], dtype=torch.int64)
            seq_cpu = torch.tensor([seq_len + n], dtype=torch.int64)
            dst = allocator.alloc_extend(
                prefix_cpu.to(device),
                prefix_cpu,
                seq_cpu.to(device),
                seq_cpu,
                last_loc,
                n,
            )
            if dst is None:
                # Decoding the page from its images alone still lands on the right
                # page; only the previous page's text is lost as context.
                logger.info(
                    "[Unlimited-OCR v2] %s: no KV for text splice; marker only", req.rid
                )
                n = 0
        if n > 0:
            req_to_token[row, seq_len : seq_len + n] = dst.to(req_to_token.dtype)
            first_blk = seq_len // bs
            last_blk = (seq_len + n - 1) // bs
            offsets = [
                max(j * bs, seq_len) - seq_len for j in range(first_blk, last_blk + 1)
            ]
            toks = [
                min(bs, seq_len + n - j * bs) for j in range(first_blk, last_blk + 1)
            ]
            pages = (
                dst[torch.tensor(offsets, dtype=torch.int64, device=dst.device)] // bs
            )
            self._enqueue(
                req.rid,
                lambda be, dst=dst: _apply_segment(
                    be,
                    dst=dst,
                    src=plan.src_slots,
                    values=tuple([v] * n for v in (-1, *plan.iso)),
                    kbar_pages=pages,
                    kbar_toks=toks,
                    coord=(plan.first_block, *plan.coord),
                ),
            )
        else:
            # The carry is read off the last token's slot, so the prompt's own tail
            # slot has to carry the page the first decode step continues from.
            tail = req_to_token[row, seq_len - 1 : seq_len].to(torch.int64)
            self._enqueue(
                req.rid, lambda be, tail=tail: _seed_continuation(be, tail, plan)
            )
        forced = plan.forced if n > 0 else plan.forced[-1:]
        req.output_ids.extend(array("q", forced))
        st = req_state(req)
        st.forced_len = len(forced)
        # The copied text sits in output_ids, where the length check counts it;
        # leaving it there cuts this window's own answer short by that many.
        if req.sampling_params.max_new_tokens is not None:
            req.sampling_params.max_new_tokens += len(forced)
        st.skip_prefill_append = True
        hide_forced_text(req)
        req.kv.allocate_decode(n)
        self._input_overrides[int(row)] = int(plan.forced[-1])
        # Queued behind the copies that read the source, so its freed slots
        # cannot be reallocated and overwritten before they are read.
        self._enqueue(None, lambda be, req=req: self._release_source(req))
        return n
