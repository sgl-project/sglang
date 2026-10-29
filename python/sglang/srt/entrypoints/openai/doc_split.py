"""Split oversized Unlimited-OCR v2 documents into page batches.

Batches are contiguous: the scheduler hands each one the previous batch's KV
tail, so a batch needs no page of its own for context. A request can set
``overlap_pages`` to carry preceding pages anyway, which is what a run without
KV reuse needs to keep the split lossless.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

PAGE_MARKER = "<PAGE>"
# Reported in response metadata; names the splitter that produced the answer.
SPLIT_VERSION = "contiguous-pages-v3"
BUDGET_POLICY = "global-cumulative-v1"
CONTEXT_RESERVE_TOKENS = 64
# Preceding pages a window carries as context. Zero because copy-splice reuse
# hands it the previous page's KV tail, which is all it reads of that page;
# carrying the page again only pays for its image twice.
OVERLAP_PAGES = 0

# Measured input-token expansion per page for this checkpoint.
_TOKENS_PER_PAGE = {
    "gundam": 1514,
    "large": 422,
    "base": 274,
    "small": 112,
    "tiny": 74,
}
_DEFAULT_TOKENS_PER_PAGE = 1514  # assume the expensive mode when unknown


# Match the processor's mode normalization before applying page budgets.
def _normalize_mode(image_mode: Optional[str]) -> str:
    return str(image_mode).strip().lower() if image_mode else ""


def tokens_per_page(image_mode: Optional[str]) -> int:
    """Return the estimated input-token count for one page in an image mode."""
    return _TOKENS_PER_PAGE.get(_normalize_mode(image_mode), _DEFAULT_TOKENS_PER_PAGE)


def count_images(messages: List[Any]) -> int:
    """How many image parts the request carries, across all messages."""
    n = 0
    for msg in messages or ():
        content = msg.get("content") if isinstance(msg, dict) else msg.content
        if not isinstance(content, (list, tuple)):
            continue
        for part in content:
            ptype = part.get("type") if isinstance(part, dict) else part.type
            if ptype in ("image_url", "image"):
                n += 1
    return n


def completion_capacity(
    *, context_length: int, image_mode: Optional[str], input_pages: int
) -> int:
    """Return the estimated completion-token capacity of one page batch."""
    return (
        context_length
        - CONTEXT_RESERVE_TOKENS
        - tokens_per_page(image_mode) * input_pages
    )


# Markers around each page and the stop marker; arbitrary headroom over them.
_PAGE_CAP_SLACK_TOKENS = 256


def requested_max_page_tokens(custom_params: Optional[dict]) -> int:
    """The request's own per-page cap, or 0 when it names none."""
    if not custom_params:
        return 0
    try:
        return max(0, int(custom_params.get("max_page_tokens") or 0))
    except (TypeError, ValueError):
        return 0


def effective_max_page_tokens(
    *, server_max_page_tokens: int, custom_params: Optional[dict]
) -> int:
    """Per-page cap the page guard enforces, as ``page_guard_params`` sets it.

    A request may lower the server's cap but not raise it: the default context
    is sized from the server's cap, and a window past it outgrows its
    ``req_to_token`` row. With no server cap the request's own applies.
    """
    own = requested_max_page_tokens(custom_params)
    if server_max_page_tokens <= 0:
        return own
    if own <= 0:
        return server_max_page_tokens
    return min(own, server_max_page_tokens)


def window_output_cap(*, own_pages: int, max_page_tokens: int) -> Optional[int]:
    """Most tokens a guarded window can write, or None when pages are uncapped.

    Asking for more than this lets an estimate of the input that runs short
    push input plus ``max_new_tokens`` past the context, which truncates it.
    """
    if max_page_tokens <= 0:
        return None
    return own_pages * max_page_tokens + _PAGE_CAP_SLACK_TOKENS


def plan_batches(
    *,
    n_pages: int,
    image_mode: Optional[str],
    context_length: int,
    max_completion_tokens: int,
    max_batch_pages: Optional[int] = None,
    overlap_pages: int = 0,
) -> Optional[List[tuple]]:
    """Plan ``(lo, hi, emit_from)`` batches, or return ``None`` when unsplit.

    Later batches include ``overlap_pages`` preceding context pages;
    completion limits are applied per batch against its remaining context
    capacity.
    """
    if n_pages <= 1 or context_length <= CONTEXT_RESERVE_TOKENS:
        return None
    if max_completion_tokens <= 0:
        return None

    if overlap_pages < 0:
        raise ValueError("overlap_pages must be non-negative")

    per_page = tokens_per_page(image_mode)
    fits = (context_length - CONTEXT_RESERVE_TOKENS - 1) // per_page
    if max_batch_pages is not None:
        fits = min(fits, max_batch_pages)
    # One page per window is allowed: with no overlap (the default) the window
    # owns its page, and KV reuse carries the previous page's tail as context.
    if fits < 1 or overlap_pages >= fits:
        return None

    policy_forces_split = max_batch_pages is not None and n_pages > max_batch_pages
    needs_budget_clamp = max_completion_tokens > completion_capacity(
        context_length=context_length,
        image_mode=image_mode,
        input_pages=n_pages,
    )
    if fits >= n_pages and not policy_forces_split:
        return [(0, n_pages, 0)] if needs_budget_clamp else None

    batches: List[tuple] = []
    emit_from = 0
    while emit_from < n_pages:
        lo = max(0, emit_from - overlap_pages) if emit_from else 0
        hi = min(lo + fits, n_pages)
        batches.append((lo, hi, emit_from))
        if hi >= n_pages:
            break
        emit_from = hi
    return batches


def split_messages(messages: List[Any], lo: int, hi: int) -> List[Any]:
    """Retain images [lo, hi); image and response page IDs must both start at zero."""
    out = []
    seen = 0
    for msg in messages:
        is_dict = isinstance(msg, dict)
        content = msg.get("content") if is_dict else msg.content
        if not isinstance(content, (list, tuple)):
            out.append(msg)
            continue
        # A message with only other batches' images keeps its text parts, which
        # may carry the instruction.
        kept_parts, kept_images = [], 0
        for part in content:
            ptype = part.get("type") if isinstance(part, dict) else part.type
            if ptype in ("image_url", "image"):
                if lo <= seen < hi:
                    kept_parts.append(part)
                    kept_images += 1
                seen += 1
            else:
                kept_parts.append(part)
        kept_parts = _retag_markers(kept_parts, kept_images)
        if is_dict:
            out.append({**msg, "content": kept_parts})
        else:
            copy = msg.model_copy(deep=False)
            try:
                copy.content = kept_parts
            except (AttributeError, ValueError):
                return messages  # cannot rewrite; caller falls back
            out.append(copy)
    return out


def _retag_markers(parts: List[Any], n_images: int) -> List[Any]:
    """Rewrite the leading marker run to match the selected images."""
    fixed = []
    for part in parts:
        is_dict = isinstance(part, dict)
        ptype = part.get("type") if is_dict else part.type
        if ptype != "text":
            fixed.append(part)
            continue
        text = part.get("text") if is_dict else part.text
        if "<image>" not in (text or ""):
            fixed.append(part)
            continue
        tail = (text or "").split("<image>")[-1]
        new_text = ("<PAGE><image>" * n_images) + tail
        if is_dict:
            fixed.append({**part, "text": new_text})
        else:
            copy = part.model_copy(deep=False)
            try:
                copy.text = new_text
            except (AttributeError, ValueError):
                fixed.append(part)
                continue
            fixed.append(copy)
    return fixed


def select_batch_output(
    *, text: str, input_pages: int, context_pages: int, leads_with_body: bool = False
) -> tuple[str, int, int]:
    """Return owned text and observed/expected page-body counts.

    Surplus leading empty bodies are stray markers and are dropped; ownership
    stays front-anchored, and the counts expose missing or extra output pages.
    ``leads_with_body`` says the text opens with a page body rather than a
    marker, as a KV-reuse window does (its first marker was spliced in and
    hidden): its first page is then a real page even when the body is empty.
    """
    if not 0 <= context_pages < input_pages:
        raise ValueError("Invalid document batch page range")
    head, *tail = text.split(PAGE_MARKER)
    bodies = [head, *tail] if leads_with_body or head.strip() else list(tail)
    # Only surplus leading markers are stray; a blank page legitimately has an empty body.
    while len(bodies) > input_pages and not bodies[0].strip():
        bodies = bodies[1:]
    owned_bodies = bodies[context_pages : context_pages + input_pages - context_pages]
    owned = "".join(PAGE_MARKER + body for body in owned_bodies)
    return owned, len(bodies), input_pages


def stream_owned_text(
    *, text: str, input_pages: int, context_pages: int, leads_with_body: bool
) -> str:
    """Owned text of a window still decoding, a prefix of every later call's.

    Same ownership as ``select_batch_output`` except its drop of surplus
    leading empty bodies, which needs the final page count; a trailing partial
    marker is held back until it completes.
    """
    for n in range(len(PAGE_MARKER) - 1, 0, -1):
        if text.endswith(PAGE_MARKER[:n]):
            text = text[:-n]
            break
    head, *tail = text.split(PAGE_MARKER)
    bodies = [head, *tail] if leads_with_body or head.strip() else list(tail)
    return "".join(PAGE_MARKER + body for body in bodies[context_pages:input_pages])


def window_incomplete(observed: int, expected: int) -> bool:
    """Whether a window failed to deliver the pages it owns.

    Only missing pages count. A surplus body is the page guard's own artifact:
    with ``<PAGE>`` masked, a model that wants another page writes the
    characters instead. Ownership is front-anchored, so the surplus is dropped
    with the text after it.
    """
    return observed < expected


def owned_page_bodies(owned: str) -> List[str]:
    """Return page bodies, preserving empties for completeness checks."""
    return owned.split(PAGE_MARKER)[1:]


def page_guard_params(
    *,
    max_page_tokens: int,
    page_token_id: Optional[int],
    eos_token_id: Optional[int],
    custom_params: Optional[dict],
    stop_only: bool = False,
) -> dict:
    """Sampler params for the page guard, or ``{}`` when it cannot apply.

    They live in ``custom_params`` because a ``CustomLogitProcessor`` enforces
    them. The threshold turns the whole guard on: refusing EOS while pages are
    owed needs a bound on how long one page may run. A request that names its
    own threshold keeps it. ``stop_only`` is for a window the scheduler may
    reuse: it has no instruction and stops only on the marker past its last
    page, so it needs the token ids even without a threshold. A request
    threshold above the server's is replaced by the server's (see
    ``effective_max_page_tokens``).
    """
    if page_token_id is None or eos_token_id is None:
        return {}
    if max_page_tokens <= 0:
        if not stop_only:
            return {}
        return {
            "page_token_id": int(page_token_id),
            "eos_token_id": int(eos_token_id),
            "page_stop_only": True,
        }
    own = requested_max_page_tokens(custom_params)
    if 0 < own <= max_page_tokens:
        return {}
    return {
        "max_page_tokens": int(max_page_tokens),
        "page_token_id": int(page_token_id),
        "eos_token_id": int(eos_token_id),
    }


# Unknown and absent modes use the conservative gundam limit.
_LIGHT_IMAGE_MODES = frozenset(_TOKENS_PER_PAGE) - {"gundam"}


def page_limit_for_mode(
    image_mode: Optional[str], gundam_limit: Optional[int], base_limit: Optional[int]
) -> Optional[int]:
    """Use the base limit for light modes; otherwise use gundam."""
    if _normalize_mode(image_mode) in _LIGHT_IMAGE_MODES:
        return base_limit
    return gundam_limit
