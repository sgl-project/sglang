"""Serve Unlimited-OCR v2 multi-page documents as a chain of page windows.

Each window is an ordinary chat request over a contiguous page range; the
scheduler splices the previous window's KV tail into it, and the answers are
concatenated by page ownership.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import uuid
from http import HTTPStatus
from typing import TYPE_CHECKING, Callable, Optional

import msgspec
import orjson
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.configs.model_config import UNLIMITED_OCR_V2_ARCH
from sglang.srt.configs.unlimited_ocr import merge_mm_processor_kwargs
from sglang.srt.configs.unlimited_ocr_v2 import (
    DEFAULT_LOOKAHEAD_PAGES,
    UocrDocumentConfig,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    PromptTokensDetails,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.sse_utils import build_sse_content
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.entrypoints.openai.utils import should_include_usage
from sglang.srt.environ import envs
from sglang.srt.models.unlimited_ocr_v2_support import (
    document_prefetch as uocr_doc_prefetch,
)
from sglang.srt.models.unlimited_ocr_v2_support import document_reuse as uocr_doc_reuse
from sglang.srt.models.unlimited_ocr_v2_support import document_split as doc_split
from sglang.srt.runtime_context import get_mm, get_serving

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

logger = logging.getLogger(__name__)


def _is_context_error_message(message: str) -> bool:
    """Whether a rejection means the window did not fit the context."""
    return any(
        marker in message
        for marker in (
            "longer than the model's context length",
            "exceeds the model's maximum context length",
            "Multimodal prompt is too long after expanding multimodal tokens",
            "exceeds the maximum allowed length",
        )
    )


def _is_doc_split_context_error(response) -> bool:
    if not isinstance(response, ORJSONResponse) or response.status_code != 400:
        return False
    try:
        message = orjson.loads(response.body).get("message", "")
    except orjson.JSONDecodeError:
        return False
    return _is_context_error_message(message)


class _PrefetchedStream:
    """An async iterator whose first step can be run ahead of consumption.

    ``start`` drives the wrapped generator up to its first chunk, which sends
    the window's request (tokenisation included); the chunk is replayed first.
    """

    def __init__(self, agen):
        self._agen = agen
        self._task = None

    async def start(self) -> None:
        """Run the wrapped generator up to its first chunk, in the background."""
        if self._task is None:
            self._task = asyncio.ensure_future(self._agen.__anext__())

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._task is not None:
            task, self._task = self._task, None
            return await task
        return await self._agen.__anext__()

    async def aclose(self) -> None:
        if self._task is not None:
            task, self._task = self._task, None
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._agen.aclose()


class _WindowStream(msgspec.Struct):
    """One streaming doc-split window: decoded text and what was sent of it."""

    text: str = ""
    sent: str = ""
    reused: bool = False
    meta: dict = {}


def _window_pages(reused: bool, *, lo: int, hi: int, emit_from: int) -> dict:
    """Ownership arguments of ``doc_split.select_batch_output`` for one window.

    A reused window hides its spliced text, so its answer opens with the body
    of its own first page; a recomputed one leads with the overlap pages.
    """
    if reused:
        return {
            "input_pages": hi - emit_from,
            "context_pages": 0,
            "leads_with_body": True,
        }
    return {
        "input_pages": hi - lo,
        "context_pages": emit_from - lo,
        "leads_with_body": False,
    }


def _images_config(request) -> dict:
    """The images_config the Unlimited-OCR v2 processor resolves for ``request``."""
    return (
        merge_mm_processor_kwargs(request.images_config, request.mm_processor_kwargs)
        or {}
    )


def _doc_image_mode(request) -> Optional[str]:
    """Image mode a split document's windows use, from its images_config."""
    cfg = _images_config(request)
    if "image_mode" in cfg:
        return cfg["image_mode"] or "gundam"
    if cfg.get("crop_mode") is not None:
        return "gundam" if cfg["crop_mode"] else "base"
    return None


def _window_lookahead(request, plan, bi) -> int:
    """Pages window ``bi`` reads past its own: the next windows' first pages.

    The last window has nothing after it. The plan's windows are
    contiguous, so the next one starts at this window's ``hi``.
    """
    if bi >= len(plan) - 1:
        return 0
    _, hi, _ = plan[bi]
    _, last_hi, _ = plan[-1]
    options = request.doc_split_options
    lookahead = (
        options.lookahead_pages
        if options is not None and options.lookahead_pages is not None
        else DEFAULT_LOOKAHEAD_PAGES
    )
    return min(lookahead, last_hi - hi)


def _estimate_window_capacity(request, plan, bi, *, context_length, image_mode) -> int:
    """Completion tokens window ``bi`` may ask for: context room, page-capped.

    The page guard bounds what a window writes, so asking for that keeps
    input plus ``max_new_tokens`` under the context even when its pages run
    longer than the per-page estimate. The cap is the one the guard applies,
    which a request may name itself.
    """
    lo, hi, _ = plan[bi]
    capacity = doc_split.completion_capacity(
        context_length=context_length,
        image_mode=image_mode,
        input_pages=hi - lo + _window_lookahead(request, plan, bi),
    )
    params = dict(request.custom_params or {})
    options = request.doc_split_options
    if options is not None and options.max_page_tokens is not None:
        params["max_page_tokens"] = options.max_page_tokens
    page_tokens = doc_split.effective_max_page_tokens(
        server_max_page_tokens=get_mm().uocr_max_page_tokens,
        custom_params=params,
    )
    cap = doc_split.window_output_cap(own_pages=hi - lo, max_page_tokens=page_tokens)
    return capacity if cap is None else min(capacity, cap)


def _window_request(
    request,
    *,
    hf_config,
    plan,
    bi,
    max_tokens,
    doc_reuse_id,
    after=None,
    remaining=None,
):
    """owned child request with page guard, reuse and optional predecessor metadata"""
    lo, hi, emit_from = plan[bi]
    lookahead = _window_lookahead(request, plan, bi)
    sub = request.model_copy(deep=True)
    sub.messages = doc_split.split_messages(sub.messages, lo, hi + lookahead)
    # Whether the scheduler reused the previous window's KV only shows up in
    # meta_info; the parent's own flag is restored before it responds.
    sub.return_meta_info = True
    if request.max_completion_tokens is not None:
        sub.max_completion_tokens = max_tokens
        sub.max_tokens = None
    else:
        sub.max_tokens = max_tokens
        sub.max_completion_tokens = None
    # Every split window, reused or recomputed, numbers RoPE pages globally.
    sub.custom_params = {
        **(sub.custom_params or {}),
        uocr_doc_reuse.KEY_PAGE_BASE: lo,
        uocr_doc_reuse.KEY_LOOKAHEAD: lookahead,
    }
    # A follow-up window may be reused without an instruction, and a window
    # with a lookahead page must stop before transcribing it; either way only
    # the page guard can stop it, whatever the length threshold.
    _apply_page_cap(
        sub,
        hf_config=hf_config,
        stop_only=lookahead > 0 or (doc_reuse_id is not None and bi > 0),
    )
    if doc_reuse_id is not None:
        # Keep this window's last page so the next one can splice its KV tail
        # instead of recomputing it.
        sub.custom_params.update(
            {
                uocr_doc_reuse.KEY_DOC_ID: doc_reuse_id,
                uocr_doc_reuse.KEY_KEEP: bi < len(plan) - 1,
                uocr_doc_reuse.KEY_REUSE: bi > 0,
                uocr_doc_reuse.KEY_OVERLAP: emit_from - lo,
            }
        )
    sub.rid = f"{doc_reuse_id or 'uocr-doc'}-w{bi}-{uuid.uuid4().hex[:8]}"
    if after is not None:
        sub.custom_params.update(
            {
                uocr_doc_prefetch.KEY_AFTER: after,
                uocr_doc_prefetch.KEY_BUDGET: remaining,
            }
        )
    return sub


def _apply_page_cap(sub, *, hf_config, stop_only: bool = False) -> None:
    """Turn on the per-page length cap for one window, per --uocr-max-page-tokens.

    The cap is enforced by a ``CustomLogitProcessor``, so a request that
    brings none gets the built-in one; a request may name a lower threshold
    but not a higher one (see ``doc_split.page_guard_params``). The processor
    searches only generated tokens for repeated n-grams.
    """
    from sglang.srt.sampling.custom_logit_processor import (
        NGRAM_OUTPUT_ONLY_KEY,
        UOCR_PAGE_GUARD_PROCESSOR,
    )

    if sub.custom_logit_processor:
        sub.custom_params = {
            **(sub.custom_params or {}),
            NGRAM_OUTPUT_ONLY_KEY: True,
        }
    # Not every checkpoint config defines eos_token_id.
    eos = getattr(hf_config, "eos_token_id", None)
    if isinstance(eos, (list, tuple, set)):
        # Several ids end a turn; the cap forces one, so take the lowest.
        eos = min(eos, default=None)
    server_cap = UocrDocumentConfig.from_args(get_mm()).uocr_max_page_tokens
    custom_params = dict(sub.custom_params or {})
    options = sub.doc_split_options
    if options is not None and options.max_page_tokens is not None:
        custom_params["max_page_tokens"] = options.max_page_tokens
    cap = doc_split.effective_max_page_tokens(
        server_max_page_tokens=server_cap, custom_params=custom_params
    )
    params = doc_split.page_guard_params(
        max_page_tokens=cap,
        page_token_id=hf_config.sparse_attention_config.get("page_token_id"),
        eos_token_id=eos,
        custom_params=None,
        stop_only=stop_only,
    )
    if not params:
        return
    if not sub.custom_logit_processor:
        sub.custom_logit_processor = UOCR_PAGE_GUARD_PROCESSOR
    params[NGRAM_OUTPUT_ONLY_KEY] = True
    sub.custom_params = {**custom_params, **params}


def _requested_max_tokens(request) -> Optional[int]:
    # Same precedence as ChatCompletionRequest.to_sampling_params.
    return request.max_completion_tokens or request.max_tokens


def _window_limit(*, remaining_tokens, context_capacity) -> tuple[int, str]:
    """A window's ``max_tokens`` and which bound set it."""
    max_tokens = min(remaining_tokens, context_capacity)
    if remaining_tokens < context_capacity:
        return max_tokens, "global_budget"
    if context_capacity < remaining_tokens:
        return max_tokens, "context_capacity"
    return max_tokens, "tied"


def _usage_error(usage, *, bi, max_tokens) -> Optional[str]:
    if usage is None:
        return f"doc-split batch {bi} returned no usage accounting"
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    if (
        type(prompt_tokens) is not int
        or prompt_tokens < 0
        or type(completion_tokens) is not int
        or completion_tokens < 0
        or completion_tokens > max_tokens
    ):
        return f"doc-split batch {bi} returned invalid usage accounting"
    return None


def _context_error_batch(
    *, bi, max_tokens, context_capacity, remaining_tokens, context_pages
) -> dict:
    """Metadata row of a window the server rejected for not fitting the context."""
    return {
        "index": bi,
        "finish": "length",
        "matched_stop": None,
        "page_count": 0,
        "incomplete": True,
        "markers": 0,
        "retractions": 0,
        "extra_pages": 0,
        "head": "",
        "tail": "",
        "max_tokens": max_tokens,
        "context_capacity": context_capacity,
        "limit_source": "context_capacity",
        "remaining_before": remaining_tokens,
        "remaining_after": remaining_tokens,
        "context_pages": context_pages,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


def _window_batch(
    choice,
    usage,
    *,
    bi,
    lo,
    hi,
    emit_from,
    max_tokens,
    context_capacity,
    limit_source,
    remaining_before,
    remaining_after,
) -> tuple[str, dict]:
    """Owned text of one finished window and its metadata row."""
    # Only the scheduler knows whether the window reused its predecessor's KV.
    reused = uocr_doc_reuse.window_reused(choice.meta_info)
    pages = _window_pages(reused, lo=lo, hi=hi, emit_from=emit_from)
    content = choice.message.content or ""
    owned, observed, expected = doc_split.select_batch_output(text=content, **pages)
    bodies = doc_split.owned_page_bodies(owned)
    batch = {
        "index": bi,
        "finish": choice.finish_reason,
        "matched_stop": choice.matched_stop,
        "page_count": observed,
        "context_pages": pages["context_pages"],
        "reused": reused,
        # Blank and image-only pages legitimately yield empty bodies, so only
        # the page count decides completeness; empties are reported separately.
        "incomplete": doc_split.window_incomplete(observed, expected),
        "empty_pages": [
            emit_from + i for i, body in enumerate(bodies) if not body.strip()
        ],
        # Markers tell a window that wrote an extra page apart from one whose
        # last body is empty; both inflate page_count.
        "markers": content.count(doc_split.PAGE_MARKER),
        "retractions": (choice.meta_info or {}).get("num_retractions", 0),
        "extra_pages": max(0, observed - expected),
        # A leading surplus marker means the spliced one leaked through the
        # detokenizer, a trailing one that the model opened a page too many.
        "head": content[:40],
        "tail": content[-40:],
        "max_tokens": max_tokens,
        "context_capacity": context_capacity,
        "limit_source": limit_source,
        "remaining_before": remaining_before,
        "remaining_after": remaining_after,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens_details": usage.prompt_tokens_details,
        "reasoning_tokens": usage.reasoning_tokens or 0,
    }
    return owned, batch


def _merged_finish_reason(finishes, *, planned, incomplete) -> Optional[str]:
    """Finish reason of the whole document, or None when every window stopped."""
    terminal = next((f for f in finishes if f in ("abort", "content_filter")), None)
    if terminal is not None:
        return terminal
    if any(f != "stop" for f in finishes) or len(finishes) < planned or incomplete:
        return "length"
    return None


def _merged_usage(batches) -> UsageInfo:
    """Token usage summed over every window that ran."""
    detail_rows = [
        batch["prompt_tokens_details"]
        for batch in batches
        if batch.get("prompt_tokens_details") is not None
    ]
    prompt_details = (
        PromptTokensDetails(
            cached_tokens=sum(row.cached_tokens for row in detail_rows),
            image_tokens=sum(row.image_tokens or 0 for row in detail_rows) or None,
            audio_tokens=sum(row.audio_tokens or 0 for row in detail_rows) or None,
            video_tokens=sum(row.video_tokens or 0 for row in detail_rows) or None,
        )
        if detail_rows
        else None
    )
    return UsageInfo(
        prompt_tokens=sum(batch["prompt_tokens"] for batch in batches),
        completion_tokens=sum(batch["completion_tokens"] for batch in batches),
        total_tokens=sum(
            batch["prompt_tokens"] + batch["completion_tokens"] for batch in batches
        ),
        prompt_tokens_details=prompt_details,
        reasoning_tokens=sum(batch.get("reasoning_tokens", 0) for batch in batches),
    )


def _split_metadata(*, plan, batches, original_max_tokens, remaining_tokens) -> dict:
    """What the split actually did. `batches` holds one dict per batch that ran."""
    incomplete = [batch["index"] for batch in batches if batch["incomplete"]]
    unstarted = list(range(len(batches), len(plan)))
    mark = {
        "doc_split_implementation": doc_split.SPLIT_VERSION,
        "doc_split_budget_policy": doc_split.BUDGET_POLICY,
        # Reuse is attempted per window and can fall back to recomputation,
        # so doc_split_reused_batches below is what actually happened.
        "doc_split_kv_reuse_mode": "copy-splice",
        "doc_split_batches": len(batches),
        "doc_split_planned_batches": len(plan),
        "doc_split_incomplete_batches": incomplete,
        # Global page indices whose owned body is empty; diagnostic only.
        "doc_split_empty_pages": [
            page for batch in batches for page in batch.get("empty_pages", ())
        ],
        "doc_split_reused_batches": [
            batch["index"] for batch in batches if batch.get("reused")
        ],
        "doc_split_batch_finish_reasons": [batch["finish"] for batch in batches],
        "doc_split_batch_matched_stops": [batch["matched_stop"] for batch in batches],
        "doc_split_complete": (
            len(batches) == len(plan)
            and not incomplete
            and all(batch["finish"] == "stop" for batch in batches)
        ),
        "doc_split_overlap_pages": (
            next(
                (
                    batch.get("context_pages", 0)
                    for batch in batches
                    if batch.get("context_pages", 0) > 0
                ),
                0,
            )
            if len(plan) > 1
            else 0
        ),
        "doc_split_batch_page_counts": [batch["page_count"] for batch in batches],
        # A window with more bodies than pages is either one extra marker
        # or one empty trailing body; these two tell them apart.
        "doc_split_batch_marker_counts": [batch["markers"] for batch in batches],
        "doc_split_batch_retractions": [batch["retractions"] for batch in batches],
        # Bodies past the window's own pages; dropped, not a failure.
        "doc_split_batch_extra_pages": [batch["extra_pages"] for batch in batches],
        "doc_split_batch_heads": [batch["head"] for batch in batches],
        "doc_split_batch_tails": [batch["tail"] for batch in batches],
        "doc_split_original_max_completion_tokens": original_max_tokens,
        "doc_split_batch_max_tokens": [batch["max_tokens"] for batch in batches],
        "doc_split_batch_context_capacities": [
            batch["context_capacity"] for batch in batches
        ],
        "doc_split_batch_completion_tokens": [
            batch["completion_tokens"] for batch in batches
        ],
        "doc_split_batch_remaining_tokens_before": [
            batch["remaining_before"] for batch in batches
        ],
        "doc_split_batch_remaining_tokens_after": [
            batch["remaining_after"] for batch in batches
        ],
        "doc_split_batch_limit_sources": [batch["limit_source"] for batch in batches],
        "doc_split_batch_truncation_sources": [
            batch["limit_source"] if batch["finish"] == "length" else None
            for batch in batches
        ],
        "doc_split_remaining_completion_tokens": remaining_tokens,
        "doc_split_budget_exhausted": bool(unstarted and remaining_tokens == 0),
        "doc_split_unstarted_batches": unstarted,
        "doc_split_global_position_inherited": False,
        "doc_split_sampling_state_inherited": False,
    }
    truncated = [batch["index"] for batch in batches if batch["finish"] == "length"]
    if truncated:
        mark["doc_split_truncated_batches"] = truncated
    return mark


def _assemble_split_response(
    response, *, request, plan, texts, batches, original_max_tokens, remaining_tokens
):
    """Rewrite the last window's response into the whole document's answer."""
    mark = _split_metadata(
        plan=plan,
        batches=batches,
        original_max_tokens=original_max_tokens,
        remaining_tokens=remaining_tokens,
    )
    choice = response.choices[0]
    choice.message.content = "".join(texts)
    if not request.return_meta_info:
        # The windows were forced to return it; the parent did not ask.
        choice.meta_info = None
    finish = _merged_finish_reason(
        [batch["finish"] for batch in batches],
        planned=len(plan),
        incomplete=any(batch["incomplete"] for batch in batches),
    )
    if finish is not None:
        choice.finish_reason = finish
        choice.matched_stop = None
    response.usage = _merged_usage(batches)
    inherited = dict(response.metadata or {})
    inherited.pop("weight_versions", None)
    inherited.pop("weight_version", None)
    response.metadata = {**inherited, **mark}
    logger.info(
        "doc-split: %d pages -> %d/%d batches, %d chars, complete=%s, remaining=%d",
        doc_split.count_images(request.messages),
        len(batches),
        len(plan),
        len(choice.message.content),
        mark["doc_split_complete"],
        remaining_tokens,
    )
    return response


class _SplitProgress(msgspec.Struct):
    """What the windows of a non-streamed split document produced so far."""

    remaining_tokens: int
    texts: list = []
    batches: list = []
    last_response: Optional[object] = None
    # A window finished with abort or content_filter; no later window runs.
    terminal: bool = False

    def add_window(self, response, *, owned: str, batch: dict) -> None:
        """Keep a finished window's owned text and metadata row."""
        self.last_response = response
        self.remaining_tokens = batch["remaining_after"]
        self.texts.append(owned)
        self.batches.append(batch)
        self.terminal = batch["finish"] in ("abort", "content_filter")


class _StreamTotals(msgspec.Struct):
    """Running accounting of a streamed split document."""

    remaining: int
    prompt: int = 0
    completion: int = 0
    cached: int = 0
    image: int = 0
    finishes: list = []
    incomplete: bool = False
    matched_stop: Optional[object] = None
    # The role chunk went out; an error after it is an SSE payload, not HTTP.
    started: bool = False

    def lose_window(self) -> None:
        """Account a window rejected for its context length."""
        self.finishes.append("length")
        self.incomplete = True


def _account_stream_window(
    window, totals, *, bi, lo, hi, emit_from, prompt_tokens
) -> tuple[str, Optional[str]]:
    """Fold a finished streamed window into ``totals``.

    Returns the owned text not streamed yet and the window's finish type.
    """
    meta = window.meta
    completion = meta.get("completion_tokens", 0)
    totals.remaining -= completion
    totals.prompt += prompt_tokens
    totals.completion += completion
    totals.cached += meta.get("cached_tokens", 0)
    totals.image += meta.get("image_tokens", 0) or 0
    owned, observed, expected = doc_split.select_batch_output(
        text=window.text,
        **_window_pages(window.reused, lo=lo, hi=hi, emit_from=emit_from),
    )
    tail = ""
    if owned.startswith(window.sent):
        tail = owned[len(window.sent) :]
    else:
        # Only a stray leading marker reorders ownership at the end;
        # streamed text cannot be taken back.
        logger.warning("doc-split stream: window %d ownership moved", bi)
    totals.incomplete = totals.incomplete or doc_split.window_incomplete(
        observed, expected
    )
    finish = meta.get("finish_reason") or {}
    totals.finishes.append(finish.get("type"))
    totals.matched_stop = finish.get("matched")
    return tail, finish.get("type")


class _SseFormatter(msgspec.Struct, frozen=True):
    """Formats the chunks of one SSE response; every chunk shares id and model."""

    chunk_id: str
    created: int
    model: Optional[str]

    def __call__(self, **kwargs) -> str:
        return build_sse_content(
            chunk_id=self.chunk_id,
            created=self.created,
            model=self.model,
            index=0,
            **kwargs,
        )


def _stream_epilogue(*, request, plan, totals, sse, include_usage):
    """The finish chunk and the optional usage chunk that close a split stream."""
    final_finish = (
        _merged_finish_reason(
            totals.finishes, planned=len(plan), incomplete=totals.incomplete
        )
        or "stop"
    )
    yield sse(
        finish_reason=final_finish,
        matched_stop=totals.matched_stop if final_finish == "stop" else None,
    )
    if include_usage:
        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=totals.prompt,
            completion_tokens=totals.completion,
            cached_tokens=(
                UsageProcessor._details_if_cached(totals.cached)
                if get_serving().enable_cache_report
                else None
            ),
            image_tokens=totals.image,
        )
        usage_chunk = ChatCompletionStreamResponse(
            id=sse.chunk_id,
            created=sse.created,
            choices=[],
            model=request.model,
            usage=usage,
        )
        yield f"data: {usage_chunk.model_dump_json()}\n\n"
    logger.info(
        "doc-split stream: %d pages -> %d/%d batches, finish=%s, remaining=%d",
        doc_split.count_images(request.messages),
        len(totals.finishes),
        len(plan),
        final_finish,
        totals.remaining,
    )


class ChatDocSplitter:
    """Chat-completions handling of Unlimited-OCR v2 multi-page documents.

    Reads the owning ``OpenAIServingChat`` but never writes to it; each window
    is served by ``handle_window``, the base class's ``handle_request``.
    """

    def __init__(
        self,
        serving: OpenAIServingChat,
        *,
        handle_window: Callable,
        config: UocrDocumentConfig,
    ):
        self._serving = serving
        self._tokenizer_manager = serving.tokenizer_manager
        self._hf_config = serving.tokenizer_manager.model_config.hf_config
        self._handle_window = handle_window
        self._config = config

    @classmethod
    def for_model(
        cls, serving: OpenAIServingChat, *, handle_window: Callable
    ) -> Optional[ChatDocSplitter]:
        """A splitter when ``serving`` runs Unlimited-OCR v2, else None."""
        architectures = serving.tokenizer_manager.model_config.hf_config.architectures
        if not architectures or architectures[0] != UNLIMITED_OCR_V2_ARCH:
            return None
        return cls(
            serving,
            handle_window=handle_window,
            config=UocrDocumentConfig.from_args(get_mm()),
        )

    async def handle_request(self, request, raw_request):
        """Serve a chat request, splitting a long document into page windows.

        A multi-window answer is not lossless and says so in its response metadata.
        """
        if request.n != 1:
            return await self._handle_window(request, raw_request)
        options = request.doc_split_options
        if (
            not self._config.uocr_doc_split
            and options is not None
            and any(
                value is not None
                for value in (
                    options.max_batch_pages,
                    options.overlap_pages,
                    options.lookahead_pages,
                )
            )
        ):
            return self._serving.create_error_response(
                "window options in doc_split_options require --uocr-doc-split"
            )
        try:
            plan = self._plan_doc_split(request)
        except ValueError as e:
            return self._serving.create_error_response(str(e))
        except Exception:
            # A planning bug must not take the request down; the base path still
            # answers (or rejects with its accurate message).
            logger.exception("doc-split planning failed; serving the request as-is")
            plan = None
        if plan is None:
            # The page guard applies to an unsplit document too; the shallow
            # copy keeps the caller's request unchanged.
            if doc_split.count_images(request.messages):
                request = request.model_copy(deep=False)
                _apply_page_cap(request, hf_config=self._hf_config)
            return await self._handle_window(request, raw_request)
        # Every multi-window document reuses the previous window's KV.
        doc_reuse_id = f"uocr-doc-{uuid.uuid4().hex}" if len(plan) > 1 else None
        if request.stream:
            # The stream outlives this call, so it releases the document itself.
            return await self._handle_split_document_stream(
                request, raw_request, plan, doc_reuse_id=doc_reuse_id
            )
        try:
            return await self._handle_split_document(
                request, raw_request, plan, doc_reuse_id=doc_reuse_id
            )
        finally:
            # A disconnect, an abort or an early stop leaves the last finished
            # window retained with no follow-up to consume it.
            if doc_reuse_id is not None:
                self._tokenizer_manager.release_uocr_doc(doc_reuse_id)

    def _plan_doc_split(self, request) -> Optional[list[tuple[int, int, int]]]:
        """``[(lo, hi, emit_from), ...]`` or None. Cheap: no tokenization."""
        if not self._config.uocr_doc_split:
            return None
        limit = self._tokenizer_manager.max_req_input_len
        if not limit:
            return None
        if not self._is_splittable(request):
            return None
        max_tokens = _requested_max_tokens(request)
        if max_tokens is None or max_tokens <= 0 or request.min_tokens > 0:
            return None
        image_mode = _doc_image_mode(request)
        options = request.doc_split_options
        # a request may tighten the server window even when the document fits
        max_batch_pages = self._config.uocr_doc_page_window
        if options and options.max_batch_pages is not None:
            if options.max_batch_pages > max_batch_pages:
                raise ValueError(
                    "doc_split_options.max_batch_pages exceeds --uocr-doc-page-window"
                )
            max_batch_pages = options.max_batch_pages
        overlap_pages = (
            options.overlap_pages
            if options and options.overlap_pages is not None
            else doc_split.OVERLAP_PAGES
        )
        if overlap_pages < 0:
            raise ValueError("doc_split overlap_pages must be non-negative")
        if max_batch_pages is not None and overlap_pages >= max_batch_pages:
            raise ValueError(
                "doc_split overlap_pages must be smaller than max_batch_pages"
            )
        return doc_split.plan_batches(
            n_pages=doc_split.count_images(request.messages),
            image_mode=image_mode,
            context_length=int(limit),
            max_completion_tokens=int(max_tokens),
            max_batch_pages=max_batch_pages,
            overlap_pages=overlap_pages,
        )

    def _is_splittable(self, request) -> bool:
        """Whether the windows' concatenated answers can stand for the request's.

        Page windows are constrained by input capacity and the page-limit
        policy; the request-wide completion limit is enforced at runtime.
        """
        serving = self._serving
        return not (
            serving.template_manager.chat_template_name != "unlimited-ocr"
            or serving._effective_tools(request)
            or request.response_format is not None
            or request.regex is not None
            or request.ebnf is not None
            or request.session_id is not None
            or request.session_params is not None
            or request.input_ids is not None
            or request.continue_final_message
            or request.logprobs
            or request.return_token_ids
            or request.return_prompt_token_ids
            or request.return_hidden_states
            or request.return_routed_experts
            or request.return_meta_info
            or serving._should_return_input_ids(request)
            or serving._should_return_output_ids(request)
            or request.return_cached_tokens_details
            or request.return_spec_tokens_details
            or serving.reasoning_parser
            or _images_config(request).get("prefix")
        )

    async def _handle_split_document(self, request, raw_request, plan, *, doc_reuse_id):
        """Run page windows in order; cancel an unconsumed successor on every exit"""
        error = self._serving._validate_request(
            request, validate_max_output_tokens=False
        )
        if error:
            return self._serving.create_error_response(error)
        original_max_tokens = _requested_max_tokens(request)
        if original_max_tokens is None or original_max_tokens <= 0:
            return self._serving.create_error_response(
                "doc-split requires a positive token budget"
            )
        progress = _SplitProgress(remaining_tokens=int(original_max_tokens))
        prefetch = (
            envs.SGLANG_ENABLE_UOCR_DOC_PREFETCH.get() and doc_reuse_id is not None
        )
        pending = None
        try:
            for bi, (lo, hi, emit_from) in enumerate(plan):
                if progress.remaining_tokens <= 0 or progress.terminal:
                    break
                context_capacity = self._window_capacity(request, plan, bi)
                if context_capacity <= 0:
                    return self._serving.create_error_response(
                        f"doc-split batch {bi} has no completion capacity"
                    )
                max_tokens, limit_source = _window_limit(
                    remaining_tokens=progress.remaining_tokens,
                    context_capacity=context_capacity,
                )
                # 1. consume the prefetched window or submit the current one
                if pending is not None:
                    (task, sub_rid), pending = pending, None
                else:
                    sub = _window_request(
                        request,
                        hf_config=self._hf_config,
                        plan=plan,
                        bi=bi,
                        max_tokens=max_tokens,
                        doc_reuse_id=doc_reuse_id,
                    )
                    sub_rid = sub.rid
                    task = asyncio.ensure_future(self._handle_window(sub, raw_request))
                # 2. send the successor before awaiting this window's response
                if prefetch and bi + 1 < len(plan):
                    pending = self._prefetch_window(
                        request,
                        raw_request,
                        plan=plan,
                        bi=bi + 1,
                        remaining=progress.remaining_tokens,
                        doc_reuse_id=doc_reuse_id,
                        after=sub_rid,
                    )
                response = await task
                if not isinstance(response, ChatCompletionResponse):
                    # a rejected window cannot release its parked successor
                    if pending is not None:
                        self._cancel_prefetch(pending)
                        pending = None
                    if not _is_doc_split_context_error(response):
                        return response
                    progress.batches.append(
                        _context_error_batch(
                            bi=bi,
                            max_tokens=max_tokens,
                            context_capacity=context_capacity,
                            remaining_tokens=progress.remaining_tokens,
                            context_pages=emit_from - lo,
                        )
                    )
                    continue
                # 3. settle usage and owned pages before advancing the budget
                usage_error = _usage_error(response.usage, bi=bi, max_tokens=max_tokens)
                if usage_error is not None:
                    return self._serving.create_error_response(usage_error)
                owned, batch = _window_batch(
                    response.choices[0],
                    response.usage,
                    bi=bi,
                    lo=lo,
                    hi=hi,
                    emit_from=emit_from,
                    max_tokens=max_tokens,
                    context_capacity=context_capacity,
                    limit_source=limit_source,
                    remaining_before=progress.remaining_tokens,
                    remaining_after=progress.remaining_tokens
                    - response.usage.completion_tokens,
                )
                progress.add_window(response, owned=owned, batch=batch)
            if progress.last_response is None:
                return self._serving.create_error_response(
                    "doc-split produced no batches"
                )
            return _assemble_split_response(
                progress.last_response,
                request=request,
                plan=plan,
                texts=progress.texts,
                batches=progress.batches,
                original_max_tokens=int(original_max_tokens),
                remaining_tokens=progress.remaining_tokens,
            )
        finally:
            if pending is not None:
                self._cancel_prefetch(pending)

    def _window_capacity(self, request, plan, bi) -> int:
        return _estimate_window_capacity(
            request,
            plan,
            bi,
            context_length=int(self._tokenizer_manager.max_req_input_len),
            image_mode=_doc_image_mode(request),
        )

    def _prefetch_window(
        self, request, raw_request, *, plan, bi, remaining, doc_reuse_id, after
    ):
        """Send window ``bi`` now; it waits in the scheduler for ``after``."""
        capacity = self._window_capacity(request, plan, bi)
        sub = _window_request(
            request,
            hf_config=self._hf_config,
            plan=plan,
            bi=bi,
            max_tokens=min(remaining, max(capacity, 1)),
            doc_reuse_id=doc_reuse_id,
            after=after,
            remaining=remaining,
        )
        task = asyncio.ensure_future(self._handle_window(sub, raw_request))
        return task, sub.rid

    def _cancel_prefetch(self, pending) -> None:
        """Abort a prefetched window in the scheduler and cancel its handler."""
        task, rid = pending
        if rid in self._tokenizer_manager.rid_to_state:
            self._tokenizer_manager.abort_request(rid)
        task.cancel()

    async def _handle_split_document_stream(
        self, request, raw_request, plan, *, doc_reuse_id
    ):
        """Stream a split document as one SSE response, window by window."""
        error = self._serving._validate_request(
            request, validate_max_output_tokens=False
        )
        if error:
            if doc_reuse_id is not None:
                self._tokenizer_manager.release_uocr_doc(doc_reuse_id)
            return self._serving.create_error_response(error)
        generator = self._split_document_stream(
            request, raw_request, plan, doc_reuse_id=doc_reuse_id
        )
        # Run the first window up to its first token so a rejected document is
        # an HTTP error, not an SSE payload.
        try:
            first_chunk = await generator.__anext__()
        except ValueError as e:
            await generator.aclose()
            return self._serving.create_error_response(str(e))

        async def prepend_first_chunk():
            try:
                yield first_chunk
                async for chunk in generator:
                    yield chunk
            finally:
                # A disconnect closes this wrapper only; the inner stream owns
                # the running window and the retained document.
                await generator.aclose()

        return StreamingResponse(prepend_first_chunk(), media_type="text/event-stream")

    async def _split_document_stream(self, request, raw_request, plan, *, doc_reuse_id):
        """Yield each window's owned pages while it decodes, then finish and usage.

        Window ``i+1`` is sent as soon as window ``i`` starts streaming: the
        scheduler parks it until ``i`` finishes and encodes its pages meanwhile,
        so the boundary costs only its prefill.
        """
        sse = _SseFormatter(
            chunk_id=f"{self._serving._request_id_prefix()}{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
        )
        include_usage, _ = should_include_usage(
            request.stream_options, get_serving().stream_response_default_include_usage
        )
        totals = _StreamTotals(remaining=int(_requested_max_tokens(request)))
        rids: list = []
        open_window = functools.partial(
            self._open_window_stream,
            request,
            raw_request,
            plan=plan,
            doc_reuse_id=doc_reuse_id,
            rids=rids,
        )
        prefetch = (
            envs.SGLANG_ENABLE_UOCR_DOC_PREFETCH.get() and doc_reuse_id is not None
        )
        current = pending = None
        try:
            pending = open_window(bi=0, remaining=totals.remaining)
            for bi, (lo, hi, emit_from) in enumerate(plan):
                current, pending = pending, None
                if current is None or totals.remaining <= 0:
                    break
                window, chunks = current
                has_next = bi + 1 < len(plan)
                try:
                    async for delta in chunks:
                        if prefetch and has_next and pending is None:
                            pending = open_window(
                                bi=bi + 1, remaining=totals.remaining, after=rids[-1]
                            )
                            await pending[1].start()
                        if not totals.started:
                            totals.started = True
                            yield sse(role="assistant", content="")
                        if delta:
                            yield sse(content=delta)
                except ValueError as e:
                    if not _is_context_error_message(str(e)):
                        raise
                    totals.lose_window()
                else:
                    tail, finish_type = _account_stream_window(
                        window,
                        totals,
                        bi=bi,
                        lo=lo,
                        hi=hi,
                        emit_from=emit_from,
                        prompt_tokens=self._serving._reported_prompt_tokens(
                            window.meta
                        ),
                    )
                    if tail:
                        yield sse(content=tail)
                    if finish_type in ("abort", "content_filter"):
                        break
                if has_next and totals.remaining > 0:
                    pending = pending or open_window(
                        bi=bi + 1, remaining=totals.remaining
                    )
            if not totals.started:
                raise ValueError("doc-split produced no batches")
            for chunk in _stream_epilogue(
                request=request,
                plan=plan,
                totals=totals,
                sse=sse,
                include_usage=include_usage,
            ):
                yield chunk
        except ValueError as e:
            if not totals.started:
                raise
            yield f"data: {self._serving.create_streaming_error_response(str(e))}\n\n"
        finally:
            # tokenization can still be running before the manager registers the rid
            for window_stream in (current, pending):
                if window_stream is not None:
                    await window_stream[1].aclose()
            self._abort_windows(rids, doc_reuse_id=doc_reuse_id)
        yield "data: [DONE]\n\n"

    def _abort_windows(self, rids, *, doc_reuse_id) -> None:
        """Abort every window still running and free the document's retained KV."""
        for rid in rids:
            if rid in self._tokenizer_manager.rid_to_state:
                self._tokenizer_manager.abort_request(rid)
        if doc_reuse_id is not None:
            self._tokenizer_manager.release_uocr_doc(doc_reuse_id)

    def _open_window_stream(
        self,
        request,
        raw_request,
        *,
        plan,
        bi,
        remaining,
        doc_reuse_id,
        rids,
        after=None,
    ):
        """Build window ``bi``'s request and a lazily started stream over it.

        ``after`` names the window it must wait for in the scheduler; the
        scheduler then caps its budget by what that window used.
        """
        lo, hi, emit_from = plan[bi]
        capacity = self._window_capacity(request, plan, bi)
        if capacity <= 0:
            raise ValueError(f"doc-split batch {bi} has no completion capacity")
        sub = _window_request(
            request,
            hf_config=self._hf_config,
            plan=plan,
            bi=bi,
            max_tokens=min(remaining, capacity),
            doc_reuse_id=doc_reuse_id,
            after=after,
            remaining=remaining,
        )
        # Reuse shows up in the internal meta_info, which every chunk
        # carries; return_meta_info is rejected on streams.
        sub.stream, sub.return_meta_info, sub.stream_options = True, False, None
        error = self._serving._validate_request(sub)
        if error:
            raise ValueError(error)
        adapted, _ = self._serving._convert_to_internal_request(sub, raw_request)
        rids.append(adapted.rid)
        window = _WindowStream()
        chunks = _PrefetchedStream(
            self._stream_window_text(
                adapted, raw_request, window, lo=lo, hi=hi, emit_from=emit_from
            )
        )
        return window, chunks

    async def _stream_window_text(
        self, adapted, raw_request, window, *, lo, hi, emit_from
    ):
        """Yield the newly owned text of one window per chunk ("" when none)."""
        incremental = get_serving().incremental_streaming_output
        async for content in self._tokenizer_manager.generate_request(
            adapted, raw_request
        ):
            window.meta = content["meta_info"]
            finish = window.meta.get("finish_reason") or {}
            if finish.get("type") == "abort" and isinstance(
                finish.get("status_code"), HTTPStatus
            ):
                raise ValueError(finish.get("message", "Generation aborted."))
            piece = content.get("text") or ""
            window.text = window.text + piece if incremental else piece
            window.reused = window.reused or uocr_doc_reuse.window_reused(window.meta)
            owned = doc_split.stream_owned_text(
                text=window.text,
                **_window_pages(window.reused, lo=lo, hi=hi, emit_from=emit_from),
            )
            delta = ""
            if len(owned) > len(window.sent) and owned.startswith(window.sent):
                delta = owned[len(window.sent) :]
                window.sent = owned
            yield delta
