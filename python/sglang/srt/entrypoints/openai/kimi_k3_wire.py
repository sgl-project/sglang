"""The Kimi-K3 shape of /v1/chat/completions responses."""

from __future__ import annotations

import hashlib
from typing import Any, Literal, Optional

from fastapi import HTTPException

from sglang.srt.entrypoints.openai.chat_encoding_adapters import reported_prompt_tokens
from sglang.srt.entrypoints.openai.kimi_token_provenance import (
    KimiTokenProvenance,
    token_byte_decoder,
)
from sglang.srt.entrypoints.openai.protocol import (
    STREAM_DELTA_CONTEXT,
    ChatCompletionRequest,
    CompletionTokensDetails,
    ErrorResponse,
    PromptTokensDetails,
    UsageInfo,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_serving

# K3's three-token assistant generation stub is model input, but the
# reference API excludes it from reported prompt tokens.
GENERATION_STUB_TOKENS = 3

# The K3 wire has no "abort"; its K3 equivalent server_interrupted lets a client
# tell retryable from terminal.
_FINISH_REASONS = {"abort": "server_interrupted"}


def _wants_internal_content(request: ChatCompletionRequest) -> bool:
    return bool(
        request.stream
        and request.stream_options
        and request.stream_options.include_internal_content
    )


class KimiK3ChatWire:
    def __init__(self, *, tokenizer: Any):
        self._tokenizer = tokenizer

    def new_response(
        self, request: ChatCompletionRequest, *, created: int
    ) -> KimiK3WireResponse:
        return KimiK3WireResponse(
            created=created,
            tokenizer=self._tokenizer if _wants_internal_content(request) else None,
        )

    def internal_content_error(self, request: ChatCompletionRequest) -> Optional[str]:
        """A request this wire cannot serve, found before generation."""
        if (
            _wants_internal_content(request)
            and token_byte_decoder(self._tokenizer) is None
        ):
            return (
                "stream_options.include_internal_content needs a tokenizer with a "
                "tiktoken byte decoder, which this server does not have."
            )
        return None

    def error_body(self, error: ErrorResponse) -> dict[str, Any]:
        # sglang's stock body puts message/type/param/code at the top level, so
        # a strict client sees no "error" key at all.
        return {"error": error.model_dump()}


class KimiK3WireResponse:
    serialization_context = STREAM_DELTA_CONTEXT
    omit_null_delta = True
    # Top-level usage only in the final choices=[] summary.
    continuous_usage = False
    # A first tool-call chunk carrying arguments is an anti-pattern clients
    # must special-case; the slot opens with arguments="".
    split_tool_call_opening = True
    # An error-only SSE frame is exactly what the K3 stream contract forbids.
    errors_end_choices = True

    def __init__(self, *, created: int, tokenizer: Any):
        """``tokenizer`` is None unless the request asked for internal_content."""
        self._created = created
        self._tokenizer = tokenizer
        self._id: Optional[str] = None
        self._headers: Optional[dict[str, str]] = None
        self._provenances: dict[int, KimiTokenProvenance] = {}
        self._prev_completion_tokens: dict[int, int] = {}
        self._replayed_chars: dict[int, int] = {}
        self._reasoning_seen: set[int] = set()
        self._boundary_sent: set[int] = set()

    def response_id(self, engine_rid: str) -> str:
        """Fixed by the first rid, since n>1 gives each choice its own rid; the
        leading 4 bytes of Unix seconds make ids sort by creation time."""
        if self._id is None:
            digest = hashlib.sha256(engine_rid.encode("utf-8")).hexdigest()
            self._id = f"chatcmpl-{self._created & 0xFFFFFFFF:08x}{digest[:16]}"
        return self._id

    def finish_reason(self, reason: Optional[str]) -> Optional[str]:
        if reason is None:
            return None
        return _FINISH_REASONS.get(reason, reason)

    def finalize_usage(self, usage: UsageInfo, *, cached_tokens: int) -> UsageInfo:
        """Both token-detail objects are mandatory on every K3 usage object;
        cached_tokens ignores --enable-cache-report, as the usage headers do."""
        usage.completion_tokens_details = CompletionTokensDetails(
            reasoning_tokens=int(usage.reasoning_tokens or 0)
        )
        if usage.prompt_tokens_details is None:
            usage.prompt_tokens_details = PromptTokensDetails(
                cached_tokens=int(cached_tokens or 0)
            )
        else:
            usage.prompt_tokens_details.cached_tokens = int(cached_tokens or 0)
        return usage

    def choice_usage(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        cached_tokens: int,
    ) -> dict[str, Any]:
        """Each choice's end frame carries its usage, whatever include_usage says."""
        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            reasoning_tokens=reasoning_tokens,
        )
        return self.finalize_usage(usage, cached_tokens=cached_tokens).model_dump()

    def headers(self) -> dict[str, str]:
        return self._headers or {}

    def check_abort(
        self, finish_reason: dict[str, Any], *, stream_started: bool
    ) -> None:
        code = finish_reason["status_code"]
        if not stream_started and code < 500:
            raise HTTPException(
                status_code=code.value,
                detail=finish_reason.get("message", "Generation aborted."),
            )

    def observe(self, index: int, content: dict[str, Any]) -> None:
        if self._headers is None:
            self._headers = self._usage_headers(content["meta_info"])
        if self._tokenizer is None:
            return
        provenance = self._provenances.get(index)
        if provenance is None:
            provenance = self._provenances[index] = KimiTokenProvenance(self._tokenizer)
        provenance.append(self._new_token_ids(index, content))

    def ends_reasoning(
        self, index: int, detector: Any, reasoning_text: str, *, finishing: bool
    ) -> bool:
        """One reasoning_content="" frame per choice, once its detector has left
        the think channel or the stream is finishing."""
        if self._tokenizer is not None:
            # Reasoning that force_nonempty_content replays as content.
            self._replayed_chars[index] = getattr(
                detector, "replayed_reasoning_chars", 0
            )
        if reasoning_text:
            self._reasoning_seen.add(index)
        if index not in self._reasoning_seen or index in self._boundary_sent:
            return False
        # Response-template detectors do not track the think channel: a step
        # that still streams reasoning is taken as inside it.
        if getattr(detector, "in_reasoning", bool(reasoning_text)) and not finishing:
            return False
        self._boundary_sent.add(index)
        return True

    def internal_content(
        self,
        index: int,
        source: Optional[str],
        *,
        kind: Literal["reasoning", "content", "tool"],
    ) -> Optional[list[dict[str, Any]]]:
        """The generated tokens behind ``source``; a tool-call fragment's source
        is its original markup, including characters turned into JSON syntax."""
        if self._tokenizer is None:
            return None
        provenance = self._provenances[index]
        if kind == "reasoning":
            return provenance.take(source, text=True, reasoning=True)
        if kind == "content":
            return provenance.take(
                source, text=True, replay_chars=self._replayed_chars.pop(index, 0)
            )
        if source is None:
            raise ValueError("K3 tool delta has no original token source")
        return provenance.take(source)

    def _usage_headers(self, meta_info: dict[str, Any]) -> dict[str, str]:
        """Prompt-side counts are fixed after prefill, so the first event's
        agree with the final usage even when the stream drops before it."""
        prefix = envs.SGLANG_USAGE_HEADER_PREFIX.get()
        if not prefix:
            return {}
        prompt = reported_prompt_tokens(meta_info, GENERATION_STUB_TOKENS)
        cached = meta_info.get("cached_tokens", 0) or 0
        return {
            f"{prefix}-Prompt-Tokens": str(prompt),
            f"{prefix}-Cached-Tokens": str(cached),
        }

    def _new_token_ids(self, index: int, content: dict[str, Any]) -> list[int]:
        all_ids = content.get("output_ids") or []
        if get_serving().incremental_streaming_output:
            return list(all_ids)
        # Non-incremental events carry the cumulative output_ids, which backlog
        # coalescing may concatenate with itself; the tail of that is still the
        # cumulative tail, so take the k tokens completion_tokens says it added.
        completion_tokens = content["meta_info"].get("completion_tokens", 0) or 0
        k = completion_tokens - self._prev_completion_tokens.get(index, 0)
        self._prev_completion_tokens[index] = completion_tokens
        return list(all_ids[-k:]) if k > 0 else []
