"""The Kimi-K3 shape of /v1/chat/completions responses: a constant ObjectId-like
id, ``server_interrupted`` for aborts, complete usage objects, null-free deltas,
an end-of-thinking boundary frame, the generated tokens behind each increment,
and the OpenAI ``{"error": {...}}`` envelope."""

from __future__ import annotations

import hashlib
from typing import Any, Literal, Optional

from fastapi import HTTPException

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

# "abort" is sglang's internal graceful-abort reason, not a value the K3 wire
# contract defines. server_interrupted is its documented K3 equivalent, so a
# client can tell retryable from terminal.
_FINISH_REASONS = {"abort": "server_interrupted"}


def _wants_internal_content(request: ChatCompletionRequest) -> bool:
    return bool(
        request.stream
        and request.stream_options
        and request.stream_options.include_internal_content
    )


class KimiK3ChatWire:
    def __init__(self, *, tokenizer: Any, uncounted_prompt_tokens: int):
        self._tokenizer = tokenizer
        self._uncounted_prompt_tokens = uncounted_prompt_tokens

    def new_response(
        self, request: ChatCompletionRequest, *, created: int
    ) -> KimiK3WireResponse:
        return KimiK3WireResponse(
            created=created,
            tokenizer=self._tokenizer if _wants_internal_content(request) else None,
            uncounted_prompt_tokens=self._uncounted_prompt_tokens,
        )

    def request_error(self, request: ChatCompletionRequest) -> Optional[str]:
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

    def __init__(self, *, created: int, tokenizer: Any, uncounted_prompt_tokens: int):
        """``tokenizer`` is None unless the request asked for internal_content."""
        self._created = created
        self._tokenizer = tokenizer
        self._uncounted_prompt_tokens = uncounted_prompt_tokens
        self._id: Optional[str] = None
        self._headers: Optional[dict[str, str]] = None
        self._provenances: dict[int, KimiTokenProvenance] = {}
        self._prev_completion_tokens: dict[int, int] = {}
        self._replayed_chars: dict[int, int] = {}
        self._reasoning_seen: set[int] = set()
        self._boundary_sent: set[int] = set()

    def response_id(self, engine_rid: str) -> str:
        """chatcmpl-<24 hex>, fixed by the first rid it sees.

        With n>1 the engine assigns a separate rid to every sampled choice, so
        a per-frame derivation would send a different id per candidate. Shaped
        like an ObjectId so ids sort by creation time: 4 bytes of Unix seconds,
        then 8 bytes hashed from the rid. Hashing, rather than filtering the rid
        down to its hex characters, keeps distinct rids ("req-a", "req-b") apart.
        """
        if self._id is None:
            digest = hashlib.sha256(engine_rid.encode("utf-8")).hexdigest()
            self._id = f"chatcmpl-{self._created & 0xFFFFFFFF:08x}{digest[:16]}"
        return self._id

    def finish_reason(self, reason: Optional[str]) -> Optional[str]:
        if reason is None:
            return None
        return _FINISH_REASONS.get(reason, reason)

    def finalize_usage(
        self, usage: UsageInfo, *, reasoning_tokens: int, cached_tokens: int
    ) -> UsageInfo:
        """Fill both token-detail objects, mandatory on every K3 usage object.

        cached_tokens ignores --enable-cache-report: the engine always reports
        the count and the flag only gates whether the stock response shows it,
        while the usage headers report it unconditionally. Gating one and not
        the other made them disagree whenever a prefix-cache hit landed.
        """
        usage.completion_tokens_details = CompletionTokensDetails(
            reasoning_tokens=int(reasoning_tokens or 0)
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
        return self.finalize_usage(
            usage, reasoning_tokens=reasoning_tokens, cached_tokens=cached_tokens
        ).model_dump()

    def headers(self) -> dict[str, str]:
        return self._headers or {}

    def check_abort(
        self, finish_reason: dict[str, Any], *, stream_started: bool
    ) -> None:
        code = finish_reason["status_code"]
        if not stream_started and code < 500:
            # A client error found before any byte is sent is a plain HTTP
            # error, not a stream.
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

    def reasoning_step(
        self, index: int, detector: Any, reasoning_text: str, *, finishing: bool
    ) -> bool:
        """One reasoning_content="" frame marks the end of thinking, after the
        last non-empty reasoning delta and before the first content delta, once
        the detector has left the think channel or the stream is finishing."""
        if self._tokenizer is not None:
            # Reasoning that force_nonempty_content replays as content.
            self._replayed_chars[index] = getattr(
                detector, "replayed_reasoning_chars", 0
            )
        if reasoning_text:
            self._reasoning_seen.add(index)
        if index not in self._reasoning_seen or index in self._boundary_sent:
            return False
        if getattr(detector, "_in_reasoning", False) and not finishing:
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
        """Usage accounting headers, named by SGLANG_USAGE_HEADER_PREFIX.

        They let clients account prompt-side tokens even when the stream drops
        before the final usage frame. Prompt-side counts are fixed after
        prefill, so the first engine event's agree with the final usage.
        """
        prefix = envs.SGLANG_USAGE_HEADER_PREFIX.get()
        if not prefix:
            return {}
        prompt = max(
            0, meta_info.get("prompt_tokens", 0) - self._uncounted_prompt_tokens
        )
        cached = meta_info.get("cached_tokens", 0) or 0
        return {
            f"{prefix}-Prompt-Tokens": str(prompt),
            f"{prefix}-Cached-Tokens": str(cached),
        }

    def _new_token_ids(self, index: int, content: dict[str, Any]) -> list[int]:
        all_ids = content.get("output_ids") or []
        if get_serving().incremental_streaming_output:
            return list(all_ids)
        # Non-incremental events carry a reference to the cumulative
        # output_ids, and backlog coalescing concatenates that same reference
        # several times, so length-based offsets would slice garbage. This
        # event added k tokens by its completion_tokens count: take the last k;
        # the tail of a duplicated concatenation is still the cumulative tail.
        completion_tokens = content["meta_info"].get("completion_tokens", 0) or 0
        k = completion_tokens - self._prev_completion_tokens.get(index, 0)
        self._prev_completion_tokens[index] = completion_tokens
        return list(all_ids[-k:]) if k > 0 else []
