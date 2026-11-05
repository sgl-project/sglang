from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, final

from sglang.srt.entrypoints.openai.protocol import (
    CompletionTokensDetails,
    PromptTokensDetails,
    UsageInfo,
)


@final
class UsageProcessor:
    """Stateless helpers that turn raw token counts into a UsageInfo."""

    @staticmethod
    def cache_write_tokens(
        value: Any, prompt_tokens: int, cached_tokens: int = 0
    ) -> Optional[int]:
        """Validate a measured write count without estimating missing telemetry.

        This is newly reusable *input* KV, not prompt minus cache hits. Some
        engine/cache paths cannot report it; absence must not become a billed
        zero or an estimate. Reject overlapping/out-of-range input counters.
        """
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value + cached_tokens > prompt_tokens
        ):
            return None
        return value

    @staticmethod
    def calculate_response_usage(
        responses: List[Dict[str, Any]],
        n_choices: int = 1,
        enable_cache_report: bool = False,
        image_tokens: int = 0,
        audio_tokens: int = 0,
        video_tokens: int = 0,
    ) -> UsageInfo:
        completion_tokens = sum(
            r["meta_info"].get("completion_tokens", 0) for r in responses
        )
        prompt_tokens = sum(
            responses[i]["meta_info"].get("prompt_tokens", 0)
            for i in range(0, len(responses), n_choices)
        )

        # some API don't have reasoning_tokens semantics
        reasoning_tokens = sum(
            r["meta_info"].get("reasoning_tokens", 0) for r in responses
        )

        cached_details = None
        if enable_cache_report:
            cached_total = sum(
                responses[i]["meta_info"].get("cached_tokens", 0)
                for i in range(0, len(responses), n_choices)
            )
            # Per-choice insertion counts are not a deduplicated per-prompt
            # total. Until the engine supplies that contract, n>1 is unknown.
            writes = (
                [
                    UsageProcessor.cache_write_tokens(
                        r["meta_info"].get("cache_write_tokens"),
                        r["meta_info"].get("prompt_tokens", 0),
                        r["meta_info"].get("cached_tokens", 0),
                    )
                    for r in responses
                ]
                if n_choices == 1
                else []
            )
            cache_write = (
                sum(writes) if writes and all(v is not None for v in writes) else None
            )
            cached_details = PromptTokensDetails(
                cached_tokens=cached_total, cache_write_tokens=cache_write
            )

        result = UsageProcessor.calculate_token_usage(
            prompt_tokens=prompt_tokens,
            reasoning_tokens=reasoning_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_details,
            image_tokens=image_tokens,
            audio_tokens=audio_tokens,
            video_tokens=video_tokens,
        )
        result._cache_write_reporting_enabled = enable_cache_report
        return result

    @staticmethod
    def calculate_streaming_usage(
        prompt_tokens: Mapping[int, int],
        reasoning_tokens: Mapping[int, int],
        completion_tokens: Mapping[int, int],
        cached_tokens: Mapping[int, int],
        n_choices: int,
        enable_cache_report: bool = False,
        image_tokens: int = 0,
        audio_tokens: int = 0,
        video_tokens: int = 0,
        cache_write_tokens: Optional[Mapping[int, Optional[int]]] = None,
    ) -> UsageInfo:
        # index % n_choices == 0 marks the first choice of a prompt
        total_prompt_tokens = sum(
            tok for idx, tok in prompt_tokens.items() if idx % n_choices == 0
        )
        total_reasoning_tokens = sum(reasoning_tokens.values())
        total_completion_tokens = sum(completion_tokens.values())

        cached_details = None
        if enable_cache_report:
            writes = (
                [
                    UsageProcessor.cache_write_tokens(
                        (cache_write_tokens or {}).get(idx),
                        tok,
                        cached_tokens.get(idx, 0),
                    )
                    for idx, tok in prompt_tokens.items()
                ]
                if n_choices == 1
                else []
            )
            cached_details = PromptTokensDetails(
                cached_tokens=sum(
                    tok for idx, tok in cached_tokens.items() if idx % n_choices == 0
                ),
                cache_write_tokens=(
                    sum(writes)
                    if writes and all(v is not None for v in writes)
                    else None
                ),
            )

        result = UsageProcessor.calculate_token_usage(
            prompt_tokens=total_prompt_tokens,
            reasoning_tokens=total_reasoning_tokens,
            completion_tokens=total_completion_tokens,
            cached_tokens=cached_details,
            image_tokens=image_tokens,
            audio_tokens=audio_tokens,
            video_tokens=video_tokens,
        )
        result._cache_write_reporting_enabled = enable_cache_report
        return result

    @staticmethod
    def calculate_token_usage(
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: Optional[int] = 0,
        cached_tokens: Optional[PromptTokensDetails] = None,
        image_tokens: int = 0,
        audio_tokens: int = 0,
        video_tokens: int = 0,
    ) -> UsageInfo:
        """Calculate cumulative token usage; final stream usage is the total."""
        details = cached_tokens
        if image_tokens or audio_tokens or video_tokens:
            if details is None:
                details = PromptTokensDetails()
            if image_tokens:
                details.image_tokens = image_tokens
            if audio_tokens:
                details.audio_tokens = audio_tokens
            if video_tokens:
                details.video_tokens = video_tokens

        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=details,
            reasoning_tokens=reasoning_tokens,
            completion_tokens_details=(
                CompletionTokensDetails(reasoning_tokens=reasoning_tokens)
                if reasoning_tokens
                else None
            ),
        )
