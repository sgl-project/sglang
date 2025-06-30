from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, final

from sglang.srt.entrypoints.openai.protocol import UsageInfo


@final
class UsageProcessor:
    """Stateless helpers that turn raw token counts into a UsageInfo."""

    @staticmethod
    def _details_if_cached(count: int) -> Optional[Dict[str, int]]:
        """Return {"cached_tokens": N} only when N > 0 (keeps JSON slim)."""
        return {"cached_tokens": count} if count > 0 else None

    @staticmethod
    def calculate_response_usage(
        responses: List[Dict[str, Any]],
        n_choices: int = 1,
        enable_cache_report: bool = False,
    ) -> UsageInfo:
        completion_tokens = sum(r["meta_info"]["completion_tokens"] for r in responses)

        prompt_tokens = sum(
            responses[i]["meta_info"]["prompt_tokens"]
            for i in range(0, len(responses), n_choices)
        )

        cached_details = None
        if enable_cache_report:
            cached_total = sum(
                r["meta_info"].get("cached_tokens", 0) for r in responses
            )
            cached_details = UsageProcessor._details_if_cached(cached_total)

        return UsageProcessor.calculate_token_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_details,
        )

    @staticmethod
    def calculate_streaming_usage(
        prompt_tokens: Mapping[int, int],
        completion_tokens: Mapping[int, int],
        cached_tokens: Mapping[int, int],
        n_choices: int,
        enable_cache_report: bool = False,
    ) -> UsageInfo:
        # index % n_choices == 0 marks the first choice of a prompt
        total_prompt_tokens = sum(
            tok for idx, tok in prompt_tokens.items() if idx % n_choices == 0
        )
        total_completion_tokens = sum(completion_tokens.values())

        cached_details = (
            UsageProcessor._details_if_cached(sum(cached_tokens.values()))
            if enable_cache_report
            else None
        )

        return UsageProcessor.calculate_token_usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            cached_tokens=cached_details,
        )

    @staticmethod
    def calculate_token_usage(
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: Optional[Dict[str, int]] = None,
    ) -> UsageInfo:
        """Calculate token usage information"""
        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=cached_tokens,
        )
