from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from sglang.srt.entrypoints.openai.utils import calculate_token_usage


def _details_if_cached(count: int) -> Optional[Dict[str, int]]:
    """Return {"cached_tokens": N} only when N > 0 (keeps JSON slim)."""
    return {"cached_tokens": count} if count > 0 else None


class UsageProcessor:
    """Stateless helpers that turn raw token counts into a UsageInfo."""

    @classmethod
    def calculate_response_usage(
        cls,
        responses: List[Dict[str, Any]],
        *,
        n_choices: int = 1,
        enable_cache_report: bool = False,
    ):
        completion = sum(r["meta_info"]["completion_tokens"] for r in responses)

        prompt = sum(
            responses[i]["meta_info"]["prompt_tokens"]
            for i in range(0, len(responses), n_choices)
        )

        cached_details = None
        if enable_cache_report:
            cached_total = sum(
                r["meta_info"].get("cached_tokens", 0) for r in responses
            )
            cached_details = _details_if_cached(cached_total)

        return calculate_token_usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=cached_details,
        )

    @classmethod
    def calculate_streaming_usage(
        cls,
        prompt_tokens: Mapping[int, int],
        completion_tokens: Mapping[int, int],
        cached_tokens: Mapping[int, int],
        *,
        n_choices: int,
        enable_cache_report: bool = False,
    ):
        # index % n_choices==0 marks the first choice of a prompt
        prompt = sum(tok for idx, tok in prompt_tokens.items() if idx % n_choices == 0)

        completion = sum(completion_tokens.values())

        cached_details = (
            _details_if_cached(sum(cached_tokens.values()))
            if enable_cache_report
            else None
        )

        return calculate_token_usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=cached_details,
        )
