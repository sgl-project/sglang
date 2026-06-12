from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, final

from sglang.srt.entrypoints.openai.protocol import (
    CompletionTokensDetails,
    PromptTokensDetails,
    UsageInfo,
)


@final
class UsageProcessor:
    """Stateless helpers that turn raw token counts into a UsageInfo."""

    @staticmethod
    def _details_if_cached(count: int) -> Optional[PromptTokensDetails]:
        """Return PromptTokensDetails only when count > 0 (keeps JSON slim)."""
        return PromptTokensDetails(cached_tokens=count) if count > 0 else None

    @staticmethod
    def _details_if_spec(
        meta_infos: Iterable[Dict[str, Any]],
    ) -> Optional[CompletionTokensDetails]:
        """Aggregate spec-decode counts from meta_info dicts into a
        CompletionTokensDetails. Returns None when no response carried spec
        metrics (i.e. speculative decoding was off or never verified)."""
        total_correct_drafts = 0
        total_proposed_drafts = 0
        any_spec_response = False
        for meta_info in meta_infos:
            if "spec_num_proposed_drafts" in meta_info:
                total_correct_drafts += meta_info.get("spec_num_correct_drafts", 0)
                total_proposed_drafts += meta_info["spec_num_proposed_drafts"]
                any_spec_response = True
        if not any_spec_response:
            return None
        return CompletionTokensDetails(
            accepted_prediction_tokens=total_correct_drafts,
            rejected_prediction_tokens=max(
                total_proposed_drafts - total_correct_drafts, 0
            ),
        )

    @staticmethod
    def calculate_response_usage(
        responses: List[Dict[str, Any]],
        n_choices: int = 1,
        enable_cache_report: bool = False,
        enable_spec_decode_usage: bool = False,
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
            cached_details = UsageProcessor._details_if_cached(cached_total)

        completion_details = (
            UsageProcessor._details_if_spec(r["meta_info"] for r in responses)
            if enable_spec_decode_usage
            else None
        )

        return UsageProcessor.calculate_token_usage(
            prompt_tokens=prompt_tokens,
            reasoning_tokens=reasoning_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_details,
            completion_tokens_details=completion_details,
        )

    @staticmethod
    def calculate_streaming_usage(
        prompt_tokens: Mapping[int, int],
        reasoning_tokens: Mapping[int, int],
        completion_tokens: Mapping[int, int],
        cached_tokens: Mapping[int, int],
        n_choices: int,
        enable_cache_report: bool = False,
        accepted_prediction_tokens: Optional[Mapping[int, int]] = None,
        rejected_prediction_tokens: Optional[Mapping[int, int]] = None,
    ) -> UsageInfo:
        # index % n_choices == 0 marks the first choice of a prompt
        total_prompt_tokens = sum(
            tok for idx, tok in prompt_tokens.items() if idx % n_choices == 0
        )
        total_reasoning_tokens = sum(reasoning_tokens.values())
        total_completion_tokens = sum(completion_tokens.values())

        cached_details = (
            UsageProcessor._details_if_cached(
                sum(tok for idx, tok in cached_tokens.items() if idx % n_choices == 0)
            )
            if enable_cache_report
            else None
        )

        completion_details: Optional[CompletionTokensDetails] = None
        if accepted_prediction_tokens or rejected_prediction_tokens:
            completion_details = CompletionTokensDetails(
                accepted_prediction_tokens=sum(
                    (accepted_prediction_tokens or {}).values()
                ),
                rejected_prediction_tokens=sum(
                    (rejected_prediction_tokens or {}).values()
                ),
            )

        return UsageProcessor.calculate_token_usage(
            prompt_tokens=total_prompt_tokens,
            reasoning_tokens=total_reasoning_tokens,
            completion_tokens=total_completion_tokens,
            cached_tokens=cached_details,
            completion_tokens_details=completion_details,
        )

    @staticmethod
    def calculate_token_usage(
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: Optional[int] = 0,
        cached_tokens: Optional[PromptTokensDetails] = None,
        completion_tokens_details: Optional[CompletionTokensDetails] = None,
    ) -> UsageInfo:
        """Calculate token usage information"""
        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            completion_tokens_details=completion_tokens_details,
        )
