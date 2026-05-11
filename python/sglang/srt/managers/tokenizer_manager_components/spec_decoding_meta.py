from __future__ import annotations

from typing import Any, Dict, Union

from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
)


def fill_spec_decoding_meta(
    meta_info: Dict[str, Any],
    *,
    recv_obj: Union[
        BatchStrOutput,
        BatchEmbeddingOutput,
        BatchTokenIDOutput,
    ],
    i: int,
    speculative_num_draft_tokens: int,
) -> None:
    """Calculate speculative decoding metrics, such as acceptance rate and acceptance length metrics."""
    if (
        hasattr(recv_obj, "spec_verify_ct")
        and recv_obj.spec_verify_ct[i] > 0
        and hasattr(recv_obj, "spec_accepted_drafts")
        and len(recv_obj.spec_accepted_drafts) > i
    ):
        # Total number of proposed draft tokens per request.
        all_drafts = recv_obj.spec_verify_ct[i] * (speculative_num_draft_tokens - 1)
        accepted_drafts = recv_obj.spec_accepted_drafts[i]

        # Calculate per-request acceptance rate and average acceptance length.
        if all_drafts > 0:
            # accept_rate: accepted_drafts / total_proposed_drafts (strict count, no bonus).
            meta_info["spec_accept_rate"] = accepted_drafts / all_drafts
            # accept_length: completion_tokens / verify_ct (includes bonus token).
            meta_info["spec_accept_length"] = (
                recv_obj.completion_tokens[i] / recv_obj.spec_verify_ct[i]
            )

            meta_info["spec_accepted_drafts"] = accepted_drafts
            meta_info["spec_proposed_drafts"] = all_drafts
            meta_info["spec_verify_ct"] = recv_obj.spec_verify_ct[i]

        # Acceptance histogram: tracks how many decoding steps accepted a certain number of draft tokens.
        if (
            recv_obj.spec_acceptance_histogram
            and len(recv_obj.spec_acceptance_histogram) > i
            and recv_obj.spec_acceptance_histogram[i]
        ):
            meta_info["spec_accept_histogram"] = recv_obj.spec_acceptance_histogram[i]
