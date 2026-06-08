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
        and hasattr(recv_obj, "spec_num_correct_drafts")
        and len(recv_obj.spec_num_correct_drafts) > i
    ):
        # Total number of proposed draft tokens per request.
        num_proposed_drafts = recv_obj.spec_verify_ct[i] * (
            speculative_num_draft_tokens - 1
        )
        num_correct_drafts = recv_obj.spec_num_correct_drafts[i]

        # Calculate per-request acceptance rate and average acceptance length.
        if num_proposed_drafts > 0:
            # accept_rate: num_correct_drafts / num_proposed_drafts (strict count, no bonus).
            meta_info["spec_accept_rate"] = num_correct_drafts / num_proposed_drafts
            # accept_length: completion_tokens / verify_ct (includes bonus token).
            meta_info["spec_accept_length"] = (
                recv_obj.completion_tokens[i] / recv_obj.spec_verify_ct[i]
            )

            meta_info["spec_num_correct_drafts"] = num_correct_drafts
            meta_info["spec_num_proposed_drafts"] = num_proposed_drafts
            meta_info["spec_verify_ct"] = recv_obj.spec_verify_ct[i]

            # FIXME: backward-compat aliases, remove in next release.
            meta_info["spec_accepted_drafts"] = num_correct_drafts
            meta_info["spec_proposed_drafts"] = num_proposed_drafts

        # Acceptance histogram: tracks how many decoding steps accepted a certain number of draft tokens.
        if (
            recv_obj.spec_correct_drafts_histogram
            and len(recv_obj.spec_correct_drafts_histogram) > i
            and recv_obj.spec_correct_drafts_histogram[i]
        ):
            meta_info["spec_correct_drafts_histogram"] = (
                recv_obj.spec_correct_drafts_histogram[i]
            )
            # FIXME: backward-compat alias, remove in next release.
            meta_info["spec_accept_histogram"] = recv_obj.spec_correct_drafts_histogram[
                i
            ]
