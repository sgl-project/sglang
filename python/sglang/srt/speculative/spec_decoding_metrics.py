"""The speculative-decoding stats a finished request reports in `meta_info`.

Python's TokenizerManager computes them from the scheduler's batch output; in
Rust frontend mode the scheduler computes them itself (its `request_stats`),
so both servers report the same keys and values.
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional


@lru_cache(maxsize=1)
def ragged_verify_cap_accept() -> bool:
    # The mode env is fixed at server launch; cache to keep it off the
    # per-request metrics path.
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )

    return read_ragged_verify_mode() is RaggedVerifyMode.CAP_ACCEPT


def populate_spec_info(
    meta_info: Dict[str, Any],
    *,
    num_draft_tokens: int,
    completion_tokens: int,
    verify_ct: int,
    num_correct_drafts: int,
    num_cap_tokens: Optional[int],
    num_block_accept_tokens: Optional[int],
    correct_drafts_histogram: Optional[List[int]],
    cap_lens_histogram: Optional[List[int]],
) -> None:
    """Add a finished request's acceptance rate and length metrics, from its
    counters (None for a counter the scheduler did not send) and the
    server's `speculative_num_draft_tokens`. A request that ran no verify
    step adds none."""
    if verify_ct <= 0:
        return
    # Total number of proposed draft tokens per request.
    num_proposed_drafts = verify_ct * (num_draft_tokens - 1)

    # Calculate per-request acceptance rate and average acceptance length.
    if num_proposed_drafts > 0:
        # accept_rate: num_correct_drafts / num_proposed_drafts (strict count, no bonus).
        meta_info["spec_accept_rate"] = num_correct_drafts / num_proposed_drafts
        # accept_length: completion_tokens / verify_ct (includes bonus token).
        meta_info["spec_accept_length"] = completion_tokens / verify_ct

        meta_info["spec_num_correct_drafts"] = num_correct_drafts
        meta_info["spec_num_proposed_drafts"] = num_proposed_drafts
        meta_info["spec_verify_ct"] = verify_ct

        if num_cap_tokens is not None and num_cap_tokens > 0:
            meta_info["spec_cap_length"] = num_cap_tokens / verify_ct
        if ragged_verify_cap_accept() and num_block_accept_tokens is not None:
            meta_info["spec_block_accept_length"] = num_block_accept_tokens / verify_ct

    # Acceptance histogram: tracks how many decoding steps accepted a certain number of draft tokens.
    if correct_drafts_histogram:
        meta_info["spec_correct_drafts_histogram"] = correct_drafts_histogram
    if cap_lens_histogram:
        meta_info["spec_cap_lens_histogram"] = cap_lens_histogram
