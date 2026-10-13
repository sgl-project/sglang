from __future__ import annotations

import dataclasses
from typing import List


@dataclasses.dataclass(slots=True, kw_only=True)
class SpecReqStats:
    """A request's speculative-decoding statistics, ``req.spec_stats``: plain
    data, written by the verify-result commit and read by the output streamer."""

    # Per-request count of verification forward passes.
    verify_ct: int = 0
    # Per-request count of accepted draft tokens (excludes the bonus token).
    num_correct_drafts: int = 0
    num_block_accept_tokens: int = 0
    num_cap_tokens: int = 0
    # Acceptance histogram for speculative decoding.
    # List index = number of accepted tokens in a step, List value = count of steps with that many accepted tokens.
    # Example: histogram[0] = 5 means 5 steps with 0 accepted tokens, histogram[3] = 10 means 10 steps with 3 accepted tokens.
    correct_drafts_histogram: List[int] = dataclasses.field(default_factory=list)
    cap_lens_histogram: List[int] = dataclasses.field(default_factory=list)

    def update_correct_drafts_histogram(self, num_correct_drafts: int):
        """Record one step accepted draft count (excludes bonus token) into the histogram."""
        if len(self.correct_drafts_histogram) <= num_correct_drafts:
            self.correct_drafts_histogram.extend(
                [0] * (num_correct_drafts - len(self.correct_drafts_histogram) + 1)
            )
        self.correct_drafts_histogram[num_correct_drafts] += 1

    def update_cap_lens_histogram(self, cap_len: int):
        cap_len = int(cap_len)
        if len(self.cap_lens_histogram) <= cap_len:
            self.cap_lens_histogram.extend(
                [0] * (cap_len - len(self.cap_lens_histogram) + 1)
            )
        self.cap_lens_histogram[cap_len] += 1
