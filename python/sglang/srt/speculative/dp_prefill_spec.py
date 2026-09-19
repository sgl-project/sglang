"""Plan token counts for prefill and speculative decoding across DP ranks."""

import os
from dataclasses import dataclass
from typing import Tuple

ENABLED = os.environ.get("SGLANG_EXPERIMENTAL_DP_PREFILL_SPEC", "0") == "1"


@dataclass(frozen=True)
class DPPrefillSpecPlan:
    # Counts from the scheduler: prefill token counts or decode request counts.
    counts: Tuple[int, ...]
    logprob_counts: Tuple[int, ...]
    prefills: Tuple[bool, ...]
    draft_width: int
    verify_width: int

    def __post_init__(self):
        if not (len(self.counts) == len(self.logprob_counts) == len(self.prefills)):
            raise ValueError("Incomplete DP prefill/spec metadata")
        if not self.counts or min(self.counts + self.logprob_counts) < 0:
            raise ValueError("Invalid DP token counts")
        if self.draft_width < 1 or self.verify_width < 1:
            raise ValueError("Invalid speculative widths")

    @property
    def heterogeneous(self):
        return any(p and n for p, n in zip(self.prefills, self.counts)) and any(
            not p and n for p, n in zip(self.prefills, self.counts)
        )

    def phase_counts(self, phase):
        if phase == "draft":
            tokens = tuple(
                0 if p else n * self.draft_width
                for p, n in zip(self.prefills, self.counts)
            )
            return tokens, tokens
        if phase not in ("target", "draft_extend"):
            raise ValueError("Unknown DP speculative phase")
        return (
            tuple(
                n if p else n * self.verify_width
                for p, n in zip(self.prefills, self.counts)
            ),
            tuple(
                n if p else n * self.verify_width
                for p, n in zip(self.prefills, self.logprob_counts)
            ),
        )

    def apply(self, batch, phase, rank):
        tokens, logprobs = self.phase_counts(phase)
        # EP-only batches retain only their local counts.
        if len(batch.global_num_tokens) == 1:
            tokens, logprobs = (tokens[rank],), (logprobs[rank],)
        elif len(batch.global_num_tokens) != len(tokens):
            raise ValueError("Unexpected DP synchronization group width")
        batch.global_num_tokens = list(tokens)
        batch.global_num_tokens_for_logprob = list(logprobs)
        batch.dp_prefill_spec_phase = phase
        batch.is_extend_in_batch = True
        batch.can_run_decode_cuda_graph = False
        batch.can_run_dp_prefill_cuda_graph = False
