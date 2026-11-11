"""Plan token counts for prefill and speculative decoding across DP ranks."""

from typing import List, Optional

import msgspec
import torch


class DPSpecPrefillCoordinationPlan(msgspec.Struct, frozen=True):
    # Counts from the scheduler: prefill token counts or decode request counts.
    counts: List[int]
    logprob_counts: List[int]
    prefills: torch.Tensor
    draft_width: int
    verify_width: int
    mixed_decode_counts: Optional[List[int]] = None

    @property
    def heterogeneous(self):
        return any(p and n for p, n in zip(self.prefills, self.counts)) and any(
            (not p and n) or mixed
            for p, n, mixed in zip(self.prefills, self.counts, self._mixed_counts)
        )

    @property
    def _mixed_counts(self):
        return self.mixed_decode_counts or [0] * len(self.counts)

    def phase_counts(self, phase):
        if phase == "draft":
            tokens = [
                (mixed if p else n) * self.draft_width
                for p, n, mixed in zip(self.prefills, self.counts, self._mixed_counts)
            ]
            return tokens, tokens
        if phase not in ("target", "draft_extend"):
            raise ValueError("Unknown DP speculative phase")
        return (
            [
                n if p else n * self.verify_width
                for p, n in zip(self.prefills, self.counts)
            ],
            [
                n if p else n * self.verify_width
                for p, n in zip(self.prefills, self.logprob_counts)
            ],
        )

    def apply(self, batch, phase, rank, *, local_only):
        tokens, logprobs = self.phase_counts(phase)
        # Local drafts and EP-only batches retain only this rank's counts.
        if local_only:
            tokens, logprobs = [tokens[rank]], [logprobs[rank]]
        batch.global_num_tokens = tokens
        batch.global_num_tokens_for_logprob = logprobs
        batch.dp_spec_prefill_coordination_applied = True
        batch.is_extend_in_batch = True
        batch.can_run_decode_cuda_graph = False
        batch.can_run_dp_prefill_cuda_graph = False
