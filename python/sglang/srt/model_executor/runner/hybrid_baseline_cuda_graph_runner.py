"""CudaGraphRunner for HYBRID_SUFFIX_MTP's NONE-backend (K=1, no spec).

Third cuda graph runner alongside:
  - main ``decode_cuda_graph_runner``        — K = ``speculative_num_draft_tokens``
                                               (SUFFIX TARGET_VERIFY)
  - ``hybrid_short_chain_cuda_graph_runner`` — K = ``speculative_num_steps + 1``
                                               (MTP TARGET_VERIFY)
  - this ``hybrid_baseline_cuda_graph_runner`` — K = 1 (NONE)

``HybridSuffixMTPWorkerV2.verify`` swaps ``model_runner.decode_cuda_graph_runner``
to the K-matching runner before the verify forward (keyed on
``spec_info.draft_token_num``), so the right graph claims each step without
any width-probing dispatch.

K=1 is set via ``DecodeCudaGraphRunner``'s ``speculative_num_draft_tokens``
constructor override. The graphs capture in TARGET_VERIFY mode — which is what
the NONE path actually runs: ``_decode_step_none`` builds a K=1
``EagleVerifyInput`` (bonus only, no spec slots) and goes through the EAGLE V2
verify scaffold, which runs in TARGET_VERIFY mode.
"""

from __future__ import annotations

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)


class HybridBaselineCudaGraphRunner(DecodeCudaGraphRunner):
    """CudaGraphRunner for the HYBRID NONE-backend path (K=1)."""

    def __init__(self, model_runner):
        # K=1 chain has 0 draft steps (bonus only) — keep the capture-time
        # EagleVerifyInput's spec_steps consistent with its draft_token_num.
        super().__init__(
            model_runner, speculative_num_steps=0, speculative_num_draft_tokens=1
        )

        assert (
            self.num_tokens_per_bs == 1
        ), f"baseline runner expected num_tokens_per_bs=1, got {self.num_tokens_per_bs}"
