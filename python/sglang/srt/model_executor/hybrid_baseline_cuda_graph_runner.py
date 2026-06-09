"""CudaGraphRunner for HYBRID_SUFFIX_MTP's NONE-backend (K=1, no spec).

Third cuda graph runner alongside:
  - main ``graph_runner``         — K = ``speculative_num_draft_tokens``
                                    (SUFFIX TARGET_VERIFY)
  - ``short_chain_graph_runner``  — K = ``speculative_num_steps + 1``
                                    (MTP TARGET_VERIFY)
  - this ``baseline_chain_graph_runner`` — K = 1 (NONE)

``ModelRunner._forward_raw`` dispatches in width-ascending order
(short_chain → baseline → main). Each runner's ``can_run`` short-circuits
when ``forward_batch.input_ids.numel()`` doesn't match its captured K, so
the right runner claims each step.

K=1 is set via ``CudaGraphRunner``'s ``speculative_num_draft_tokens``
constructor override. The graphs capture in TARGET_VERIFY mode — which is
what the NONE path actually runs: ``_decode_step_none`` builds a K=1
``EagleVerifyInput`` (bonus only, no spec slots) and goes through
``prepare_for_v2_verify``, which sets ``forward_mode = TARGET_VERIFY``.
The DSv4 attention backend's TARGET_VERIFY metadata cache is keyed by
``(bs, K)``, so K=1 coexists with the other two runners' K values.
"""

from __future__ import annotations

from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
)


class HybridBaselineCudaGraphRunner(CudaGraphRunner):
    """CudaGraphRunner specialized for HYBRID NONE-backend path (K=1)."""

    def __init__(self, model_runner):
        # K=1 chain has 0 draft steps (bonus only) — keep the capture-time
        # EagleVerifyInput's spec_steps consistent with its draft_token_num.
        super().__init__(
            model_runner, speculative_num_steps=0, speculative_num_draft_tokens=1
        )

        assert (
            self.num_tokens_per_bs == 1
        ), f"baseline runner expected num_tokens_per_bs=1, got {self.num_tokens_per_bs}"

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        # Claim batches whose total input_ids width == bs (K=1 per req). The
        # wider SUFFIX (bs*K_suffix) and MTP (bs*K_mtp) batches naturally
        # fall through to their own runners.
        bs = forward_batch.batch_size
        numel = forward_batch.input_ids.numel()
        if bs != numel:
            return False
        return super().can_run(forward_batch)
