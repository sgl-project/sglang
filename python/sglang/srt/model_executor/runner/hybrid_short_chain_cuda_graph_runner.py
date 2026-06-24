"""CudaGraphRunner for HYBRID_SUFFIX_MTP's MTP-backend path.

Captures TARGET_VERIFY graphs at K = ``speculative_num_steps + 1`` (the EAGLE
draft-chain width) so the MTP backend runs on a cuda graph instead of falling
back to eager.

Coexists with two sibling runners that share the same ``model_runner`` and
``attn_backend`` (only their static input buffers + graph dict are
per-instance):
  - main ``decode_cuda_graph_runner`` — K = ``speculative_num_draft_tokens``
    (the wide SUFFIX chain)
  - ``hybrid_baseline_cuda_graph_runner`` — K = 1 (NONE path)

``HybridSuffixMTPWorkerV2.verify`` selects the matching runner by setting
``model_runner.decode_cuda_graph_runner`` to it immediately before the verify
forward (keyed on ``spec_info.draft_token_num``), then restores the main
runner afterward — so there is no width-probing dispatch to maintain.

K is set via ``DecodeCudaGraphRunner``'s ``speculative_num_draft_tokens``
constructor override; ``get_num_tokens_per_bs_for_target_verify`` returns it
verbatim for the HYBRID plugin algorithm. ``capture_hidden_mode`` is forced to
FULL automatically because HYBRID reports ``is_eagle()`` True, so the parent's
``get_spec_info`` takes the EAGLE branch (EagleVerifyInput, FULL hidden mode +
hidden_states buffer) — the same input the MTP keep-up path needs at replay.
"""

from __future__ import annotations

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)


class HybridShortChainCudaGraphRunner(DecodeCudaGraphRunner):
    """CudaGraphRunner for the HYBRID MTP path verify (K = num_steps + 1)."""

    def __init__(self, model_runner):
        short_k = model_runner.server_args.speculative_num_steps + 1
        super().__init__(model_runner, speculative_num_draft_tokens=short_k)

        # Sanity-check: parent picked up K=short_k from the ctor override.
        assert self.num_tokens_per_bs == short_k, (
            f"short-chain runner expected num_tokens_per_bs={short_k}, "
            f"got {self.num_tokens_per_bs}"
        )
        assert self.capture_forward_mode == ForwardMode.TARGET_VERIFY, (
            f"short-chain runner expected TARGET_VERIFY mode, "
            f"got {self.capture_forward_mode}"
        )
