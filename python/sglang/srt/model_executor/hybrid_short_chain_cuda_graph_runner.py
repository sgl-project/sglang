"""CudaGraphRunner for HYBRID_SUFFIX_MTP's MTP-backend path.

Captures TARGET_VERIFY graphs at K = ``speculative_num_steps + 1`` (EAGLE
draft chain width) so the MTP backend runs on cuda graph instead of
falling back to eager.

Coexists with the main runner (K = ``speculative_num_draft_tokens``,
SUFFIX path) and the baseline runner (K = 1, NONE path). All three share
the same ``model_runner`` and ``attn_backend``; only their input/output
buffers and graph dict are per-instance. The DSv4 attention backend's
``cuda_graph_metadata_of_bucket_and_bs`` is keyed by ``(bs, K)`` for the
TARGET_VERIFY bucket so this runner and the main runner don't overwrite
each other.

K is set via ``CudaGraphRunner``'s ``speculative_num_draft_tokens``
constructor override (the parent's ``get_num_tokens_per_bs_for_target_verify``
returns it verbatim for plugin algorithms).

``capture_hidden_mode=FULL`` is set globally for HYBRID by the parent
class (see ``cuda_graph_runner.py``); EAGLE's ``EagleVerifyInput`` also
requests FULL by default, so the captured graphs match what MTP verify
asks for at replay.
"""

from __future__ import annotations

from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
)


class HybridShortChainCudaGraphRunner(CudaGraphRunner):
    """CudaGraphRunner specialized for HYBRID MTP path verify (K = num_steps + 1)."""

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

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        # Only claim batches whose total input_ids width matches bs * K.
        # The SUFFIX path (bs * K_suffix) and NONE path (bs * 1) fall
        # through to their own runners.
        if (
            forward_batch.batch_size * self.num_tokens_per_bs
            != forward_batch.input_ids.numel()
        ):
            return False
        return super().can_run(forward_batch)
