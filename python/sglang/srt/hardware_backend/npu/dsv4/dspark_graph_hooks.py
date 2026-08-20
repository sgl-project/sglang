"""NPU-only CUDA-graph hooks for DSpark on the DSV4 backend."""

from __future__ import annotations

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput


def make_dspark_verify_epilogue_capture_hook(epilogue):
    """Build the NPU target-verify tail hook for folded DSpark epilogue.

    DSV4 graph replay keeps two sequence-length domains: ``seq_lens`` used by
    the target Attention/KV view may include the verify window, while
    ``forward_metadata.start_pos`` is refreshed from ``live_seq_lens`` before
    replay. Accept finalization must use the latter committed prefix.

    Keep this adaptation out of the shared epilogue so CUDA continues to use
    its original ``forward_batch.seq_lens`` path.
    """

    def capture_hook(runner, out, forward_batch, num_tokens) -> None:
        if runner.model_runner.is_draft_worker or not runner.ragged_verify_mode:
            return
        if (
            not isinstance(out, LogitsProcessorOutput)
            or out.next_token_logits is None
            or out.hidden_states is None
        ):
            return

        metadata = runner.model_runner.attn_backend.forward_metadata
        live_prefix_lens = getattr(metadata, "start_pos", None)
        if live_prefix_lens is None:
            raise RuntimeError(
                "DSpark NPU folded epilogue requires DSV4 "
                "forward_metadata.start_pos."
            )
        epilogue(
            compact_logits=out.next_token_logits,
            compact_hidden=out.hidden_states,
            input_ids=forward_batch.input_ids,
            seq_lens=live_prefix_lens.to(torch.int64),
            req_pool_indices=forward_batch.req_pool_indices,
            bs=forward_batch.batch_size,
        )

    return capture_hook
