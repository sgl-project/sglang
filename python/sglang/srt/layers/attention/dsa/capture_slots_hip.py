"""KV slots for the requests a Kimi-K3.1 prefill graph captures with on ROCm under DCP."""

import torch

from sglang.srt.runtime_context import get_parallel


def fill_by_dcp_owner(model_runner, capture_seq_lens) -> None:
    """Write slot p % dcp_size at each capture request's position p. Sharded DSA
    picks a rank's KV rows by slot owner (slot % dcp_size) but sizes its gather by
    position; these slots keep the two consistent and only touch each rank's
    padding row 0."""
    dcp_size = get_parallel().attn_dcp_size
    if dcp_size <= 1:
        return
    req_to_token = model_runner.req_to_token_pool.req_to_token
    for row, seq_len in enumerate(capture_seq_lens):
        req_to_token[row, :seq_len] = (
            torch.arange(seq_len, device=req_to_token.device, dtype=req_to_token.dtype)
            % dcp_size
        )
