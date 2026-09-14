"""Breakable prefill bridge for the request-dependent pooled-key indexer."""

import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)


def _kpool_indexer_prefill_with_output(
    indexer,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    layer_id: int,
) -> None:
    # Metadata, write counts and cache destinations change between requests.
    # Resolve the live batch inside the eager break, never from capture args.
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    n = forward_batch.extend_num_tokens
    if n is None or not 0 <= n <= x.shape[0]:
        raise ValueError(f"Invalid pooled-indexer prefill token count: {n}")
    if n > q_lora.shape[0] or n > positions.shape[0]:
        raise ValueError("Pooled-indexer prefill inputs have inconsistent rows")
    return_indices = output.shape[0] != 0
    result = indexer._forward_cuda_impl(
        x=x[:n],
        q_lora=q_lora[:n],
        positions=positions[:n],
        forward_batch=forward_batch,
        layer_id=layer_id,
        return_indices=return_indices,
    )
    if not return_indices:
        return
    if result is None or result.shape != (n, output.shape[1]):
        raise ValueError("Pooled-indexer prefill returned an unexpected top-k shape")
    # The following captured attention segment reads this stable padded buffer.
    output[:n].copy_(result)
    output[n:].fill_(-1)


def _kpool_indexer_prefill_capture_stub(
    indexer,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    layer_id: int,
) -> None:
    output.fill_(-1)


bcg_kpool_indexer_prefill_with_output = eager_on_graph(
    True, capture_stub=_kpool_indexer_prefill_capture_stub
)(_kpool_indexer_prefill_with_output)
