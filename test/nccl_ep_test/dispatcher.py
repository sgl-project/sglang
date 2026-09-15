"""Synthetic expert arithmetic for real dispatcher regression tests."""

import torch

from .oracle import dequantize_fp8


def forward_layer(dispatcher, x, ids, weights, rank, *, identity=False):
    """Only GPU work: safe to use as a runner callback once EP capture is enabled.

    Snapshot receive counters, but retain the dispatcher's actual combined
    output so multi-layer tests can detect scratch aliasing. CPU oracle checks
    run after the complete forward, outside capture and the staged transaction.
    """
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    dispatched = dispatcher.dispatch(x, StandardTopKOutput(weights, ids, None))
    received = dequantize_fp8(dispatched.hidden_states, dispatched.hidden_states_scale)
    counters = dispatched.masked_m.clone()
    factors = torch.arange(rank * 2 + 1, rank * 2 + 3, device=x.device, dtype=x.dtype)
    expert_output = received if identity else received * factors[:, None, None]
    combined = dispatcher.combine(
        DeepEPLLCombineInput(
            expert_output, dispatched.topk_ids, dispatched.topk_weights
        )
    )
    return received, counters, combined
