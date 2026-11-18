"""Hand AITER's fused_moe the BM=16 sort the radix-4 router writes on ROCm.

At decode and verify batch sizes AITER's fused_moe spends one or two launches on
moe_sorting, which also zeroes its output. route_radix4_with_sort emits both from
the routing launch; register_presorted_moe (ROCm/aiter#6484) hands that sort to
the next fused_moe call, which uses it when the call's topk tensors, M and
block_m match and ignores it otherwise.

Only a Kimi-K3 MoE layer whose experts take the sort asks for it, through the
consume-once stash in route_quant_handoff, so the shared routing functions keep
their signatures:

    KimiK3MoE.forward           stage() before routing, clear() after the layer
    biased_grouped_topk_gpu     try_route_and_register() replaces route_radix4
                                when a staged request covers the batch
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.moe import moe_route_radix4
from sglang.srt.layers import zero_copy_context
from sglang.srt.layers.moe import route_quant_handoff
from sglang.srt.runtime_context import get_buffer
from sglang.srt.utils import is_device_stream_capturing

logger = logging.getLogger(__name__)

# AITER uses the sort only where its tuned kernel for the batch is BM=16 and sorts
# again elsewhere. Its Kimi-K3 A8W4 (at or past ROCm/aiter#6464) and A16W4 tables
# are BM=16 up to 512 tokens, and so is the AgentX A4W4 recipe's replaced table,
# where 512 measured faster than 64. AITER's stock A4W4 table is BM=32 at 65-256
# tokens, where the sort is computed and dropped (4.2-6.1 us per layer on MI355X);
# telling those apart needs AITER to report the block_m it will pick.
_MAX_TOKENS = 512


@cache_once
def _registrar() -> Optional[Callable[[Optional[dict]], None]]:
    try:
        from aiter.fused_moe import register_presorted_moe
    except ImportError:
        return None
    return register_presorted_moe


def available() -> bool:
    """Whether this AITER can take a presort (register_presorted_moe)."""
    return _registrar() is not None


def stage() -> None:
    """Ask the next radix-4 routing call for the fused sort. The caller pairs
    this with clear() after the layer (try/finally)."""
    route_quant_handoff.stage_aiter_presort()


def clear() -> None:
    """Drop the request and a registration that no fused_moe consumed, as
    AITER's caller contract requires."""
    route_quant_handoff.clear()
    _registrar()(None)


class _SortWorkspaces:
    """The sort's arrival counters per (device, stream): launches in flight on
    two streams must not share them. A captured graph keeps its capture stream's
    counters, so it must not replay alongside eager launches on that stream."""

    def __init__(self) -> None:
        self.by_stream: dict[tuple[torch.device, int], torch.Tensor] = {}

    def get(self, device: torch.device) -> Optional[torch.Tensor]:
        """The current stream's counters, or None when they would first be
        allocated inside a graph capture, where they are only zeroed when that
        one graph replays."""
        key = (device, torch.cuda.current_stream(device).cuda_stream)
        ws = self.by_stream.get(key)
        if ws is None:
            if is_device_stream_capturing(device):
                _warn_captured_without_workspace()
                return None
            ws = moe_route_radix4.new_sort_workspace(device)
            self.by_stream[key] = ws
        return ws


@cache_once
def _warn_captured_without_workspace() -> None:
    # A graph captured here keeps the unfused router (and AITER's own sort) on
    # every replay, which otherwise only shows up as a slower step.
    logger.warning(
        "radix-4 fused MoE sort skipped: first use on this stream is a CUDA graph "
        "capture; run the router once eagerly on the capture stream first"
    )


def _published_moe_output(n_tokens: int) -> Optional[torch.Tensor]:
    """The MoE output buffer the model published, when the kernel can zero it.

    AITER only trusts the zeroing when this is also the output fused_moe gets.
    """
    out = zero_copy_context.ctx.moe_output
    if (
        out is not None
        and out.dim() == 2
        and out.shape[0] == n_tokens
        and out.shape[1] % 8 == 0
        and out.dtype == torch.bfloat16
        and out.is_contiguous()
        and out.data_ptr() % 16 == 0
    ):
        return out
    return None


def try_route_and_register(
    scores: torch.Tensor,
    bias: torch.Tensor,
    *,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Route with the fused sort and register it for the next fused_moe call
    when a staged request covers this batch. Returns (weights, ids) on a hit,
    None otherwise (the caller runs route_radix4).

    Caller must have checked moe_route_radix4.covered().
    """
    requested, num_token_non_padded = route_quant_handoff.take_aiter_presort()
    if not requested or not 1 <= scores.shape[0] <= _MAX_TOKENS:
        return None
    sort_ws = get_buffer("aiter_presort_sort_ws", _SortWorkspaces).get(scores.device)
    if sort_ws is None:
        return None
    weights, ids, aux = moe_route_radix4.route_radix4_with_sort(
        scores=scores,
        bias=bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        sort_ws=sort_ws,
        moe_buf=_published_moe_output(scores.shape[0]),
        num_token_non_padded=num_token_non_padded,
    )
    _registrar()(aux)
    return weights, ids
