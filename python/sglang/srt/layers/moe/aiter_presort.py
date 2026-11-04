"""Hand AITER's fused_moe the BM=16 sort the radix-4 router writes on ROCm.

At decode batch sizes AITER's fused_moe spends a launch on moe_sorting and
another on zeroing its output. route_radix4_with_sort emits both from the
routing launch; register_presorted_moe (ROCm/aiter#6484) hands that sort to the
next fused_moe call, which uses it when the call's topk tensors, M and block_m
match and ignores it otherwise. Without that AITER API the router runs unfused.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.moe import moe_route_radix4
from sglang.srt.environ import envs
from sglang.srt.layers import zero_copy_context

logger = logging.getLogger(__name__)


@cache_once
def _registrar() -> Optional[Callable[[Optional[dict]], None]]:
    try:
        from aiter.fused_moe import register_presorted_moe
    except ImportError:
        return None
    return register_presorted_moe


def clear_registration() -> None:
    """Drop a registration that no fused_moe consumed, as AITER's caller contract requires."""
    registrar = _registrar()
    if registrar is not None:
        registrar(None)


def enabled(scores: torch.Tensor) -> bool:
    return (
        envs.SGLANG_OPT_HIP_K3_RADIX4_FUSE_SORT.get()
        and 1 <= scores.shape[0] <= moe_route_radix4.FUSED_SORT_MAX_TOKENS
        and _registrar() is not None
        and moe_route_radix4.fused_sort_ready(scores.device)
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


def route_and_register(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    num_token_non_padded: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Route with the fused sort and register it for the next fused_moe call.

    Caller must have checked moe_route_radix4.covered() and enabled().
    """
    weights, ids, aux = moe_route_radix4.route_radix4_with_sort(
        scores,
        bias,
        topk,
        renormalize,
        routed_scaling_factor,
        moe_buf=_published_moe_output(scores.shape[0]),
        num_token_non_padded=num_token_non_padded,
    )
    _registrar()(aux)
    return weights, ids
