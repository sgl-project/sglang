"""Hand AITER's fused_moe the BM=16 sort the radix-4 router writes on ROCm.

At decode and verify batch sizes AITER's fused_moe spends one or two launches on
moe_sorting, which also zeroes its output. route_radix4_with_sort emits both from
the routing launch; register_presorted_moe (ROCm/aiter#6484) hands that sort to
the next fused_moe call, which uses it when the call's topk tensors, M and
block_m match and ignores it otherwise. Without that AITER API the router runs
unfused. Only a layer whose TopKConfig sets fused_aiter_presort asks for the sort.
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
        and 1
        <= scores.shape[0]
        <= envs.SGLANG_OPT_HIP_K3_RADIX4_FUSE_SORT_MAX_TOKENS.get()
        and _registrar() is not None
        and _workspace_ready(scores.device)
    )


def _workspace_ready(device: torch.device) -> bool:
    ready = moe_route_radix4.fused_sort_ready(device)
    if not ready:
        _warn_captured_without_workspace()
    return ready


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


def route_and_register(
    *,
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
        scores=scores,
        bias=bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        moe_buf=_published_moe_output(scores.shape[0]),
        num_token_non_padded=num_token_non_padded,
    )
    _registrar()(aux)
    return weights, ids
