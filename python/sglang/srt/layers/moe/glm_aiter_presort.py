"""Hand AITER's fused_moe the sort the GLM-5.2 router writes on ROCm.

At decode and verify batch sizes the aiter path spends three launches between
the router GEMM and the first expert GEMM: the top-k, the shared-expert append
and moe_sorting (which also zeroes fused_moe's output). glm_route_sort does all
of it in one, and register_presorted_moe hands the sort and the zeroed output
to the next fused_moe call, which checks them against its own arguments (topk
tensors, M, block_m, expert mask) and re-sorts on any mismatch.
"""

from __future__ import annotations

import functools
import logging
from typing import Optional, Tuple

import torch

from sglang.kernels.ops.moe import glm_route_sort
from sglang.srt.environ import envs
from sglang.srt.layers.moe.aiter_presort import _registrar
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

# The output buffer the last route_and_register zeroed, until the runner takes it.
_pending_output: Optional[torch.Tensor] = None
_announced = False


def kernel_ready(router_logits: torch.Tensor, correction_bias: torch.Tensor) -> bool:
    return (
        envs.SGLANG_OPT_HIP_GLM_ROUTE_SORT.get()
        and glm_route_sort.covered(router_logits, correction_bias)
        and _registrar() is not None
        and glm_route_sort.available()
        and glm_route_sort.ready(router_logits.device, router_logits.shape[0])
    )


def _block_m(n_tokens: int) -> int:
    """The block size fused_moe sorted this many tokens with last time.

    fused_moe picks it per token count from its tuned configs. Every GLM MoE
    layer shares one configuration, so the previous call's choice holds; until
    a call has been seen (the first layer of the first eager pass) the env
    default stands, and a wrong guess only makes fused_moe sort for itself."""
    lookup = _block_m_lookup()
    if lookup is not None:
        seen = lookup(n_tokens, glm_route_sort.NUM_EXPERTS + 1, glm_route_sort.TOPK)
        if seen is not None:
            return seen
    return envs.SGLANG_OPT_HIP_GLM_ROUTE_SORT_BLOCK_M.get()


@functools.cache
def _block_m_lookup():
    try:
        from aiter.fused_moe import plain_sort_block_m
    except ImportError:
        return None
    return plain_sort_block_m


def route_and_register(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    shared_weight: float,
    renormalize: bool,
    num_token_non_padded: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Route with the shared expert appended and register the sort.

    Returns (weights [M, 9], ids [M, 9]) as _post_process_topk_ids would leave
    them on HIP, padding rows included. Caller must have checked kernel_ready().
    """
    global _pending_output, _announced
    parallel = get_parallel()
    ep_size = parallel.moe_ep_size
    num_routed = glm_route_sort.NUM_EXPERTS // ep_size
    routed_begin = parallel.moe_ep_rank * num_routed
    M = router_logits.shape[0]
    block_m = _block_m(M)
    if not _announced:
        _announced = True
        logger.info(
            f"GLM route+sort presort on: experts [{routed_begin}, "
            f"{routed_begin + num_routed}) + shared"
        )
    moe_out = torch.empty(
        (M, hidden_states.shape[-1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    weights, ids, sort = glm_route_sort.glm_route_sort(
        router_logits,
        correction_bias,
        routed_begin=routed_begin,
        num_routed=num_routed,
        block_m=block_m,
        routed_scaling_factor=routed_scaling_factor,
        shared_weight=shared_weight,
        renormalize=renormalize,
        moe_buf=moe_out,
        num_token_non_padded=num_token_non_padded,
    )
    _registrar()(
        {
            "layout": "plain",
            "topk_ids_ptr": ids.data_ptr(),
            "topk_weights_ptr": weights.data_ptr(),
            "M": M,
            "block_m": block_m,
            "num_experts": glm_route_sort.NUM_EXPERTS + 1,
            "topk": glm_route_sort.TOPK,
            "ep_layout": (
                (routed_begin, num_routed, glm_route_sort.NUM_EXPERTS, 1)
                if ep_size > 1
                else None
            ),
            "moe_buf_zeroed": True,
            "moe_buf_ptr": moe_out.data_ptr(),
            **sort,
        }
    )
    _pending_output = moe_out
    return weights, ids


def take_zeroed_output(
    shape: torch.Size, dtype: torch.dtype, device: torch.device
) -> Optional[torch.Tensor]:
    """The output route_and_register zeroed, for the fused_moe its sort is for."""
    global _pending_output
    out, _pending_output = _pending_output, None
    if (
        out is not None
        and out.shape == shape
        and out.dtype == dtype
        and out.device == device
    ):
        return out
    return None
