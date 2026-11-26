"""Hand AITER's fused_moe the sort the GLM-5.2 router writes on ROCm.

At decode and verify batch sizes the aiter path spends three launches between
the router GEMM and the first expert GEMM: the top-k, the shared-expert append
and moe_sorting (which also zeroes fused_moe's output). glm_route_sort does all
of it in one, and route_and_stage keeps the sort and the zeroed output here.

The AITER runner takes them at the top of its next call (take_staged), which
drops them unless that call got these exact topk tensors, and registers the
sort with register_presorted_moe right before its fused_moe. fused_moe checks
it against its own arguments (topk tensors, M, block_m, expert mask) and
re-sorts on any mismatch. A runner call that returns before fused_moe never
registers it, so no later fused_moe can pick up a sort written for another call.
"""

from __future__ import annotations

import functools
import logging
from typing import Callable, Optional, Tuple

import torch

from sglang.kernels.ops.moe import glm_route_sort
from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

# The sort the last route_and_stage wrote, until the AITER runner takes it.
_staged: Optional[dict] = None
_announced = False


@functools.cache
def _registrar() -> Optional[Callable[[Optional[dict]], None]]:
    # An AITER whose fused_moe cannot match a plain-layout sort (ROCm/aiter#8045)
    # still has register_presorted_moe, but would drop every registration.
    try:
        from aiter.fused_moe import register_presorted_moe
        from aiter.presorted_moe import match_plain_presorted_moe  # noqa: F401
    except ImportError:
        return None
    return register_presorted_moe


def kernel_ready(router_logits: torch.Tensor, correction_bias: torch.Tensor) -> bool:
    return (
        envs.SGLANG_OPT_HIP_GLM_ROUTE_SORT.get()
        # The handoff is host state, which a compiled graph would not replay.
        and not torch.compiler.is_compiling()
        # An all-to-all dispatch hands the runner topk tensors of its own.
        and get_moe_a2a_backend().is_none()
        and glm_route_sort.covered(router_logits, correction_bias)
        and _registrar() is not None
        and _block_m(router_logits.shape[0]) is not None
        and glm_route_sort.available()
        and glm_route_sort.ready(router_logits.device, router_logits.shape[0])
    )


def _block_m(n_tokens: int) -> Optional[int]:
    """The block size fused_moe gave its general-path sort of this many tokens
    last time, or None if it has not sorted them on that path.

    fused_moe picks the pipeline and block size per token count from its tuned
    configs, and only its general moe_sorting path takes a plain-layout sort;
    the BM=16 aux-sort pipeline, its fused sort+quant and FLAT sort for
    themselves. Every GLM MoE layer shares one configuration, so the previous
    call's choice holds. A batch size fused_moe has not sorted on that path (the
    first layer of the first pass at it, or one it serves another way) keeps
    the unfused router rather than pay for a sort fused_moe would redo."""
    lookup = _block_m_lookup()
    if lookup is None:
        return None
    return lookup(n_tokens, glm_route_sort.NUM_EXPERTS + 1, glm_route_sort.TOPK)


@functools.cache
def _block_m_lookup():
    try:
        from aiter.fused_moe import plain_sort_block_m
    except ImportError:
        return None
    return plain_sort_block_m


def route_and_stage(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    shared_weight: float,
    renormalize: bool,
    num_token_non_padded: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Route with the shared expert appended and stage the sort for the runner.

    Returns (weights [M, 9], ids [M, 9]) as _post_process_topk_ids would leave
    them on HIP, padding rows included. Caller must have checked kernel_ready().
    """
    global _staged, _announced
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
    _staged = {
        "layout": "plain",
        # AITER matches on these addresses; holding the tensors keeps a
        # later same-size allocation from reusing them while staged.
        "topk_ids": ids,
        "topk_weights": weights,
        "moe_buf": moe_out,
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
    return weights, ids


def _same_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.data_ptr() == b.data_ptr()
        and a.dtype == b.dtype
        and a.shape == b.shape
        and a.stride() == b.stride()
    )


def take_staged(topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> Optional[dict]:
    """Consume the staged sort, returning it if it was written for these topk
    tensors. The staged sort holds them, so an equal address is the same data."""
    global _staged
    if torch.compiler.is_compiling():
        return None
    staged, _staged = _staged, None
    if (
        staged is None
        or not _same_tensor(staged["topk_ids"], topk_ids)
        or not _same_tensor(staged["topk_weights"], topk_weights)
    ):
        return None
    return staged


def zeroed_output(
    staged: dict, shape: torch.Size, dtype: torch.dtype, device: torch.device
) -> Optional[torch.Tensor]:
    """The output the router zeroed for the fused_moe call its sort is for."""
    out = staged["moe_buf"]
    if out.shape == shape and out.dtype == dtype and out.device == device:
        return out
    return None


def register(staged: dict) -> None:
    """Hand the sort to the fused_moe call that follows on this thread."""
    _registrar()(staged)
