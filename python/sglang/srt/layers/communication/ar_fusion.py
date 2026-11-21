"""Push-plane fused all-reduce over the TP group's symmetric workspace: its
communicator, the launch configuration, and the custom ops the models call.

Kernels: ``kernels.ops.communication.fused_all_reduce.push``. Register the
workspace's communicator once with :func:`register_comm`; the ops key on
``world_size`` alone and take the thin ops' arguments in the same order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication import fused_all_reduce
from sglang.kernels.ops.communication.comm_registry import CommRegistry
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from sglang.kernels.ops.communication.all_reduce import Communicator

_COMMS = CommRegistry("ar_fusion")

# 2-shot from this many rows; measured on GB300 at hidden 6144.
_TWO_SHOT_MIN_TOKENS = {4: 44, 8: 17}
_TWO_SHOT_MIN_TOKENS_DEFAULT = 44

_MAX_CLUSTER_SIZE = 8  # portable cluster size limit


def register_comm(provider: Callable[[], Communicator], *, world_size: int) -> None:
    """Register the provider of the communicator the ops launch over; it runs on
    the first launch (see ``CommRegistry``)."""
    _COMMS.register(provider, world_size=world_size)


def is_registered(world_size: int) -> bool:
    return world_size in _COMMS


def fits_push_slot(max_push_size: int, num_tokens: int, hidden_size: int) -> bool:
    """Whether a ``[num_tokens, hidden_size]`` bf16 row view fits one push slot."""
    return 0 < num_tokens * hidden_size * 2 <= max_push_size


@cache_once
def get_cluster_size(hidden_size: int) -> int:
    """CTAs per row: 1024 bf16 per CTA up to 8 CTAs, else 512 up to 7, else the
    whole-warp split closest to 128 threads per CTA."""
    if hidden_size % 1024 == 0 and hidden_size <= 8192:
        return hidden_size // 1024
    if hidden_size % 512 == 0 and hidden_size <= 3584:
        return hidden_size // 512
    row_vecs = hidden_size // 8
    candidates = [
        c
        for c in range(1, _MAX_CLUSTER_SIZE + 1)
        if hidden_size % 8 == 0
        and row_vecs % c == 0
        and (row_vecs // c) % 32 == 0
        and row_vecs // c <= 1024
    ]
    if not candidates:
        raise ValueError(f"hidden_size={hidden_size} is not a multiple of 256")
    return min(candidates, key=lambda c: (abs(row_vecs // c - 128), c))


def _two_shot(world_size: int, num_tokens: int) -> bool:
    return num_tokens >= _TWO_SHOT_MIN_TOKENS.get(
        world_size, _TWO_SHOT_MIN_TOKENS_DEFAULT
    )


# Residual + RMSNorm epilogue (MiniMax M3). The fakes mirror their op's signature,
# so they bind positional and keyword calls alike.


def _residual_norm_fake(world_size, partial, residual, norm_weight, eps):
    return torch.empty_like(residual), torch.empty_like(residual)


def _moe_residual_norm_fake(
    world_size,
    gemm2_out,
    expanded_idx_to_permuted_idx,
    expert_weights,
    shared_output,
    residual,
    norm_weight,
    eps,
):
    return torch.empty_like(residual), torch.empty_like(residual)


@register_custom_op(
    op_name="push_all_reduce_residual_norm", fake_impl=_residual_norm_fake
)
def all_reduce_residual_norm(
    world_size: int,
    partial: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rmsnorm(r), r)`` with ``r = allreduce(partial) + residual``, both new tensors."""
    num_tokens, hidden_size = partial.shape
    return fused_all_reduce.push.all_reduce_residual_norm(
        _COMMS.get(world_size),
        partial,
        residual,
        norm_weight,
        eps,
        cluster_size=get_cluster_size(hidden_size),
        two_shot=_two_shot(world_size, num_tokens),
        residual_out=torch.empty_like(residual),
    )


@register_custom_op(
    op_name="push_moe_finalize_all_reduce_residual_norm",
    fake_impl=_moe_residual_norm_fake,
)
def moe_finalize_all_reduce_residual_norm(
    world_size: int,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`all_reduce_residual_norm` of the deferred MoE finalize, plus
    ``shared_output`` when given."""
    num_tokens, hidden_size = residual.shape
    return fused_all_reduce.push.moe_finalize_all_reduce_residual_norm(
        _COMMS.get(world_size),
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        shared_output,
        residual,
        norm_weight,
        eps,
        cluster_size=get_cluster_size(hidden_size),
        two_shot=_two_shot(world_size, num_tokens),
        residual_out=torch.empty_like(residual),
    )


# mHC post-mix epilogue (DeepSeek V4, HC = 4); residual is [T, 4, H]


def _mhc_outputs(residual, norm: bool, quant: bool):
    stream = torch.empty_like(residual[:, 0])
    outs = [stream, torch.empty_like(residual)]
    if norm:
        outs.append(torch.empty_like(stream))
    if quant:
        outs.append(torch.empty_like(stream, dtype=torch.float8_e4m3fn))
        outs.append(
            residual.new_empty(residual.shape[-1] // 32 * 128, dtype=torch.uint8)
        )
    return tuple(outs)


def _mhc_norm_fake(world_size, partial, residual, post, comb, pre, norm_weight, eps):
    return _mhc_outputs(residual, norm=True, quant=False)


def _moe_mhc_fake(
    world_size,
    gemm2_out,
    expanded_idx_to_permuted_idx,
    expert_weights,
    shared_output,
    residual,
    post,
    comb,
):
    return _mhc_outputs(residual, norm=False, quant=False)


def _moe_mhc_norm_quant_fake(
    world_size,
    gemm2_out,
    expanded_idx_to_permuted_idx,
    expert_weights,
    shared_output,
    residual,
    post,
    comb,
    pre,
    norm_weight,
    eps,
):
    return _mhc_outputs(residual, norm=True, quant=True)


@register_custom_op(
    op_name="push_all_reduce_mhc_post_combine_norm", fake_impl=_mhc_norm_fake
)
def all_reduce_mhc_post_combine_norm(
    world_size: int,
    partial: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(reduced, mhc_out, normalized)``."""
    return fused_all_reduce.push.all_reduce_mhc_post_combine_norm(
        _COMMS.get(world_size),
        partial,
        residual,
        post,
        comb,
        pre,
        norm_weight,
        eps,
        cluster_size=get_cluster_size(partial.shape[1]),
    )


@register_custom_op(
    op_name="push_moe_finalize_all_reduce_mhc_post", fake_impl=_moe_mhc_fake
)
def moe_finalize_all_reduce_mhc_post(
    world_size: int,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(reduced, mhc_out)``."""
    return fused_all_reduce.push.moe_finalize_all_reduce_mhc_post(
        _COMMS.get(world_size),
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        shared_output,
        residual,
        post,
        comb,
        cluster_size=get_cluster_size(residual.shape[-1]),
    )


@register_custom_op(
    op_name="push_moe_finalize_all_reduce_mhc_post_combine_norm_quant",
    fake_impl=_moe_mhc_norm_quant_fake,
)
def moe_finalize_all_reduce_mhc_post_combine_norm_quant(
    world_size: int,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(reduced, mhc_out, normalized, quantized, scales)``; at most 128 tokens."""
    return fused_all_reduce.push.moe_finalize_all_reduce_mhc_post_combine_norm_quant(
        _COMMS.get(world_size),
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        shared_output,
        residual,
        post,
        comb,
        pre,
        norm_weight,
        eps,
        cluster_size=get_cluster_size(residual.shape[-1]),
    )


# Sandwich norm epilogue (Beam): no custom op, the caller owns the communicator


def get_sandwich_cluster_size(world_size: int, num_tokens: int) -> int:
    # Measured on 8xH200: 5-CTA clusters are faster at TP8 up to 16 rows; one CTA
    # per row is faster at TP2 and TP4.
    return 5 if world_size == 8 and num_tokens <= 16 else 1


def all_reduce_sandwich_norm(
    comm: Communicator,
    partial: torch.Tensor,
    residual: torch.Tensor,
    post_norm_weight: torch.Tensor,
    next_norm_weight: Optional[torch.Tensor],
    eps: float,
    scales: tuple[float, float],
    *,
    routed_scale: float = 1.0,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``(combined, normalized)`` of Beam's TP boundary; a ``[T, K, H]`` partial holds
    the routing-weighted expert rows, summed and scaled by ``routed_scale`` first."""
    cluster_size = get_sandwich_cluster_size(comm.world_size, partial.shape[0])
    args = (comm, partial, residual, post_norm_weight, next_norm_weight, eps, scales)
    if partial.ndim == 3:
        return fused_all_reduce.push.topk_sum_all_reduce_sandwich_norm(
            *args, routed_scale=routed_scale, cluster_size=cluster_size
        )
    return fused_all_reduce.push.all_reduce_sandwich_norm(
        *args, cluster_size=cluster_size
    )
