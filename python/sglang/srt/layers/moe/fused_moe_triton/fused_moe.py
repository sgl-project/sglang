# NOTE: this file will be separated into sglang/srt/layers/moe/moe_runner/triton_utils.py
# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/model_executor/layers/fused_moe/fused_moe.py

"""Fused MoE kernel."""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import triton
import triton.language as tl
from sgl_kernel import machete_mm
from sgl_kernel.scalar_type import ScalarType, scalar_types

from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.utils import (
    cpu_has_amx_support,
    direct_register_custom_op,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
)

from .fused_moe_triton_config import get_config_dtype_str, try_get_optimal_moe_config
from .fused_moe_triton_kernels import invoke_fused_moe_kernel, moe_sum_reduce_triton
from .moe_align_block_size import moe_align_block_size

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import StandardTopKOutput

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _is_cuda:
    from sgl_kernel import gelu_and_mul, silu_and_mul
elif _is_cpu and _is_cpu_amx_available:
    pass
elif _is_hip:
    from sgl_kernel import gelu_and_mul, silu_and_mul

    if _use_aiter:
        try:
            from aiter import moe_sum
        except ImportError:
            raise ImportError("aiter is required when SGLANG_USE_AITER is set to True")
    else:
        from vllm import _custom_ops as vllm_ops

padding_size = 128 if bool(int(os.getenv("SGLANG_MOE_PADDING", "0"))) else 0


def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
) -> None:
    fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        b1,
        b2,
        True,
        activation,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        False,
        routed_scaling_factor,
        gemm1_alpha,
        gemm1_limit,
    )


def inplace_fused_experts_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
) -> None:
    pass


direct_register_custom_op(
    op_name="inplace_fused_experts",
    op_func=inplace_fused_experts,
    mutates_args=["hidden_states"],
    fake_impl=inplace_fused_experts_fake,
)


def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
) -> torch.Tensor:
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        b1,
        b2,
        False,
        activation,
        apply_router_weight_on_input,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        a1_scale,
        a2_scale,
        block_shape,
        no_combine=no_combine,
        routed_scaling_factor=routed_scaling_factor,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
    )


def outplace_fused_experts_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="outplace_fused_experts",
    op_func=outplace_fused_experts,
    mutates_args=[],
    fake_impl=outplace_fused_experts_fake,
)


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output: StandardTopKOutput,
    moe_runner_config: MoeRunnerConfig,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
):
    topk_weights, topk_ids, _ = topk_output
    if moe_runner_config.inplace:
        assert not moe_runner_config.no_combine, "no combine + inplace makes no sense"
        torch.ops.sglang.inplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            b1,
            b2,
            moe_runner_config.activation,
            moe_runner_config.apply_router_weight_on_input,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1_scale,
            a2_scale,
            block_shape,
            moe_runner_config.routed_scaling_factor,
            moe_runner_config.gemm1_alpha,
            moe_runner_config.gemm1_clamp_limit,
        )
        return hidden_states
    else:
        return torch.ops.sglang.outplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            b1,
            b2,
            moe_runner_config.activation,
            moe_runner_config.apply_router_weight_on_input,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1_scale,
            a2_scale,
            block_shape,
            no_combine=moe_runner_config.no_combine,
            routed_scaling_factor=moe_runner_config.routed_scaling_factor,
            gemm1_alpha=moe_runner_config.gemm1_alpha,
            gemm1_limit=moe_runner_config.gemm1_clamp_limit,
        )


# _moe_sum_reduce_kernel kernel modified from https://github.com/ModelTC/lightllm/blob/main/lightllm/common/fused_moe/moe_sum_reduce.py
@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    token_start = token_block_id * BLOCK_M
    token_end = min((token_block_id + 1) * BLOCK_M, token_num)

    dim_start = dim_block_id * BLOCK_DIM
    dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)

    offs_dim = dim_start + tl.arange(0, BLOCK_DIM)

    for token_index in range(token_start, token_end):
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        input_t_ptr = input_ptr + token_index * input_stride_0 + offs_dim
        for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
            tmp = tl.load(
                input_t_ptr + i * input_stride_1, mask=offs_dim < dim_end, other=0.0
            )
            accumulator += tmp
        accumulator = accumulator * routed_scaling_factor
        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
        tl.store(
            store_t_ptr,
            accumulator.to(input_ptr.dtype.element_ty),
            mask=offs_dim < dim_end,
        )


@triton.jit
def _moe_sum_reduce_with_reorder_topk_weight_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    output_ptr,
    output_stride_0,
    output_stride_1,
    reorder_token_pos_ptr,
    topk_weight_ptr,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    token_start = token_block_id * BLOCK_M
    token_end = min((token_block_id + 1) * BLOCK_M, token_num)

    dim_start = dim_block_id * BLOCK_DIM
    dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)

    offs_dim = dim_start + tl.arange(0, BLOCK_DIM)

    for token_index in range(token_start, token_end):
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        input_t_ptr = input_ptr + offs_dim
        token_offset_0 = token_index * topk_num
        for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
            token_offset = token_offset_0 + i
            i_real = tl.load(reorder_token_pos_ptr + token_offset)
            token_weight = tl.load(topk_weight_ptr + token_offset)

            tmp = tl.load(
                input_t_ptr + i_real * input_stride_0,
                mask=offs_dim < dim_end,
                other=0.0,
            )
            accumulator += tmp * token_weight
        accumulator = accumulator * routed_scaling_factor
        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
        tl.store(
            store_t_ptr,
            accumulator.to(input_ptr.dtype.element_ty),
            mask=offs_dim < dim_end,
        )


def moe_sum_reduce_triton(
    input: torch.Tensor,
    output: torch.Tensor,
    routed_scaling_factor: float,
    reorder_token_pos: torch.Tensor = None,
    topk_weights: torch.Tensor = None,
):
    assert input.is_contiguous()
    assert output.is_contiguous()

    if reorder_token_pos is None and topk_weights is None:
        token_num, topk_num, hidden_dim = input.shape
        assert output.shape[0] == token_num and output.shape[1] == hidden_dim

        BLOCK_M = 1
        BLOCK_DIM = 2048
        NUM_STAGE = 1
        num_warps = 8

        grid = (
            triton.cdiv(token_num, BLOCK_M),
            triton.cdiv(hidden_dim, BLOCK_DIM),
        )

        _moe_sum_reduce_kernel[grid](
            input,
            *input.stride(),
            output,
            *output.stride(),
            token_num=token_num,
            topk_num=topk_num,
            hidden_dim=hidden_dim,
            routed_scaling_factor=routed_scaling_factor,
            BLOCK_M=BLOCK_M,
            BLOCK_DIM=BLOCK_DIM,
            NUM_STAGE=NUM_STAGE,
            num_warps=num_warps,
        )
    else:
        token_num, topk_num, hidden_dim = (
            reorder_token_pos.shape[0],
            reorder_token_pos.shape[1],
            input.shape[1],
        )
        BLOCK_M = 1
        BLOCK_DIM = 2048
        NUM_STAGE = 1
        num_warps = 8

        grid = (
            triton.cdiv(token_num, BLOCK_M),
            triton.cdiv(hidden_dim, BLOCK_DIM),
        )

        _moe_sum_reduce_with_reorder_topk_weight_kernel[grid](
            input,
            *input.stride(),
            output,
            *output.stride(),
            reorder_token_pos_ptr=reorder_token_pos,
            topk_weight_ptr=topk_weights,
            token_num=token_num,
            topk_num=topk_num,
            hidden_dim=hidden_dim,
            routed_scaling_factor=routed_scaling_factor,
            BLOCK_M=BLOCK_M,
            BLOCK_DIM=BLOCK_DIM,
            NUM_STAGE=NUM_STAGE,
            num_warps=num_warps,
        )


@torch.compile
def moe_sum_reduce_torch_compile(x, out, routed_scaling_factor):
    torch.sum(x, dim=1, out=out)
    out.mul_(routed_scaling_factor)


@torch.compile
def swiglu_with_alpha_and_limit(x, gemm1_alpha, gemm1_limit):
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=gemm1_limit)
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    return gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1)


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
):
    padded_size = padding_size
    if not (use_fp8_w8a8 or use_int8_w8a8) or block_shape is not None or _use_aiter:
        padded_size = 0

    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.shape[1] // 2 == w1.shape[2], "Hidden size mismatch"
    else:
        assert (
            hidden_states.shape[1] == w1.shape[2] - padded_size
        ), f"Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = 64 * 1024
    M = min(num_tokens, CHUNK_SIZE)
    config_dtype = get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        dtype=hidden_states.dtype,
    )

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        topk_ids.shape[1],
        config_dtype,
        block_shape=block_shape,
    )

    config = get_config_func(M)

    cache = torch.empty(
        M * topk_ids.shape[1] * max(N, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache[: M * topk_ids.shape[1] * N].view(
        (M, topk_ids.shape[1], N),
    )
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = cache[: M * topk_ids.shape[1] * w2.shape[1]].view(
        (M, topk_ids.shape[1], w2.shape[1]),
    )

    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, topk_ids.shape[1], w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[
                : tokens_in_chunk * topk_ids.shape[1]
            ]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            curr_topk_ids, config["BLOCK_SIZE_M"], E
        )

        invoke_fused_moe_kernel(
            curr_hidden_states,
            w1,
            b1,
            intermediate_cache1,
            a1_scale,
            w1_scale,
            w1_zp,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            topk_ids.shape[1],
            config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
        )
        if activation == "silu":
            if gemm1_alpha is not None:
                assert gemm1_limit is not None
                intermediate_cache2 = swiglu_with_alpha_and_limit(
                    intermediate_cache1.view(-1, N),
                    gemm1_alpha,
                    gemm1_limit,
                )
            elif _is_cuda or _is_hip:
                silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
            else:
                vllm_ops.silu_and_mul(
                    intermediate_cache2, intermediate_cache1.view(-1, N)
                )
        elif activation == "gelu":
            assert gemm1_alpha is None, "gemm1_alpha is not supported for gelu"
            assert gemm1_limit is None, "gemm1_limit is not supported for gelu"
            if _is_cuda or _is_hip:
                gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
            else:
                vllm_ops.gelu_and_mul(
                    intermediate_cache2, intermediate_cache1.view(-1, N)
                )
        else:
            raise ValueError(f"Unsupported activation: {activation=}")

        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2,
            b2,
            (
                intermediate_cache3
                if not no_combine and topk_ids.shape[1] != 1
                else out_hidden_states[begin_chunk_idx:end_chunk_idx].unsqueeze(0)
            ),
            a2_scale,
            w2_scale,
            w2_zp,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
        )

        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        if no_combine:
            pass
        elif _is_cuda:
            if topk_ids.shape[1] == 1 and routed_scaling_factor == 1.0:
                pass  # we write directly into out_hidden_states
            elif topk_ids.shape[1] == 2 and routed_scaling_factor == 1.0:
                torch.add(
                    intermediate_cache3[:, 0],
                    intermediate_cache3[:, 1],
                    out=out_hidden_states[begin_chunk_idx:end_chunk_idx],
                ).squeeze(dim=1)
            else:
                # According to micro benchmark results, torch.compile can get better performance for small token.
                if tokens_in_chunk <= 32:
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
                else:
                    moe_sum_reduce_triton(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
        elif _is_hip:
            if _use_aiter:
                moe_sum(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx],
                )
            else:
                # According to micro benchmark results, torch.compile can get better performance for small token.
                if tokens_in_chunk <= 32:
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
                else:
                    moe_sum_reduce_triton(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
        else:
            vllm_ops.moe_sum(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states[begin_chunk_idx:end_chunk_idx],
            )

    return out_hidden_states


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output: StandardTopKOutput,
    moe_runner_config: MoeRunnerConfig = MoeRunnerConfig(),
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - topk_output (StandardTopKOutput): The top-k output of the experts.
    - moe_runner_config (MoeRunnerConfig): The configuration for the MoE runner.
    - b1 (Optional[torch.Tensor]): Optional bias for w1.
    - b2 (Optional[torch.Tensor]): Optional bias for w2.
    - use_fp8_w8a8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a8 (bool): If True, use int8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a16 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int4_w4a16 (bool): If True, use matmul of int4 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.
    - a1_scale (Optional[torch.Tensor]): Optional scale to be used for
        a1.
    - a2_scale (Optional[torch.Tensor]): Optional scale to be used for
        a2.
    - block_shape: (Optional[List[int]]): Optional block size for block-wise
        quantization.
    - gemm1_alpha (Optional[float]): Optional gemm1_alpha for the activation
        function.
    - gemm1_limit (Optional[float]): Optional gemm1_limit for the swiglu activation
        function.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """

    return fused_experts(
        hidden_states,
        w1,
        w2,
        topk_output,
        moe_runner_config=moe_runner_config,
        b1=b1,
        b2=b2,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
    )


#######################################################################################


@triton.jit
def fill_moe_hidden_fp8_kernel(
    x_fp8_ptr,
    reorder_token_pos_ptr,
    x_ptr,
    sorted_ids_ptr,
    sorted_ids_len,
    BLOCK_M: tl.constexpr,
    invalid_id: tl.constexpr,
    num_topk: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    if tl.load(sorted_ids_ptr + pid * BLOCK_M) == invalid_id:
        return
    token_id = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_token = token_id < tl.load(sorted_ids_len)

    ids = tl.load(sorted_ids_ptr + token_id, mask=mask_token, other=invalid_id)
    ids = ids.to(tl.int64)

    valid_mask = ids < invalid_id

    for k_idx in range(0, hidden_dim, BLOCK_K):
        k_range = k_idx + tl.arange(0, BLOCK_K)
        k_mask = k_range < hidden_dim

        a_ptr = x_ptr + (ids[:, None] // num_topk * hidden_dim + k_range[None, :])

        a = tl.load(a_ptr, mask=valid_mask[:, None] & k_mask[None, :], other=0.0)

        row_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        dst_ptr = x_fp8_ptr + (row_idx[:, None] * hidden_dim + k_range[None, :])
        # dst_ptr = x_fp8_ptr + (tl.arange(0, BLOCK_M)[:, None] * hidden_dim + k_range[None, :])

        tl.store(dst_ptr, a, mask=mask_token[:, None] & k_mask[None, :])

    # for reorder_token_pos
    # ids: [0,1,invalid,invalid,...,invalid]
    reorder_pos = reorder_token_pos_ptr + ids
    tl.store(reorder_pos, token_id, mask=valid_mask)


def fill_moe_hidden_fp8(
    x_fp8,
    reorder_token_pos,
    x,
    sorted_ids,
    valid_sorted_ids_len,
    BLOCK_M,
    invalid_id,
    num_topk,
):
    BLOCK_M = 1
    hidden_dim = x.shape[1]
    # sorted_ids_len = sorted_ids.numel()  # 使用元素总数
    sorted_ids_len = valid_sorted_ids_len

    grid = ((sorted_ids.shape[0] + BLOCK_M - 1) // BLOCK_M,)

    BLOCK_K = 512
    fill_moe_hidden_fp8_kernel[grid](
        x_fp8,
        reorder_token_pos,
        x,
        sorted_ids,
        sorted_ids_len,
        BLOCK_M,
        invalid_id,
        num_topk,
        hidden_dim,
        BLOCK_K,
    )


@triton.jit
def silu_and_mul_with_mask_kernel(
    x_ptr,
    mask_ptr,
    output_ptr,
    row_num,
    dim,
    stride_x_row,
    stride_x_col,
    stride_mask,
    stride_output_row,
    stride_output_col,
    BLOCK_SIZE_DIM: tl.constexpr,
    INVALID_ID: tl.constexpr,
):
    row_idx = tl.program_id(0)

    if row_idx >= row_num:
        return

    mask_val = tl.load(mask_ptr + row_idx * stride_mask)

    if mask_val == INVALID_ID:
        return

    for dim_offset in range(0, dim, BLOCK_SIZE_DIM):
        col_idx = dim_offset + tl.arange(0, BLOCK_SIZE_DIM)
        col_mask = col_idx < dim

        offs_x1 = row_idx * stride_x_row + col_idx * stride_x_col

        x1_ptr = x_ptr + offs_x1
        x1 = tl.load(x1_ptr, mask=col_mask, other=0.0).to(tl.float32)

        x2_ptr = x1_ptr + dim
        x2 = tl.load(x2_ptr, mask=col_mask, other=0.0).to(tl.float32)

        silu_val = x1 * tl.sigmoid(x1)  # SiLU(x) = x * sigmoid(x)
        result_val = silu_val.to(x2.dtype) * x2

        offs_out = row_idx * stride_output_row + col_idx * stride_output_col
        output_ptr_pos = output_ptr + offs_out
        tl.store(output_ptr_pos, result_val, mask=col_mask)


def silu_and_mul_with_mask(
    output: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    invalid_id: int,
    BLOCK_SIZE_DIM: int = 256,
) -> torch.Tensor:
    """
    compute SiLU(x[:, :dim]) * x[:, dim:] with mask

    Args:
        x: [row_num, 2*dim]
        mask: [row_num]
        invalid_id: invalid id in mask
        BLOCK_SIZE_DIM: BLOCK_M size

    Returns: [row_num, dim]
    """
    assert x.dim() == 2, "x.dim() is not 2"
    row_num, two_dim = x.shape
    assert two_dim % 2 == 0, "x.dim[-1] % 2 != 0"
    dim = two_dim // 2

    grid = (row_num,)

    stride_x_row, stride_x_col = x.stride()
    stride_mask = mask.stride()[0] if mask.dim() > 1 else 0
    stride_output_row, stride_output_col = output.stride()

    BLOCK_SIZE_DIM = 256
    silu_and_mul_with_mask_kernel[grid](
        x,
        mask,
        output,
        row_num,
        dim,
        stride_x_row,
        stride_x_col,
        stride_mask,
        stride_output_row,
        stride_output_col,
        BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
        INVALID_ID=invalid_id,
    )

    return output


@dataclass
class MacheteTypes:
    act_type: torch.dtype
    weight_type: ScalarType
    output_type: Optional[torch.dtype]
    group_scale_type: Optional[torch.dtype]
    group_zero_type: Optional[torch.dtype]
    channel_scale_type: Optional[torch.dtype]
    token_scale_type: Optional[torch.dtype]


def fused_experts_machete_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    no_combine: bool = False,
    has_zp: bool = True,
    routed_scaling_factor: Optional[float] = None,
    schedule: str = None,
):
    a_type = hidden_states.dtype
    types = MacheteTypes(
        act_type=torch.float8_e4m3fn,
        weight_type=scalar_types.uint4b8,
        output_type=a_type,
        group_scale_type=a_type,
        group_zero_type=None,
        channel_scale_type=None,
        token_scale_type=None,
    )

    num_tokens, _ = hidden_states.shape
    E, _, N = w1.shape  # N = 2 * moe_intermediate_size
    group_size = 128

    finfo = torch.finfo(torch.float8_e4m3fn)

    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, topk_ids.shape[1], w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    # TODO: get_best_scheduler by tuning
    schedule_gateup = "256x16_2x1x1_TmaMI__TmaCoop_PersistentScheduler"
    schedule_down = schedule_gateup
    if schedule:
        schedule_gateup = schedule
        schedule_down = schedule
    BLOCK_SIZE_M = int(schedule_gateup.split("_")[0].split("x")[1])
    BLOCK_SIZE_M2 = int(schedule_down.split("_")[0].split("x")[1])
    if BLOCK_SIZE_M >= BLOCK_SIZE_M2:
        block_ratio_1 = 1
        block_ratio_2 = BLOCK_SIZE_M // BLOCK_SIZE_M2
    else:
        block_ratio_1 = BLOCK_SIZE_M2 // BLOCK_SIZE_M
        block_ratio_2 = 1

    def div_ceil(a, b):
        return (a + b - 1) // b * b

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, BLOCK_SIZE_M, E
    )
    valid_sorted_token_len = num_tokens_post_padded
    # hidden_states_fp8 = torch.empty([div_ceil(valid_sorted_token_len, BLOCK_SIZE_M), hidden_states.shape[1]], dtype=torch.float8_e4m3fn, device=hidden_states.device)
    hidden_states_fp8 = torch.empty(
        [div_ceil(sorted_token_ids.shape[0], BLOCK_SIZE_M), hidden_states.shape[1]],
        dtype=torch.float8_e4m3fn,
        device=hidden_states.device,
    )
    invalid_id = topk_ids.numel()
    num_topk = topk_ids.shape[1]
    reorder_token_pos = torch.empty(
        [num_tokens, num_topk], dtype=torch.int32, device=hidden_states.device
    )

    fill_moe_hidden_fp8(
        hidden_states_fp8,
        reorder_token_pos,
        hidden_states,
        sorted_token_ids,
        valid_sorted_token_len,
        BLOCK_SIZE_M,
        invalid_id,
        num_topk,
    )
    group_layout = expert_ids

    intermediate_cache1 = machete_mm(
        a=hidden_states_fp8,
        b_q=w1,
        b_type=types.weight_type,
        b_group_scales=w1_scale,
        b_group_zeros=w1_zp,
        b_group_size=group_size,
        b_channel_scales=None,
        a_token_scales=None,
        out_type=types.output_type,
        schedule=schedule_gateup,
        group_layout=group_layout,
        group_stride=block_ratio_1,
        valid_len=valid_sorted_token_len,
    )

    # mul_silu
    is_cache2_cast_fp8 = True
    dtype_cache2 = (
        intermediate_cache1.dtype if not is_cache2_cast_fp8 else torch.float8_e4m3fn
    )
    intermediate_cache2 = torch.empty(
        [intermediate_cache1.shape[0], N // 2],
        dtype=dtype_cache2,
        device=intermediate_cache1.device,
    )

    silu_and_mul_with_mask(
        intermediate_cache2,
        intermediate_cache1.view(-1, N),
        sorted_token_ids,
        invalid_id,
    )

    # down
    if not is_cache2_cast_fp8:
        intermediate_cache2_fp8 = torch.clamp(
            intermediate_cache2, finfo.min, finfo.max
        ).to(torch.float8_e4m3fn)
    else:
        intermediate_cache2_fp8 = intermediate_cache2
    intermediate_cache3 = machete_mm(
        a=intermediate_cache2_fp8,
        b_q=w2,
        b_type=types.weight_type,
        b_group_scales=w2_scale,
        b_group_zeros=w2_zp,
        b_group_size=group_size,
        b_channel_scales=None,
        a_token_scales=None,
        out_type=types.output_type,
        schedule=schedule_down,
        group_layout=group_layout,
        group_stride=block_ratio_2,
        valid_len=valid_sorted_token_len,
    )

    # reduce
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    # moe_reduce
    moe_sum_reduce_triton(
        intermediate_cache3.view(*intermediate_cache3.shape),
        out_hidden_states,
        routed_scaling_factor,
        reorder_token_pos=reorder_token_pos,
        topk_weights=topk_weights,
    )
    return out_hidden_states
