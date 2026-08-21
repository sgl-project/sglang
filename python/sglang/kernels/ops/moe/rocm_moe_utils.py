# Adapted from https://github.com/vllm-project/vllm/blob/v0.9.1rc2/vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import IntEnum
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.utils import get_bool_env_var, is_hip
from sglang.srt.utils.custom_op import register_custom_op

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip


class ActivationMethod(IntEnum):
    # This allows interfacing with AITER ActivationType enum
    # without importing the ActivationType enum from AITER globally.
    SILU = 0
    GELU = 1


# NOTE: for non _use_aiter case, use lazy registration to avoid overhead
# (registration may not be trigger actually, since it will not be called)
@register_custom_op(out_shape="hidden_states", eager=_use_aiter)
def rocm_aiter_asm_moe_tkw1(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    fc1_scale: Optional[torch.Tensor] = None,
    fc2_scale: Optional[torch.Tensor] = None,
    fc1_smooth_scale: Optional[torch.Tensor] = None,
    fc2_smooth_scale: Optional[torch.Tensor] = None,
    a16: bool = False,
    per_tensor_quant_scale: Optional[torch.Tensor] = None,
    expert_mask: Optional[torch.Tensor] = None,
    activation_method: int = ActivationMethod.SILU.value,
) -> torch.Tensor:

    from aiter import ActivationType
    from aiter.fused_moe_bf16_asm import asm_moe_tkw1

    activation = ActivationType(activation_method)

    return asm_moe_tkw1(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        fc1_scale=fc1_scale,
        fc2_scale=fc2_scale,
        fc1_smooth_scale=fc1_smooth_scale,
        fc2_smooth_scale=fc2_smooth_scale,
        a16=a16,
        per_tensor_quant_scale=per_tensor_quant_scale,
        expert_mask=expert_mask,
        activation=activation,
    )


def rocm_fused_experts_tkw1(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:

    activation_method = (
        ActivationMethod.SILU if activation == "silu" else ActivationMethod.GELU
    )
    # All AITER Fused MoE kernels are expecting the following datatypes
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    # w8a8 per-channel quantization
    if per_channel_quant and apply_router_weight_on_input and use_fp8_w8a8:
        # AITER tkw1 kernel for FP8 models with `apply_router_weight_on_input`
        # This applies topk_weights on the GEMM output of the first FC layer
        #  rather than the second FC.
        assert (
            topk_weights.dim() == 2
        ), "`topk_weights` should be in shape (num_tokens, topk)"
        assert topk_weights.shape[-1] == 1, (
            "Only support topk=1 when" " `apply_router_weight_on_input` is True"
        )

        return rocm_aiter_asm_moe_tkw1(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            fc1_scale=w1_scale,
            fc2_scale=w2_scale,
            fc1_smooth_scale=None,
            fc2_smooth_scale=None,
            a16=False,
            per_tensor_quant_scale=None,
            expert_mask=None,
            activation_method=activation_method,
        )
    else:
        assert False, "This should not be called."


@triton.jit
def upscale_kernel(
    A_ptr,  # *fp16 / *fp32
    scale_ptr,  # *fp16 / *fp32
    Out_ptr,  # *fp16 / *fp32
    M,
    N,
    recv_token_num,
    stride_am,
    stride_an,
    stride_sm,
    stride_sn,
    stride_om,
    stride_on,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # row id
    pid_n = tl.program_id(1)  # block id along N

    recv_token_num_val = tl.load(recv_token_num)

    if pid_m >= recv_token_num_val:
        return

    # column offsets
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N

    # A[m, n]
    a_ptrs = A_ptr + pid_m * stride_am + offs_n * stride_an
    a = tl.load(a_ptrs, mask=mask, other=0.0)

    # scale index: n // 128
    scale_idx = offs_n // 128
    s_ptrs = scale_ptr + pid_m * stride_sm + scale_idx * stride_sn
    s = tl.load(s_ptrs, mask=mask, other=1.0)

    out = a * s

    out_ptrs = Out_ptr + pid_m * stride_om + offs_n * stride_on
    tl.store(out_ptrs, out, mask=mask)


def upscale(hidden_state, hidden_state_scale, recv_token_num, output_dtype):
    M, N = hidden_state.shape

    Out = torch.empty_like(hidden_state, dtype=output_dtype)

    BLOCK_N = 256

    grid = (M, triton.cdiv(N, BLOCK_N))

    upscale_kernel[grid](
        hidden_state,
        hidden_state_scale,
        Out,
        M,
        N,
        recv_token_num,
        hidden_state.stride(0),
        hidden_state.stride(1),
        hidden_state_scale.stride(0),
        hidden_state_scale.stride(1),
        Out.stride(0),
        Out.stride(1),
        BLOCK_N=BLOCK_N,
    )

    return Out


@triton.jit
def upscale_fp4x2_block32_kernel(
    A_u8_ptr,  # *uint8  (view from float4_e2m1fn_x2)
    S_u8_ptr,  # *uint8  (view from float8_e8m0fnu), shape (M, N_fp4/32)
    Out_ptr,  # *fp16/fp32/bf16, shape (M, N_fp4)
    N_FP4: tl.constexpr,
    recv_token_num,
    stride_am,
    stride_an,  # A strides (in uint8 elements) for (M, packed_N)
    stride_sm,
    stride_sn,  # S strides (in uint8 elements) for (M, N_FP4/32)
    stride_om,
    stride_on,  # Out strides (in output elements) for (M, N_FP4)
    BLOCK_N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,  # tl.float16 / tl.float32 / tl.bfloat16
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    recv_token_num_val = tl.load(recv_token_num)
    if pid_m >= recv_token_num_val:
        return

    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N_FP4

    # --------------------------
    # Load packed fp4x2 byte
    # --------------------------
    byte_idx = offs >> 1  # offs // 2
    is_hi = (offs & 1) != 0  # select high nibble?

    a_ptrs = A_u8_ptr + pid_m * stride_am + byte_idx * stride_an
    a_byte = tl.load(a_ptrs, mask=mask, other=0).to(tl.int32)

    lo = a_byte & 0xF
    hi = (a_byte >> 4) & 0xF
    code = tl.where(is_hi, hi, lo).to(tl.int32)  # 0..15

    # --------------------------
    # Decode float4_e2m1fn
    # layout: [sign|exp(2)|mant(1)]
    # bias=1, finite-only
    # --------------------------
    sign = (code >> 3) & 0x1
    exp = (code >> 1) & 0x3
    mant = code & 0x1

    mant_f = mant.to(tl.float32) * 0.5
    is_sub = exp == 0

    # normal: 2^(exp-bias) * (1 + mant/2), bias=1
    e_norm = (exp - 1).to(tl.float32)
    val_norm = tl.exp2(e_norm) * (1.0 + mant_f)

    # subnorm/zero: mant/2 * 2^(1-bias) = mant/2
    val_sub = mant_f

    val = tl.where(is_sub, val_sub, val_norm)
    val = tl.where(sign != 0, -val, val)  # apply sign

    # --------------------------
    # Per-token block32 scale: scale_idx = offs // 32
    # scale dtype: float8_e8m0fnu stored in uint8
    # decode: e==0 -> 0
    #         e in [1..254] -> 2^(e-127)
    #         e==255 -> clamp to 254
    # --------------------------
    scale_idx = offs >> 5  # offs // 32

    s_ptrs = S_u8_ptr + pid_m * stride_sm + scale_idx * stride_sn
    e = tl.load(s_ptrs, mask=mask, other=0).to(tl.int32)

    e = tl.minimum(e, 254)  # clamp 255->254
    is_zero = e == 0
    exp_s = (e - 127).to(tl.float32)
    s = tl.exp2(exp_s)
    s = tl.where(is_zero, 0.0, s)

    out = (val * s).to(OUT_DTYPE)

    out_ptrs = Out_ptr + pid_m * stride_om + offs * stride_on
    tl.store(out_ptrs, out, mask=mask)


def upscale_mxfp4(hidden_state, hidden_state_scale, recv_token_num, output_dtype):
    """
    hidden_state: (M, packed_N) torch.float4_e2m1fn_x2
    hidden_state_scale: (M, packed_N*2/32) = (M, N_fp4/32) torch.float8_e8m0fnu
    output: (M, N_fp4) output_dtype
    """
    assert hidden_state.dtype == torch.float4_e2m1fn_x2, hidden_state.dtype
    assert hidden_state_scale.dtype == torch.float8_e8m0fnu, hidden_state_scale.dtype
    assert hidden_state.is_contiguous() or True  # stride-based load OK

    M, packed_N = hidden_state.shape
    N_fp4 = packed_N * 2

    # scale second dim must be N_fp4/32
    assert hidden_state_scale.shape[0] == M
    assert hidden_state_scale.shape[1] == (N_fp4 // 32), (
        hidden_state_scale.shape,
        N_fp4,
    )

    # Triton doesn't (reliably) accept torch.float4/float8 pointers directly.
    # Use raw uint8 views.
    A_u8 = hidden_state.view(torch.uint8)
    S_u8 = hidden_state_scale.view(torch.uint8)

    Out = torch.empty((M, N_fp4), dtype=output_dtype, device=hidden_state.device)

    BLOCK_N = 256
    grid = (M, triton.cdiv(N_fp4, BLOCK_N))

    OUT_TL = (
        tl.float16
        if output_dtype == torch.float16
        else tl.bfloat16 if output_dtype == torch.bfloat16 else tl.float32
    )

    upscale_fp4x2_block32_kernel[grid](
        A_u8,
        S_u8,
        Out,
        N_FP4=N_fp4,
        recv_token_num=recv_token_num,
        stride_am=A_u8.stride(0),
        stride_an=A_u8.stride(1),
        stride_sm=S_u8.stride(0),
        stride_sn=S_u8.stride(1),
        stride_om=Out.stride(0),
        stride_on=Out.stride(1),
        BLOCK_N=BLOCK_N,
        OUT_DTYPE=OUT_TL,
        num_warps=4,
    )
    return Out
