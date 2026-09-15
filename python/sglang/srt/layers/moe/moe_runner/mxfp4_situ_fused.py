# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Derived from aiter (commit 9127c94a):
#   aiter/ops/triton/moe/moe_op_mxfp4_silu_fused.py               (wrapper)
#   aiter/ops/triton/_triton_kernels/moe/moe_op_mxfp4_silu_fused.py (kernel)
#
# CHANGE vs upstream: the fused epilogue gains a compile-time activation switch.
#
#   ACT_SITU = False  ->  SwiGLU, byte-identical to upstream:
#                             out = silu(gate) * up
#   ACT_SITU = True   ->  Kimi-K3's SituGLU (sglang SituAndMul, beta=4.0,
#                         linear_beta=25.0):
#                             gate_out = beta        * tanh(gate/beta) * sigmoid(gate)
#                             up_out   = linear_beta * tanh(up/linear_beta)
#                             out      = gate_out * up_out
#
# SiTU *is* SwiGLU with both branches soft-clipped by tanh; as beta -> inf it
# degenerates to SwiGLU exactly.  Implemented with tanh(x) = 2*sigmoid(2x) - 1
# so it reuses the same exp2-based sigmoid the SwiGLU epilogue already needs:
# 3 exp2 + 3 reciprocal per output element instead of 1 + 1, on an epilogue
# amortised over K=3584 -- i.e. off the critical path.  The GEMM loop, the
# pointer arithmetic, the N-interleaving trick and the store are untouched.

from typing import Any, Dict

import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.moe.moe_op_mxfp4_silu_fused import (
    get_scaled_dot_format_string,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.moe_common import _write_zeros_to_output
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils.types import torch_to_triton_dtype

LOG2E: tl.constexpr = 1.44269504089


@triton.jit
def _sigmoid_exp2(x):
    """1/(1+e^-x) via exp2 -- same transform aiter's _silu_exp2 uses."""
    return 1.0 / (1.0 + tl.exp2(-(x * 1.44269504089)))


@triton.jit
def _silu_exp2(x):
    # verbatim from aiter/ops/triton/_triton_kernels/activation.py
    return x / (1.0 + tl.exp2(-(x * 1.44269504089)))


@triton.jit
def _situ_and_mul(gate, up, BETA: tl.constexpr, LINEAR_BETA: tl.constexpr):
    """Kimi-K3 SituGLU.

    gate_out = BETA * tanh(gate/BETA) * sigmoid(gate)
    up_out   = LINEAR_BETA * tanh(up/LINEAR_BETA)

    tanh(x) = 2*sigmoid(2x) - 1, and sigmoid is the exp2 form above, so the
    whole thing is 3 exp2 + 3 reciprocal + 5 multiplies.  The two 2/BETA
    factors are folded into constants at compile time.
    """
    TWO_OVER_BETA: tl.constexpr = 2.0 / BETA
    TWO_OVER_LBETA: tl.constexpr = 2.0 / LINEAR_BETA
    # tanh(gate/BETA) = 2*sigmoid(2*gate/BETA) - 1
    tanh_g = 2.0 * _sigmoid_exp2(gate * TWO_OVER_BETA) - 1.0
    gate_out = (BETA * tanh_g) * _sigmoid_exp2(gate)
    # tanh(up/LINEAR_BETA)
    tanh_u = 2.0 * _sigmoid_exp2(up * TWO_OVER_LBETA) - 1.0
    up_out = LINEAR_BETA * tanh_u
    return gate_out * up_out


_fused_moe_kernel_mxfp4_act_repr = make_kernel_repr(
    "_fused_moe_kernel_mxfp4_act",
    [
        "A_DTYPE_FORMAT",
        "B_DTYPE_FORMAT",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "EVEN_K",
        "MUL_ROUTED_WEIGHT",
        "top_k",
        "compute_type",
        "SWIZZLE_MX_A",
        "SWIZZLE_MX_B",
        "ACT_SITU",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit(repr=_fused_moe_kernel_mxfp4_act_repr)
def _fused_moe_kernel_mxfp4_act(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    a_mx_scale_ptr,
    b_mx_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_amxm,
    stride_amxk,
    stride_bmxe,
    stride_bmxk,
    stride_bmxn,
    # Meta-parameters
    A_DTYPE_FORMAT: tl.constexpr,
    B_DTYPE_FORMAT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    SWIZZLE_MX_A: tl.constexpr,  # TODO add swizzle support
    SWIZZLE_MX_B: tl.constexpr,  # TODO add swizzle support
    ACT_SITU: tl.constexpr = False,
    SITU_BETA: tl.constexpr = 4.0,
    SITU_LINEAR_BETA: tl.constexpr = 25.0,
):
    """MoE GEMM with MXFP4 weights and a fused gated activation.

    Identical to aiter's `_fused_moe_kernel_mxfp4_silu` except for the last
    four lines of the epilogue (see ACT_SITU).
    """
    is_a_microscaled_format: tl.constexpr = a_mx_scale_ptr is not None
    is_b_microscaled_format: tl.constexpr = b_mx_scale_ptr is not None
    MX_PACK_DIVISOR: tl.constexpr = 32
    if is_a_microscaled_format:
        a_type: tl.constexpr = a_ptr.dtype.element_ty
        tl.static_assert(
            a_type == tl.uint8 or (a_type == tl.float8e4nv or a_type == tl.float8e5),
            "mx_weight_ptr must be 1 byte",
        )
        tl.static_assert(
            a_mx_scale_ptr.dtype.element_ty == tl.uint8, "a_mx_scale_ptr must be uint8"
        )
        tl.static_assert(
            BLOCK_SIZE_K % MX_PACK_DIVISOR == 0,
            "BLOCK_SIZE_K must be a multiple of MX_PACK_DIVISOR",
        )
    if is_b_microscaled_format:
        b_type: tl.constexpr = b_ptr.dtype.element_ty
        tl.static_assert(
            b_type == tl.uint8 or (b_type == tl.float8e4nv or b_type == tl.float8e5),
            "mx_weight_ptr must be 1 byte",
        )
        tl.static_assert(
            b_mx_scale_ptr.dtype.element_ty == tl.uint8, "b_mx_scale_ptr must be uint8"
        )
        tl.static_assert(
            BLOCK_SIZE_K % MX_PACK_DIVISOR == 0,
            "BLOCK_SIZE_K must be a multiple of MX_PACK_DIVISOR",
        )

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    NUM_XCDS: tl.constexpr = 8

    GRID_MN = num_pid_n * num_pid_m
    if pid < GRID_MN:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS)
    else:
        return  # rest of the tiles are dummy paddings
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    BLOCK_SIZE_HALF: tl.constexpr = BLOCK_SIZE_N // 2

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        # Write back zeros when the expert is not in this EP rank.
        #
        # BUGFIX vs upstream `moe_op_mxfp4_silu_fused.py`: this is a
        # gate/up-FUSED kernel, so C has N//2 columns and the real store below
        # writes BLOCK_SIZE_HALF of them at `pid_n * BLOCK_SIZE_HALF`.  Upstream
        # passes the *unfused* N / BLOCK_SIZE_N here, so on the EP sentinel path
        # it writes twice as many columns at twice the offset -- past the end of
        # every row of C, corrupting the following rows.  Nothing exercised it
        # until MXFP4 + EP>1 met on gfx942.  Only the zero-write is changed; the
        # arithmetic epilogue is untouched.
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N // 2,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_HALF,
            compute_type,
        )
        return

    i = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    # [0, 0, 1, 1, ..., BLOCK_SIZE_HALF - 1, BLOCK_SIZE_HALF - 1]
    i_floor = i // 2
    offs_half = ((pid_n * (BLOCK_SIZE_N // 2) + i_floor) % (N // 2)).to(tl.int64)
    # even lanes take a column from the FIRST half of N (gate), odd lanes from
    # the SECOND half (up) -- so the reshape+split below needs no permute.
    offs_b_n = ((offs_half + (i % 2) * (N // 2)) % N).to(tl.int64)

    # Load a_scale, b_scale
    a_scale = tl.load(a_scale_ptr)
    b_scale = tl.load(b_scale_ptr + off_expert)
    # Set offsets of B on dim N
    offs_b_n = tl.max_contiguous(
        tl.multiple_of(offs_b_n % N, BLOCK_SIZE_N), BLOCK_SIZE_N
    )
    # Load a_mx_scale
    if is_a_microscaled_format:
        A_PACK_DIVISOR: tl.constexpr = 2 if a_ptr.dtype.element_ty == tl.uint8 else 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_SIZE_K // A_PACK_DIVISOR
        MX_SCALE_BLOCK_K_A: tl.constexpr = BLOCK_SIZE_K // MX_PACK_DIVISOR

        if SWIZZLE_MX_A:
            tl.static_assert(BLOCK_SIZE_M % 128 == 0)
            tl.static_assert(MX_SCALE_BLOCK_K_A % 4 == 0)
            PACKED_MX_BLOCK_A: tl.constexpr = (MX_SCALE_BLOCK_K_A // 4) * 32 * 4 * 4
            offs_inner = tl.arange(0, PACKED_MX_BLOCK_A)
            offs_scale_m = (
                pid_m * (BLOCK_SIZE_M // 128) + tl.arange(0, BLOCK_SIZE_M // 128)
            ) % N
            offs_scale_m = tl.max_contiguous(
                tl.multiple_of(offs_scale_m, BLOCK_SIZE_M // 128), BLOCK_SIZE_M // 128
            )

            a_mx_scale_ptrs = (
                a_mx_scale_ptr
                + offs_scale_m.to(tl.int64)[:, None] * stride_amxm
                + offs_inner[None, :]
            )
        else:
            offs_scale_ak = tl.arange(0, MX_SCALE_BLOCK_K_A)
            offs_scale_m = offs_token
            a_mx_scale_ptrs = (
                a_mx_scale_ptr
                + offs_scale_ak.to(tl.int64)[None, :] * stride_amxk
                + offs_scale_m.to(tl.int64)[:, None] // top_k * stride_amxm
            )
    else:
        a_mx_scale_ptrs = None
        A_PACK_DIVISOR: tl.constexpr = 1
        MX_SCALE_BLOCK_K_A: tl.constexpr = 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_SIZE_K
    # Load b_mx_scale
    if is_b_microscaled_format:
        B_PACK_DIVISOR: tl.constexpr = 2 if b_ptr.dtype.element_ty == tl.uint8 else 1
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_SIZE_K // B_PACK_DIVISOR
        MX_SCALE_BLOCK_K_B: tl.constexpr = BLOCK_SIZE_K // MX_PACK_DIVISOR

        b_mx_scale_ptr += off_expert * stride_bmxe

        if SWIZZLE_MX_B:
            tl.static_assert(BLOCK_SIZE_N % 128 == 0)
            tl.static_assert(MX_SCALE_BLOCK_K_B % 4 == 0)
            PACKED_MX_BLOCK_B: tl.constexpr = (MX_SCALE_BLOCK_K_B // 4) * 32 * 4 * 4
            offs_inner = tl.arange(0, PACKED_MX_BLOCK_B)
            offs_scale_n = (
                pid_n * (BLOCK_SIZE_N // 128) + tl.arange(0, BLOCK_SIZE_N // 128)
            ) % N
            offs_scale_n = tl.max_contiguous(
                tl.multiple_of(offs_scale_n, BLOCK_SIZE_N // 128), BLOCK_SIZE_N // 128
            )

            b_mx_scale_ptrs = (
                b_mx_scale_ptr
                + offs_scale_n.to(tl.int64)[:, None]
                * PACKED_MX_BLOCK_B
                * (K // MX_SCALE_BLOCK_K_B // (MX_PACK_DIVISOR // B_PACK_DIVISOR))
                + offs_inner[None, :]
            )
        else:
            offs_scale_bk = tl.arange(0, MX_SCALE_BLOCK_K_B)
            offs_scale_n = offs_b_n
            b_mx_scale_ptrs = (
                b_mx_scale_ptr
                + offs_scale_bk.to(tl.int64)[None, :] * stride_bmxk
                + offs_scale_n.to(tl.int64)[:, None] * stride_bmxn
            )
    else:
        b_mx_scale_ptrs = None
        B_PACK_DIVISOR: tl.constexpr = 1
        MX_SCALE_BLOCK_K_B: tl.constexpr = 1
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_SIZE_K

    offs_a_k = tl.arange(0, PACKED_BLOCK_K_A)
    offs_b_k = tl.arange(0, PACKED_BLOCK_K_B)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_a_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_expert * stride_be
        + (offs_b_k[:, None] * stride_bk + offs_b_n[None, :] * stride_bn)
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, PACKED_BLOCK_K_A)):
        if EVEN_K:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None]
                & (offs_a_k[None, :] < (K - k * PACKED_BLOCK_K_A)),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=offs_b_k[:, None] < (K - k * PACKED_BLOCK_K_B),
                other=0.0,
            )
        # We accumulate along the K dimension.
        if is_a_microscaled_format or is_b_microscaled_format:
            if is_a_microscaled_format:
                mask_ak_scale = offs_scale_ak < (K - k * PACKED_BLOCK_K_A) // (
                    MX_PACK_DIVISOR // A_PACK_DIVISOR
                )
                a_mx_scales = tl.load(
                    a_mx_scale_ptrs, mask=mask_ak_scale[None, :], other=0.0
                )
            else:
                a_mx_scales = None
            mask_bk_scale = offs_scale_bk < (K - k * PACKED_BLOCK_K_B) // (
                MX_PACK_DIVISOR // B_PACK_DIVISOR
            )
            b_mx_scales = tl.load(
                b_mx_scale_ptrs, mask=mask_bk_scale[None, :], other=0.0
            )

            accumulator = tl.dot_scaled(
                a,
                a_mx_scales,
                A_DTYPE_FORMAT,
                b,
                b_mx_scales,
                B_DTYPE_FORMAT,
                acc=accumulator,
                fast_math=True,
            )

            if is_a_microscaled_format:
                if SWIZZLE_MX_A:
                    a_mx_scale_ptrs += MX_SCALE_BLOCK_K_A // 4 * stride_amxk
                else:
                    a_mx_scale_ptrs += MX_SCALE_BLOCK_K_A * stride_amxk
            if SWIZZLE_MX_B:
                b_mx_scale_ptrs += MX_SCALE_BLOCK_K_B // 4 * 512
            else:
                b_mx_scale_ptrs += MX_SCALE_BLOCK_K_B * stride_bmxk
        # Advance the ptrs to the next K block.
        a_ptrs += PACKED_BLOCK_K_A * stride_ak
        b_ptrs += PACKED_BLOCK_K_B * stride_bk

    # Multiply with the scalar weight
    accumulator *= a_scale * b_scale
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    gate_acc, up_acc = (
        accumulator.to(tl.float32).reshape(BLOCK_SIZE_M, BLOCK_SIZE_HALF, 2).split()
    )

    # ------------------------- THE ONLY DIVERGENCE FROM UPSTREAM -------------
    if ACT_SITU:
        accumulator = _situ_and_mul(gate_acc, up_acc, SITU_BETA, SITU_LINEAR_BETA).to(
            compute_type
        )
    else:
        accumulator = (_silu_exp2(gate_acc) * up_acc).to(compute_type)
    # -------------------------------------------------------------------------

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_HALF + tl.arange(0, BLOCK_SIZE_HALF)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N // 2)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def fused_moe_mxfp4_act(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    A_mx_scale: torch.Tensor,
    B_mx_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    swizzle_mx_a: bool,
    swizzle_mx_b: bool,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    activation: str = "silu",
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
) -> None:
    """Drop-in replacement for aiter's `fused_moe_mxfp4_silu` with an extra
    `activation` argument.

    activation="silu"  -> upstream SwiGLU epilogue, byte-identical output.
    activation="situ"  -> Kimi-K3 SituGLU (see module docstring).

    swizzle_mx_a / swizzle_mx_b MUST be False on gfx942 -- preshuffled scale
    layouts are not emulated off CDNA4.
    """
    assert activation in ("silu", "situ"), activation
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    assert A_scale is not None
    assert B_scale is not None
    if A.dtype == torch.uint8:
        assert A_mx_scale is not None, "A_mx_scale should exist when A is mxfp4"
        A_mx_scale_strid_m, A_mx_scale_strid_k = A_mx_scale.stride()
    else:
        assert A_mx_scale is None, "A_mx_scale should not exist when A is not mxfp4"
        A_mx_scale_strid_m, A_mx_scale_strid_k = None, None
    # NOTE: Only supports B_mx_scale
    assert B_mx_scale is not None

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size (same heuristic as upstream)
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config["BLOCK_SIZE_M"])

    A_tl_dtype = torch_to_triton_dtype[A.dtype]
    A_DTYPE_FORMAT = get_scaled_dot_format_string(A_tl_dtype)
    B_tl_dtype = torch_to_triton_dtype[B.dtype]
    B_DTYPE_FORMAT = get_scaled_dot_format_string(B_tl_dtype)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    _fused_moe_kernel_mxfp4_act[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        A_mx_scale,
        B_mx_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        A.shape[1],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        A_mx_scale_strid_m,
        A_mx_scale_strid_k,
        B_mx_scale.stride(0),
        B_mx_scale.stride(2),
        B_mx_scale.stride(1),
        A_DTYPE_FORMAT=A_DTYPE_FORMAT,
        B_DTYPE_FORMAT=B_DTYPE_FORMAT,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        SWIZZLE_MX_A=swizzle_mx_a,
        SWIZZLE_MX_B=swizzle_mx_b,
        ACT_SITU=(activation == "situ"),
        SITU_BETA=situ_beta,
        SITU_LINEAR_BETA=situ_linear_beta,
        **config,
    )


def fused_moe_mxfp4_situ(*args, **kwargs) -> None:
    """`fused_moe_mxfp4_act` with activation defaulted to SiTU."""
    kwargs.setdefault("activation", "situ")
    return fused_moe_mxfp4_act(*args, **kwargs)
