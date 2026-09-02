# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# The GEMM body is derived from ROCm/AITER's batched_gemm_a16wfp4 kernel;
# the flattened MXFP4 output epilogue is specific to SGLang's MLA path.

from __future__ import annotations

import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a16wfp4 import (
    _get_config,
)
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd

from sglang.srt.layers.quantization.rocm_mxfp4_utils import (
    batched_gemm_afp4wfp4_pre_quant,
    fused_flatten_mxfp4_quant,
)
from sglang.srt.utils.common import direct_register_custom_op

_MXFP4_QUANT_BLOCK_SIZE = 32
# Triton only lets @triton.jit kernels read module-level globals that are
# instantiated as tl.constexpr, so device code uses this alias while host code
# keeps the plain int.
_MXFP4_QUANT_BLOCK_SIZE_TL = tl.constexpr(_MXFP4_QUANT_BLOCK_SIZE)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (
            args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0
            and args["K"] % (args["SPLITK_BLOCK_SIZE"] // 2) == 0
        ),
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _batched_gemm_a16wfp4_flatten_mxfp4_quant_kernel(
    a_ptr,
    b_ptr,
    b_scales_ptr,
    out_ptr,
    out_scales_ptr,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bn,
    stride_bk,
    stride_bsb,
    stride_bsn,
    stride_bsk,
    stride_om,
    stride_on,
    stride_osm,
    stride_osn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """Batched A16W4 GEMM with a flattened MXFP4 output epilogue."""
    pid_batch = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # This epilogue must see the complete accumulator before quantizing it.
    tl.static_assert(NUM_KSPLIT == 1)
    tl.static_assert(BLOCK_SIZE_N % _MXFP4_QUANT_BLOCK_SIZE_TL == 0)

    remap_xcd(pid, GRID_MN)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    stride_ab = tl.cast(stride_ab, tl.int64)
    stride_bb = tl.cast(stride_bb, tl.int64)
    pid_batch_i64 = tl.cast(pid_batch, tl.int64)

    offs_k_bf16 = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    a_ptrs = (
        a_ptr
        + pid_batch_i64 * stride_ab
        + offs_am[:, None] * stride_am
        + offs_k_bf16[None, :] * stride_ak
    )

    offs_k = tl.arange(0, BLOCK_SIZE_K // 2)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    b_ptrs = (
        b_ptr
        + pid_batch_i64 * stride_bb
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    scale_group_size: tl.constexpr = _MXFP4_QUANT_BLOCK_SIZE_TL
    offs_ks = tl.arange(0, BLOCK_SIZE_K // scale_group_size)
    b_scale_ptrs = (
        b_scales_ptr
        + pid_batch_i64 * stride_bsb
        + offs_bn[:, None] * stride_bsn
        + offs_ks[None, :] * stride_bsk
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_k_iter: tl.constexpr = tl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)
    for k in range(0, num_k_iter):
        b_scales = tl.load(b_scale_ptrs)
        if EVEN_K:
            a_bf16 = tl.load(a_ptrs)
            b = tl.load(b_ptrs, cache_modifier=cache_modifier)
        else:
            logical_k = 2 * K
            a_bf16 = tl.load(
                a_ptrs,
                mask=offs_k_bf16[None, :] < logical_k - k * BLOCK_SIZE_K,
                other=0,
            )
            b = tl.load(
                b_ptrs,
                mask=offs_k[:, None] < K - k * (BLOCK_SIZE_K // 2),
                other=0,
            )

        a, a_scales = _mxfp4_quant_op(
            a_bf16,
            BLOCK_SIZE_K,
            BLOCK_SIZE_M,
            scale_group_size,
        )
        accumulator = tl.dot_scaled(
            a,
            a_scales,
            "e2m1",
            b,
            b_scales,
            "e2m1",
            acc=accumulator,
        )

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        b_scale_ptrs += (BLOCK_SIZE_K // scale_group_size) * stride_bsk

    # Preserve the split path's BF16 boundary before dynamic output quantization.
    rounded = accumulator.to(tl.bfloat16).to(tl.float32)
    out, out_scales = _mxfp4_quant_op(
        rounded,
        BLOCK_SIZE_N,
        BLOCK_SIZE_M,
        scale_group_size,
    )

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = (
        pid_batch * (N // 2)
        + pid_n * (BLOCK_SIZE_N // 2)
        + tl.arange(0, BLOCK_SIZE_N // 2)
    )
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_out_n[None, :] * stride_on,
        out,
        mask=offs_m[:, None] < M,
    )

    num_quant_blocks: tl.constexpr = BLOCK_SIZE_N // _MXFP4_QUANT_BLOCK_SIZE_TL
    offs_scale_n = (
        pid_batch * (N // _MXFP4_QUANT_BLOCK_SIZE_TL)
        + pid_n * num_quant_blocks
        + tl.arange(0, num_quant_blocks)
    )
    tl.store(
        out_scales_ptr
        + offs_m[:, None] * stride_osm
        + offs_scale_n[None, :] * stride_osn,
        out_scales,
        mask=offs_m[:, None] < M,
    )


def _get_fused_config(x: torch.Tensor, w: torch.Tensor) -> dict | None:
    _, m, logical_k = x.shape
    _, n, packed_k = w.shape
    if logical_k != 2 * packed_k or n % _MXFP4_QUANT_BLOCK_SIZE != 0:
        return None

    config, _ = _get_config(m, n, packed_k)
    if (
        config["NUM_KSPLIT"] != 1
        or config["BLOCK_SIZE_N"] % _MXFP4_QUANT_BLOCK_SIZE != 0
        or n % config["BLOCK_SIZE_N"] != 0
    ):
        return None
    return config


def _batched_gemm_a16wfp4_flatten_mxfp4_quant(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scales: torch.Tensor,
    out: torch.Tensor,
    out_scales: torch.Tensor,
) -> None:
    batch, m, _ = x.shape
    _, n, packed_k = w.shape
    config = _get_fused_config(x, w)
    if config is None:
        bf16_output = torch.empty((m, batch, n), dtype=torch.bfloat16, device=x.device)
        batched_gemm_afp4wfp4_pre_quant(
            x,
            w,
            w_scales,
            torch.bfloat16,
            bf16_output.transpose(0, 1),
        )
        split_out, split_scales = fused_flatten_mxfp4_quant(bf16_output)
        out.copy_(split_out)
        out_scales.copy_(split_scales)
        return

    config = dict(config)
    config["SPLITK_BLOCK_SIZE"] = 2 * packed_k
    if config["BLOCK_SIZE_K"] >= 2 * packed_k:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * packed_k)

    grid = lambda meta: (  # noqa: E731
        batch,
        triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
    )
    _batched_gemm_a16wfp4_flatten_mxfp4_quant_kernel[grid](
        x,
        w,
        w_scales,
        out,
        out_scales,
        m,
        n,
        packed_k,
        *x.stride(),
        *w.stride(),
        *w_scales.stride(),
        *out.stride(),
        *out_scales.stride(),
        **config,
    )


def _batched_gemm_a16wfp4_flatten_mxfp4_quant_fake(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scales: torch.Tensor,
    out: torch.Tensor,
    out_scales: torch.Tensor,
) -> None:
    return None


direct_register_custom_op(
    op_name="batched_gemm_a16wfp4_flatten_mxfp4_quant",
    op_func=_batched_gemm_a16wfp4_flatten_mxfp4_quant,
    mutates_args=["out", "out_scales"],
    fake_impl=_batched_gemm_a16wfp4_flatten_mxfp4_quant_fake,
)


def can_fuse_mla_value_bmm_mxfp4_quant(
    x: torch.Tensor, w: torch.Tensor, w_scales: torch.Tensor
) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and w.dtype == torch.uint8
        and w_scales.dtype == torch.uint8
        and x.device == w.device == w_scales.device
        and x.dim() == w.dim() == w_scales.dim() == 3
        and x.shape[0] == w.shape[0] == w_scales.shape[0]
        and w.shape[1] == w_scales.shape[1]
        and w.shape[2] * 2 == x.shape[2]
        and w_scales.shape[2] * _MXFP4_QUANT_BLOCK_SIZE == x.shape[2]
        # Keep Python-side graph tracing independent of AITER's config lookup.
        # The pinned Kimi shape has a non-split config; the custom op verifies it
        # again with concrete runtime dimensions before launching.
        and w.shape[1] == 128
        and x.shape[2] == 512
    )


def batched_gemm_a16wfp4_flatten_mxfp4_quant(
    x: torch.Tensor, w: torch.Tensor, w_scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute batched A16W4 GEMM and emit flattened per-1x32 MXFP4."""
    if not can_fuse_mla_value_bmm_mxfp4_quant(x, w, w_scales):
        raise ValueError(
            "Fused MLA value BMM requires BF16 x, uint8 W/scales, compatible "
            "batched dimensions, and the Kimi N=128, K=512 value shape"
        )

    batch, m, _ = x.shape
    n = w.shape[1]
    out = torch.empty(
        (m, batch * n // 2),
        dtype=torch.uint8,
        device=x.device,
    )
    # Match fused_flatten_mxfp4_quant's transposed scale layout, which the
    # following o_proj GEMM consumes without materialization.
    out_scales = torch.empty(
        (batch * n // _MXFP4_QUANT_BLOCK_SIZE, m),
        dtype=torch.uint8,
        device=x.device,
    ).T
    torch.ops.sglang.batched_gemm_a16wfp4_flatten_mxfp4_quant(
        x, w, w_scales, out, out_scales
    )
    return out, out_scales


__all__ = [
    "batched_gemm_a16wfp4_flatten_mxfp4_quant",
    "can_fuse_mla_value_bmm_mxfp4_quant",
]
