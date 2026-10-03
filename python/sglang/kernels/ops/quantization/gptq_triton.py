# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM
# vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py.
"""Triton W4A16 kernels for GPTQ checkpoints on ROCm.

GPTQ checkpoints pack weights along K as [K/8, N]. They are repacked once at
load time to the ExLlama-shuffled [N, K/8] layout read by the GEMM below.
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl

GPTQ_TRITON_SUPPORTED_GROUP_SIZES = {-1, 32, 64, 128, 256}

# Offline-tuned on Radeon 8060S (gfx1151), ROCm 7.2.4, Triton 3.5.1.
# Key: (M bucket, K, N), value: (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages).
_GFX1151_W4A16_CONFIGS = {
    (8, 5120, 16384): (16, 64, 128, 4, 1),
    (8, 17408, 5120): (16, 32, 128, 2, 1),
    (12, 5120, 16384): (16, 64, 128, 4, 1),
    (12, 5120, 14336): (32, 16, 128, 2, 1),
    (12, 5120, 34816): (16, 32, 128, 2, 1),
    (12, 17408, 5120): (16, 32, 128, 2, 1),
    (16, 5120, 16384): (16, 64, 128, 4, 1),
    (16, 5120, 34816): (16, 32, 128, 2, 1),
    (16, 17408, 5120): (16, 32, 128, 2, 1),
    (64, 5120, 16384): (64, 32, 128, 4, 1),
    (64, 5120, 96): (128, 64, 64, 8, 2),
    (64, 6144, 5120): (128, 32, 128, 4, 1),
    (64, 5120, 14336): (64, 64, 128, 4, 1),
    (64, 5120, 34816): (64, 64, 128, 4, 1),
    (64, 17408, 5120): (64, 64, 128, 4, 1),
    (128, 5120, 16384): (128, 64, 64, 8, 2),
    (128, 5120, 96): (64, 32, 128, 4, 1),
    (128, 6144, 5120): (128, 32, 128, 4, 1),
    (128, 5120, 14336): (128, 64, 64, 8, 2),
    (128, 5120, 34816): (128, 64, 64, 8, 2),
    (128, 17408, 5120): (128, 64, 64, 8, 2),
    (512, 5120, 16384): (128, 128, 64, 8, 2),
    (512, 6144, 5120): (128, 64, 64, 8, 2),
    (512, 5120, 14336): (128, 128, 64, 8, 2),
    (512, 17408, 5120): (128, 256, 32, 8, 2),
    (1024, 5120, 16384): (128, 128, 64, 8, 2),
    (1024, 6144, 5120): (128, 128, 64, 8, 2),
    (1024, 5120, 14336): (128, 256, 32, 8, 2),
    (1024, 5120, 34816): (128, 512, 32, 16, 2),
    (1024, 17408, 5120): (128, 128, 64, 8, 2),
    (4096, 5120, 16384): (128, 256, 32, 8, 2),
    (4096, 5120, 96): (64, 128, 32, 4, 3),
    (4096, 5120, 14336): (128, 256, 32, 8, 2),
    (4096, 5120, 34816): (128, 256, 32, 8, 2),
}


@functools.cache
def _is_gfx1151() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    return arch.startswith("gfx1151")


def _gfx1151_w4a16_config(
    m: int, k: int, n: int, group_size: int
) -> tuple[int, int, int, int, int] | None:
    if (
        os.getenv("SGLANG_GFX1151_W4A16_TUNING", "1").lower() not in ("1", "true")
        or group_size != 128
        or not _is_gfx1151()
    ):
        return None
    if m in (8, 12, 16):
        bucket = m
    elif m <= 32:
        return None
    elif m <= 64:
        bucket = 64
    elif m <= 128:
        bucket = 128
    elif m <= 512:
        bucket = 512
    elif m <= 1024:
        bucket = 1024
    elif m <= 4096:
        bucket = 4096
    else:
        return None
    return _GFX1151_W4A16_CONFIGS.get((bucket, k, n))


@triton.jit
def _repack_gptq_w4_to_skinny_kernel(
    source,
    destination,
    K8,
    N,
    BLOCK_K8: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Transpose GPTQ words and apply the ExLlama nibble shuffle."""
    offs_k8 = tl.program_id(0) * BLOCK_K8 + tl.arange(0, BLOCK_K8)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    source_offsets = offs_k8[:, None] * N + offs_n[None, :]
    word = tl.load(
        source + source_offsets,
        mask=(offs_k8[:, None] < K8) & (offs_n[None, :] < N),
        other=0,
    )

    shuffled = word & 0x0000000F
    shuffled |= (word & 0x00000F00) >> 4
    shuffled |= (word & 0x000F0000) >> 8
    shuffled |= (word & 0x0F000000) >> 12
    shuffled |= (word & 0x000000F0) << 12
    shuffled |= (word & 0x0000F000) << 8
    shuffled |= (word & 0x00F00000) << 4
    shuffled |= word & -0x10000000

    destination_offsets = offs_n[None, :] * K8 + offs_k8[:, None]
    tl.store(
        destination + destination_offsets,
        shuffled,
        mask=(offs_k8[:, None] < K8) & (offs_n[None, :] < N),
    )


def repack_gptq_w4_to_skinny(qweight: torch.Tensor, k: int) -> torch.Tensor:
    """Convert [K/8, N] sequential GPTQ words to [N, K/8] ExLlama words."""
    if qweight.dtype != torch.int32 or qweight.ndim != 2:
        raise ValueError("GPTQ skinny kernels require a 2D int32 qweight")
    if k % 8 != 0 or qweight.shape[0] != k // 8:
        raise ValueError(
            f"Unexpected GPTQ qweight shape {tuple(qweight.shape)} for K={k}"
        )

    k8, n = qweight.shape
    output = torch.empty((n, k8), dtype=qweight.dtype, device=qweight.device)
    block_k8 = 32
    block_n = 32
    grid = (triton.cdiv(k8, block_k8), triton.cdiv(n, block_n))
    _repack_gptq_w4_to_skinny_kernel[grid](
        qweight,
        output,
        k8,
        n,
        BLOCK_K8=block_k8,
        BLOCK_N=block_n,
    )
    return output


@triton.jit
def _gptq_w4a16_skinny_gemm_kernel(
    a_ptr,
    b_ptr,
    scales_ptr,
    zeros_ptr,
    c_ptr,
    M,
    N,
    K,
    K8,
    num_groups,
    group_size,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """W4A16 GEMM reading ExLlama-shuffled [N, K/8] weights."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    shifts = (tl.arange(0, 8) // 2) * 4 + (tl.arange(0, 8) % 2) * 16
    shifts = tl.reshape(
        tl.broadcast_to(shifts[None, :], (BLOCK_K // 8, 8)),
        (BLOCK_K,),
    )
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_N, BLOCK_K))
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(
            a_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & mask_k[None, :],
            other=0.0,
        )

        offs_k8 = k_start * (BLOCK_K // 8) + tl.arange(0, BLOCK_K // 8)
        b_packed = tl.load(
            b_ptr + offs_n[:, None] * K8 + offs_k8[None, :],
            mask=(offs_n[:, None] < N) & (offs_k8[None, :] < K8),
            other=0,
        )
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = (b >> shifts) & 0xF

        group_idx = (k_start * BLOCK_K) // group_size
        scales = tl.load(
            scales_ptr + offs_n * num_groups + group_idx,
            mask=offs_n < N,
            other=1.0,
        )
        if HAS_ZP:
            zero_word = tl.load(
                zeros_ptr + (offs_n // 8) * num_groups + group_idx,
                mask=offs_n < N,
                other=0,
            )
            zero = (zero_word >> (4 * (offs_n % 8))) & 0xF
            b = (b - zero[:, None]).to(scales.dtype) * scales[:, None]
        else:
            b = (b - ZP_BIAS).to(scales.dtype) * scales[:, None]

        accumulator += tl.dot(a, tl.trans(b), out_dtype=tl.float32)

    output = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(
        c_ptrs,
        output,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def gptq_w4a16_skinny_gemm(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    qzeros: torch.Tensor | None = None,
    zp_bias: int = 8,
) -> torch.Tensor:
    """Run W4A16 GEMM from the shared HIP-skinny weight layout."""
    if input.ndim != 2 or input.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("GPTQ skinny Triton expects a 2D fp16/bf16 input")
    input = input.contiguous()
    if not qweight.is_contiguous() or not scales.is_contiguous():
        raise ValueError("GPTQ skinny weights and scales must be contiguous")

    m, k = input.shape
    n, k8 = qweight.shape
    effective_group_size = k if group_size == -1 else group_size
    if k8 != k // 8:
        raise ValueError(
            f"Unexpected skinny qweight shape {tuple(qweight.shape)} for K={k}"
        )
    if k % effective_group_size != 0:
        raise ValueError(
            f"GPTQ input size {k} is not divisible by group size {effective_group_size}"
        )
    num_groups = k // effective_group_size
    if scales.shape != (n, num_groups):
        raise ValueError(
            f"Unexpected skinny scales shape {tuple(scales.shape)}; "
            f"expected {(n, num_groups)}"
        )
    if qzeros is not None and qzeros.shape != (n // 8, num_groups):
        raise ValueError(
            f"Unexpected skinny qzeros shape {tuple(qzeros.shape)}; "
            f"expected {(n // 8, num_groups)}"
        )

    tuned_config = _gfx1151_w4a16_config(m, k, n, effective_group_size)
    num_stages = None
    if tuned_config is not None:
        block_m, block_n, block_k, num_warps, num_stages = tuned_config
    else:
        if m <= 32:
            block_m, block_n, block_k, num_warps = 32, 32, 128, 4
        elif m <= 64:
            block_m, block_n, block_k, num_warps = 64, 64, 32, 4
        elif m <= 128:
            if k >= 2 * n:
                block_m, block_n, block_k, num_warps = 64, 16, 64, 1
            elif n > k:
                block_m, block_n, block_k, num_warps = 64, 64, 64, 4
            else:
                block_m, block_n, block_k, num_warps = 64, 32, 64, 4
        elif m <= 1024:
            if k >= 2 * n:
                block_m, block_n, block_k, num_warps = 64, 64, 64, 4
            elif n >= 4 * k:
                block_m, block_n, block_k, num_warps = 128, 64, 64, 8
            else:
                block_m, block_n, block_k, num_warps = 64, 128, 32, 4
        elif k >= 2 * n:
            block_m, block_n, block_k, num_warps = 128, 512, 32, 16
        else:
            block_m, block_n, block_k, num_warps = 128, 64, 64, 8
    block_k = min(block_k, effective_group_size)

    output = torch.empty((m, n), dtype=input.dtype, device=input.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    launch_kwargs = {} if num_stages is None else {"num_stages": num_stages}
    _gptq_w4a16_skinny_gemm_kernel[grid](
        input,
        qweight,
        scales,
        qzeros if qzeros is not None else scales,
        output,
        m,
        n,
        k,
        k8,
        num_groups,
        effective_group_size,
        HAS_ZP=qzeros is not None,
        ZP_BIAS=zp_bias,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        **launch_kwargs,
    )
    return output
