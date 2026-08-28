"""Triton prefill GEMM for canonical ModelOpt NVFP4 weights on RDNA4."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_e2m1(bits):
    magnitude = bits & 0x7
    value = tl.where(
        magnitude <= 4,
        magnitude.to(tl.float32) * 0.5,
        tl.where(
            magnitude == 5,
            3.0,
            tl.where(magnitude == 6, 4.0, 6.0),
        ),
    )
    return tl.where((bits & 0x8) != 0, -value, value)


@triton.jit
def _rdna4_nvfp4_prefill_kernel(
    input_ptr,
    weight_ptr,
    weight_scale_ptr,
    weight_global_scale_ptr,
    output_ptr,
    size_m,
    size_n,
    size_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # int64 row/column offsets: `offsets_m * size_n` reaches past 2**31 for a
    # large prefill chunk against a wide projection, and int32 would wrap.
    offsets_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offsets_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(0, tl.cdiv(size_k, BLOCK_K)):
        k = k_tile * BLOCK_K + offsets_k
        activation = tl.load(
            input_ptr + offsets_m[:, None] * size_k + k[None, :],
            mask=(offsets_m[:, None] < size_m) & (k[None, :] < size_k),
            other=0.0,
        )
        packed_weight = tl.load(
            weight_ptr + offsets_n[None, :] * (size_k // 2) + (k[:, None] // 2),
            mask=(offsets_n[None, :] < size_n) & (k[:, None] < size_k),
            other=0,
        )
        nibble = tl.where(
            (k[:, None] & 1) == 0,
            packed_weight & 0xF,
            packed_weight >> 4,
        )
        block_scale = tl.load(
            weight_scale_ptr + offsets_n[None, :] * (size_k // 16) + (k[:, None] // 16),
            mask=(offsets_n[None, :] < size_n) & (k[:, None] < size_k),
            other=0.0,
        )
        weight = (_decode_e2m1(nibble) * block_scale).to(activation.dtype)
        accumulator = tl.dot(activation, weight, accumulator)

    output = accumulator * tl.load(weight_global_scale_ptr)
    tl.store(
        output_ptr + offsets_m[:, None] * size_n + offsets_n[None, :],
        output,
        mask=(offsets_m[:, None] < size_m) & (offsets_n[None, :] < size_n),
    )


def rdna4_nvfp4_prefill(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Launch a matrix-instruction prefill kernel for validated tensors."""
    size_m, size_k = input.shape
    size_n = weight.shape[0]
    if size_m >= 128:
        block_m, block_n, num_warps = 128, 64, 4
    else:
        block_m, block_n, num_warps = 64, 32, 4
    block_k = 32
    grid = (triton.cdiv(size_m, block_m), triton.cdiv(size_n, block_n))
    _rdna4_nvfp4_prefill_kernel[grid](
        input,
        weight,
        weight_scale,
        weight_global_scale,
        output,
        size_m,
        size_n,
        size_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
