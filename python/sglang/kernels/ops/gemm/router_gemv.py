# SPDX-License-Identifier: Apache-2.0
"""Skinny router-logits GEMV: [M, K] bf16 x [N, K] bf16 or fp32 -> [M, N] fp32.

MoE router gates are tiny (N ~ 128 experts, K ~ hidden) but hipblaslt's
solutions for M<=8, N=128 run at ~0.1 TB/s on gfx950 (~12us for 1.6MB). This
split-K kernel reads the gate weight once at near-roofline. Up to _FIXUP_MAX_M
rows the split-K reduction runs in the same kernel via a last-CTA fixup with a
self-cleaning counter (reset to 0 by the reducing CTA), so no separate zero-init
or reduce launch is needed (every extra launch costs ~2us on gfx950). Past it a
second launch sums the partials in the same fixed order.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.aiter_fp8_quant_triton import (
    swiglu_oai_fp8_outputs,
    swiglu_oai_fp8_rows,
)
from sglang.kernels.ops.quantization.bf16x3_split import split_fp32_to_bf16x3
from sglang.srt.utils import is_gfx95_supported, is_hip

_is_hip = is_hip()
_is_gfx95_supported = _is_hip and is_gfx95_supported()

# Past 16 rows the fixup's cross-XCD release / acquire costs more than a reduce launch;
# past 512 hipblaslt with an fp32 output is faster (MiniMax-M3 gate, cold weight).
_FIXUP_MAX_M = 16
_MAX_M = 512

# (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps) per M bucket.
_CONFIGS = (
    (8, (16, 32, 256, 12, 8)),
    (_FIXUP_MAX_M, (16, 16, 512, 8, 4)),
    (128, (16, 32, 256, 12, 4)),
    (256, (32, 32, 256, 12, 4)),
    (320, (64, 32, 256, 12, 4)),
    (_MAX_M, (64, 32, 256, 8, 4)),
)
# FP32 weights on CUDA (FP32 router gates) use the FP32 row path only, up to 8 rows,
# a conservative cap: for N=256, K=5120 on GB300 it beats bf16x3 through 14 rows.
_CUDA_FP32_CONFIGS = ((8, (1, 16, 128, 8, 8)),)
_REDUCE_BLOCK = 512


def _configs(w_dtype: torch.dtype):
    return _CUDA_FP32_CONFIGS if w_dtype == torch.float32 else _CONFIGS


def _config(m: int, w_dtype: torch.dtype):
    configs = _configs(w_dtype)
    for max_m, cfg in configs:
        if m <= max_m:
            return cfg
    return configs[-1][1]


@triton.jit
def _router_gemv_tile(
    x_ptr,
    w_ptr,
    out_ptr,
    partials_ptr,
    counter_ptr,
    M,
    K,
    N,
    stride_xm,
    stride_wn,
    stride_om,
    pid_n,
    pid_m,
    pid_k,
    num_m_tiles,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    FIXUP: tl.constexpr,
):
    """The (n-block, row tile, split) program of _router_gemv_kernel."""
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if BLOCK_M == 1:
        acc = tl.zeros((BLOCK_N,), tl.float32)
        for k0 in range(pid_k * BLOCK_K, K, BLOCK_K * SPLIT_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            xv = tl.load(x_ptr + pid_m * stride_xm + offs_k).to(tl.float32)
            wv = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]).to(
                tl.float32
            )
            acc += tl.sum(wv * xv[None, :], axis=1)
        acc = acc[None, :]
        offs_m = pid_m + tl.zeros((1,), tl.int32)
    else:
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(pid_k * BLOCK_K, K, BLOCK_K * SPLIT_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            xv = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                mask=mask_m[:, None],
                other=0.0,
            )
            wv = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :])
            acc += tl.dot(xv, tl.trans(wv), out_dtype=tl.float32)

    part_base = (pid_k * num_m_tiles + pid_m) * (BLOCK_M * N)
    tl.store(
        partials_ptr + part_base + tl.arange(0, BLOCK_M)[:, None] * N + offs_n, acc
    )
    if not FIXUP:
        return

    # Last CTA for this (n-block, row tile) reduces all SPLIT_K partials and
    # resets the counter, keeping the buffer reusable with no zeroing launch.
    count = tl.atomic_add(counter_ptr + pid_n * num_m_tiles + pid_m, 1, sem="acq_rel")
    if count == SPLIT_K - 1:
        total = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for s in range(SPLIT_K):
            total += tl.load(
                partials_ptr
                + (s * num_m_tiles + pid_m) * (BLOCK_M * N)
                + tl.arange(0, BLOCK_M)[:, None] * N
                + offs_n
            )
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n,
            total,
            mask=(offs_m < M)[:, None],
        )
        tl.atomic_xchg(counter_ptr + pid_n * num_m_tiles + pid_m, 0)


@triton.jit
def _router_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    partials_ptr,  # [SPLIT_K, M, N] fp32 scratch (always fully overwritten)
    counter_ptr,  # [num_n_blocks * M] int32, all-zero between launches
    M,
    K,
    N,
    stride_xm,
    stride_wn,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    FIXUP: tl.constexpr,
):
    _router_gemv_tile(
        x_ptr,
        w_ptr,
        out_ptr,
        partials_ptr,
        counter_ptr,
        M,
        K,
        N,
        stride_xm,
        stride_wn,
        stride_om,
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
        tl.num_programs(1),
        BLOCK_M,
        BLOCK_K,
        BLOCK_N,
        SPLIT_K,
        FIXUP,
    )


@triton.jit
def _router_gemv_swiglu_kernel(
    x_ptr,
    w_ptr,
    partials_ptr,
    M,
    K,
    N,
    stride_xm,
    stride_wn,
    num_n_blocks,
    num_m_tiles,
    gu_ptr,
    q_ptr,
    scale_ptr,
    gu_n,
    stride_gu,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ALPHA: tl.constexpr,
    LIMIT: tl.constexpr,
    GROUP: tl.constexpr,
    TRANSPOSE_SCALE: tl.constexpr,
    SW_BLOCK_M: tl.constexpr,
    SW_BLOCK_N: tl.constexpr,
):
    """router_gemv_partials' programs (in their grid order), then swiglu_oai_fp8's."""
    pid = tl.program_id(0)
    num_gemv = num_n_blocks * num_m_tiles * SPLIT_K
    if pid < num_gemv:
        # without the fixup, out_ptr, counter_ptr and stride_om are never read
        _router_gemv_tile(
            x_ptr,
            w_ptr,
            partials_ptr,
            partials_ptr,
            partials_ptr,
            M,
            K,
            N,
            stride_xm,
            stride_wn,
            0,
            pid % num_n_blocks,
            pid // num_n_blocks % num_m_tiles,
            pid // (num_n_blocks * num_m_tiles),
            num_m_tiles,
            BLOCK_M,
            BLOCK_K,
            BLOCK_N,
            SPLIT_K,
            False,
        )
    else:
        swiglu_oai_fp8_rows(
            gu_ptr,
            q_ptr,
            scale_ptr,
            M,
            gu_n,
            stride_gu,
            pid - num_gemv,
            ALPHA,
            LIMIT,
            GROUP,
            TRANSPOSE_SCALE,
            SW_BLOCK_M,
            SW_BLOCK_N,
        )


@triton.jit
def _router_gemv_reduce_kernel(
    partials_ptr,  # [SPLIT_K, >= M, N] fp32, row tiles contiguous per split
    out_ptr,
    numel,
    stride_split,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    total = tl.zeros((BLOCK,), tl.float32)
    for s in tl.static_range(SPLIT_K):
        total += tl.load(partials_ptr + s * stride_split + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, total, mask=mask)


_scratch: Dict[Tuple[torch.device, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_scratch(device: torch.device, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (device, n)
    if key not in _scratch:
        max_split_k = max(cfg[3] for _, cfg in _CONFIGS + _CUDA_FP32_CONFIGS)
        _scratch[key] = (
            torch.empty(max_split_k * _MAX_M * n, dtype=torch.float32, device=device),
            torch.zeros(n * _FIXUP_MAX_M // 16, dtype=torch.int32, device=device),
        )
    return _scratch[key]


def router_gemv_supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    m, k = x.shape
    n = w.shape[0]
    _, block_n, block_k, _, _ = _config(m, w.dtype)
    # BF16 gates are tuned for gfx950 only; FP32 gates for CUDA only.
    platform_ok = (
        _is_gfx95_supported if w.dtype == torch.bfloat16 else not _is_hip
    ) and w.dtype in (torch.bfloat16, torch.float32)
    return (
        platform_ok
        and x.device.type == "cuda"
        and w.device == x.device
        and x.dtype == torch.bfloat16
        and x.stride(1) == 1
        and w.stride(1) == 1
        and n % block_n == 0
        and k % block_k == 0
        and m <= _configs(w.dtype)[-1][0]
    )


def _launch(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, fixup: bool) -> int:
    """Runs the split-K kernel; returns the rows of each split in the partials scratch."""
    m, k = x.shape
    n = w.shape[0]
    block_m, block_n, block_k, split_k, num_warps = _config(m, w.dtype)
    partials, counter = _get_scratch(x.device, n)
    m_tiles = triton.cdiv(m, block_m)
    _router_gemv_kernel[(n // block_n, m_tiles, split_k)](
        x,
        w,
        out,
        partials,
        counter,
        m,
        k,
        n,
        x.stride(0),
        w.stride(0),
        out.stride(0),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_N=block_n,
        SPLIT_K=split_k,
        FIXUP=fixup,
        num_warps=num_warps,
    )
    return m_tiles * block_m


def router_gemv_partials_supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    """router_gemv_supported() at the row counts router_gemv_partials() takes."""
    return x.shape[0] > _FIXUP_MAX_M and router_gemv_supported(x, w)


def router_gemv_partials(x: torch.Tensor, w: torch.Tensor) -> Optional[torch.Tensor]:
    """The [SPLIT_K, M, N] fp32 split-K partials router_gemv sums over dim 0 in order, past
    _FIXUP_MAX_M rows (None below); valid until the next router_gemv call on the device.
    Caller guards with router_gemv_supported()."""
    m = x.shape[0]
    if m <= _FIXUP_MAX_M:
        return None
    n = w.shape[0]
    partials, _ = _get_scratch(x.device, n)
    rows = _launch(x, w, partials, fixup=False)
    split_k = _config(m, w.dtype)[3]
    return partials[: split_k * rows * n].view(split_k, rows, n)[:, :m]


def router_gemv_partials_swiglu(
    x: torch.Tensor,
    w: torch.Tensor,
    gate_up: torch.Tensor,
    alpha: float,
    limit: float,
    group_size: Optional[int] = None,
    transpose_scale: bool = False,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """(router_gemv_partials(x, w), swiglu_oai_fp8(gate_up, alpha, limit, group_size,
    transpose_scale)) in one launch, bitwise equal to the two calls: both are latency-bound
    and come from the same MoE layer input (router and shared expert). Caller guards with
    router_gemv_partials_supported()."""
    m, k = x.shape
    n = w.shape[0]
    assert _FIXUP_MAX_M < m and gate_up.shape[0] == m
    block_m, block_n, block_k, split_k, num_warps = _config(m, w.dtype)
    q, scale, sw_block_m, sw_block_n = swiglu_oai_fp8_outputs(
        gate_up, group_size, transpose_scale
    )
    # both kernels launch with 4 warps on their own; sharing that count keeps their layouts
    assert num_warps == 4
    partials, _ = _get_scratch(x.device, n)
    m_tiles = triton.cdiv(m, block_m)
    num_gemv = (n // block_n) * m_tiles * split_k
    _router_gemv_swiglu_kernel[(num_gemv + triton.cdiv(m, sw_block_m),)](
        x,
        w,
        partials,
        m,
        k,
        n,
        x.stride(0),
        w.stride(0),
        n // block_n,
        m_tiles,
        gate_up,
        q,
        scale,
        q.shape[1],
        gate_up.stride(0),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_N=block_n,
        SPLIT_K=split_k,
        ALPHA=alpha,
        LIMIT=limit,
        GROUP=group_size or 0,
        TRANSPOSE_SCALE=transpose_scale,
        SW_BLOCK_M=sw_block_m,
        SW_BLOCK_N=sw_block_n,
        num_warps=num_warps,
    )
    rows = m_tiles * block_m
    parts = partials[: split_k * rows * n].view(split_k, rows, n)[:, :m]
    return parts, (q, scale.t() if group_size and transpose_scale else scale)


def router_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Router logits in fp32; caller guards with router_gemv_supported()."""
    m, n = x.shape[0], w.shape[0]
    out = torch.empty(m, n, device=x.device, dtype=torch.float32)
    partials = router_gemv_partials(x, w)
    if partials is None:
        _launch(x, w, out, fixup=True)
        return out
    _router_gemv_reduce_kernel[(triton.cdiv(m * n, _REDUCE_BLOCK),)](
        partials,
        out,
        m * n,
        partials.stride(0),
        SPLIT_K=partials.shape[0],
        BLOCK=_REDUCE_BLOCK,
        num_warps=4,
    )
    return out


def router_gemm_bf16x3_cublas(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """[M, K] bf16 x [N, K] fp32 router logits on BF16 tensor cores.

    The fp32 weight is split per call, so in-place weight updates take effect
    without any refresh. Each bf16 x bf16 product is exact within fp32's exponent
    range, but tensor-core accumulation is not IEEE fp32 summation.
    """
    n = w.shape[0]
    w3 = split_fp32_to_bf16x3(w)
    return torch.mm(x, w3.t(), out_dtype=torch.float32).view(-1, 3, n).sum(dim=1)
