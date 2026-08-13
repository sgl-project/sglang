# SPDX-License-Identifier: Apache-2.0
"""Skinny router-logits GEMV: [M, K] bf16 x [N, K] bf16 -> [M, N] fp32.

MoE router gates are tiny (N ~ 128 experts, K ~ hidden) but hipblaslt's
solutions for M<=8, N=128 run at ~0.1 TB/s on gfx950 (~12us for 1.6MB). This
split-K kernel reads the gate weight once at near-roofline. The split-K
reduction runs in the same kernel via a last-CTA fixup with a self-cleaning
counter (reset to 0 by the reducing CTA), so no separate zero-init or reduce
launch is needed — every extra launch costs ~2us on gfx950.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import triton
import triton.language as tl

_MAX_M = 64

# (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps) per M bucket.
_CONFIGS = (
    (8, (1, 16, 512, 4, 16)),
    (16, (16, 16, 512, 8, 4)),
    (32, (16, 32, 512, 8, 4)),
    (_MAX_M, (32, 32, 512, 8, 4)),
)


def _config(m: int):
    for max_m, cfg in _CONFIGS:
        if m <= max_m:
            return cfg
    return _CONFIGS[-1][1]


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
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
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

    part_base = (pid_k * tl.num_programs(1) + pid_m) * (BLOCK_M * N)
    tl.store(
        partials_ptr + part_base + tl.arange(0, BLOCK_M)[:, None] * N + offs_n, acc
    )

    # Last CTA for this (n-block, row tile) reduces all SPLIT_K partials and
    # resets the counter, keeping the buffer reusable with no zeroing launch.
    count = tl.atomic_add(
        counter_ptr + pid_n * tl.num_programs(1) + pid_m, 1, sem="acq_rel"
    )
    if count == SPLIT_K - 1:
        total = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for s in range(SPLIT_K):
            total += tl.load(
                partials_ptr
                + (s * tl.num_programs(1) + pid_m) * (BLOCK_M * N)
                + tl.arange(0, BLOCK_M)[:, None] * N
                + offs_n
            )
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n,
            total,
            mask=(offs_m < M)[:, None],
        )
        tl.atomic_xchg(counter_ptr + pid_n * tl.num_programs(1) + pid_m, 0)


_scratch: Dict[Tuple[torch.device, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_scratch(device: torch.device, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (device, n)
    if key not in _scratch:
        max_split_k = max(cfg[3] for _, cfg in _CONFIGS)
        _scratch[key] = (
            torch.empty(max_split_k * _MAX_M * n, dtype=torch.float32, device=device),
            torch.zeros(n * _MAX_M // 16, dtype=torch.int32, device=device),
        )
    return _scratch[key]


def router_gemv_supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    m, k = x.shape
    n = w.shape[0]
    _, block_n, block_k, _, _ = _config(m)
    return (
        x.stride(1) == 1
        and w.stride(1) == 1
        and n % block_n == 0
        and k % block_k == 0
        and m <= _MAX_M
    )


def router_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Router logits in fp32; caller guards with router_gemv_supported()."""
    m, k = x.shape
    n = w.shape[0]
    block_m, block_n, block_k, split_k, num_warps = _config(m)
    partials, counter = _get_scratch(x.device, n)
    out = torch.empty(m, n, device=x.device, dtype=torch.float32)
    grid = (n // block_n, triton.cdiv(m, block_m), split_k)
    _router_gemv_kernel[grid](
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
        num_warps=num_warps,
    )
    return out
