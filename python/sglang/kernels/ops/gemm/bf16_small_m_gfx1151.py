# SPDX-License-Identifier: Apache-2.0
"""bf16 ``x @ w.T`` for a few rows (``M <= 16``: an LM head or a speculative drafter
projection during a verify or draft block) on gfx1151.

At M = 8, hipBLASLt reads bf16 weights at only 30-160 GB/s on gfx1151; a plain ``tl.dot``
kernel with M padded to 16 and a shape-dependent tile reaches 120-210 GB/s. Shapes with
few output tiles may split K across programs with fp32 atomics (nondeterministic summation
order); ``deterministic=True`` (the default) never splits.
"""

import functools

import torch
import triton
import triton.language as tl

from sglang.srt.utils import use_gfx1151_kernels


@triton.jit
def _bf16_small_m_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    rm = tl.arange(0, 16)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    KS: tl.constexpr = K // SPLIT
    k0 = ps * KS
    # no masks in the loop (a masked load is a branch plus a wait): clamp the rows instead
    a = x_ptr + tl.minimum(rm, M - 1)[:, None] * K + (k0 + rk)[None, :]
    b = w_ptr + tl.minimum(rn, N - 1)[:, None] * K + (k0 + rk)[None, :]
    acc = tl.zeros((16, BN), tl.float32)
    for _ in range(0, KS, BK):
        acc = tl.dot(tl.load(a), tl.trans(tl.load(b)), acc)
        a += BK
        b += BK
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    if SPLIT == 1:
        tl.store(
            out_ptr + rm[:, None] * N + rn[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=mask,
        )
    else:
        tl.atomic_add(
            out_ptr + rm[:, None] * N + rn[None, :], acc, mask=mask, sem="relaxed"
        )


@functools.lru_cache(maxsize=None)
def pick_config(n: int, k: int, deterministic: bool):
    """``(BN, BK, warps, stages, split)`` from a sweep on gfx1151 at M = 8. GB/s, torch -> this: LM head 157 -> 212, drafter gate_up 117 -> 165,
    down 116 -> 154 (split 4; 126 without), q 59 -> 91 (split 8), kv 52 -> 80 (split 8),
    o 35 -> 40."""
    if n >= 16384:
        return 32, 128, 4, 1, 1
    if not deterministic:
        if k >= 16384:
            return 32, 64, 2, 1, 4
        if n <= 4096:
            return 128, 64, 2, 1, 8
    if k <= 4096:
        return 32, 128, 8, 1, 1
    return 32, 128, 2, 2, 1


def covered(x: torch.Tensor, w: torch.Tensor) -> bool:
    """Whether ``bf16_small_m_gemm`` takes ``x [M, K] @ w[N, K].T`` (gfx1151, contiguous
    bf16, ``M <= 16``, ``K % 128 == 0``)."""
    return (
        x.dim() == 2
        and 0 < x.shape[0] <= 16
        and use_gfx1151_kernels()
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and w.dim() == 2
        and w.shape[1] == x.shape[1]
        and x.shape[1] % 128 == 0
        and x.is_contiguous()
        and w.is_contiguous()
    )


def bf16_small_m_gemm(
    x,
    w,
    deterministic=True,
    bn=None,
    bk=None,
    num_warps=None,
    num_stages=None,
    split=None,
):
    """``x @ w.T`` (bf16) for ``M <= 16`` rows; see ``covered``. ``deterministic`` forbids
    split-K; the keyword arguments override the tuned config (for benchmarking)."""
    m, k = x.shape
    n = w.shape[0]
    c_bn, c_bk, c_w, c_s, c_split = pick_config(n, k, deterministic)
    bn, bk = bn or c_bn, bk or c_bk
    num_warps, num_stages, split = num_warps or c_w, num_stages or c_s, split or c_split
    assert k % (bk * split) == 0
    if split == 1:
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    else:
        out = torch.zeros((m, n), device=x.device, dtype=torch.float32)
    _bf16_small_m_kernel[(triton.cdiv(n, bn), split)](
        x,
        w,
        out,
        m,
        n,
        K=k,
        BN=bn,
        BK=bk,
        SPLIT=split,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out if split == 1 else out.to(x.dtype)
