"""Split-K bf16 GEMV for single-row decode on RDNA APUs.

Enabled on gfx1151 (Strix Halo); the tuning constants below were measured there and
have not been re-measured on the wider RDNA parts, which have enough compute units
that the rocBLAS tile already covers the machine.

Motivation
----------
A single-row (M=1) decode projection ``x[1, K] @ W[N, K].T`` is served by rocBLAS
as a tiled GEMM. The tile is fixed at 96 rows/columns, so the number of CTA
groups that can run at once is pinned by ``N / 96``: for the decode projections of
a 0.6B-9B model that is 32-64 concurrent CTAs, far below what is needed to cover
memory latency on a 20-CU APU. Measured on gfx1151 (Strix Halo, 20 CU): 45-93 GB/s,
i.e. 20-40% of the achievable read bandwidth.

Splitting K gives every k-chunk its own CTA group. For M=1 the fp32 partials are
``N * 4 B * split_k``, which is 0.2-0.4% of the weight bytes, so the extra traffic
is negligible next to the 2-4x occupancy gain.

Note that this kernel deliberately reduces with ``tl.sum`` instead of ``tl.dot``:
``tl.dot`` needs at least 16 rows, so at M=1 it spends 15/16 of the loaded weight
tile on padding and is measurably slower here (76 GB/s vs 202 GB/s on gfx1151 for
``4096x1024``).

Numerics
--------
Accumulation is fp32 and the K-split reduction order is static, so the result does
not depend on the batch composition (batch-invariant).
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx1151_supported

_BLOCK_N = 32
_BLOCK_K = 512
# Measured on gfx1151 with cold weights (200 MB working set): K splits of 8-50 all
# beat rocBLAS (1.37x at split_k=34 for 5120x17408, 1.27x at split_k=50 for
# 5120x25600), so the cap only guards against pathological K rather than a
# bandwidth crossover.
_MAX_SPLIT_K = 64
_SUPPORTED_ROWS = 1
# Above this many n-tiles the GEMM is already wide enough to saturate the bus, and
# splitting K only adds the partial/reduce traffic. Measured on gfx1151 with cold
# weights: the 151936 vocabulary head reaches 140 GB/s via rocBLAS and drops to
# 121 GB/s with split-K; a 34816x5120 gate_up (1088 tiles) is 145.7 vs 133.0 GB/s,
# i.e. the crossover sits right at this bound.
_MAX_N_TILES = 1024


@triton.jit
def _gemv_split_k_kernel(
    x_ptr,
    w_ptr,
    part_ptr,
    stride_wn,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One CTA per (n-tile, k-chunk). ``N % BLOCK_N == 0`` is guaranteed by the gate."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :])
    x = tl.load(x_ptr + offs_k)
    acc = tl.sum(w.to(tl.float32) * x[None, :].to(tl.float32), axis=1)
    tl.store(part_ptr + pid_k * N + offs_n, acc)


@triton.jit
def _reduce_split_k_kernel(
    part_ptr,
    out_ptr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + s * N + offs)
    tl.store(out_ptr + offs, acc.to(tl.bfloat16))


@lru_cache(maxsize=1)
def _is_disabled() -> bool:
    return envs.SGLANG_ROCM_DISABLE_RDNA_SPLITK_GEMV.get()


def use_rdna_splitk_gemv(m: int, n: int, k: int, weight_dtype: torch.dtype) -> bool:
    """Whether a single-row bf16 GEMV should be routed to the split-K kernel."""
    if _is_disabled():
        return False
    if m != _SUPPORTED_ROWS or weight_dtype != torch.bfloat16:
        return False
    if n % _BLOCK_N != 0 or k % _BLOCK_K != 0:
        return False
    if n // _BLOCK_N > _MAX_N_TILES:
        return False
    split_k = k // _BLOCK_K
    if split_k < 2 or split_k > _MAX_SPLIT_K:
        # split_k == 1 has no occupancy problem to fix; a very large split_k turns
        # the reduction into the bottleneck.
        return False
    return is_gfx1151_supported()


def rdna_splitk_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x[1, K] @ weight[N, K].T`` for bf16 weights, returning bf16 ``[1, N]``."""
    n, k = weight.shape
    split_k = k // _BLOCK_K
    # The kernel has no tail mask: it reads one contiguous K-row per weight row and
    # writes whole n-tiles.
    assert (
        weight.is_contiguous()
        and x.numel() == k
        and n % _BLOCK_N == 0
        and k % _BLOCK_K == 0
    ), (weight.shape, weight.stride(), x.shape)
    x_flat = x.reshape(-1)
    partial = torch.empty((split_k, n), dtype=torch.float32, device=x.device)
    out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
    _gemv_split_k_kernel[(triton.cdiv(n, _BLOCK_N), split_k)](
        x_flat,
        weight,
        partial,
        weight.stride(0),
        N=n,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=4,
    )
    _reduce_split_k_kernel[(triton.cdiv(n, _BLOCK_N),)](
        partial,
        out,
        N=n,
        SPLIT_K=split_k,
        BLOCK_N=_BLOCK_N,
        num_warps=4,
    )
    return out
