"""BF16 GEMMs at 17..64 rows from a tile-contiguous weight copy (gfx950).

``W [N, K]`` is copied once at load into ``[N/BN, K/BK, BN, BK]`` so each program streams one contiguous slab
(its BN columns over its share of K); split-K partials are FP32 and summed by a second kernel. Same products
and FP32 accumulation as a plain GEMM; only the K summation order differs.

``LAYOUT`` / ``LAUNCH`` are the MI355X tuning data (Muse Spark TP8 shapes), kept only where the tiled kernel
beat the incumbent path at that width with HBM-resident weights: the MoE pooler at 17-64 rows (vs hipBLASLt:
16.8 vs 20.6 us at 32, 20.4 vs 27.5 at 64), the attention O projection at 33-64 (6.5 vs 8.8), the DFlash draft
gate/up and down at 17-64 (24.5 vs 26.2 / 29.8 vs 33.1, 14.4 vs 18.3 / 16.9 vs 19.9).
"""

import torch
import triton
import triton.language as tl

MIN_M = 17
MAX_M = 64

# (N, K) -> (BN, BK) of the stored copy; (N, K, BM) -> (SK, warps).
LAYOUT = {
    (4096, 8192): (32, 128),
    (8192, 1024): (32, 256),
    (7168, 8192): (64, 128),
    (8192, 3584): (32, 256),
}
LAUNCH = {
    (4096, 8192, 32): (4, 4),
    (4096, 8192, 64): (4, 4),
    (8192, 1024, 64): (1, 8),
    (7168, 8192, 32): (4, 8),
    (7168, 8192, 64): (4, 8),
    (8192, 3584, 32): (2, 4),
    (8192, 3584, 64): (1, 4),
}


@triton.jit
def _tiled_partial(
    X,
    Wt,
    P,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
):
    pn, pk = tl.program_id(0), tl.program_id(1)
    KT: tl.constexpr = K // BK
    KT_PER: tl.constexpr = KT // SK
    m = tl.arange(0, BM)
    kk = tl.arange(0, BK)
    nn = tl.arange(0, BN)
    bp = (
        Wt
        + (pn * KT + pk * KT_PER).to(tl.int64) * (BN * BK)
        + nn[None, :] * BK
        + kk[:, None]
    )
    ap = X + m[:, None] * K + (pk * KT_PER * BK + kk)[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(KT_PER):
        a = tl.load(ap, mask=m[:, None] < M, other=0.0)
        b = tl.load(bp, cache_modifier=".cg")
        acc = tl.dot(a, b, acc)
        ap += BK
        bp += BN * BK
    n = pn * BN + nn
    tl.store(
        P + pk * M * N + m[:, None] * N + n[None, :],
        acc.to(P.dtype.element_ty),
        mask=m[:, None] < M,
    )


@triton.jit
def _sum_partials(P, Y, MN, SK: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    s = tl.arange(0, SK)
    p = tl.load(P + s[:, None] * MN + i[None, :], mask=i[None, :] < MN, other=0.0)
    tl.store(Y + i, tl.sum(p, axis=0).to(Y.dtype.element_ty), mask=i < MN)


@torch.no_grad()
def prepare_tiled_weight(weight: torch.Tensor):
    """Tile-contiguous copy of a [N, K] BF16 weight, or None when the shape has no tuned layout."""
    n, k = weight.shape
    layout = LAYOUT.get((n, k))
    if layout is None or weight.dtype != torch.bfloat16:
        return None
    bn, bk = layout
    return weight.view(n // bn, bn, k // bk, bk).permute(0, 2, 1, 3).contiguous()


def _bm(m: int) -> int:
    return max(16, 1 << (m - 1).bit_length())


def tiled_gemm_applies(x: torch.Tensor, tiled) -> bool:
    """Whether ``tiled_gemm(x, tiled, ...)`` covers these operands (tuned launch for this row count)."""
    if (
        tiled is None
        or x.dim() != 2
        or not x.is_contiguous()
        or not MIN_M <= x.shape[0] <= MAX_M
    ):
        return False
    n = tiled.shape[0] * tiled.shape[2]
    k = tiled.shape[1] * tiled.shape[3]
    return (n, k, _bm(x.shape[0])) in LAUNCH


def tiled_gemm(x: torch.Tensor, tiled: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """``x @ W.T`` (BF16 out) from the tile-contiguous copy of ``W [n, k]``."""
    m = x.shape[0]
    bm = _bm(m)
    bn, bk = LAYOUT[(n, k)]
    sk, warps = LAUNCH[(n, k, bm)]
    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    p = y if sk == 1 else torch.empty((sk, m, n), device=x.device, dtype=torch.float32)
    _tiled_partial[(n // bn, sk)](
        x,
        tiled,
        p,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        sk,
        num_warps=warps,
        num_stages=2,
        matrix_instr_nonkdim=16,
    )
    if sk > 1:
        _sum_partials[(triton.cdiv(m * n, 256),)](p, y, m * n, sk, 256, num_warps=4)
    return y
