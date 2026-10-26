"""BF16 GEMMs at 17..128 rows (row-major: from 1 row where tuned) on gfx950, split-K over programs that each
stream BN weight rows.

Tiled: ``W [N, K]`` is copied once at load into ``[N/BN, K/BK, BN, BK]`` so each program streams one contiguous
slab. Row-major: the same programs read the stored ``W`` (BN rows of BK contiguous elements per step), no copy.
Split-K partials are FP32 and summed by a second kernel. Same products and FP32 accumulation as a plain GEMM;
only the K summation order differs.

``LAYOUT`` / ``LAUNCH`` are the MI355X tuning data of the tiled path (Muse Spark TP8 shapes), kept only where it
beat the incumbent path at that width with HBM-resident weights: the attention O projection at 33-64 (6.5 vs
8.8 us), the DFlash draft gate/up and down at 17-64 (24.5 vs 26.2 / 29.8 vs 33.1, 14.4 vs 18.3 / 16.9 vs 19.9).
``ROW_MAJOR`` is the row-major path's (measured on MI355X, HBM-resident weights, vs hipBLASLt / the tiled copy):
the MoE pooler [4096, 8192] (15.6 vs 16.2 tiled at 32 rows, 17.8 vs 19.8 at 64, 21.7 vs 22.6 hipBLASLt at 96),
the MoE unpooler [8192, 4096] (16.6 vs 19.7 at 64, 20.7 vs 26.3 at 96), the TP4 attention O projection
[8192, 2048] (9.6 vs 12.3 at 64, 12.2 vs 17.6 at 96) and the joint pooler + split router [4608, 8192] at 64 rows
(23.4 us with the sum; one GEMM in place of the pooler and the FP32 router GEMV).
"""

import torch
import triton
import triton.language as tl

MIN_M = 17
MAX_M = 64

# (N, K) -> (BN, BK) of the stored copy; (N, K, BM) -> (SK, warps).
LAYOUT = {
    (8192, 1024): (32, 256),
    (7168, 8192): (64, 128),
    (8192, 3584): (32, 256),
}
LAUNCH = {
    (8192, 1024, 64): (1, 8),
    (7168, 8192, 32): (4, 8),
    (7168, 8192, 64): (4, 8),
    (8192, 3584, 32): (2, 4),
    (8192, 3584, 64): (1, 4),
}
ROW_MAX_M = 128
# (N, K, BM) -> (BN, BK, SK, warps, stages) of the row-major path.
ROW_MAJOR = {
    (4096, 8192, 32): (64, 128, 8, 8, 2),
    (4096, 8192, 64): (64, 128, 8, 8, 2),
    (4096, 8192, 128): (64, 128, 4, 8, 3),
    (4608, 8192, 64): (128, 128, 4, 4, 3),
    (8192, 4096, 64): (32, 256, 1, 4, 3),
    (8192, 4096, 128): (64, 128, 2, 8, 3),
    (8192, 2048, 64): (32, 256, 1, 4, 3),
    (8192, 2048, 128): (32, 128, 1, 8, 3),
    # TP8 O-proj (tune_tp8_shapes.py, 10-23)
    (8192, 1024, 32): (32, 256, 1, 8, 3),
    (8192, 1024, 64): (32, 256, 1, 4, 3),
    (8192, 1024, 128): (32, 128, 1, 4, 3),
    # TP4 DFlash draft (10-24, 1.5 GB of rotating weights per replay, vs aiter tgemm / hipBLASLt):
    # conv kernel projection [2048, 8192] (9.4 vs 12.8 at 16, 9.9 vs 12.8 at 32, 11.8 vs 18.3 at 64, 14.7 vs 20.5
    # at 128, with the sum),
    (2048, 8192, 16): (64, 128, 8, 4, 3),
    (2048, 8192, 32): (64, 128, 8, 4, 3),
    (2048, 8192, 64): (32, 256, 4, 4, 3),
    (2048, 8192, 128): (64, 128, 8, 4, 3),
    # QKV [2560, 8192] (12.6 vs 14.6 at 32, 15.4 vs 15.8 at 64, 20.1 vs 21.1 at 128),
    (2560, 8192, 32): (128, 128, 8, 8, 3),
    (2560, 8192, 64): (128, 128, 8, 8, 3),
    (2560, 8192, 128): (64, 128, 4, 4, 3),
    # MLP down [8192, 7168] (20.0 vs 25.0 at 8, 20.0 vs 23.3 at 16, 21.7 vs 23.3 at 32, 25.8 vs 30.0 at 64,
    # 35.2 vs 36.8 hipBLASLt at 128),
    (8192, 7168, 16): (32, 256, 1, 4, 3),
    (8192, 7168, 32): (32, 256, 1, 4, 3),
    (8192, 7168, 64): (32, 256, 1, 4, 3),
    (8192, 7168, 128): (128, 64, 4, 4, 3),
    # MLP gate/up [14336, 8192] up to 32 rows (40.4 vs 59.3 at 8, 40.4 vs 45.6 at 16, 40.1 vs 46.2 at 32; wider,
    # hipBLASLt is faster).
    (14336, 8192, 16): (64, 256, 1, 4, 3),
    (14336, 8192, 32): (64, 256, 1, 8, 3),
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
def _rowmajor_partial(
    X,
    W,
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
    KT_PER: tl.constexpr = K // BK // SK
    m = tl.arange(0, BM)
    kk = tl.arange(0, BK)
    n = pn * BN + tl.arange(0, BN)
    bp = W + n[None, :].to(tl.int64) * K + (pk * KT_PER * BK + kk)[:, None]
    ap = X + m[:, None] * K + (pk * KT_PER * BK + kk)[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(KT_PER):
        a = tl.load(ap, mask=m[:, None] < M, other=0.0)
        b = tl.load(bp, cache_modifier=".cg")
        acc = tl.dot(a, b, acc)
        ap += BK
        bp += BK
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


def rowmajor_gemm_applies(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether ``rowmajor_gemm(x, weight)`` covers these operands (tuned launch for this shape and row count)."""
    return (
        x.dim() == 2
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.dtype == weight.dtype == torch.bfloat16
        and 0 < x.shape[0] <= ROW_MAX_M
        and (*weight.shape, _bm(x.shape[0])) in ROW_MAJOR
        and weight.shape[1] == x.shape[1]
    )


def rowmajor_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ W.T`` (BF16 out) reading the stored row-major ``W [n, k]``."""
    p = rowmajor_gemm_partials(x, weight)
    if p.dtype == torch.bfloat16:
        return p
    m, n = p.shape[1:]
    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    _sum_partials[(triton.cdiv(m * n, 256),)](p, y, m * n, p.shape[0], 256, num_warps=4)
    return y


def rowmajor_gemm_partials_split(m: int, n: int, k: int) -> int:
    """K partitions (SK) of the tuned row-major launch for ``m`` rows of an ``[n, k]`` weight (one if untuned)."""
    cfg = ROW_MAJOR.get((n, k, _bm(m)))
    return cfg[2] if cfg else 1


def rowmajor_gemm_partials(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """FP32 split-K partials ``[SK, m, n]`` of ``x @ W.T`` for a consumer that sums them itself, or the BF16
    product ``[m, n]`` when the tuned launch does not split K."""
    m, k = x.shape
    n = weight.shape[0]
    bm = _bm(m)
    bn, bk, sk, warps, stages = ROW_MAJOR[(n, k, bm)]
    p = (
        torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        if sk == 1
        else torch.empty((sk, m, n), device=x.device, dtype=torch.float32)
    )
    _rowmajor_partial[(n // bn, sk)](
        x,
        weight,
        p,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        sk,
        num_warps=warps,
        num_stages=stages,
        matrix_instr_nonkdim=16,
    )
    return p
