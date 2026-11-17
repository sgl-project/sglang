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
    # the joint pooler + split router at 65-128 rows (SGLANG_ROCM_MUSE_JOINT_ROUTER_MAX_ROWS)
    (4608, 8192, 128): (64, 128, 4, 8, 3),
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
# (N, K, BM) -> per-N-block K-start rotation stride (coprime to the per-split K-tile count, 16 or 32 here) of the
# long-K launches, so concurrent workgroups stream different K tiles; 0 or absent: the canonical in-order K loop.
# Measured per bucket over its M range: KT_PER <= 8 launches and every [8192, 7168] bucket lose with rotation.
ROW_MAJOR_ROT = {
    (8192, 4096, 64): 7,
    (8192, 4096, 128): 5,
    (8192, 2048, 128): 5,
    (4096, 8192, 128): 5,
    (4608, 8192, 64): 3,
    (4608, 8192, 128): 3,
    (2560, 8192, 128): 3,
    (14336, 8192, 16): 7,
    (14336, 8192, 32): 7,
}


# gfx1250 (wave32): row-major launches tuned at the Muse TP8 decode / verify shapes (M 8 / 32 / 64 / 128).
_GFX1250_ROW_MAJOR = {
    (8192, 4096, 16): (64, 256, 1, 4, 3),
    (8192, 4096, 32): (64, 256, 1, 4, 3),
    (8192, 4096, 64): (128, 256, 1, 4, 3),
    (8192, 4096, 128): (128, 128, 1, 4, 3),
    (4096, 8192, 16): (64, 128, 4, 4, 3),
    (4096, 8192, 32): (64, 128, 4, 4, 3),
    (4096, 8192, 64): (64, 128, 4, 4, 3),
    (4096, 8192, 128): (64, 128, 4, 4, 2),
    (4608, 8192, 16): (64, 128, 4, 4, 3),
    (4608, 8192, 32): (64, 128, 4, 4, 3),
    (4608, 8192, 64): (64, 128, 4, 4, 3),
    (4608, 8192, 128): (64, 128, 4, 4, 2),
    (8192, 1024, 16): (128, 256, 1, 8, 3),
    (8192, 1024, 32): (128, 256, 1, 4, 3),
    (8192, 1024, 64): (128, 256, 1, 4, 3),
    (8192, 1024, 128): (128, 128, 1, 4, 3),
    # DFlash draft shapes (attention [2560, 8192], down [8192, 7168], gate_up [14336, 8192]); the down projection
    # at 128 rows stays on hipBLASLt.
    (2560, 8192, 16): (64, 128, 8, 4, 3),
    (2560, 8192, 32): (64, 256, 4, 8, 3),
    (2560, 8192, 64): (128, 256, 4, 4, 3),
    (2560, 8192, 128): (128, 256, 4, 4, 2),
    (8192, 7168, 16): (64, 256, 1, 4, 3),
    (8192, 7168, 32): (32, 256, 1, 4, 3),
    (8192, 7168, 64): (128, 256, 1, 4, 3),
    (14336, 8192, 16): (64, 128, 1, 4, 3),
    (14336, 8192, 32): (64, 128, 1, 4, 3),
    (14336, 8192, 64): (64, 128, 1, 4, 3),
    (14336, 8192, 128): (256, 128, 1, 4, 3),
    # DFlash draft context projection at TP8, faster than hipBLASLt at M 8 / 64 / 128: stacked K/V of the 5 layers
    # [1280, 8192] (also against hipBLASLt on the transposed copy) and this rank's fc slice [1024, 40960] (32 rows take
    # the 64-row launch).
    (1280, 8192, 16): (128, 256, 16, 8, 2),
    (1280, 8192, 32): (64, 128, 8, 4, 3),
    (1280, 8192, 64): (64, 128, 8, 4, 3),
    (1280, 8192, 128): (64, 256, 4, 4, 3),
    (1024, 40960, 16): (64, 128, 16, 4, 3),
    (1024, 40960, 32): (64, 128, 16, 4, 3),
    (1024, 40960, 64): (64, 128, 16, 4, 3),
}
# gfx1250: aiter's FlyDSL a16w16 GEMM (TDM pipeline, split-K summed in-kernel in a fixed order) where it beats the
# incumbent, (N, K, BM) -> (out dtype, launch): the joint router / pooler [4608, 8192] with an FP32 output at 16 rows
# (one split for the epilogue to sum instead of eight), the unpooler [8192, 4096] at 16 and 64 rows, and the DFlash
# draft gate/up [7168, 8192] and down [8192, 3584] at 65..128 rows (vs aiter tgemm).
FLYDSL: dict = {}
_GFX1250_FLYDSL = {
    (4608, 8192, 16): (
        torch.float32,
        dict(tile_m=16, tile_n=64, tile_k=256, m_warp=1, n_warp=2, num_buffers=6),
    ),
    (8192, 4096, 16): (
        torch.bfloat16,
        dict(tile_m=16, tile_n=64, tile_k=128, m_warp=1, n_warp=4, num_buffers=6),
    ),
    (8192, 4096, 64): (
        torch.bfloat16,
        dict(
            tile_m=64,
            tile_n=64,
            tile_k=128,
            m_warp=2,
            n_warp=2,
            num_buffers=3,
            split_k=2,
        ),
    ),
    (7168, 8192, 128): (
        torch.bfloat16,
        dict(tile_m=64, tile_n=160, tile_k=128, m_warp=1, n_warp=2, num_buffers=4),
    ),
    (8192, 3584, 128): (
        torch.bfloat16,
        dict(
            tile_m=64,
            tile_n=192,
            tile_k=128,
            m_warp=4,
            n_warp=2,
            num_buffers=3,
            variant="compute_bound",
        ),
    ),
}
try:
    if (
        torch.cuda.is_available()
        and "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    ):
        ROW_MAJOR.update(_GFX1250_ROW_MAJOR)
        FLYDSL.update(_GFX1250_FLYDSL)
except Exception:
    pass


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
    ROT: tl.constexpr,
):
    pn, pk = tl.program_id(0), tl.program_id(1)
    KT_PER: tl.constexpr = K // BK // SK
    m = tl.arange(0, BM)
    kk = tl.arange(0, BK)
    n = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    if ROT > 0:
        # Long K loops: rotate each N-block's K start so the workgroups do not all stream the same K tile at once.
        start = (pn * ROT) % KT_PER
        for i in range(KT_PER):
            kt = (start + i) % KT_PER
            k0 = (pk * KT_PER + kt) * BK
            a = tl.load(
                X + m[:, None] * K + (k0 + kk)[None, :], mask=m[:, None] < M, other=0.0
            )
            b = tl.load(
                W + n[None, :].to(tl.int64) * K + (k0 + kk)[:, None],
                cache_modifier=".cg",
            )
            acc = tl.dot(a, b, acc)
    else:
        bp = W + n[None, :].to(tl.int64) * K + (pk * KT_PER * BK + kk)[:, None]
        ap = X + m[:, None] * K + (pk * KT_PER * BK + kk)[None, :]
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
    # Flat 1-D reduction: each thread owns contiguous elements, all SK partial loads are independent and in flight.
    i = tl.program_id(0) * B + tl.arange(0, B)
    msk = i < MN
    acc = tl.load(P + i, mask=msk, other=0.0)
    for s in tl.static_range(1, SK):
        acc += tl.load(P + s * MN + i, mask=msk, other=0.0)
    tl.store(Y + i, acc.to(Y.dtype.element_ty), mask=msk)


def _sum_block(mn: int):
    """(elements per program, warps) of the partial sum: 4 FP32 per thread (one 16 B load per partial); one-wave
    programs below 64K elements so small products still spread over the CUs."""
    return (1024, 4) if mn >= 1 << 16 else (256, 1)


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


def flydsl_gemm_applies(
    x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype
) -> bool:
    """Whether ``flydsl_gemm(x, weight)`` has a tuned launch with this output dtype."""
    cfg = FLYDSL.get((*weight.shape, _bm(x.shape[0])))
    return (
        cfg is not None
        and cfg[0] == out_dtype
        and x.dim() == 2
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.dtype == weight.dtype == torch.bfloat16
        and weight.shape[1] == x.shape[1]
    )


def flydsl_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ W.T`` with FP32 accumulate on the tuned FlyDSL launch (``flydsl_gemm_applies``)."""
    from aiter.ops.flydsl.gemm_a16w16_gfx1250 import gemm_a16w16

    out_dtype, launch = FLYDSL[(*weight.shape, _bm(x.shape[0]))]
    return gemm_a16w16(x, weight, dtype=out_dtype, **launch)


def rowmajor_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ W.T`` (BF16 out) reading the stored row-major ``W [n, k]``."""
    if flydsl_gemm_applies(x, weight, torch.bfloat16):
        return flydsl_gemm(x, weight)
    p = rowmajor_gemm_partials(x, weight)
    if p.dtype == torch.bfloat16:
        return p
    m, n = p.shape[1:]
    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    b, w = _sum_block(m * n)
    _sum_partials[(triton.cdiv(m * n, b),)](p, y, m * n, p.shape[0], b, num_warps=w)
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
        ROW_MAJOR_ROT.get((n, k, bm), 0),
        num_warps=warps,
        num_stages=stages,
        matrix_instr_nonkdim=16,
    )
    return p
