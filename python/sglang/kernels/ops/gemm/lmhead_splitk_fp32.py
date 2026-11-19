"""FP32-output LM head for small M on gfx1250: logits[M, N] = X_bf16 @ W_bf16.T with FP32 accumulate.

hipBLASLt picks a large-tile kernel for the Muse TP8 head (M <= 128, N 25256, K 8192, out_dtype fp32) that is
far from bandwidth-bound at decode widths. This split-K Triton GEMM masks N (25256 has no power-of-two tile), writes FP32 partials and
sums them in a fixed order; it differs from the hipBLASLt FP32 output only in the K summation order.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _lmhead_partial(
    X,
    W,
    P,
    M,
    N,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
):
    pn, pm, pk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    k = pk * (K // SK) + tl.arange(0, BK)
    ap = X + m[:, None] * K + k[None, :]
    bp = W + n[None, :] * K + k[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(K // SK // BK):
        a = tl.load(ap, m[:, None] < M, other=0)
        b = tl.load(bp, n[None, :] < N, other=0.0, cache_modifier=".cg")
        acc = tl.dot(a, b, acc)
        ap += BK
        bp += BK
    tl.store(
        P + pk * M * N + m[:, None] * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _lmhead_sum(P, Y, MN, SK: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < MN
    acc = tl.zeros((BLOCK,), tl.float32)
    for s in tl.static_range(SK):
        acc += tl.load(P + s * MN + i, mask=mask, other=0.0)
    tl.store(Y + i, acc, mask=mask)


# (BM-bucket) -> (BN, BK, SK, warps, stages); tuned on gfx1250 at the TP8 head shape.
CONFIGS = {
    16: (64, 128, 1, 4, 3),
    32: (64, 64, 1, 4, 3),
    64: (64, 128, 1, 4, 3),
    128: (64, 128, 1, 4, 3, 128),
}
# [25256, 8192] per TP8 rank: a 128-row tile reads the weight once at 128 rows; there the head is BF16-compute-bound.
MAX_M = 128
# 129..512 rows (wide verify), FP32 out: aiter's FlyDSL gfx1250 a16w16 GEMM. For [25256, 8192] per TP8 rank it is
# far faster than hipBLASLt's FP32-out pick at 256 / 496 rows; the split-K Triton head is slower there. 272-column
# tiles cover a row tile in one wave of workgroups.
WIDE_MAX_M = 512
_WIDE_CONFIG = dict(
    tile_m=256,
    tile_n=272,
    tile_k=32,
    m_warp=4,
    n_warp=1,
    num_buffers=4,
    variant="bandwidth_bound",
)
# 65..128 rows without an explicit launch: the same FlyDSL GEMM with 128 x 192 tiles beats the split-K Triton head.
_MID_CONFIG = dict(
    tile_m=128,
    tile_n=192,
    tile_k=64,
    m_warp=1,
    n_warp=4,
    num_buffers=3,
    variant="compute_bound",
)


def _bm(m: int) -> int:
    return max(16, 1 << (m - 1).bit_length())


def applies(x: torch.Tensor, w: torch.Tensor) -> bool:
    return (
        x.dim() == 2
        and 0 < x.shape[0] <= MAX_M
        and x.dtype == w.dtype == torch.bfloat16
        and x.is_contiguous()
        and w.stride(1) == 1
        and w.stride(0) == w.shape[1]
        and x.shape[1] == w.shape[1]
        and _bm(x.shape[0]) in CONFIGS
        and w.shape[1] % (CONFIGS[_bm(x.shape[0])][1] * CONFIGS[_bm(x.shape[0])][2])
        == 0
    )


def wide_applies(x: torch.Tensor, w: torch.Tensor) -> bool:
    """Rows above ``MAX_M`` that ``lmhead_fp32`` serves with an FP32 output (BF16 operands only)."""
    return (
        x.dim() == 2
        and MAX_M < x.shape[0] <= WIDE_MAX_M
        and x.dtype == w.dtype == torch.bfloat16
        and x.is_contiguous()
        and w.stride(1) == 1
        and x.shape[1] == w.shape[1]
    )


def lmhead_fp32(
    x: torch.Tensor, w: torch.Tensor, cfg=None, out_dtype=torch.float32
) -> torch.Tensor:
    """``x @ w.T`` with FP32 accumulate, stored as ``out_dtype`` (FP32 for the head, BF16 for the DFlash candidates)."""
    m, k = x.shape
    n = w.shape[0]
    if m > MAX_M:
        assert out_dtype == torch.float32 and wide_applies(x, w)
        from aiter.ops.flydsl.gemm_a16w16_gfx1250 import gemm_a16w16

        return gemm_a16w16(x, w, dtype=torch.float32, **_WIDE_CONFIG)
    if cfg is None and m > 64:
        from aiter.ops.flydsl.gemm_a16w16_gfx1250 import gemm_a16w16

        return gemm_a16w16(x, w, dtype=out_dtype, **_MID_CONFIG)
    cfg = cfg or CONFIGS[_bm(m)]
    bn, bk, sk, warps, stages = cfg[:5]
    # Row tile: optional 6th entry; 64 by default (larger M then re-reads the weight once per 64-row tile).
    bm = cfg[5] if len(cfg) > 5 else min(_bm(m), 64)
    if sk == 1:
        y = torch.empty((m, n), device=x.device, dtype=out_dtype)
        _lmhead_partial[(triton.cdiv(n, bn), triton.cdiv(m, bm), 1)](
            x, w, y, m, n, k, bm, bn, bk, 1, num_warps=warps, num_stages=stages
        )
        return y
    p = torch.empty((sk, m, n), device=x.device, dtype=torch.float32)
    _lmhead_partial[(triton.cdiv(n, bn), triton.cdiv(m, bm), sk)](
        x, w, p, m, n, k, bm, bn, bk, sk, num_warps=warps, num_stages=stages
    )
    y = torch.empty((m, n), device=x.device, dtype=out_dtype)
    _lmhead_sum[(triton.cdiv(m * n, 1024),)](p, y, m * n, sk, 1024, num_warps=4)
    return y
