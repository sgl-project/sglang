"""FP8 small-M GEMMs on gfx950 (Muse Spark TP4 decode / verify widths).

``W [N, K]`` is quantized once at load to FP8 e4m3 with one FP32 scale per output channel; the activation rows
arrive already quantized (per-row scale) by the kernel that produced them, so the GEMM streams half the weight
bytes of the BF16 row-major kernel on the same split-K tiling and nothing else changes. The sum kernel applies
both scales. Quantizing the activation inside the GEMM instead (per K tile, or one amax pass per program) is
VALU-bound at these widths and slower than BF16 (30 / 26 us vs 23 at [4608, 8192] x 64), hence the producer-side
quantization. Measured on MI355X with HBM-resident weights: QKV [4608, 8192] 23.4 -> 14.4 us at 64 rows (27.8 ->
17.6 at 96), unpooler [8192, 4096] 16.5 -> 11.7, pooler [4096, 8192] 17.6 -> 11.6, O [8192, 2048] 9.6 -> 8.5.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.bf16_tiled_small_m_gemm import _bm

F8 = torch.float8_e4m3fn
F8_MAX = 448.0
MIN_M = 17
MAX_M = 128
# (N, K, BM) -> (BN, BK, SK, warps, stages), tuned on MI355X (bench_fp8_rowmajor.py / bench_fp8w_v2.py).
CONFIGS = {
    (4608, 8192, 32): (128, 256, 4, 8, 3),
    (4608, 8192, 64): (128, 256, 4, 8, 3),
    (4608, 8192, 128): (64, 256, 2, 8, 3),
    (8192, 2048, 32): (32, 512, 1, 4, 3),
    (8192, 2048, 64): (32, 512, 1, 4, 3),
    (8192, 2048, 128): (32, 256, 1, 8, 3),
    (4096, 8192, 32): (64, 256, 4, 8, 3),
    (4096, 8192, 64): (64, 256, 4, 8, 3),
    (4096, 8192, 128): (64, 256, 4, 8, 3),
    (8192, 4096, 32): (64, 256, 2, 8, 3),
    (8192, 4096, 64): (64, 256, 2, 8, 3),
    (8192, 4096, 128): (64, 256, 2, 8, 3),
    # TP8 QKV (tune_tp8_shapes.py, 10-23): 8.7 / 10.6 / 13.2 us at 32 / 64 / 128 rows
    (2304, 8192, 32): (64, 256, 4, 8, 3),
    (2304, 8192, 64): (64, 256, 4, 8, 3),
    (2304, 8192, 128): (64, 256, 4, 4, 3),
}


@triton.jit
def _fp8_rowmajor_partial(
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
    tl.store(P + pk * M * N + m[:, None] * N + n[None, :], acc, mask=m[:, None] < M)


@triton.jit
def _sum_scaled(P, XS, WS, Y, M, N: tl.constexpr, SK: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    ok = i < M * N
    s = tl.arange(0, SK)
    p = tl.load(P + s[:, None] * M * N + i[None, :], mask=ok[None, :], other=0.0)
    y = (
        tl.sum(p, axis=0)
        * tl.load(XS + i // N, mask=ok, other=0.0)
        * tl.load(WS + i % N, mask=ok, other=0.0)
    )
    tl.store(Y + i, y.to(Y.dtype.element_ty), mask=ok)


@torch.no_grad()
def quantize_weight(weight: torch.Tensor):
    """(FP8 e4m3 copy [N, K], FP32 per-output-channel scales [N]) of a BF16 weight, or None for other dtypes; the
    copy also serves ``fp8_prefill_gemm`` (any shape), ``fp8_gemm_applies`` keeps the small-M kernel to its tuned shapes."""
    if weight.dtype != torch.bfloat16 or weight.dim() != 2:
        return None
    scale = weight.float().abs().amax(1).clamp_min(1e-12) / F8_MAX
    return (weight.float() / scale[:, None]).to(F8).contiguous(), scale.contiguous()


def quantize_rows(x: torch.Tensor):
    """Per-row FP8 e4m3 of ``x [M, K]`` and its FP32 scales [M] (torch; the fused producers do this in-kernel)."""
    scale = x.float().abs().amax(1).clamp_min(1e-12) / F8_MAX
    return (x.float() / scale[:, None]).to(F8), scale


def fp8_gemm_applies(xq: torch.Tensor, fp8_weight) -> bool:
    """Whether ``fp8_gemm(xq, ...)`` covers these operands (tuned launch for this row count)."""
    if fp8_weight is None or xq.dim() != 2 or not xq.is_contiguous() or xq.dtype != F8:
        return False
    if not MIN_M <= xq.shape[0] <= MAX_M:
        return False
    wq = fp8_weight[0]
    return (*wq.shape, _bm(xq.shape[0])) in CONFIGS and wq.shape[1] == xq.shape[1]


def fp8_gemm(xq: torch.Tensor, xs: torch.Tensor, fp8_weight) -> torch.Tensor:
    """``(xq * xs) @ (wq * ws).T`` in BF16 from the per-row-quantized activation and ``quantize_weight(W)``."""
    wq, ws = fp8_weight
    m, k = xq.shape
    n = wq.shape[0]
    bn, bk, sk, warps, stages = CONFIGS[(n, k, _bm(m))]
    p = torch.empty((sk, m, n), device=xq.device, dtype=torch.float32)
    _fp8_rowmajor_partial[(n // bn, sk)](
        xq,
        wq,
        p,
        m,
        n,
        k,
        _bm(m),
        bn,
        bk,
        sk,
        num_warps=warps,
        num_stages=stages,
        matrix_instr_nonkdim=16,
    )
    y = torch.empty((m, n), device=xq.device, dtype=torch.bfloat16)
    _sum_scaled[(triton.cdiv(m * n, 512),)](p, xs, ws, y, m, n, sk, 512, num_warps=4)
    return y
