"""Weight-streaming BF16 small-M GEMM partials on gfx950 (Muse Spark TP4 decode / verify widths up to 32 rows).

Every program streams one contiguous K slice of BN weight rows once for all rows of the activation (two
power-of-two dot tiles M1 + M2) and writes FP32 split-K partials ``[SK, M, N]`` that the consumer sums. BF16
operands and FP32 accumulation as in ``bf16_splitk_gemm``; only the order of the K summation differs.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl


class StreamConfig(NamedTuple):
    bn: int
    bk: int
    sk: int
    warps: int
    stages: int


# (N, K) -> ((max rows, launch), ...), first bucket that holds the rows; tuned on gfx950 with cold weights.
STREAM_LAUNCH = {
    (4608, 8192): (
        (24, StreamConfig(64, 128, 8, 4, 2)),
        (32, StreamConfig(32, 64, 8, 4, 2)),
    ),
    (8192, 4096): (
        (16, StreamConfig(64, 128, 4, 4, 2)),
        (24, StreamConfig(64, 256, 4, 4, 2)),
        (32, StreamConfig(64, 128, 4, 8, 2)),
    ),
}


def stream_launch(n: int, k: int, m: int) -> StreamConfig | None:
    """The tuned launch for ``m`` rows of an ``[n, k]`` BF16 weight, or None."""
    for rows, cfg in STREAM_LAUNCH.get((n, k), ()):
        if 0 < m <= rows:
            return cfg
    return None


def row_split(m: int) -> tuple[int, int]:
    """(M1, M2): the two power-of-two dot tiles covering ``m`` rows (M2 = 0 when one tile is exact)."""
    m1 = 1 << (max(m, 16).bit_length() - 1)
    if m1 >= m:
        return max(m1, 16), 0
    return m1, max(16, triton.next_power_of_2(m - m1))


@triton.jit
def _stream_partial(
    X,
    W,
    P,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    M1: tl.constexpr,
    M2: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
):
    pid = tl.program_id(0)
    G0: tl.constexpr = N // BN
    TOT: tl.constexpr = G0 * SK
    if TOT % 8 == 0:
        # Program ids are dealt round-robin over the 8 XCDs; give each XCD a contiguous range of programs.
        PER: tl.constexpr = TOT // 8
        lin = (pid % 8) * PER + pid // 8
    else:
        lin = pid
    pk = lin // G0
    pn = lin % G0
    KT: tl.constexpr = K // BK // SK
    kk = tl.arange(0, BK)
    nn = tl.arange(0, BN)
    k0 = pk * (KT * BK)
    bp = W + (pn * BN + nn)[None, :].to(tl.int64) * K + (k0 + kk)[:, None]
    m1 = tl.arange(0, M1)
    a1p = X + m1[:, None] * K + (k0 + kk)[None, :]
    acc1 = tl.zeros((M1, BN), tl.float32)
    if M2 > 0:
        m2 = M1 + tl.arange(0, M2)
        a2p = X + m2[:, None] * K + (k0 + kk)[None, :]
        acc2 = tl.zeros((M2, BN), tl.float32)
    for _ in range(KT):
        b = tl.load(bp, cache_modifier=".cs")
        a1 = tl.load(a1p, mask=m1[:, None] < M, other=0.0)
        acc1 = tl.dot(a1, b, acc1)
        a1p += BK
        if M2 > 0:
            a2 = tl.load(a2p, mask=m2[:, None] < M, other=0.0)
            acc2 = tl.dot(a2, b, acc2)
            a2p += BK
        bp += BK
    n = pn * BN + nn
    base = P + pk * M * N
    tl.store(base + m1[:, None] * N + n[None, :], acc1, mask=m1[:, None] < M)
    if M2 > 0:
        tl.store(base + m2[:, None] * N + n[None, :], acc2, mask=m2[:, None] < M)


def stream_gemm_partials(
    x: torch.Tensor, w: torch.Tensor, n: int, k: int, cfg: StreamConfig
) -> torch.Tensor:
    """FP32 split-K partials ``[SK, m, n]`` of ``x @ W.T``; ``w`` is the row-major ``[n, k]`` weight."""
    m = x.shape[0]
    m1, m2 = row_split(m)
    p = torch.empty((cfg.sk, m, n), device=x.device, dtype=torch.float32)
    _stream_partial[(n // cfg.bn * cfg.sk,)](
        x,
        w,
        p,
        m,
        n,
        k,
        m1,
        m2,
        cfg.bn,
        cfg.bk,
        cfg.sk,
        num_warps=cfg.warps,
        num_stages=cfg.stages,
        matrix_instr_nonkdim=16,
    )
    return p
