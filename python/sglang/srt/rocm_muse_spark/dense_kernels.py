"""BF16 split-K projections for Muse."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _dense_partial(
    X,
    W,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
    CACHE: tl.constexpr,
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
        b = tl.load(bp, cache_modifier=CACHE)
        acc = tl.dot(a, b, acc)
        ap += BK
        bp += BK
    tl.store(P + pk * M * N + m[:, None] * N + n[None, :], acc, m[:, None] < M)


@tr.jit
def _dense_reduce(P, Y, MN: tl.constexpr, SK: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    s = tl.arange(0, SK)
    p = tl.load(P + s[:, None] * MN + i[None, :], i[None, :] < MN, other=0)
    tl.store(Y + i, tl.sum(p, axis=0), i < MN)


def dense(x, weight, cfg):
    m, k = x.shape
    n = weight.shape[0]
    sk = cfg["sk"]
    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    p = y if sk == 1 else torch.empty((sk, m, n), device=x.device, dtype=torch.float32)
    _dense_partial[(n // cfg["bn"], tr.cdiv(m, 16), sk)](
        x,
        weight,
        p,
        m,
        n,
        k,
        16,
        cfg["bn"],
        cfg["bk"],
        sk,
        cfg.get("cache", ".cg"),
        num_warps=cfg.get("warps", 4),
        num_stages=cfg.get("stages", 2),
        matrix_instr_nonkdim=cfg.get("mfma", 16),
    )
    if sk > 1:
        _dense_reduce[(tr.cdiv(m * n, 256),)](p, y, m * n, sk, 256, num_warps=4)
    return y
