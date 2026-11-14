"""BF16 split-K GEMM partials for small M: P[s] = X @ W[:, K-slice s].T in FP32.

``W`` is ``[N, K]`` BF16 (an ``nn.Linear`` weight); every program owns a ``BM x BN`` output tile and one of
``SK`` K slices, so the partials ``[SK, M, N]`` FP32 are summed by the consumer's epilogue (the Muse Spark
QKV/QK-norm/RoPE/cache, unpool/boundary-norm and router/pooler kernels), which also fixes the BF16 rounding
point of the projection. Tuned on MI355X for decode / verify widths (M <= 64).
"""

import torch
import triton
import triton.language as tl


@triton.jit
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


def bf16_splitk_gemm_partials(
    x: torch.Tensor,
    weight: torch.Tensor,
    partials: torch.Tensor,
    *,
    block_n: int,
    block_k: int,
    split_k: int,
    cache: str = ".cg",
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Write ``partials[s] = x @ weight[:, s * K / split_k : (s + 1) * K / split_k].T`` (FP32, [split_k, M, N])."""
    m, k = x.shape
    n = weight.shape[0]
    _dense_partial[(n // block_n, triton.cdiv(m, 16), split_k)](
        x,
        weight,
        partials,
        m,
        n,
        k,
        16,
        block_n,
        block_k,
        split_k,
        cache,
        num_warps=num_warps,
        num_stages=num_stages,
        matrix_instr_nonkdim=16,
    )
    return partials
