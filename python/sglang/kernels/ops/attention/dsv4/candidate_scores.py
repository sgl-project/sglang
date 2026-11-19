"""Index scores of each row over its candidate blocks only."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.candidate_table import CANDIDATE_BLOCK_SIZE
from sglang.srt.utils.device import is_cuda


@triton.jit
def _candidate_matmul(
    Q,
    K,
    B,
    L,
    S,
    LC,
    N,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    h = tl.arange(0, BH)
    d = tl.arange(0, BD)
    block = tl.load(B + row * BS0 + (n // BLOCK) * BS1, n < N, -1).to(tl.int64)
    pos = block * BLOCK + n % BLOCK
    length = tl.load(L + row)
    valid = (n < N) & (block >= 0) & (pos < length) & (pos < LC)
    q = tl.load(
        Q + row * QS0 + h[:, None] * QS1 + d[None, :] * QS2,
        (h[:, None] < H) & (d[None, :] < D),
        0,
    )
    k = tl.load(
        K + pos[:, None] * KS0 + d[None, :] * KS1, valid[:, None] & (d[None, :] < D), 0
    )
    score = tl.dot(q, tl.trans(k)).to(tl.bfloat16)
    tl.store(
        S + row * H * N + h[:, None] * N + n[None, :],
        score,
        (h[:, None] < H) & (n[None, :] < N),
    )


@triton.jit
def _mask_scores(
    S,
    B,
    L,
    LC,
    N,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    BLOCK: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    block = tl.load(B + row * BS0 + (n // BLOCK) * BS1, n < N, -1).to(tl.int64)
    pos = block * BLOCK + n % BLOCK
    length = tl.load(L + row)
    valid = (block >= 0) & (pos < length) & (pos < LC)
    value = tl.load(S + row * N + n, n < N, 0)
    tl.store(S + row * N + n, tl.where(valid, value, float("-inf")), n < N)


def can_use_candidate_scores(q, k, weights, candidate_width: int, lc: int) -> bool:
    """Whether candidate scoring supports these operands and reduces the score width."""
    return (
        candidate_width > 0
        and candidate_width * 2 <= lc
        and q.is_cuda
        and is_cuda()
        and q.dtype == k.dtype == weights.dtype == torch.bfloat16
        and 0 < q.shape[1] <= 64
        and q.shape[2] == 128
    )


def candidate_scores(q, k, weights, blocks, lens, block_size=CANDIDATE_BLOCK_SIZE):
    rows, heads, dim = q.shape
    width = blocks.shape[1] * block_size
    assert q.dtype == k.dtype == weights.dtype == torch.bfloat16
    assert blocks.shape[0] == lens.shape[0] == rows and k.shape[1] == dim
    per_head = torch.empty((rows, heads, width), dtype=torch.bfloat16, device=q.device)
    if rows and width:
        _candidate_matmul[(rows, triton.cdiv(width, 64))](
            q,
            k,
            blocks,
            lens,
            per_head,
            k.shape[0],
            width,
            *q.stride(),
            *k.stride(),
            *blocks.stride(),
            heads,
            dim,
            block_size,
            max(16, triton.next_power_of_2(heads)),
            triton.next_power_of_2(dim),
            64,
            num_warps=4,
        )
    scores = (per_head.relu() * weights.unsqueeze(-1)).sum(dim=1).float()
    if rows and width:
        _mask_scores[(rows, triton.cdiv(width, 256))](
            scores,
            blocks,
            lens,
            k.shape[0],
            width,
            *blocks.stride(),
            block_size,
            256,
            num_warps=4,
        )
    return scores
