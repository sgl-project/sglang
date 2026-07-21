"""Single-pass online row logsumexp.

Reads the input exactly once with fp32 accumulation and writes per-row
``(max, log_sum_exp - max)`` fp32 pairs. Keeping the two terms separate lets
callers compute ``logprob[i] = (logit[i] - max) - log_sum`` the same
shift-invariant way log-softmax does: folding them into one absolute
normalizer would round the small log-sum term away whenever rows carry a
large common offset.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _combine_lse(m_a, l_a, m_b, l_b):
    """Merge two (max, sum_exp) accumulators.

    The tl.where guards keep fully -inf inputs nan-free: exp(-inf - -inf)
    would otherwise poison the sum.
    """
    m_new = tl.maximum(m_a, m_b)
    l_new = l_a * tl.exp(tl.where(m_a == m_new, 0.0, m_a - m_new)) + l_b * tl.exp(
        tl.where(m_b == m_new, 0.0, m_b - m_new)
    )
    return m_new, l_new


@triton.jit(do_not_specialize=["num_rows"])
def _row_logsumexp_kernel(
    x_ptr,
    out_max_ptr,
    out_log_sum_ptr,
    row_stride,
    col_stride,
    num_rows,
    # int rather than tl.constexpr so triton does not unroll the loop
    num_cols,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if row >= num_rows:
        return
    row_ptr = x_ptr + row * row_stride

    m_i = float("-inf")
    l_i = 0.0
    for start in range(0, num_cols, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        x = tl.load(
            row_ptr + offs * col_stride, mask=offs < num_cols, other=float("-inf")
        ).to(tl.float32)
        # Per-block max first: a single dominant logit stays exact instead of
        # being renormalized against a stale running max.
        m_blk = tl.max(x)
        l_blk = tl.sum(tl.exp(x - m_blk))
        # +/-inf block max: exp(x - m_blk) is nan there; the true sum_exp
        # normalized by m_blk is 1.
        l_blk = tl.where((m_blk == float("inf")) | (m_blk == float("-inf")), 1.0, l_blk)
        m_i, l_i = _combine_lse(m_i, l_i, m_blk, l_blk)

    tl.store(out_max_ptr + row, m_i)
    tl.store(out_log_sum_ptr + row, tl.log(l_i))


def row_logsumexp(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row ``(max, logsumexp - max)`` of a 2D tensor, as fp32."""
    assert x.ndim == 2
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)

    num_rows, num_cols = x.shape
    out_max = torch.empty(num_rows, device=x.device, dtype=torch.float32)
    out_log_sum = torch.empty(num_rows, device=x.device, dtype=torch.float32)
    if num_rows == 0:
        return out_max, out_log_sum
    if num_cols == 0:
        return out_max.fill_(float("-inf")), out_log_sum.zero_()

    BLOCK_N = triton.next_power_of_2(min(num_cols, 16384))
    _row_logsumexp_kernel[(num_rows,)](
        x,
        out_max,
        out_log_sum,
        x.stride(0),
        x.stride(1),
        num_rows,
        num_cols,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    return out_max, out_log_sum
