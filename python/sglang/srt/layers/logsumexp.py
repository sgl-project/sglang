"""Single-pass online row logsumexp, optionally fused with a small top-k.

Reads the input exactly once with fp32 accumulation and writes per-row
``(max, log_sum_exp - max)`` fp32 pairs. Keeping the two terms separate lets
callers compute ``logprob[i] = (logit[i] - max) - log_sum`` the same
shift-invariant way log-softmax does: folding them into one absolute
normalizer would round the small log-sum term away whenever rows carry a
large common offset.

``row_logsumexp_topk`` additionally maintains a running top-k (k <= 32) in
registers during the same pass, so callers that need both the normalizer and
the top-k tokens pay for a single read of the input.
"""

import torch
import triton
import triton.language as tl

# Maximum k for the fused top-k kernel; larger k should fall back to
# torch.topk. The per-block selection cost grows with k: measured on GB300
# at vocab=151936 the fused kernel beats a separate logsumexp + top-k up to
# k~6 and drops below plain torch.topk near k=32.
FUSED_TOPK_MAX_K = 8


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


@triton.jit
def _pack_key(vals, idxs):
    """Pack fp32 value + int32 index into one order-encoding int64 key.

    Key order == candidate order: higher value wins, value ties go to the
    LOWER index. The fp32 bits are mapped through the standard
    sign-flip transform so unsigned bit order matches float order, then
    placed above the complemented index; all keys land in [0, 2**63), so
    signed int64 comparison is the candidate comparison. The 2147483647
    (int32 max) index sentinel yields the smallest key for a given value,
    so "empty" (-inf, sentinel) candidates lose against everything,
    including genuine -inf lanes.
    """
    bits = vals.to(tl.int32, bitcast=True)
    sortable = bits ^ ((bits >> 31) | -2147483648)
    return ((sortable.to(tl.int64) & 0xFFFFFFFF) << 31) | (2147483647 - idxs).to(
        tl.int64
    )


@triton.jit
def _unpack_val(keys):
    sortable = ((keys >> 31) & 0xFFFFFFFF).to(tl.int32)
    bits = sortable ^ ((~sortable >> 31) | -2147483648)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _unpack_idx(keys):
    return 2147483647 - (keys & 0x7FFFFFFF).to(tl.int32)


@triton.jit(do_not_specialize=["num_rows"])
def _row_logsumexp_topk_kernel(
    x_ptr,
    out_max_ptr,
    out_log_sum_ptr,
    out_top_vals_ptr,
    out_top_idx_ptr,
    row_stride,
    col_stride,
    num_rows,
    # int rather than tl.constexpr so triton does not unroll the loop
    num_cols,
    K: tl.constexpr,
    # K rounded up to a power of 2: register tensors need pow2 shapes. The
    # padding slots hold (-inf, sentinel) and lose every comparison.
    K_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if row >= num_rows:
        return
    row_ptr = x_ptr + row * row_stride

    m_i = float("-inf")
    l_i = 0.0
    slots = tl.arange(0, K_PAD)
    # Running top-K as packed keys, sorted descending. Slots beyond the
    # genuine entries hold the (-inf, sentinel) key, which loses everything.
    run_keys = _pack_key(
        tl.full((K_PAD,), float("-inf"), tl.float32),
        tl.full((K_PAD,), 2147483647, tl.int32),
    )
    kth_key = tl.min(run_keys)

    for start in range(0, num_cols, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        x = tl.load(
            row_ptr + offs * col_stride, mask=offs < num_cols, other=float("-inf")
        ).to(tl.float32)

        # Same op sequence as _row_logsumexp_kernel so the (max, log_sum)
        # outputs stay bitwise identical to the non-fused kernel.
        m_blk = tl.max(x)
        l_blk = tl.sum(tl.exp(x - m_blk))
        # +/-inf block max: exp(x - m_blk) is nan there; the true sum_exp
        # normalized by m_blk is 1.
        l_blk = tl.where((m_blk == float("inf")) | (m_blk == float("-inf")), 1.0, l_blk)
        m_i, l_i = _combine_lse(m_i, l_i, m_blk, l_blk)

        # Top-k maintenance. Blocks are visited in ascending column order and
        # the wrapper guarantees the first block holds >= K in-row lanes, so
        # after the first block every running slot carries a genuine index
        # smaller than any later block's. A later block whose max does not
        # strictly beat the running k-th value therefore cannot contribute:
        # on equal values the running entry's lower index wins. The first
        # block must merge unconditionally to displace the sentinel slots.
        # View the block as K_PAD contiguous segments of SEG lanes (a
        # row-major reshape, so no data movement) and surface up to K_PAD
        # candidates per pass: the best available lane of each segment, via
        # minor-axis reductions. Random rows spread their top-K across
        # segments, so a contributing block usually needs one merge pass;
        # the worst case (all K in one segment) needs K.
        SEG: tl.constexpr = BLOCK_N // K_PAD
        x2 = tl.reshape(x, (K_PAD, SEG))
        i2 = start + tl.arange(0, K_PAD)[:, None] * SEG + tl.arange(0, SEG)[None, :]
        avail2 = i2 < num_cols
        go = (start == 0) | (m_blk > _unpack_val(kth_key))
        while go:
            xa2 = tl.where(avail2, x2, float("-inf"))
            cand_vals = tl.max(xa2, axis=1)
            attain = avail2 & (xa2 == cand_vals[:, None])
            cand_idxs = tl.min(tl.where(attain, i2, 2147483647), axis=1)
            # Surfaced lanes are spent either way: winners enter the running
            # top-K and losers lost against a set that only ever improves,
            # so neither can matter again.
            avail2 = avail2 & (i2 != cand_idxs[:, None])

            cand_keys = _pack_key(cand_vals, cand_idxs)
            go = tl.max(cand_keys) > kth_key
            if go:
                # Sorted-sequence merge: max(A, reverse(B)) holds exactly the
                # top-K_PAD of the union (both inputs sorted descending, keys
                # distinct), then re-sort. Compare-exchange networks only; no
                # reductions.
                cand_sorted = tl.sort(cand_keys, descending=True)
                merged = tl.maximum(run_keys, tl.flip(cand_sorted, 0))
                run_keys = tl.sort(merged, descending=True)
                kth_key = tl.max(tl.where(slots == K - 1, run_keys, -1))

    tl.store(out_max_ptr + row, m_i)
    tl.store(out_log_sum_ptr + row, tl.log(l_i))
    out_mask = slots < K
    tl.store(out_top_vals_ptr + row * K + slots, _unpack_val(run_keys), mask=out_mask)
    tl.store(
        out_top_idx_ptr + row * K + slots,
        _unpack_idx(run_keys).to(tl.int64),
        mask=out_mask,
    )


def row_logsumexp_topk(
    x: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row ``(max, logsumexp - max, top_vals, top_idx)`` in one read.

    ``top_vals`` are the RAW input values upcast to fp32 (normalize with
    ``(v - max) - log_sum``; the subtraction is shift-invariant), sorted
    descending with value ties broken by the lowest index. ``top_idx`` is
    int64 like torch.topk's. The (max, log_sum) pair is bitwise identical to
    ``row_logsumexp`` on the same input.

    k is a tl.constexpr, so each distinct k compiles its own kernel: k is the
    per-batch max of the requested top-logprob counts and rarely varies
    within a deployment, while padding every launch to k=32 would make the
    common k<=2 case do 16x the selection work.
    """
    assert x.ndim == 2
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    num_rows, num_cols = x.shape
    # Like torch.topk, selecting more entries than a row holds is an error.
    assert 1 <= k <= min(FUSED_TOPK_MAX_K, num_cols), (k, num_cols)

    out_max = torch.empty(num_rows, device=x.device, dtype=torch.float32)
    out_log_sum = torch.empty(num_rows, device=x.device, dtype=torch.float32)
    top_vals = torch.empty((num_rows, k), device=x.device, dtype=torch.float32)
    top_idx = torch.empty((num_rows, k), device=x.device, dtype=torch.int64)
    if num_rows == 0:
        return out_max, out_log_sum, top_vals, top_idx

    # next_power_of_2(num_cols) >= num_cols >= k when num_cols < 16384, and
    # 16384 > FUSED_TOPK_MAX_K otherwise: the kernel's first block always
    # sees at least k in-row lanes.
    BLOCK_N = triton.next_power_of_2(min(num_cols, 16384))
    _row_logsumexp_topk_kernel[(num_rows,)](
        x,
        out_max,
        out_log_sum,
        top_vals,
        top_idx,
        x.stride(0),
        x.stride(1),
        num_rows,
        num_cols,
        K=k,
        # Floor 2: K_PAD=1 would degenerate the kernel's [BLOCK_N // K_PAD,
        # K_PAD] view (and flip/sort of single-lane tensors).
        K_PAD=max(2, triton.next_power_of_2(k)),
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    return out_max, out_log_sum, top_vals, top_idx
