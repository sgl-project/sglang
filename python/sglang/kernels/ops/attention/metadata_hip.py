"""ROCm-only attention metadata kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def get_num_kv_splits_by_length_triton(
    num_kv_splits_ptr,
    seq_lens_ptr,
    num_seq,
    num_group,
    max_splits,
    MIN_TOKENS: tl.constexpr,
    MAX_NUM_SEQ: tl.constexpr,
    INDPTR: tl.constexpr = False,
):
    """Each request takes cdiv(seq_len, MIN_TOKENS) splits, at least 1 and at most
    max_splits (the caller's bound for this batch). With INDPTR, seq_lens_ptr is a
    kv_indptr and the lengths are its differences."""
    offs_seq = tl.arange(0, MAX_NUM_SEQ)
    mask_seq = offs_seq < num_seq
    if INDPTR:
        seq_lens = tl.load(
            seq_lens_ptr + offs_seq + 1, mask=mask_seq, other=1
        ) - tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=0)
    else:
        seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=1)
    num_kv_splits = tl.minimum(tl.maximum(tl.cdiv(seq_lens, MIN_TOKENS), 1), max_splits)
    offs_token = offs_seq * num_group
    mask_token = offs_token < num_seq * num_group
    for i in range(0, num_group):
        tl.store(num_kv_splits_ptr + i + offs_token, num_kv_splits, mask=mask_token)


@triton.jit
def _verify_indptrs_kernel(
    SeqLens,
    QoIndptr,
    KvIndptr,
    MaskIndptr,
    WindowIndptr,
    WindowLens,
    WindowStart,
    WindowOffsets,
    bs,
    tokens_per_req,
    window,
    HAS_WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    valid = i < bs
    seq = tl.load(SeqLens + i, valid, other=0).to(tl.int64)
    qo = i.to(tl.int64) * tokens_per_req
    tl.store(QoIndptr + i, qo.to(QoIndptr.dtype.element_ty), i <= bs)
    kv = tl.cumsum(seq, 0)
    tl.store(KvIndptr + 1 + i, kv.to(KvIndptr.dtype.element_ty), valid)
    # Lanes past bs follow the valid ones, so they never enter a stored prefix sum.
    mask = tl.cumsum(tokens_per_req * (seq + tokens_per_req), 0)
    tl.store(MaskIndptr + 1 + i, mask.to(MaskIndptr.dtype.element_ty), valid)
    if HAS_WINDOW:
        lens = tl.minimum(seq, window)
        tl.store(
            WindowIndptr + 1 + i,
            tl.cumsum(lens, 0).to(WindowIndptr.dtype.element_ty),
            valid,
        )
        tl.store(WindowLens + i, lens.to(WindowLens.dtype.element_ty), valid)
        start = seq - lens
        tl.store(WindowStart + i, start.to(WindowStart.dtype.element_ty), valid)
        tl.store(WindowOffsets + i, start.to(WindowOffsets.dtype.element_ty), valid)


def verify_indptrs(
    seq_lens,
    tokens_per_req,
    qo_indptr,
    kv_indptr,
    mask_indptr,
    window=None,
    window_kv_indptr=None,
    window_kv_offsets=None,
):
    """Target-verify indptrs in one launch, as the separate torch ops compute them:
    qo_indptr[: bs + 1] = arange * tokens_per_req, kv_indptr[1:] = cumsum(seq_lens),
    mask_indptr[1:] = cumsum(tokens_per_req * (seq_lens + tokens_per_req)); with a
    window, also window_kv_indptr[1:] = cumsum(min(seq_lens, window)) and the
    window's start offsets (into window_kv_offsets[:bs]). Returns (window lengths,
    window starts) in seq_lens' dtype, or (None, None) without a window."""
    bs = seq_lens.shape[0]
    lens = start = None
    if window is not None:
        lens, start = torch.empty_like(seq_lens), torch.empty_like(seq_lens)
    _verify_indptrs_kernel[(1,)](
        seq_lens,
        qo_indptr,
        kv_indptr,
        mask_indptr,
        window_kv_indptr if window is not None else kv_indptr,
        lens if window is not None else seq_lens,
        start if window is not None else seq_lens,
        window_kv_offsets if window is not None else seq_lens,
        bs,
        tokens_per_req,
        window if window is not None else 0,
        window is not None,
        max(16, triton.next_power_of_2(bs + 1)),
    )
    return lens, start
