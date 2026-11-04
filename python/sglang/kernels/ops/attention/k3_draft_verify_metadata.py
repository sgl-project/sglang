"""One-launch TARGET_VERIFY metadata fill for the DFlash draft (Triton backend).

Outputs:
  qo_indptr[0..bs]         = b * L
  kv_indptr[0..bs]         = cumsum(seq_lens)
  kv_indices[kv_indptr[b] + i] = req_to_token[rpi[b], i],          i < seq[b]
  window_kv_indptr[0..bs]  = cumsum(min(seq, W))
  window_kv_offsets[b]     = seq[b] - min(seq[b], W)
  window_kv_indices[w_indptr[b] + i] = req_to_token[rpi[b], off[b] + i]
  mask_indptr[0..bs]       = cumsum(L * (seq + L))
"""

import triton
import triton.language as tl


@triton.jit
def _draft_verify_meta_kernel(
    seq_lens_ptr,
    rpi_ptr,
    r2t_ptr,
    stride_r2t,
    qo_indptr_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    w_indptr_ptr,
    w_indices_ptr,
    w_offsets_ptr,
    mask_indptr_ptr,
    bs,
    L,
    W,
    HAS_WINDOW: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    BLOCK: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    offs_bs = tl.arange(0, BLOCK_BS)
    seqs = tl.load(seq_lens_ptr + offs_bs, mask=offs_bs < bs, other=0).to(tl.int64)
    before = offs_bs < b
    seq_b = tl.load(seq_lens_ptr + b).to(tl.int64)
    kv_start = tl.sum(tl.where(before, seqs, 0), 0)
    if HAS_WINDOW:
        wls = tl.minimum(seqs, W)
        w_start = tl.sum(tl.where(before, wls, 0), 0)
        wl_b = tl.minimum(seq_b, W)
        off_b = seq_b - wl_b
    if c == 0:
        if b == 0:
            tl.store(qo_indptr_ptr, 0)
            tl.store(kv_indptr_ptr, 0)
            tl.store(mask_indptr_ptr, 0)
            if HAS_WINDOW:
                tl.store(w_indptr_ptr, 0)
        tl.store(qo_indptr_ptr + b + 1, (b + 1) * L)
        tl.store(kv_indptr_ptr + b + 1, (kv_start + seq_b).to(tl.int32))
        incl = offs_bs <= b
        mlen = tl.sum(tl.where(incl, L * (seqs + L), 0), 0)
        tl.store(mask_indptr_ptr + b + 1, mlen)
        if HAS_WINDOW:
            tl.store(w_indptr_ptr + b + 1, (w_start + wl_b).to(tl.int32))
            tl.store(w_offsets_ptr + b, off_b.to(tl.int32))
    row = tl.load(rpi_ptr + b).to(tl.int64) * stride_r2t
    for start in range(c * BLOCK, seq_b, NCHUNK * BLOCK):
        i = start + tl.arange(0, BLOCK)
        m = i < seq_b
        tok = tl.load(r2t_ptr + row + i, mask=m, other=0)
        tl.store(kv_indices_ptr + kv_start + i, tok.to(tl.int64), mask=m)
    if HAS_WINDOW:
        for start in range(c * BLOCK, wl_b, NCHUNK * BLOCK):
            i = start + tl.arange(0, BLOCK)
            m = i < wl_b
            tok = tl.load(r2t_ptr + row + off_b + i, mask=m, other=0)
            tl.store(w_indices_ptr + w_start + i, tok.to(tl.int64), mask=m)


def fill_draft_verify_metadata(
    *,
    bs,
    seq_lens,
    req_pool_indices,
    req_to_token,
    num_tokens_per_req,
    qo_indptr,
    kv_indptr,
    kv_indices,
    mask_indptr,
    window_size=None,
    window_kv_indptr=None,
    window_kv_indices=None,
    window_kv_offsets=None,
):
    has_window = window_size is not None and window_size > 0
    BLOCK_BS = max(16, triton.next_power_of_2(bs))
    NCHUNK = 4
    dummy = kv_indptr
    _draft_verify_meta_kernel[(bs, NCHUNK)](
        seq_lens,
        req_pool_indices,
        req_to_token,
        req_to_token.stride(0),
        qo_indptr,
        kv_indptr,
        kv_indices,
        window_kv_indptr if has_window else dummy,
        window_kv_indices if has_window else kv_indices,
        window_kv_offsets if has_window else dummy,
        mask_indptr,
        bs,
        int(num_tokens_per_req),
        int(window_size) if has_window else 0,
        HAS_WINDOW=has_window,
        BLOCK_BS=BLOCK_BS,
        BLOCK=1024,
        NCHUNK=NCHUNK,
        num_warps=4,
    )
