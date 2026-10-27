"""Prefill-side gather of the packed NVFP4 (MSA-v1) main K/V cache into the paged
shuffle layout the Gluon sparse prefill reads, dequantized as it is gathered."""

import torch
import triton
import triton.language as tl

from ..common.nvfp4 import Nvfp4Cache, nvfp4_load_rows, nvfp4_slot_offsets


@triton.jit
def _gather_nvfp4_to_shuffle_kernel(
    k_payload_ptr,
    k_scale_ptr,
    v_payload_ptr,
    v_scale_ptr,
    k_out_ptr,  # [num_pages, 1, HEAD_DIM // X, PAGE, X] contiguous
    v_out_ptr,  # [num_pages, 1, PAGE // X, HEAD_DIM, X] contiguous
    req_to_token_ptr,  # [max_reqs, max_kv_len]
    req_pool_indices_ptr,  # [batch]
    seq_lens_ptr,  # [batch] total K length (prefix + current chunk)
    page_req_ptr,  # [total_pages] -> batch index owning this page
    page_pos_ptr,  # [total_pages] -> page index within the request
    max_slots,
    pay_page,
    pay_head,
    pay_row,
    sc_page,
    sc_head,
    sc_row,
    stride_r2t_b,
    HEAD_DIM: tl.constexpr,
    PAGE: tl.constexpr,
    X: tl.constexpr,
    POOL_PAGE_SIZE: tl.constexpr,
):
    """The NVFP4 twin of `_gather_nhd_to_shuffle_kernel`: dequantize while gathering."""
    pid = tl.program_id(0)
    req = tl.load(page_req_ptr + pid)
    ppos = tl.load(page_pos_ptr + pid)
    pool_id = tl.load(req_pool_indices_ptr + req).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + req)
    off_t = tl.arange(0, PAGE)
    tok = ppos * PAGE + off_t
    tmask = tok < seq_len
    slots = tl.load(
        req_to_token_ptr + pool_id * stride_r2t_b + tok, mask=tmask, other=0
    ).to(tl.int64)
    slots = (slots + max_slots) % max_slots
    pay_off = nvfp4_slot_offsets(slots, pay_page, pay_head, pay_row, 0, POOL_PAGE_SIZE)
    sc_off = nvfp4_slot_offsets(slots, sc_page, sc_head, sc_row, 0, POOL_PAGE_SIZE)
    page_base = pid.to(tl.int64) * (HEAD_DIM * PAGE)
    d = tl.arange(0, HEAD_DIM)
    # Unmasked: tiles are zero past seq_len, so no previous layer's data survives.
    k_rows = nvfp4_load_rows(
        k_payload_ptr, k_scale_ptr, pay_off, sc_off, tmask, HEAD_DIM
    )
    # K shuffle page [1, D//X, PAGE, X]: elem(d, t) at (d//X)*(PAGE*X) + t*X + d%X
    k_off = (d[None, :] // X) * (PAGE * X) + off_t[:, None] * X + (d[None, :] % X)
    tl.store(k_out_ptr + page_base + k_off, k_rows.to(k_out_ptr.dtype.element_ty))
    v_rows = nvfp4_load_rows(
        v_payload_ptr, v_scale_ptr, pay_off, sc_off, tmask, HEAD_DIM
    )
    # V transposed page [1, PAGE//X, D, X]: elem(d, t) at (t//X)*(D*X) + d*X + t%X
    v_off = (
        (off_t[:, None] // X) * (HEAD_DIM * X) + d[None, :] * X + (off_t[:, None] % X)
    )
    tl.store(v_out_ptr + page_base + v_off, v_rows.to(v_out_ptr.dtype.element_ty))


def gather_nvfp4_to_shuffle(
    *,
    k_cache: Nvfp4Cache,
    v_cache: Nvfp4Cache,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    page_req: torch.Tensor,
    page_pos: torch.Tensor,
    total_pages: int,
    page: int,
    x: int,
) -> None:
    assert k_cache.num_heads == 1 and k_cache.payload.shape == v_cache.payload.shape
    pool_page_size = k_cache.page_size
    _gather_nvfp4_to_shuffle_kernel[(total_pages,)](
        k_cache.payload,
        k_cache.scales,
        v_cache.payload,
        v_cache.scales,
        k_out,
        v_out,
        req_to_token,
        req_pool_indices,
        seq_lens,
        page_req,
        page_pos,
        k_cache.payload.shape[0] * pool_page_size,
        *k_cache.payload_strides(),
        *k_cache.scale_strides(),
        req_to_token.stride(0),
        HEAD_DIM=k_cache.payload.shape[-1] * 2,
        PAGE=page,
        X=x,
        POOL_PAGE_SIZE=pool_page_size,
    )
