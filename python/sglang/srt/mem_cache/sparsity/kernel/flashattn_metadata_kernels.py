import torch
import triton
import triton.language as tl


@triton.jit
def _update_page_table_kernel(
    page_table_ptr,
    physical_pages_ptr,
    valid_lengths_ptr,
    sparse_mask_ptr,
    stride_pt_b: tl.constexpr,
    stride_pt_s: tl.constexpr,
    stride_phys_b: tl.constexpr,
    stride_phys_s: tl.constexpr,
    max_selected: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_COLS)
    valid_len = tl.load(valid_lengths_ptr + b)
    sparse_val = tl.load(sparse_mask_ptr + b)
    col_mask = (offs < max_selected) & (offs < valid_len) & (sparse_val != 0)
    pt_row = page_table_ptr + b * stride_pt_b
    phys_row = physical_pages_ptr + b * stride_phys_b
    vals = tl.load(phys_row + offs * stride_phys_s, mask=col_mask, other=0)
    tl.store(pt_row + offs * stride_pt_s, vals, mask=col_mask)


@triton.jit
def _compute_sparse_seqlens_kernel(
    out_ptr,
    seq_lens_ptr,
    valid_lengths_ptr,
    sparse_mask_ptr,
    orig_ptr,
    page_size: tl.constexpr,
):
    b = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + b)
    valid_len = tl.load(valid_lengths_ptr + b)
    sparse_val = tl.load(sparse_mask_ptr + b)
    pos_in_page = (seq_len - 1) % page_size
    diff = page_size - pos_in_page - 1
    sparse_seq = valid_len * page_size - diff
    orig = tl.load(orig_ptr + b)
    out = tl.where(sparse_val != 0, sparse_seq, orig)
    tl.store(out_ptr + b, out)


def update_page_table_triton(
    page_table: torch.Tensor,
    physical_pages: torch.Tensor,
    valid_lengths: torch.Tensor,
    sparse_mask: torch.Tensor,
) -> None:
    batch = page_table.shape[0]
    max_selected = physical_pages.shape[1]
    block_cols = 128
    grid = (batch,)
    _update_page_table_kernel[grid](
        page_table,
        physical_pages,
        valid_lengths,
        sparse_mask,
        page_table.stride(0),
        page_table.stride(1),
        physical_pages.stride(0),
        physical_pages.stride(1),
        max_selected,
        block_cols,
    )


def compute_sparse_seqlens_triton(
    seq_lens: torch.Tensor,
    valid_lengths: torch.Tensor,
    sparse_mask: torch.Tensor,
    original_cache_seqlens: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    out = torch.empty_like(original_cache_seqlens)
    grid = (seq_lens.shape[0],)
    _compute_sparse_seqlens_kernel[grid](
        out,
        seq_lens,
        valid_lengths,
        sparse_mask,
        original_cache_seqlens,
        page_size,
    )
    return out
