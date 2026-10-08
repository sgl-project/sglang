"""Zero the V rows past seq_len on a request's last KV page.

TRT-LLM-gen attention returns NaN when those rows hold NaN
(flashinfer-ai/flashinfer#6246); in paged serving they hold the previous
owner's data.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _zero_v_page_tails_kernel(
    v_ptrs,  # uint64 [num_layers]: per-layer V buffer addresses
    page_table,  # int32 [bs, max_pages]
    seq_lens,  # int32 [bs]: KV length after this forward
    cu_seqlens_q,  # int32 [bs + 1]: tokens this forward writes per request
    table_stride,
    page_stride,
    head_stride,
    row_stride,
    num_heads,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    EVERY_FORWARD: tl.constexpr,
):
    req = tl.program_id(0)
    layer = tl.program_id(1)
    seq_len = tl.load(seq_lens + req)
    q_len = tl.load(cu_seqlens_q + req + 1) - tl.load(cu_seqlens_q + req)
    tail = seq_len % PAGE_SIZE
    page_start = seq_len - tail
    # Otherwise only a page that starts in this forward can hold another request's rows.
    if (tail != 0) & ((page_start >= seq_len - q_len) | EVERY_FORWARD):
        page = tl.load(page_table + req * table_stride + page_start // PAGE_SIZE)
        # Page 0 is the CUDA-graph padding page.
        if page > 0:
            dims = tl.arange(0, HEAD_DIM_PAD)[None, :]
            # 2-byte KV dtypes only: the zero bit pattern is 0.0.
            values = tl.load(v_ptrs + layer).to(tl.pointer_type(tl.int16))
            for head in range(num_heads):
                base = page.to(tl.int64) * page_stride + head * head_stride
                for row_start in range(0, PAGE_SIZE, BLOCK_ROWS):
                    rows = row_start + tl.arange(0, BLOCK_ROWS)[:, None]
                    mask = (rows >= tail) & (rows < PAGE_SIZE) & (dims < HEAD_DIM)
                    tl.store(values + base + rows * row_stride + dims, 0, mask=mask)


def zero_v_page_tails(
    *,
    v_ptrs: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    v_paged_strides: tuple[int, int, int, int],
    num_heads: int,
    page_size: int,
    head_dim: int,
    every_forward: bool,
) -> None:
    """Zero V rows `seq_len % page_size` .. `page_size - 1` of each request's last page.

    With `every_forward=False`, only pages that start in this forward are zeroed.
    Covers every layer in `v_ptrs`; the V buffers must have a 2-byte dtype.
    `v_paged_strides` are the element strides of one layer's
    [num_pages, num_heads, page_size, head_dim] view; the last must be 1.
    """
    batch_size = seq_lens.numel()
    if batch_size == 0:
        return
    page_stride, head_stride, row_stride, dim_stride = v_paged_strides
    assert dim_stride == 1, "the head dimension must be contiguous"
    head_dim_pad = triton.next_power_of_2(head_dim)
    _zero_v_page_tails_kernel[(batch_size, v_ptrs.numel())](
        v_ptrs,
        page_table,
        seq_lens,
        cu_seqlens_q,
        page_table.stride(0),
        page_stride,
        head_stride,
        row_stride,
        num_heads,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        HEAD_DIM_PAD=head_dim_pad,
        BLOCK_ROWS=min(triton.next_power_of_2(page_size), max(1, 4096 // head_dim_pad)),
        EVERY_FORWARD=every_forward,
    )
