"""Build Ascend attention page tables in the graph's existing buffers."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_pages", "batch_size"])
def _write_graph_block_table_kernel(
    req_to_token,
    req_pool_indices,
    block_table,
    num_pages,
    batch_size,
    REQ_COUNT: tl.constexpr,
    REQ_ROW_STRIDE: tl.constexpr,
    REQ_COL_STRIDE: tl.constexpr,
    REQ_INDEX_STRIDE: tl.constexpr,
    TABLE_ROW_STRIDE: tl.constexpr,
    TABLE_COL_STRIDE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    row = tl.program_id(0)
    pages = tl.program_id(1) * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)
    req = tl.load(
        req_pool_indices + row * REQ_INDEX_STRIDE,
        mask=(row < batch_size) & (num_pages > 0),
        other=0,
    ).to(tl.int64)
    req = tl.where(req < 0, req + REQ_COUNT, req)
    mapped = (row < batch_size) & (pages < num_pages) & (pages < TABLE_WIDTH)
    slot = tl.load(
        req_to_token + req * REQ_ROW_STRIDE + pages * PAGE_SIZE * REQ_COL_STRIDE,
        mask=mapped & (req >= 0) & (req < REQ_COUNT),
        other=0,
    )
    # Match PyTorch's floor division for signed slot values.
    page_id = slot // PAGE_SIZE - ((slot < 0) & (slot % PAGE_SIZE != 0)).to(tl.int32)
    tl.store(
        block_table + row * TABLE_ROW_STRIDE + pages * TABLE_COL_STRIDE,
        page_id,
        mask=pages < TABLE_WIDTH,
    )


def write_graph_block_table(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    block_table: torch.Tensor,
    batch_size: int,
    num_pages: int,
    page_size: int,
) -> None:
    """Write the page IDs and zero the unused rows and columns in place."""
    table_rows, table_width = block_table.shape
    if table_rows == 0 or table_width == 0:
        return
    block_pages = 256
    _write_graph_block_table_kernel[(table_rows, triton.cdiv(table_width, block_pages))](
        req_to_token,
        req_pool_indices,
        block_table,
        num_pages,
        batch_size,
        REQ_COUNT=req_to_token.shape[0],
        REQ_ROW_STRIDE=req_to_token.stride(0),
        REQ_COL_STRIDE=req_to_token.stride(1),
        REQ_INDEX_STRIDE=req_pool_indices.stride(0),
        TABLE_ROW_STRIDE=block_table.stride(0),
        TABLE_COL_STRIDE=block_table.stride(1),
        TABLE_WIDTH=table_width,
        PAGE_SIZE=page_size,
        BLOCK_PAGES=block_pages,
    )
