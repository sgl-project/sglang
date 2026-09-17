"""Zero a set of KV rows across every buffer of a pool in one launch.

Same shape as ``copy_all_layer_kv_cache_tiled`` (a pointer table plus a
per-buffer row stride, one launch covering every layer and both K and V) with
the load deleted. Per-buffer launches are not a viable alternative: a 62-layer
fp8 pool is 124 buffers, and at a few microseconds each that is ~0.5 ms per
free against ~28 us measured for the fused kernel on B200.

Rows are addressed as int64 words, like the sibling ``zero_pages``; the plan
builder in ``sglang.srt.mem_cache.kv_zeroize`` rejects any pool whose base
pointer or row size is not 8-byte aligned.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _zero_kv_rows_kernel(
    base_ptrs,  # int64 [B] base address of each buffer
    row_words_ptr,  # int64 [B] int64-words per row, per buffer
    row_ids_ptr,  # int64 [N] rows to clear (slot ids, or page ids)
    num_rows,
    ROWS_PER_BLOCK: tl.constexpr,
    WORDS_PER_TILE: tl.constexpr,
):
    bid = tl.program_id(0)
    rid = tl.program_id(1)
    tid = tl.program_id(2)

    row_words = tl.load(row_words_ptr + bid)
    base = tl.cast(tl.load(base_ptrs + bid), tl.pointer_type(tl.int64))

    # Masked against THIS buffer's row, not the table's widest one: the grid is
    # sized for the widest row and buffers differ in row size.
    word_off = tid * WORDS_PER_TILE + tl.arange(0, WORDS_PER_TILE)
    mask_word = word_off < row_words

    row_idx = rid * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
    mask_row = row_idx < num_rows
    rows = tl.load(row_ids_ptr + row_idx, mask=mask_row, other=0)

    ptr = base + rows[:, None] * row_words + word_off[None, :]
    tl.store(ptr, 0, mask=mask_row[:, None] & mask_word[None, :])


_ROWS_PER_BLOCK = 8
_WORDS_PER_TILE = 256


def zero_kv_rows(
    base_ptrs: torch.Tensor,
    row_words: torch.Tensor,
    row_ids: torch.Tensor,
    max_row_words: int,
) -> None:
    """Zero ``row_ids`` in every buffer of the pointer table, in one launch."""
    num_rows = int(row_ids.numel())
    if num_rows == 0:
        return
    grid = (
        int(base_ptrs.numel()),
        triton.cdiv(num_rows, _ROWS_PER_BLOCK),
        triton.cdiv(max_row_words, _WORDS_PER_TILE),
    )
    _zero_kv_rows_kernel[grid](
        base_ptrs,
        row_words,
        row_ids,
        num_rows,
        ROWS_PER_BLOCK=_ROWS_PER_BLOCK,
        WORDS_PER_TILE=_WORDS_PER_TILE,
    )


def warmup_zero_kv_rows(device: torch.device) -> None:
    """Compile the kernel before the engine reports ready.

    ``sglang.srt.utils.triton_load_watch`` warns on a Triton kernel first
    loading after startup, and ``SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY=1``
    turns that into a crash.
    """
    scratch = torch.zeros(8, dtype=torch.int64, device=device)
    zero_kv_rows(
        base_ptrs=torch.tensor([scratch.data_ptr()], dtype=torch.int64, device=device),
        row_words=torch.tensor([1], dtype=torch.int64, device=device),
        row_ids=torch.zeros(1, dtype=torch.int64, device=device),
        max_row_words=1,
    )
