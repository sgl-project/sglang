import torch
import triton
import triton.language as tl


@triton.jit
def _gather_repeat_kv_heads_kernel(
    src_ptr,  # (num_slots, ROW_WORDS) int64 view of the cache
    idx_ptr,  # (n_rows,) cache slot of each output row
    dst_ptr,  # (n_rows, ROW_WORDS * REPEAT) int64 view of the output
    n_rows,
    ROW_WORDS: tl.constexpr,  # 8-byte words per cache row (all KV heads)
    HEAD_WORDS: tl.constexpr,  # 8-byte words per head
    REPEAT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_W: tl.constexpr,  # next power of two >= ROW_WORDS
):
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    row_mask = rows < n_rows
    slots = tl.load(idx_ptr + rows, mask=row_mask, other=0).to(tl.int64)
    cols = tl.arange(0, BLOCK_W)
    mask = row_mask[:, None] & (cols < ROW_WORDS)[None, :]
    vals = tl.load(src_ptr + slots[:, None] * ROW_WORDS + cols[None, :], mask=mask)
    # Output head h * REPEAT + r is a copy of cache head h.
    head = cols // HEAD_WORDS
    word = cols % HEAD_WORDS
    dst_rows = dst_ptr + rows.to(tl.int64)[:, None] * (ROW_WORDS * REPEAT)
    for r in tl.static_range(REPEAT):
        tl.store(
            dst_rows + ((head * REPEAT + r) * HEAD_WORDS + word)[None, :],
            vals,
            mask=mask,
        )


def gather_repeat_kv_heads(
    cache: torch.Tensor, idx: torch.Tensor, repeat: int
) -> torch.Tensor:
    """``cache[idx]`` with each KV head repeated ``repeat`` times, in one pass.

    cache: (num_slots, num_kv_heads, head_dim), any dtype, with rows of whole
    8-byte words (e.g. FP8 head_dim 256); idx: (n,) int32/int64 slots.
    Returns (n, num_kv_heads * repeat, head_dim) where head h * repeat + r is
    head h of the gathered row: byte for byte the result of index_select
    followed by expand + reshape, without the intermediate gathered copy.
    """
    num_slots, num_kv_heads, head_dim = cache.shape
    head_bytes = head_dim * cache.element_size()
    assert cache.is_contiguous() and head_bytes % 8 == 0, (cache.shape, cache.dtype)
    n = idx.numel()
    out = torch.empty(
        (n, num_kv_heads * repeat, head_dim), dtype=cache.dtype, device=cache.device
    )
    if n == 0:
        return out
    head_words = head_bytes // 8
    row_words = num_kv_heads * head_words
    block_t = 16
    _gather_repeat_kv_heads_kernel[(triton.cdiv(n, block_t),)](
        cache.view(num_slots, -1).view(torch.int64),
        idx,
        out.view(n, -1).view(torch.int64),
        n,
        ROW_WORDS=row_words,
        HEAD_WORDS=head_words,
        REPEAT=repeat,
        BLOCK_T=block_t,
        BLOCK_W=triton.next_power_of_2(row_words),
        num_warps=4,
    )
    return out
