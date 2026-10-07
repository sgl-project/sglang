"""Token copies of the DeepSeek-V4.1 per-request SWA window (encoder SWA replay).

Each layer gathers the window's history into the attention workspace before the
layer and commits the step's new tokens back after it. One launch per copy keeps
the cost per layer flat in the batch size; the index math runs in the kernel.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout


@triton.jit
def _copy_token(
    src,
    dst,
    src_row,
    dst_row,
    src_page_words,
    dst_page_words,
    PAGE_SIZE: tl.constexpr,
    DATA_WORDS: tl.constexpr,
    SCALE_WORDS: tl.constexpr,
    BLOCK_DATA: tl.constexpr,
    BLOCK_SCALE: tl.constexpr,
):
    # A page holds PAGE_SIZE data rows, then PAGE_SIZE scale rows.
    src_page, src_slot = src_row // PAGE_SIZE, src_row % PAGE_SIZE
    dst_page, dst_slot = dst_row // PAGE_SIZE, dst_row % PAGE_SIZE
    offs = tl.arange(0, BLOCK_DATA)
    mask = offs < DATA_WORDS
    src_data = src + src_page * src_page_words + src_slot * DATA_WORDS
    dst_data = dst + dst_page * dst_page_words + dst_slot * DATA_WORDS
    tl.store(dst_data + offs, tl.load(src_data + offs, mask=mask), mask=mask)
    offs = tl.arange(0, BLOCK_SCALE)
    mask = offs < SCALE_WORDS
    scale_base = PAGE_SIZE * DATA_WORDS
    src_scale = src + src_page * src_page_words + scale_base + src_slot * SCALE_WORDS
    dst_scale = dst + dst_page * dst_page_words + scale_base + dst_slot * SCALE_WORDS
    tl.store(dst_scale + offs, tl.load(src_scale + offs, mask=mask), mask=mask)


@triton.jit
def _gather_history_kernel(
    state,
    workspace,
    history_req,
    history_pos,
    history_valid,
    history_loc,
    capacity,
    zero_row,
    state_page_words,
    workspace_page_words,
    PAGE_SIZE: tl.constexpr,
    DATA_WORDS: tl.constexpr,
    SCALE_WORDS: tl.constexpr,
    BLOCK_DATA: tl.constexpr,
    BLOCK_SCALE: tl.constexpr,
):
    i = tl.program_id(0)
    req = tl.load(history_req + i).to(tl.int64)
    pos = tl.load(history_pos + i).to(tl.int64)
    valid = tl.load(history_valid + i)
    src_row = tl.where(valid, req * capacity + pos % capacity, zero_row)
    dst_row = tl.load(history_loc + i).to(tl.int64)
    _copy_token(
        state,
        workspace,
        src_row,
        dst_row,
        state_page_words,
        workspace_page_words,
        PAGE_SIZE,
        DATA_WORDS,
        SCALE_WORDS,
        BLOCK_DATA,
        BLOCK_SCALE,
    )


@triton.jit
def _commit_tokens_kernel(
    workspace,
    state,
    tags,
    write_loc,
    req,
    pos,
    commit_mask,
    capacity,
    sink_row,
    workspace_page_words,
    state_page_words,
    PAGE_SIZE: tl.constexpr,
    DATA_WORDS: tl.constexpr,
    SCALE_WORDS: tl.constexpr,
    BLOCK_DATA: tl.constexpr,
    BLOCK_SCALE: tl.constexpr,
):
    j = tl.program_id(0)
    token_req = tl.load(req + j).to(tl.int64)
    token_pos = tl.load(pos + j).to(tl.int64)
    keep = tl.load(commit_mask + j)
    dst_row = tl.where(keep, token_req * capacity + token_pos % capacity, sink_row)
    src_row = tl.load(write_loc + j).to(tl.int64)
    _copy_token(
        workspace,
        state,
        src_row,
        dst_row,
        workspace_page_words,
        state_page_words,
        PAGE_SIZE,
        DATA_WORDS,
        SCALE_WORDS,
        BLOCK_DATA,
        BLOCK_SCALE,
    )
    tl.store(tags + dst_row, token_pos)


def _words(buf: torch.Tensor, layout: KVLayout, page_size: int) -> torch.Tensor:
    """A contiguous paged byte buffer, viewed as int32 words."""
    assert buf.dtype == torch.uint8 and buf.dim() == 2 and buf.is_contiguous()
    assert buf.shape[1] >= page_size * layout.bytes_per_token
    return buf.view(torch.int32)


def _assert_dense(*tensors: torch.Tensor) -> None:
    # The kernels index these with the program id, ignoring strides.
    for t in tensors:
        assert t.dim() == 1 and t.is_contiguous(), (t.shape, t.stride())


def _meta(layout: KVLayout, page_size: int) -> dict:
    assert layout.data_bytes % 4 == 0 and layout.scale_bytes % 4 == 0
    data_words, scale_words = layout.data_bytes // 4, layout.scale_bytes // 4
    return dict(
        PAGE_SIZE=page_size,
        DATA_WORDS=data_words,
        SCALE_WORDS=scale_words,
        BLOCK_DATA=triton.next_power_of_2(data_words),
        BLOCK_SCALE=triton.next_power_of_2(scale_words),
    )


def gather_window_history(
    state: torch.Tensor,
    workspace: torch.Tensor,
    *,
    history_req: torch.Tensor,
    history_pos: torch.Tensor,
    history_valid: torch.Tensor,
    history_loc: torch.Tensor,
    capacity: int,
    zero_row: int,
    page_size: int,
    layout: KVLayout,
) -> None:
    """workspace[history_loc[i]] = state[history_req[i] * capacity +
    history_pos[i] % capacity], or state[zero_row] where history_valid[i] is
    false."""
    n = history_loc.numel()
    if n == 0:
        return
    _assert_dense(history_req, history_pos, history_valid, history_loc)
    state_words = _words(state, layout, page_size)
    workspace_words = _words(workspace, layout, page_size)
    _gather_history_kernel[(n,)](
        state_words,
        workspace_words,
        history_req,
        history_pos,
        history_valid,
        history_loc,
        capacity,
        zero_row,
        state_words.shape[1],
        workspace_words.shape[1],
        **_meta(layout, page_size),
    )


def commit_window_tokens(
    workspace: torch.Tensor,
    state: torch.Tensor,
    tags: torch.Tensor,
    *,
    write_loc: torch.Tensor,
    req: torch.Tensor,
    pos: torch.Tensor,
    commit_mask: torch.Tensor,
    capacity: int,
    sink_row: int,
    page_size: int,
    layout: KVLayout,
) -> None:
    """Copy workspace[write_loc[j]] to the request row req[j] * capacity +
    pos[j] % capacity (sink_row where commit_mask[j] is false) and record
    pos[j] in tags at that row."""
    n = write_loc.numel()
    if n == 0:
        return
    assert tags.dtype == torch.int64 and tags.is_contiguous()
    _assert_dense(write_loc, req, pos, commit_mask)
    workspace_words = _words(workspace, layout, page_size)
    state_words = _words(state, layout, page_size)
    _commit_tokens_kernel[(n,)](
        workspace_words,
        state_words,
        tags,
        write_loc,
        req,
        pos,
        commit_mask,
        capacity,
        sink_row,
        workspace_words.shape[1],
        state_words.shape[1],
        **_meta(layout, page_size),
    )
