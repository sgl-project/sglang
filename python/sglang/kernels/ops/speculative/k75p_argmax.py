from __future__ import annotations

import torch
import triton
import triton.language as tl

# One pass covers a TP16 shard of the Qwen3.5 vocab (15,520 ids).
_CHUNK = 16384
_NO_ID = tl.constexpr(2147483647)


@triton.jit
def _k75p_shard_argmax_kernel(
    logits,
    pairs,
    logits_row_stride,
    vocab_start,
    SHARD: tl.constexpr,
    CHUNK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    TORCH_NAN: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    best_v = tl.max(tl.full([CHUNK], -float("inf"), tl.float32), axis=0)
    best_i = tl.min(tl.full([CHUNK], 0, tl.int32), axis=0)
    nan_i = tl.min(tl.full([CHUNK], _NO_ID, tl.int32), axis=0)
    for c in tl.static_range(NUM_CHUNKS):
        offs = c * CHUNK + tl.arange(0, CHUNK)
        v = tl.load(
            logits + row * logits_row_stride + offs,
            mask=offs < SHARD,
            other=-float("inf"),
        ).to(tl.float32)
        if TORCH_NAN:
            # torch.argmax ranks NaN above every value (first NaN wins).
            nan_i = tl.minimum(nan_i, tl.min(tl.where(v != v, offs, _NO_ID), axis=0))
            v = tl.where(v == v, v, -float("inf"))
        else:
            # Same rule as _draft_topk1_partial_argmax_kernel.
            v = tl.where(v == v, v, -1e30)
        m = tl.max(v, axis=0)
        i = (tl.argmax(v, axis=0) + c * CHUNK).to(tl.int32)
        # Strict: an earlier chunk keeps an equal maximum (lowest id).
        upd = m > best_v
        best_i = tl.where(upd, i, best_i)
        best_v = tl.where(upd, m, best_v)
    if TORCH_NAN:
        has_nan = nan_i < _NO_ID
        best_v = tl.where(has_nan, float("nan"), best_v)
        best_i = tl.where(has_nan, nan_i, best_i)
    tl.store(pairs + row * 2, best_v.to(tl.int32, bitcast=True))
    tl.store(pairs + row * 2 + 1, best_i + vocab_start)


@triton.jit
def _k75p_select_kernel(
    gathered,
    token_ids,
    row_start,
    num_rows,
    total_rows,
    world,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_WORLD: tl.constexpr,
    TORCH_NAN: tl.constexpr,
):
    b = tl.arange(0, BLOCK_ROWS)
    s = tl.arange(0, BLOCK_WORLD)
    mask = (b[:, None] < num_rows) & (s[None, :] < world)
    offs = (s[None, :] * total_rows + row_start + b[:, None]) * 2
    vals = tl.load(gathered + offs, mask=mask, other=0).to(tl.float32, bitcast=True)
    ids = tl.load(gathered + offs + 1, mask=mask, other=_NO_ID)
    vals = tl.where(mask, vals, -float("inf"))
    if TORCH_NAN:
        nan_id = tl.min(tl.where(vals != vals, ids, _NO_ID), axis=1)
        vals = tl.where(vals == vals, vals, -float("inf"))
    m = tl.max(vals, axis=1)
    tok = tl.min(tl.where(vals == m[:, None], ids, _NO_ID), axis=1)
    if TORCH_NAN:
        tok = tl.where(nan_id < _NO_ID, nan_id, tok)
    tl.store(token_ids + b, tok.to(tl.int64), mask=b < num_rows)


@triton.jit
def _k75p_topk1_finish_kernel(
    token_ids,
    topk_p,
    topk_index,
    positions,
    draft_tokens,
    draft_tokens_stride,
    draft_token_column,
    num_rows,
    WRITE_DRAFT_TOKEN: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    b = tl.arange(0, BLOCK_ROWS)
    mask = b < num_rows
    tok = tl.load(token_ids + b, mask=mask)
    tl.store(topk_index + b, tok, mask=mask)
    tl.store(topk_p + b, tl.full([BLOCK_ROWS], 1.0, tl.float32), mask=mask)
    if WRITE_DRAFT_TOKEN:
        tl.store(
            draft_tokens + b * draft_tokens_stride + draft_token_column, tok, mask=mask
        )
    position = tl.load(positions + b, mask=mask)
    tl.store(positions + b, position + 1, mask=mask)


def k75p_shard_argmax(
    logits: torch.Tensor, pairs: torch.Tensor, vocab_start: int, torch_nan: bool
) -> None:
    """Per row of the shard logits [rows, shard]: (fp32 max as int32 bits, vocab id) into
    pairs [rows, 2] int32, ties to the lowest id; NaN as split argmax or torch.argmax."""
    assert logits.ndim == 2 and logits.stride(1) == 1
    assert pairs.shape == (logits.shape[0], 2) and pairs.dtype == torch.int32
    shard = logits.shape[1]
    chunk = min(_CHUNK, triton.next_power_of_2(shard))
    _k75p_shard_argmax_kernel[(logits.shape[0],)](
        logits,
        pairs,
        logits.stride(0),
        vocab_start,
        SHARD=shard,
        CHUNK=chunk,
        NUM_CHUNKS=triton.cdiv(shard, chunk),
        TORCH_NAN=torch_nan,
        num_warps=8,
    )


def k75p_select(
    gathered: torch.Tensor, row_start: int, num_rows: int, torch_nan: bool
) -> torch.Tensor:
    """Token ids [num_rows, 1] int64 for rows row_start.. of the gathered pairs
    [world, total_rows, 2]: largest value, lowest id among equals (NaN first if torch_nan)."""
    world, total_rows, _ = gathered.shape
    token_ids = torch.empty((num_rows, 1), dtype=torch.int64, device=gathered.device)
    _k75p_select_kernel[(1,)](
        gathered,
        token_ids,
        row_start,
        num_rows,
        total_rows,
        world,
        BLOCK_ROWS=max(triton.next_power_of_2(num_rows), 2),
        BLOCK_WORLD=max(triton.next_power_of_2(world), 2),
        TORCH_NAN=torch_nan,
        num_warps=4,
    )
    return token_ids


def k75p_topk1_postprocess(
    next_token_ids: torch.Tensor,
    positions: torch.Tensor,
    draft_tokens: torch.Tensor | None = None,
    draft_token_column: int = 0,
):
    """draft_topk1_postprocess for token ids picked in the LM head: same outputs and
    side effects (topk_p = 1, draft_tokens[:, column], positions + 1)."""
    num_rows = next_token_ids.shape[0]
    topk_p = torch.empty((num_rows, 1), dtype=torch.float32, device=positions.device)
    topk_index = torch.empty((num_rows, 1), dtype=torch.int64, device=positions.device)
    write_draft_token = draft_tokens is not None
    _k75p_topk1_finish_kernel[(1,)](
        next_token_ids,
        topk_p,
        topk_index,
        positions,
        draft_tokens if write_draft_token else topk_index,
        draft_tokens.stride(0) if write_draft_token else 0,
        draft_token_column,
        num_rows,
        WRITE_DRAFT_TOKEN=write_draft_token,
        BLOCK_ROWS=max(triton.next_power_of_2(num_rows), 2),
        num_warps=1,
    )
    return topk_p, topk_index
