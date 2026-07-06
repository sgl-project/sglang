from __future__ import annotations

import torch
import triton
import triton.language as tl


_DRAFT_TOPK1_BLOCK = 8192
_SELECT_TOPK1_BLOCK = 128


@triton.jit
def _draft_topk1_partial_argmax_kernel(
    logits,
    partial_vals,
    partial_indices,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    vals = tl.load(
        logits + row * vocab_size + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    max_val = tl.max(vals, axis=0)
    local_index = tl.argmax(vals, axis=0)
    out_offset = row * num_splits + split
    tl.store(partial_vals + out_offset, max_val)
    tl.store(partial_indices + out_offset, split * BLOCK + local_index)


@triton.jit
def _draft_topk1_finalize_kernel(
    partial_vals,
    partial_indices,
    topk_p,
    topk_index,
    positions,
    num_splits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_splits
    vals = tl.load(
        partial_vals + row * num_splits + offsets,
        mask=mask,
        other=-float("inf"),
    )

    split = tl.argmax(vals, axis=0)
    index = tl.load(partial_indices + row * num_splits + split).to(tl.int64)
    tl.store(topk_index + row, index)
    tl.store(topk_p + row, 1.0)

    position = tl.load(positions + row)
    tl.store(positions + row, position + 1)


@triton.jit
def _select_top_k_tokens_topk1_later_kernel(
    topk_p,
    topk_index,
    first_topk_index,
    scores,
    out_scores,
    parents,
    draft_tokens,
    parent_id: tl.constexpr,
    n_elements: tl.constexpr,
    topk_index_stride: tl.constexpr,
    first_topk_index_stride: tl.constexpr,
    draft_tokens_stride: tl.constexpr,
    WRITE_DRAFT_TOKENS: tl.constexpr,
    WRITE_FIRST_DRAFT_TOKEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    next_scores = tl.load(scores + offsets, mask=mask) * tl.load(
        topk_p + offsets, mask=mask
    )
    tl.store(out_scores + offsets, next_scores, mask=mask)
    tl.store(parents + offsets, parent_id, mask=mask)
    if WRITE_DRAFT_TOKENS:
        tokens = tl.load(topk_index + offsets * topk_index_stride, mask=mask)
        tl.store(
            draft_tokens + offsets * draft_tokens_stride + parent_id,
            tokens,
            mask=mask,
        )
        if WRITE_FIRST_DRAFT_TOKEN:
            first_tokens = tl.load(
                first_topk_index + offsets * first_topk_index_stride,
                mask=mask,
            )
            tl.store(
                draft_tokens + offsets * draft_tokens_stride,
                first_tokens,
                mask=mask,
            )


def select_top_k_tokens_topk1_later(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    draft_tokens: torch.Tensor | None = None,
    first_topk_index: torch.Tensor | None = None,
):
    input_ids = topk_index.flatten()
    n_elements = topk_index.numel()
    next_scores = torch.empty_like(scores)
    parents = torch.empty_like(topk_index)
    write_draft_tokens = draft_tokens is not None
    write_first_draft_token = write_draft_tokens and first_topk_index is not None
    if n_elements == 0:
        return input_ids, hidden_states, next_scores, (
            next_scores.unsqueeze(1),
            topk_index,
            parents,
        )

    grid = (triton.cdiv(n_elements, _SELECT_TOPK1_BLOCK),)
    _select_top_k_tokens_topk1_later_kernel[grid](
        topk_p,
        topk_index,
        first_topk_index if write_first_draft_token else topk_index,
        scores,
        next_scores,
        parents,
        draft_tokens if write_draft_tokens else topk_index,
        i,
        n_elements,
        topk_index.stride(0),
        first_topk_index.stride(0) if write_first_draft_token else 0,
        draft_tokens.stride(0) if write_draft_tokens else 0,
        write_draft_tokens,
        write_first_draft_token,
        BLOCK=_SELECT_TOPK1_BLOCK,
    )
    tree_info = (
        next_scores.unsqueeze(1),  # (b, 1, 1)
        topk_index,  # (b, 1)
        parents,  # (b, 1)
    )
    return input_ids, hidden_states, next_scores, tree_info


def draft_topk1_postprocess(
    next_token_logits: torch.Tensor,
    positions: torch.Tensor,
):
    """Argmax draft logits for topk=1 and advance positions.

    The logits row is the TP-local vocab shard. Inductor lowers this argmax to
    one CTA per row and loops over the whole vocab dimension. This split
    reduction gives the vocab dimension enough parallelism during CUDA graph
    replay.
    """
    bs, vocab_size = next_token_logits.shape
    topk_p = torch.empty((bs, 1), dtype=torch.float32, device=next_token_logits.device)
    topk_index = torch.empty((bs, 1), dtype=torch.int64, device=next_token_logits.device)
    if bs == 0:
        return topk_p, topk_index

    block = _DRAFT_TOPK1_BLOCK
    num_splits = triton.cdiv(vocab_size, block)
    partial_vals = torch.empty(
        (bs, num_splits), dtype=torch.float32, device=next_token_logits.device
    )
    partial_indices = torch.empty(
        (bs, num_splits), dtype=torch.int32, device=next_token_logits.device
    )

    _draft_topk1_partial_argmax_kernel[(bs, num_splits)](
        next_token_logits,
        partial_vals,
        partial_indices,
        vocab_size,
        num_splits,
        BLOCK=block,
        num_warps=8,
    )
    _draft_topk1_finalize_kernel[(bs,)](
        partial_vals,
        partial_indices,
        topk_p,
        topk_index,
        positions,
        num_splits,
        BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return topk_p, topk_index
