"""CUDA kernels and tensor transforms for simple QSA."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


def average_pool_qsa_keys(key_groups: torch.Tensor) -> torch.Tensor:
    """FP32-average complete key groups shaped ``[groups, ratio, kv_heads, dim]``."""

    if key_groups.ndim != 4:
        raise ValueError(
            "QSA key groups must be [groups, ratio, kv_heads, head_dim], "
            f"got {key_groups.shape}"
        )
    return key_groups.float().mean(dim=1).to(key_groups.dtype)


def qsa_fast_topk(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Select compressed blocks, with a compatibility fallback for top-k 512."""

    lengths = (row_ends - row_starts).to(device=logits.device, dtype=torch.int32)
    starts = row_starts.to(device=logits.device, dtype=torch.int32)
    if logits.is_cuda:
        if topk == 512:
            # Prefer the JIT kernel: it ships with the sglang python package,
            # so top-k 512 works regardless of the installed sgl_kernel version.
            from sglang.kernels.ops.elementwise.fast_topk import fast_topk

            return fast_topk(logits, lengths, topk=512, row_starts=starts)

        from sgl_kernel import top_k as top_k_module

        supported_topk = getattr(
            top_k_module, "_FAST_TOPK_SUPPORTED_K", (2048,)
        )
        if topk in supported_topk:
            return top_k_module.fast_topk_v2(
                logits, lengths, topk=topk, row_starts=starts
            )
        if topk != 512 or 2048 not in supported_topk:
            raise ValueError(
                f"QSA top-k {topk} is unsupported by sgl_kernel; "
                f"supported values are {supported_topk}"
            )
        candidates = top_k_module.fast_topk_v2(
            logits, lengths, topk=2048, row_starts=starts
        )
        return _rerank_qsa_topk_candidates(logits, candidates, starts, topk)

    # CPU/reference path mirrors the CUDA operator's fixed-width, relative output.
    output = torch.full(
        (logits.shape[0], topk),
        -1,
        dtype=torch.int32,
        device=logits.device,
    )
    for row in range(logits.shape[0]):
        start = int(starts[row])
        length = int(lengths[row])
        width = min(length, topk)
        if width:
            output[row, :width] = torch.topk(
                logits[row, start : start + length], width
            ).indices.to(torch.int32)
    return output


def _rerank_qsa_topk_candidates(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    row_starts: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Rerank relative candidate indices with their original logits."""

    valid = candidates >= 0
    absolute = candidates.long() + row_starts.long().unsqueeze(1)
    safe_absolute = absolute.clamp(min=0, max=max(logits.shape[1] - 1, 0))
    scores = torch.gather(logits, 1, safe_absolute)
    scores = scores.masked_fill(~valid, float("-inf"))
    selected = torch.topk(scores, k=topk, dim=1).indices
    return torch.gather(candidates, 1, selected).to(torch.int32)


def torch_expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """Expand compressed block indices into fixed-width logical token indices."""

    block_topk = (token_topk + compress_ratio - 1) // compress_ratio
    final_topk = token_topk + compress_ratio - 1
    if block_indices.ndim != 2 or block_indices.shape[1] != block_topk:
        raise ValueError(
            f"expected block indices [M, {block_topk}], got "
            f"{tuple(block_indices.shape)}"
        )
    rows = block_indices.shape[0]
    if query_positions.numel() != rows or sequence_lengths.numel() != rows:
        raise ValueError("query positions and sequence lengths must match top-k rows")

    device = block_indices.device
    blocks = block_indices.long()
    offsets = torch.arange(compress_ratio, device=device, dtype=torch.long)
    expanded = blocks.unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(
        blocks.unsqueeze(-1) >= 0, expanded, torch.full_like(expanded, -1)
    ).reshape(rows, block_topk * compress_ratio)
    expanded = expanded[:, :token_topk]

    query_positions = query_positions.to(device=device, dtype=torch.long)
    sequence_lengths = sequence_lengths.to(device=device, dtype=torch.long)
    expanded = torch.where(
        (expanded >= 0) & (expanded < sequence_lengths.unsqueeze(1)),
        expanded,
        torch.full_like(expanded, -1),
    )

    tail_offsets = torch.arange(compress_ratio - 1, device=device, dtype=torch.long)
    visible_tokens = query_positions + 1
    tail_start = (
        torch.div(visible_tokens, compress_ratio, rounding_mode="floor")
        * compress_ratio
    )
    tail_count = visible_tokens - tail_start
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_valid = (tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1)) & (
        tail < sequence_lengths.unsqueeze(1)
    )
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))

    result = torch.cat([expanded, tail], dim=1)
    # Keep all valid entries contiguous. This is required by the FA2 packing path.
    order = torch.arange(final_topk, device=device).unsqueeze(0).expand(rows, -1)
    sort_key = torch.where(result >= 0, order, order + final_topk)
    return result.gather(1, torch.argsort(sort_key, dim=1, stable=True)).to(torch.int32)


@triton.jit
def _expand_qsa_block_indices_kernel(
    block_indices,
    query_positions,
    sequence_lengths,
    output,
    block_stride: tl.constexpr,
    output_stride: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_TOPK: tl.constexpr,
    FINAL_TOPK: tl.constexpr,
    OUTPUT_BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, OUTPUT_BLOCK_SIZE)
    sequence_length = tl.load(sequence_lengths + row)

    source_block_cols = cols // COMPRESS_RATIO
    offsets = cols % COMPRESS_RATIO
    blocks = tl.load(
        block_indices + row * block_stride + source_block_cols,
        mask=(cols < TOKEN_TOPK) & (source_block_cols < BLOCK_TOPK),
        other=-1,
    )
    expanded = blocks * COMPRESS_RATIO + offsets
    expanded_valid = (
        (cols < TOKEN_TOPK)
        & (blocks >= 0)
        & (expanded >= 0)
        & (expanded < sequence_length)
    )

    valid_block_count = tl.sum(
        (
            (cols < BLOCK_TOPK)
            & (
                tl.load(
                    block_indices + row * block_stride + cols,
                    mask=cols < BLOCK_TOPK,
                    other=-1,
                )
                >= 0
            )
        ).to(tl.int32),
        axis=0,
    )
    valid_token_count = tl.minimum(
        valid_block_count * COMPRESS_RATIO, TOKEN_TOPK
    )

    query_position = tl.load(query_positions + row)
    visible_tokens = query_position + 1
    tail_start = (visible_tokens // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_offset = cols - valid_token_count
    tail_count = visible_tokens - tail_start
    tail = tail_start + tail_offset
    tail_valid = (
        (tail_offset >= 0)
        & (tail_offset < COMPRESS_RATIO - 1)
        & (tail_offset < tail_count)
        & (tail < sequence_length)
    )

    result = tl.where(
        expanded_valid & (cols < valid_token_count),
        expanded,
        tl.where(tail_valid, tail, -1),
    )
    tl.store(
        output + row * output_stride + cols,
        result,
        mask=cols < FINAL_TOPK,
    )


def triton_expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """CUDA fast path for fast_topk_v2 output (valid blocks precede -1 padding)."""
    rows, block_topk = block_indices.shape
    final_topk = token_topk + compress_ratio - 1
    output = torch.empty(
        (rows, final_topk), dtype=torch.int32, device=block_indices.device
    )
    if rows == 0:
        return output
    _expand_qsa_block_indices_kernel[(rows,)](
        block_indices,
        query_positions,
        sequence_lengths,
        output,
        block_indices.stride(0),
        output.stride(0),
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_TOPK=token_topk,
        FINAL_TOPK=final_topk,
        OUTPUT_BLOCK_SIZE=triton.next_power_of_2(final_topk),
        num_warps=8,
    )
    return output


def expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """Expand compressed blocks with Triton on CUDA and Torch elsewhere."""

    block_topk = (token_topk + compress_ratio - 1) // compress_ratio
    if block_indices.ndim != 2 or block_indices.shape[1] != block_topk:
        raise ValueError(
            f"expected block indices [M, {block_topk}], got "
            f"{tuple(block_indices.shape)}"
        )
    rows = block_indices.shape[0]
    if query_positions.numel() != rows or sequence_lengths.numel() != rows:
        raise ValueError("query positions and sequence lengths must match top-k rows")
    if block_indices.is_cuda:
        # The Triton kernel loads positions/lengths as scalars, so any integer
        # dtype works; skip the int64 conversion copies.
        return triton_expand_qsa_block_indices(
            block_indices.contiguous(),
            query_positions.to(device=block_indices.device).contiguous(),
            sequence_lengths.to(device=block_indices.device).contiguous(),
            compress_ratio,
            token_topk,
        )
    return torch_expand_qsa_block_indices(
        block_indices,
        query_positions,
        sequence_lengths,
        compress_ratio,
        token_topk,
    )


def qsa_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_slots: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Torch reference for sparse GQA over physical token slots."""

    if q.ndim != 3 or k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("q, k_cache and v_cache must be rank-3 tensors")
    if token_slots.ndim != 2 or token_slots.shape[0] != q.shape[0]:
        raise ValueError(
            "token slots must be [query_tokens, selected_tokens], got "
            f"{token_slots.shape}"
        )
    if q.shape[-1] != k_cache.shape[-1] or q.shape[-1] != v_cache.shape[-1]:
        raise ValueError("Q/K/V head dimensions must match")
    if q.shape[1] % k_cache.shape[1] != 0:
        raise ValueError("query heads must be divisible by KV heads")
    return qsa_sparse_attention_reference(
        q, k_cache, v_cache, token_slots, softmax_scale
    )


def qsa_sparse_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_slots: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Device-agnostic sparse GQA reference."""

    scale = softmax_scale or q.shape[-1] ** -0.5
    outputs = []
    repeats = q.shape[1] // k_cache.shape[1]
    for row in range(q.shape[0]):
        valid = token_slots[row] >= 0
        slots = token_slots[row, valid].long()
        if slots.numel() == 0:
            outputs.append(torch.zeros_like(q[row]))
            continue
        keys = k_cache.index_select(0, slots).repeat_interleave(repeats, dim=1)
        values = v_cache.index_select(0, slots).repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys.float()) * scale
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(
            torch.einsum("hk,khd->hd", probabilities, values.float()).to(q.dtype)
        )
    return torch.stack(outputs)


__all__ = [
    "average_pool_qsa_keys",
    "expand_qsa_block_indices",
    "torch_expand_qsa_block_indices",
    "triton_expand_qsa_block_indices",
    "qsa_fast_topk",
    "qsa_sparse_attention",
    "qsa_sparse_attention_reference",
]
