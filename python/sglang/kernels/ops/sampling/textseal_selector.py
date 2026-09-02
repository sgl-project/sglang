from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK_SIZE = 8192
_M = tl.constexpr(8191)
_P2 = tl.constexpr(100000007)
_P3 = tl.constexpr(500001713)
_P4 = tl.constexpr(15485863)
_MIXING_PRIME = tl.constexpr(40499)
_MIXING_SHIFT = tl.constexpr(13)


@triton.jit
def _textseal_partial_argmax_kernel(
    probs,
    token_ids,
    weighted_contexts,
    keys,
    partial_scores,
    partial_token_ids,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    offsets = split * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offsets < vocab_size
    row_offsets = row * vocab_size + offsets
    candidate_probs = tl.load(probs + row_offsets, mask=in_bounds, other=0.0).to(
        tl.float32
    )
    out_offset = row * num_splits + split

    if tl.max(candidate_probs, axis=0) > 0.0:
        candidate_token_ids = tl.load(
            token_ids + row_offsets, mask=in_bounds, other=0
        ).to(tl.int64)
        weighted_context = tl.load(weighted_contexts + row).to(tl.int64)
        key = tl.load(keys + row).to(tl.int64)
        hashed = (weighted_context + _P2 * candidate_token_ids + _P3 * key) * _P4
        hashed = hashed * _MIXING_PRIME
        hashed = hashed ^ (hashed >> _MIXING_SHIFT)
        hashed = hashed % _M
        hashed = tl.where(hashed < 0, hashed + _M, hashed)
        uniform_scores = hashed.to(tl.float32) / _M
        is_candidate = in_bounds & (candidate_probs > 0.0)
        safe_probs = tl.where(is_candidate, candidate_probs, 1.0)
        selection_scores = tl.where(
            is_candidate,
            tl.log(uniform_scores + 1e-30) / safe_probs,
            -float("inf"),
        )
        local_index = tl.argmax(selection_scores, axis=0, tie_break_left=True)
        best_score = tl.max(selection_scores, axis=0)
        best_token_id = tl.load(
            token_ids + row * vocab_size + split * BLOCK_SIZE + local_index
        )
        tl.store(partial_scores + out_offset, best_score)
        tl.store(partial_token_ids + out_offset, best_token_id)
    else:
        tl.store(partial_scores + out_offset, -float("inf"))
        tl.store(partial_token_ids + out_offset, 0)


@triton.jit
def _textseal_finalize_argmax_kernel(
    partial_scores,
    partial_token_ids,
    output_token_ids,
    num_splits: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_splits
    scores = tl.load(
        partial_scores + row * num_splits + offsets,
        mask=mask,
        other=-float("inf"),
    )
    split = tl.argmax(scores, axis=0, tie_break_left=True)
    token_id = tl.load(partial_token_ids + row * num_splits + split)
    tl.store(output_token_ids + row, token_id)


def select_textseal_tokens_triton(
    effective_probs: torch.Tensor,
    token_ids: torch.Tensor,
    weighted_contexts: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    if effective_probs.ndim != 2:
        raise ValueError("effective_probs must be a 2D tensor")
    if effective_probs.dtype != torch.float32:
        raise ValueError("effective_probs must use float32")
    if not effective_probs.is_cuda:
        raise ValueError("TextSeal Triton selection requires CUDA tensors")
    if token_ids.shape != effective_probs.shape or token_ids.dtype != torch.int64:
        raise ValueError("token_ids must be int64 and match effective_probs")
    batch_size, vocab_size = effective_probs.shape
    if (
        weighted_contexts.shape != (batch_size,)
        or weighted_contexts.dtype != torch.int64
    ):
        raise ValueError("weighted_contexts must be int64 with one value per row")
    if keys.shape != (batch_size,) or keys.dtype != torch.int64:
        raise ValueError("keys must be int64 with one value per row")
    if not all(
        tensor.is_cuda and tensor.is_contiguous()
        for tensor in (effective_probs, token_ids, weighted_contexts, keys)
    ):
        raise ValueError("TextSeal Triton inputs must be contiguous CUDA tensors")

    output_token_ids = torch.empty(
        batch_size, dtype=torch.int32, device=effective_probs.device
    )
    if batch_size == 0:
        return output_token_ids

    num_splits = triton.cdiv(vocab_size, _BLOCK_SIZE)
    partial_scores = torch.empty(
        (batch_size, num_splits), dtype=torch.float32, device=effective_probs.device
    )
    partial_token_ids = torch.empty(
        (batch_size, num_splits), dtype=torch.int32, device=effective_probs.device
    )
    _textseal_partial_argmax_kernel[(batch_size, num_splits)](
        effective_probs,
        token_ids,
        weighted_contexts,
        keys,
        partial_scores,
        partial_token_ids,
        vocab_size=vocab_size,
        num_splits=num_splits,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
    )
    _textseal_finalize_argmax_kernel[(batch_size,)](
        partial_scores,
        partial_token_ids,
        output_token_ids,
        num_splits=num_splits,
        BLOCK_SIZE=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return output_token_ids
