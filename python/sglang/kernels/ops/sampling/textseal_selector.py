# SPDX-License-Identifier: Apache-2.0
# Adapted from facebookresearch/textseal's Apache-2.0 selector implementation.

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.sampling.murmur_hash import fmix32, murmur3_mix

_BLOCK_SIZE = 8192
_UINT32_SCALE = tl.constexpr(float(1 << 32))


@triton.jit
def _watermark_partial_argmax_kernel(
    probabilities,
    context_hashes,
    keys,
    partial_scores,
    partial_token_ids,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    token_ids = split * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = token_ids < vocab_size
    candidate_probabilities = tl.load(
        probabilities + row * vocab_size + token_ids,
        mask=in_bounds,
        other=0.0,
    ).to(tl.float32)

    key = tl.load(keys + row).to(tl.uint64)
    state: tl.uint32 = 0
    state = murmur3_mix(state, (key & 0xFFFFFFFF).to(tl.uint32))
    state = murmur3_mix(state, ((key >> 32) & 0xFFFFFFFF).to(tl.uint32))
    state = murmur3_mix(state, tl.load(context_hashes + row).to(tl.uint32))
    state = murmur3_mix(state, token_ids.to(tl.uint32))
    hashed = fmix32(state ^ 16)
    uniform = (hashed.to(tl.float32) + 0.5) / _UINT32_SCALE

    is_candidate = in_bounds & (candidate_probabilities > 0.0)
    safe_probabilities = tl.where(is_candidate, candidate_probabilities, 1.0)
    scores = tl.where(
        is_candidate,
        tl.log(uniform) / safe_probabilities,
        -float("inf"),
    )
    local_index = tl.argmax(scores, axis=0, tie_break_left=True)
    output_offset = row * num_splits + split
    tl.store(partial_scores + output_offset, tl.max(scores, axis=0))
    tl.store(partial_token_ids + output_offset, split * BLOCK_SIZE + local_index)


@triton.jit
def _watermark_finalize_argmax_kernel(
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


def select_watermark_tokens_triton(
    probabilities: torch.Tensor,
    context_hashes: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    if probabilities.ndim != 2 or probabilities.dtype != torch.float32:
        raise ValueError("probabilities must be a 2D float32 tensor")
    if not probabilities.is_cuda:
        raise ValueError("Triton watermark selection requires CUDA tensors")
    batch_size, vocab_size = probabilities.shape
    if context_hashes.shape != (batch_size,) or context_hashes.dtype != torch.int64:
        raise ValueError("context_hashes must be int64 with one value per row")
    if keys.shape != (batch_size,) or keys.dtype != torch.int64:
        raise ValueError("keys must be int64 with one value per row")
    if not all(
        tensor.is_cuda and tensor.is_contiguous()
        for tensor in (probabilities, context_hashes, keys)
    ):
        raise ValueError("Triton watermark inputs must be contiguous CUDA tensors")

    output_token_ids = torch.empty(
        batch_size, dtype=torch.int32, device=probabilities.device
    )
    if batch_size == 0:
        return output_token_ids

    num_splits = triton.cdiv(vocab_size, _BLOCK_SIZE)
    partial_scores = torch.empty(
        (batch_size, num_splits), dtype=torch.float32, device=probabilities.device
    )
    partial_token_ids = torch.empty(
        (batch_size, num_splits), dtype=torch.int32, device=probabilities.device
    )
    _watermark_partial_argmax_kernel[(batch_size, num_splits)](
        probabilities,
        context_hashes,
        keys,
        partial_scores,
        partial_token_ids,
        vocab_size=vocab_size,
        num_splits=num_splits,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
    )
    _watermark_finalize_argmax_kernel[(batch_size,)](
        partial_scores,
        partial_token_ids,
        output_token_ids,
        num_splits=num_splits,
        BLOCK_SIZE=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return output_token_ids
