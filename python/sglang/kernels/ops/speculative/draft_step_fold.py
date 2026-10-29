"""FN draft top1 finalization -> the next captured BF16 embedding slot.

The vocabulary projection and native split argmax are unchanged. Each capture
owns its scratch; nothing is memoized across requests or graph variants.
"""
import logging
import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.topk1 import (
    _DRAFT_TOPK1_BLOCK,
    _draft_topk1_partial_argmax_kernel,
)


class FoldSlot(NamedTuple):
    weight: torch.Tensor
    embedding: torch.Tensor
    topk_p: torch.Tensor
    topk_index: torch.Tensor
    partial_values: torch.Tensor
    partial_indices: torch.Tensor


def make_fold_slots(model, batch_size, num_steps):
    """Called before capture; unsupported models/shapes keep the native path."""
    if os.environ.get("Q38_MTP_STEP_INPUT_FOLD", "0") != "1":
        return None
    from sglang.srt.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )
    from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP

    if not isinstance(model, Qwen4ExpForCausalLMMTP):
        return None
    embed = model.model.embed_tokens
    weight = embed.weight
    if not (
        model.hc_count == 4
        and num_steps == 3
        and 1 <= batch_size <= 4
        and embed.tp_size == 1
        and isinstance(embed.quant_method, UnquantizedEmbeddingMethod)
        and embed.output_dtype in (None, torch.bfloat16)
        and embed.num_embeddings == embed.org_vocab_size
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
        and weight.ndim == 2
        and weight.shape[1] == 2560
        and weight.is_contiguous()
    ):
        return None
    # K3 has two draft forwards. Only the first finalizer feeds another forward;
    # the last finalizer and the draft-extend boundary remain native in v1.
    splits = triton.cdiv(embed.num_embeddings, _DRAFT_TOPK1_BLOCK)
    device = weight.device
    slot = FoldSlot(
        weight,
        torch.empty((batch_size, 2560), dtype=torch.bfloat16, device=device),
        torch.empty((batch_size, 1), dtype=torch.float32, device=device),
        torch.empty((batch_size, 1), dtype=torch.int64, device=device),
        torch.empty((batch_size, splits), dtype=torch.float32, device=device),
        torch.empty((batch_size, splits), dtype=torch.int32, device=device),
    )
    logging.getLogger(__name__).info(
        "Q38_MTP_STEP_INPUT_FOLD capture slot attached: rows=%s HC4 BF16 H2560",
        batch_size,
    )
    return (slot,)


@triton.jit
def _finalize_and_embed(
    partial_vals,
    partial_indices,
    topk_p,
    topk_index,
    positions,
    draft_tokens,
    draft_tokens_stride,
    draft_token_column,
    weight,
    embedding,
    weight_stride,
    mrope,
    mrope_axis_stride,
    mrope_row_stride,
    num_splits: tl.constexpr,
    WRITE_MROPE: tl.constexpr,
    HIDDEN: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
    EMBED_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, REDUCE_BLOCK)
    vals = tl.load(
        partial_vals + row * num_splits + offsets,
        mask=offsets < num_splits,
        other=-float("inf"),
    )
    # Keep the mature finalizer's exact reduction and first-split tie break.
    split = tl.argmax(vals, axis=0)
    index = tl.load(partial_indices + row * num_splits + split).to(tl.int64)
    tl.store(topk_index + row, index)
    tl.store(topk_p + row, 1.0)
    tl.store(draft_tokens + row * draft_tokens_stride + draft_token_column, index)
    position = tl.load(positions + row)
    tl.store(positions + row, position + 1)
    if WRITE_MROPE:
        for axis in tl.static_range(3):
            ptr = mrope + axis * mrope_axis_stride + row * mrope_row_stride
            tl.store(ptr, tl.load(ptr) + 1)

    col = tl.arange(0, EMBED_BLOCK)
    value = tl.load(weight + index * weight_stride + col, col < HIDDEN, other=0)
    tl.store(embedding + row * HIDDEN + col, value, col < HIDDEN)


def finalize_next_input(
    logits, positions, draft_tokens, column, slot, mrope_positions=None
):
    """Original partial argmax plus finalization/gather into a capture-owned slot."""
    bs, vocab = logits.shape
    splits = slot.partial_values.shape[1]
    assert bs == slot.embedding.shape[0]
    assert vocab <= slot.weight.shape[0]
    assert splits == triton.cdiv(vocab, _DRAFT_TOPK1_BLOCK)
    _draft_topk1_partial_argmax_kernel[(bs, splits)](
        logits,
        slot.partial_values,
        slot.partial_indices,
        logits.stride(0),
        vocab,
        splits,
        BLOCK=_DRAFT_TOPK1_BLOCK,
        num_warps=8,
    )
    _finalize_and_embed[(bs,)](
        slot.partial_values,
        slot.partial_indices,
        slot.topk_p,
        slot.topk_index,
        positions,
        draft_tokens,
        draft_tokens.stride(0),
        column,
        slot.weight,
        slot.embedding,
        slot.weight.stride(0),
        mrope_positions,
        mrope_positions.stride(0) if mrope_positions is not None else 0,
        mrope_positions.stride(1) if mrope_positions is not None else 0,
        splits,
        WRITE_MROPE=mrope_positions is not None,
        HIDDEN=2560,
        REDUCE_BLOCK=triton.next_power_of_2(splits),
        EMBED_BLOCK=4096,
        num_warps=4,
    )
    return slot.topk_p, slot.topk_index
