# SPDX-License-Identifier: Apache-2.0
"""Clover block penalties based only on committed response history."""

import torch
import triton
import triton.language as tl


@triton.jit
def _clover_penalty(
    logits,
    additive,
    scaling,
    row_stride,
    VOCAB: tl.constexpr,
    WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
    ADDITIVE: tl.constexpr,
    SCALING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    columns = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if ADDITIVE:
        biases = tl.load(additive + columns, columns < VOCAB, other=0)
    if SCALING:
        scales = tl.load(scaling + columns, columns < VOCAB, other=1)
    rows = tl.arange(0, ROWS)
    offsets = rows[:, None] * row_stride + columns[None, :]
    mask = (rows[:, None] < WIDTH) & (columns[None, :] < VOCAB)
    values = tl.load(logits + offsets, mask, other=0)
    if ADDITIVE:
        values = values + biases[None, :]
    if SCALING:
        scale = scales[None, :]
        values = tl.where(values < 0, values * scale, values / scale)
    tl.store(logits + offsets, values, mask)


def clover_penalty(logits, additive=None, scaling=None):
    """In-place FP32 [width,vocab] penalties from native per-request buffers.

    Broadcast additive penalties across the block, then apply sign-dependent
    repetition scaling.
    """
    assert logits.is_cuda and logits.dtype == torch.float32 and logits.ndim == 2
    width, vocab = logits.shape
    assert 1 <= width <= 128 and vocab > 0 and logits.stride(1) == 1
    for buf in (additive, scaling):
        if buf is not None:
            assert buf.shape == (vocab,) and buf.is_contiguous()
            assert buf.dtype == torch.float32 and buf.device == logits.device
    _clover_penalty[(triton.cdiv(vocab, 256),)](
        logits,
        additive,
        scaling,
        logits.stride(0),
        VOCAB=vocab,
        WIDTH=width,
        ROWS=triton.next_power_of_2(width),
        ADDITIVE=additive is not None,
        SCALING=scaling is not None,
        BLOCK=256,
        num_warps=4 if width <= 16 else 8,
        enable_fp_fusion=False,
    )
