# SPDX-License-Identifier: Apache-2.0
"""Clover block penalties based only on committed response history."""

import torch
import triton
import triton.language as tl


@triton.jit
def _clover_penalty(
    logits,
    counts,
    penalty_values,
    row_stride,
    VOCAB: tl.constexpr,
    WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
    FREQUENCY: tl.constexpr,
    PRESENCE: tl.constexpr,
    REPETITION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    columns = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows = tl.arange(0, ROWS)
    history = tl.load(counts + columns, columns < VOCAB, other=0)
    seen = history > 0
    offsets = rows[:, None] * row_stride + columns[None, :]
    mask = (rows[:, None] < WIDTH) & (columns[None, :] < VOCAB)
    values = tl.load(logits + offsets, mask, other=0)
    if FREQUENCY:
        frequency = tl.load(penalty_values)
        values = values - (history * frequency)[None, :]
    if PRESENCE:
        presence = tl.load(penalty_values + 1)
        values = values - tl.where(seen, presence, 0.0)[None, :]
    if REPETITION:
        repetition = tl.load(penalty_values + 2)
        scale = tl.where(seen, repetition, 1.0)[None, :]
        values = tl.where(values < 0, values * scale, values / scale)
    tl.store(logits + offsets, values, mask)


def clover_penalty(
    logits,
    counts,
    params,
    *,
    frequency=True,
    presence=True,
    repetition=True,
):
    """In-place FP32 [width,vocab] penalties using committed counts only.

    Params is a device FP32 vector [frequency, presence, repetition]. Additive
    operations retain their rounding boundaries (FMA disabled), followed by
    sign-dependent division/multiplication for repetition penalties.
    """
    assert logits.is_cuda and logits.dtype == torch.float32 and logits.ndim == 2
    width, vocab = logits.shape
    assert 1 <= width <= 128 and vocab > 0 and logits.stride(1) == 1
    assert counts.shape == (vocab,) and counts.is_contiguous()
    assert counts.dtype == params.dtype == torch.float32
    assert params.shape == (3,) and params.is_contiguous()
    assert counts.device == params.device == logits.device
    _clover_penalty[(triton.cdiv(vocab, 256),)](
        logits,
        counts,
        params,
        logits.stride(0),
        VOCAB=vocab,
        WIDTH=width,
        ROWS=triton.next_power_of_2(width),
        FREQUENCY=frequency,
        PRESENCE=presence,
        REPETITION=repetition,
        BLOCK=256,
        num_warps=4 if width <= 16 else 8,
        enable_fp_fusion=False,
    )
