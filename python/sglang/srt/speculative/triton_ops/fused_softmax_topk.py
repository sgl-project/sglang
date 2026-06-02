# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fused softmax top-k for speculative draft decoding.

Avoids materializing the full ``[batch_size, vocab_size]`` probability tensor.
Each Triton program processes one row: it streams through the logits in tiles,
computes the online log-sum-exp denominator and simultaneously finds the
argmax, then writes ``(softmax_prob, index)`` for the top element.

For ``topk == 1`` (the default in MTP) this is a single-pass fused kernel.
For ``topk > 1`` we fall back to ``torch.softmax`` + ``torch.topk`` since
the register-resident heap approach is fragile across Triton versions.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Triton kernel: fused softmax + argmax (topk=1 specialisation)
# ---------------------------------------------------------------------------


@triton.jit
def _fused_softmax_top1_kernel(
    logits_ptr,
    out_val_ptr,
    out_idx_ptr,
    vocab_size: tl.constexpr,
    stride_row: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """One program per row.

    Pass 1 — find ``row_max`` and ``argmax`` (streaming tiles).
    Pass 2 — accumulate ``sum(exp(x - row_max))`` for the softmax denominator.
    Emit  — ``prob = exp(row_max - (row_max + log(denom))) = 1 / denom * ...``
             simplified to ``prob = 1 / sum(exp(x_i - x_max))`` which equals
             ``softmax(x)[argmax]``.  Equivalently ``exp(0) / denom = 1/denom``
             ... but that's wrong — we need the actual probability of the max
             element which is ``exp(x_max - row_max) / denom = 1.0 / denom``.
             Wait, ``x_max - row_max = 0`` so ``prob = 1.0 / denom``.  That IS
             ``softmax(x)[argmax]``.

    Single-pass variant that finds argmax + accumulates denom simultaneously
    is also possible but the two-pass version is cleaner and the second pass
    is fully memory-bound (same data, no branches).
    """
    row = tl.program_id(0)
    base = row * stride_row

    # -- pass 1: row-max + argmax -----------------------------------------
    row_max = float("-inf")
    best_idx = tl.cast(0, tl.int64)

    for start in range(0, vocab_size, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < vocab_size
        vals = tl.load(logits_ptr + base + offs, mask=mask, other=float("-inf"))
        tile_max = tl.max(vals)
        if tile_max > row_max:
            # Find position of the max within this tile.
            is_max = vals == tile_max
            pos = tl.argmin(tl.where(is_max, 0, 1), axis=0)
            best_idx = tl.cast(start + pos, tl.int64)
            row_max = tile_max

    # -- pass 2: softmax denominator (numerically stable) ------------------
    denom = 0.0
    for start in range(0, vocab_size, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < vocab_size
        vals = tl.load(logits_ptr + base + offs, mask=mask, other=float("-inf"))
        denom += tl.sum(tl.exp(vals - row_max))

    # prob(argmax) = exp(x_max - x_max) / denom = 1.0 / denom
    tl.store(out_val_ptr + row, 1.0 / denom)
    tl.store(out_idx_ptr + row, best_idx)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def fused_softmax_topk(
    logits: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused softmax top-k without materializing the full probability tensor.

    Args:
        logits: ``[batch_size, vocab_size]`` float32 or bfloat16 tensor.
        topk: Number of top elements to return.

    Returns:
        ``(topk_probs, topk_indices)`` each of shape ``[batch_size, topk]``.
        ``topk_probs`` is float32; ``topk_indices`` is int64.

    For ``topk == 1`` uses a fused Triton kernel (no full-vocab allocation).
    For ``topk > 1`` falls back to ``torch.softmax`` + ``torch.topk``.
    """
    if topk != 1 or not logits.is_cuda:
        # Fallback: materialise full softmax.
        probs = torch.softmax(logits.float(), dim=-1)
        return torch.topk(probs, topk, dim=-1)

    batch_size, vocab_size = logits.shape

    # Cast to float32 for numerically stable softmax accumulation.
    if logits.dtype != torch.float32:
        logits = logits.float()

    # Ensure contiguous row-major layout.
    if not logits.is_contiguous():
        logits = logits.contiguous()

    out_vals = torch.empty((batch_size, 1), dtype=torch.float32, device=logits.device)
    out_idxs = torch.empty((batch_size, 1), dtype=torch.int64, device=logits.device)

    # Choose tile size: power-of-2 that balances occupancy vs register usage.
    if vocab_size >= 65536:
        BLOCK_V = 4096
    elif vocab_size >= 16384:
        BLOCK_V = 2048
    else:
        BLOCK_V = 1024

    grid = (batch_size,)
    _fused_softmax_top1_kernel[grid](
        logits,
        out_vals,
        out_idxs,
        vocab_size=vocab_size,
        stride_row=logits.stride(0),
        BLOCK_V=BLOCK_V,
    )
    return out_vals, out_idxs
