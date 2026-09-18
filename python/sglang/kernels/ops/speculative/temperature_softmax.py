# Copyright 2023-2024 SGLang Team
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


"""Split-row temperature softmax for the speculative sampling shapes."""

import torch
import triton
import triton.language as tl

# torch reduces each softmax row with one block (cunn_SoftMaxForwardGmem at a
# 154880-wide vocab), so the speculative row counts -- batch_size per draft step,
# batch_size * num_draft_tokens at verify -- occupy 1 to 12 blocks of a 256-CU
# GPU and the row is read three times. These kernels split each row across CTAs
# and fold the temperature divide in, which also drops the separate div pass
# that materialized logits / temperatures.
#
# MI355X, vocab 154880, timed under CUDA graph replay -- the path decode runs
# on -- as torch -> split us/call: 59.8 -> 13.1 at 1 row (4.6x), 59.9 -> 13.4 at
# 2, 62.4 -> 15.6 at 12, 70.4 -> 25.7 at 32 (2.7x). Eager (launch cost not
# amortized) is only 1.1-1.4x, so the win is a graph-replay win.
_MIN_BLOCK = 2048
_TARGET_CTAS = 512
# Above this many rows one block per row already fills the GPU, and the split
# only adds two launches plus the partial buffers.
ROW_LIMIT = 32


@triton.jit
def _partial_max_sumexp_kernel(
    logits,
    temperatures,
    partial_max,
    partial_sum,
    logits_row_stride,
    vocab_size,
    num_splits,
    BLOCK: tl.constexpr,
):
    # int64 row base: row * stride overflows int32 once rows * vocab reaches 2^31.
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    t = tl.load(temperatures + row)
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    x = tl.load(
        logits + row * logits_row_stride + offsets,
        mask=mask,
        other=-float("inf"),
    )
    x = x / t
    block_max = tl.max(x, axis=0)
    block_sum = tl.sum(tl.where(mask, tl.exp(x - block_max), 0.0), axis=0)
    out_offset = row * num_splits + split
    tl.store(partial_max + out_offset, block_max)
    tl.store(partial_sum + out_offset, block_sum)


@triton.jit
def _combine_kernel(
    partial_max,
    partial_sum,
    row_max,
    row_sum,
    num_splits,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_splits
    m = tl.load(
        partial_max + row * num_splits + offsets, mask=mask, other=-float("inf")
    )
    s = tl.load(partial_sum + row * num_splits + offsets, mask=mask, other=0.0)
    m_all = tl.max(m, axis=0)
    # Rescale each block's sum to the row max before adding: the blocks
    # subtracted their own max, exactly as an online softmax merge does.
    s_all = tl.sum(tl.where(mask, s * tl.exp(m - m_all), 0.0), axis=0)
    tl.store(row_max + row, m_all)
    tl.store(row_sum + row, s_all)


@triton.jit
def _normalize_kernel(
    logits,
    temperatures,
    row_max,
    row_sum,
    probs,
    logits_row_stride,
    probs_row_stride,
    vocab_size,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    t = tl.load(temperatures + row)
    m = tl.load(row_max + row)
    s = tl.load(row_sum + row)
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    x = tl.load(logits + row * logits_row_stride + offsets, mask=mask, other=0.0)
    p = tl.exp(x / t - m) / s
    tl.store(probs + row * probs_row_stride + offsets, p, mask=mask)


def _split_geometry(rows: int, vocab_size: int):
    """Block width and split count that keep the grid near _TARGET_CTAS.

    Cost is flat from 76 to 606 CTAs (the three launches dominate once the row
    is spread at all), so the block floor only has to stay wide enough to keep
    the loads coalesced.
    """
    splits = min(
        triton.cdiv(_TARGET_CTAS, rows), max(1, triton.cdiv(vocab_size, _MIN_BLOCK))
    )
    block = max(_MIN_BLOCK, triton.next_power_of_2(triton.cdiv(vocab_size, splits)))
    return block, triton.cdiv(vocab_size, block)


def temperature_softmax(
    logits: torch.Tensor, temperatures: torch.Tensor
) -> torch.Tensor:
    """``softmax(logits / temperatures, dim=-1)`` with the row split across CTAs.

    ``temperatures`` is one value per row, in any shape that flattens to the row
    count. Falls back to the torch pair for the shapes the split does not pay
    for, so callers can use it unconditionally.
    """
    if (
        logits.ndim != 2
        or logits.dtype != torch.float32
        or logits.shape[0] == 0
        or logits.shape[0] > ROW_LIMIT
        or logits.stride(1) != 1
        or not logits.is_cuda
    ):
        return torch.softmax(logits / temperatures, dim=-1)

    rows, vocab_size = logits.shape
    t = temperatures.reshape(-1)
    if t.shape[0] != rows or t.stride(0) != 1 or t.dtype != logits.dtype:
        return torch.softmax(logits / temperatures, dim=-1)

    block, num_splits = _split_geometry(rows, vocab_size)
    probs = torch.empty_like(logits)
    partial_max = torch.empty(
        (rows, num_splits), dtype=torch.float32, device=logits.device
    )
    partial_sum = torch.empty(
        (rows, num_splits), dtype=torch.float32, device=logits.device
    )
    row_max = torch.empty((rows,), dtype=torch.float32, device=logits.device)
    row_sum = torch.empty((rows,), dtype=torch.float32, device=logits.device)

    _partial_max_sumexp_kernel[(rows, num_splits)](
        logits,
        t,
        partial_max,
        partial_sum,
        logits.stride(0),
        vocab_size,
        num_splits,
        BLOCK=block,
        num_warps=8,
    )
    _combine_kernel[(rows,)](
        partial_max,
        partial_sum,
        row_max,
        row_sum,
        num_splits,
        BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    _normalize_kernel[(rows, num_splits)](
        logits,
        t,
        row_max,
        row_sum,
        probs,
        logits.stride(0),
        probs.stride(0),
        vocab_size,
        BLOCK=block,
        num_warps=8,
    )
    return probs
