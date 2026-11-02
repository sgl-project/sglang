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
"""Sparse trie child masks over a codebook, for trie-conditioned decoding."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _materialize_mask_kernel(
    pair_rows,
    offsets,
    values,
    output,
    output_row_stride: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    pair = tl.load(pair_rows + row).to(tl.int64)
    start = tl.load(offsets + pair).to(tl.int64)
    end = tl.load(offsets + pair + 1).to(tl.int64)
    positions = tl.arange(0, block_size)
    valid = positions < end - start
    children = tl.load(
        values + start + positions,
        mask=valid,
        other=0,
    ).to(tl.int64)
    tl.store(
        output + row * output_row_stride + children,
        1,
        mask=valid,
    )


@triton.jit
def _mask_logits_kernel(
    logits,
    pair_rows,
    offsets,
    values,
    output,
    logits_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    pair = tl.load(pair_rows + row).to(tl.int64)
    start = tl.load(offsets + pair).to(tl.int64)
    end = tl.load(offsets + pair + 1).to(tl.int64)
    positions = tl.arange(0, block_size)
    valid = positions < end - start
    children = tl.load(
        values + start + positions,
        mask=valid,
        other=0,
    ).to(tl.int64)
    scores = tl.load(
        logits + row * logits_row_stride + children,
        mask=valid,
    )
    tl.store(
        output + row * output_row_stride + children,
        scores,
        mask=valid,
    )


@triton.jit
def _mask_sparse_prefix_logits_kernel(
    logits,
    prefixes,
    keys,
    offsets,
    values,
    output,
    fallback,
    valid_rows,
    num_keys,
    codebook_size: tl.constexpr,
    logits_row_stride: tl.constexpr,
    logits_column_stride: tl.constexpr,
    prefix_row_stride: tl.constexpr,
    prefix_column_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    child_block: tl.constexpr,
):
    row = tl.program_id(0)
    first = tl.load(prefixes + row * prefix_row_stride).to(tl.int64)
    second = tl.load(prefixes + row * prefix_row_stride + prefix_column_stride).to(
        tl.int64
    )
    key = (first * codebook_size + second).to(keys.dtype.element_ty)
    low = 0
    high = num_keys
    while low < high:
        middle = (low + high) // 2
        found = tl.load(keys + middle)
        less = found < key
        low = tl.where(less, middle + 1, low)
        high = tl.where(less, high, middle)
    safe = tl.minimum(low, num_keys - 1)
    present = (low < num_keys) & (tl.load(keys + safe) == key)
    start = tl.load(offsets + safe).to(tl.int64)
    end = tl.load(offsets + safe + 1).to(tl.int64)
    first_child = tl.load(values + start).to(tl.int64)
    positions = tl.arange(0, child_block)
    valid = present & (positions < end - start)
    children = tl.load(values + start + positions, mask=valid, other=0).to(tl.int64)
    scores = tl.load(
        logits + row * logits_row_stride + children * logits_column_stride,
        mask=valid,
        other=0,
    )
    tl.store(output + row * output_row_stride + children, scores, mask=valid)
    tl.store(fallback + row, first_child)
    tl.store(valid_rows + row, present)


def mask_sparse_prefix_logits(
    logits: torch.Tensor,
    prefixes: torch.Tensor,
    keys: torch.Tensor,
    offsets: torch.Tensor,
    values: torch.Tensor,
    max_children: int,
    codebook_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        logits.ndim != 2
        or logits.shape[1] != codebook_size
        or prefixes.shape != (logits.shape[0], 2)
        or not keys.numel()
    ):
        raise ValueError("Sparse trie prefixes and logits do not align")
    output = torch.full_like(logits, -torch.inf, memory_format=torch.contiguous_format)
    fallback = torch.empty(
        (logits.shape[0], 1), dtype=torch.int64, device=logits.device
    )
    valid_rows = torch.empty(logits.shape[0], dtype=torch.bool, device=logits.device)
    if logits.shape[0]:
        _mask_sparse_prefix_logits_kernel[(logits.shape[0],)](
            logits,
            prefixes,
            keys,
            offsets,
            values,
            output,
            fallback,
            valid_rows,
            keys.numel(),
            codebook_size=codebook_size,
            logits_row_stride=logits.stride(0),
            logits_column_stride=logits.stride(1),
            prefix_row_stride=prefixes.stride(0),
            prefix_column_stride=prefixes.stride(1),
            output_row_stride=output.stride(0),
            child_block=triton.next_power_of_2(max_children),
            num_warps=8,
        )
        torch._assert_async(
            torch.all(valid_rows), "Trie prefix contains an unknown AB pair"
        )
    return output, fallback


def materialize_sparse_mask(
    pair_rows: torch.Tensor,
    offsets: torch.Tensor,
    values: torch.Tensor,
    codebook_size: int,
    max_children: int,
) -> torch.Tensor:
    output = torch.zeros(
        (pair_rows.shape[0], codebook_size),
        dtype=torch.bool,
        device=pair_rows.device,
    )
    block_size = triton.next_power_of_2(max_children)
    _materialize_mask_kernel[(pair_rows.shape[0],)](
        pair_rows,
        offsets,
        values,
        output,
        output_row_stride=output.stride(0),
        block_size=block_size,
        num_warps=8,
    )
    return output


def mask_sparse_logits(
    logits: torch.Tensor,
    pair_rows: torch.Tensor,
    offsets: torch.Tensor,
    values: torch.Tensor,
    max_children: int,
) -> torch.Tensor:
    output = torch.full_like(logits, -torch.inf)
    block_size = triton.next_power_of_2(max_children)
    _mask_logits_kernel[(pair_rows.shape[0],)](
        logits,
        pair_rows,
        offsets,
        values,
        output,
        logits_row_stride=logits.stride(0),
        output_row_stride=output.stride(0),
        block_size=block_size,
        num_warps=8,
    )
    return output
