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
