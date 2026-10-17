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
"""CUDA kernels for trie-conditioned decoding."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Grouped GEMM does less math but only wins once launch/partition costs amortize.
_GROUPED_PROJECTION_MIN_ROWS = 1024
_GROUPED_PROJECTION_CODEBOOKS = 3
_SINGLE_BLOCK_PARTITION_MAX_ROWS = 8192
_PARTITION_BLOCK_SIZE = 1024


@triton.jit
def _stable_depth_partition_kernel(
    depths,
    permutation,
    inverse_permutation,
    group_offsets,
    num_rows,
    block_size: tl.constexpr,
):
    rows = tl.arange(0, block_size)
    valid = rows < num_rows
    row_depths = tl.load(depths + rows, mask=valid, other=-1)
    is_depth_0 = valid & (row_depths <= 0)
    is_depth_1 = valid & (row_depths == 1)
    is_depth_2 = valid & (row_depths >= 2)
    count_0 = tl.sum(is_depth_0.to(tl.int32))
    count_1 = tl.sum(is_depth_1.to(tl.int32))
    count_2 = tl.sum(is_depth_2.to(tl.int32))
    rank_0 = tl.cumsum(is_depth_0.to(tl.int32)) - 1
    rank_1 = tl.cumsum(is_depth_1.to(tl.int32)) - 1
    rank_2 = tl.cumsum(is_depth_2.to(tl.int32)) - 1
    destinations = tl.where(
        is_depth_0,
        rank_0,
        tl.where(is_depth_1, count_0 + rank_1, count_0 + count_1 + rank_2),
    )
    tl.store(permutation + destinations, rows, mask=valid)
    tl.store(inverse_permutation + rows, destinations, mask=valid)
    tl.store(group_offsets, count_0)
    tl.store(group_offsets + 1, count_0 + count_1)
    tl.store(group_offsets + 2, count_0 + count_1 + count_2)


@triton.jit
def _depth_block_counts_kernel(
    depths,
    block_counts,
    num_rows,
    block_size: tl.constexpr,
):
    block = tl.program_id(0)
    rows = block * block_size + tl.arange(0, block_size)
    valid = rows < num_rows
    row_depths = tl.load(depths + rows, mask=valid, other=-1)
    tl.store(
        block_counts + block * 3,
        tl.sum((valid & (row_depths <= 0)).to(tl.int32)),
    )
    tl.store(
        block_counts + block * 3 + 1,
        tl.sum((row_depths == 1).to(tl.int32)),
    )
    tl.store(
        block_counts + block * 3 + 2,
        tl.sum((valid & (row_depths >= 2)).to(tl.int32)),
    )


@triton.jit
def _depth_block_offsets_kernel(
    block_counts,
    block_offsets,
    group_offsets,
    num_blocks,
    block_size: tl.constexpr,
):
    blocks = tl.arange(0, block_size)
    valid = blocks < num_blocks
    counts_0 = tl.load(block_counts + blocks * 3, mask=valid, other=0)
    counts_1 = tl.load(block_counts + blocks * 3 + 1, mask=valid, other=0)
    counts_2 = tl.load(block_counts + blocks * 3 + 2, mask=valid, other=0)
    total_0 = tl.sum(counts_0)
    total_1 = tl.sum(counts_1)
    total_2 = tl.sum(counts_2)
    tl.store(
        block_offsets + blocks * 3,
        tl.cumsum(counts_0) - counts_0,
        mask=valid,
    )
    tl.store(
        block_offsets + blocks * 3 + 1,
        total_0 + tl.cumsum(counts_1) - counts_1,
        mask=valid,
    )
    tl.store(
        block_offsets + blocks * 3 + 2,
        total_0 + total_1 + tl.cumsum(counts_2) - counts_2,
        mask=valid,
    )
    tl.store(group_offsets, total_0)
    tl.store(group_offsets + 1, total_0 + total_1)
    tl.store(group_offsets + 2, total_0 + total_1 + total_2)


@triton.jit
def _stable_depth_scatter_kernel(
    depths,
    block_offsets,
    permutation,
    inverse_permutation,
    num_rows,
    block_size: tl.constexpr,
):
    block = tl.program_id(0)
    rows = block * block_size + tl.arange(0, block_size)
    valid = rows < num_rows
    row_depths = tl.load(depths + rows, mask=valid, other=-1)
    is_depth_0 = valid & (row_depths <= 0)
    is_depth_1 = valid & (row_depths == 1)
    is_depth_2 = valid & (row_depths >= 2)
    rank_0 = tl.cumsum(is_depth_0.to(tl.int32)) - 1
    rank_1 = tl.cumsum(is_depth_1.to(tl.int32)) - 1
    rank_2 = tl.cumsum(is_depth_2.to(tl.int32)) - 1
    base_0 = tl.load(block_offsets + block * 3)
    base_1 = tl.load(block_offsets + block * 3 + 1)
    base_2 = tl.load(block_offsets + block * 3 + 2)
    destinations = tl.where(
        is_depth_0,
        base_0 + rank_0,
        tl.where(is_depth_1, base_1 + rank_1, base_2 + rank_2),
    )
    tl.store(permutation + destinations, rows, mask=valid)
    tl.store(inverse_permutation + rows, destinations, mask=valid)


def _stable_depth_partition(
    depths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_rows = depths.shape[0]
    permutation = torch.empty(num_rows, dtype=torch.int64, device=depths.device)
    inverse_permutation = torch.empty_like(permutation)
    group_offsets = torch.empty(3, dtype=torch.int32, device=depths.device)
    if num_rows <= _SINGLE_BLOCK_PARTITION_MAX_ROWS:
        _stable_depth_partition_kernel[(1,)](
            depths,
            permutation,
            inverse_permutation,
            group_offsets,
            num_rows,
            block_size=triton.next_power_of_2(num_rows),
            num_warps=8,
        )
        return permutation, inverse_permutation, group_offsets

    num_blocks = triton.cdiv(num_rows, _PARTITION_BLOCK_SIZE)
    block_counts = torch.empty((num_blocks, 3), dtype=torch.int32, device=depths.device)
    block_offsets = torch.empty_like(block_counts)
    _depth_block_counts_kernel[(num_blocks,)](
        depths,
        block_counts,
        num_rows,
        block_size=_PARTITION_BLOCK_SIZE,
        num_warps=8,
    )
    _depth_block_offsets_kernel[(1,)](
        block_counts,
        block_offsets,
        group_offsets,
        num_blocks,
        block_size=triton.next_power_of_2(num_blocks),
        num_warps=4,
    )
    _stable_depth_scatter_kernel[(num_blocks,)](
        depths,
        block_offsets,
        permutation,
        inverse_permutation,
        num_rows,
        block_size=_PARTITION_BLOCK_SIZE,
        num_warps=8,
    )
    return permutation, inverse_permutation, group_offsets


def _supports_grouped_projection(
    hidden_states: torch.Tensor,
    codebook_weights: torch.Tensor,
    depths: torch.Tensor,
) -> bool:
    return (
        hasattr(torch, "_grouped_mm")
        and torch.version.cuda is not None
        and hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and codebook_weights.dtype == torch.bfloat16
        and codebook_weights.is_contiguous()
        and depths.is_contiguous()
        and hidden_states.shape[0] >= _GROUPED_PROJECTION_MIN_ROWS
        and torch.cuda.get_device_capability(hidden_states.device)[0] == 9
    )


def project_active_codebook_logits(
    hidden_states: torch.Tensor,
    codebook_weights: torch.Tensor,
    depths: torch.Tensor,
    codebook_size: int,
) -> torch.Tensor | None:
    """Project only each row's active codebook when grouped GEMM is profitable."""
    num_rows, hidden_size = hidden_states.shape
    num_codebooks = codebook_weights.shape[0] // codebook_size
    if (
        num_codebooks != _GROUPED_PROJECTION_CODEBOOKS
        or codebook_weights.shape != (num_codebooks * codebook_size, hidden_size)
        or depths.shape != (num_rows,)
        or not _supports_grouped_projection(
            hidden_states,
            codebook_weights,
            depths,
        )
    ):
        return None

    permutation, inverse_permutation, group_offsets = _stable_depth_partition(depths)

    grouped_weights = codebook_weights.view(
        num_codebooks,
        codebook_size,
        hidden_size,
    ).transpose(1, 2)
    sorted_logits = torch._grouped_mm(
        hidden_states[permutation],
        grouped_weights,
        group_offsets,
    )
    return sorted_logits[inverse_permutation]


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
