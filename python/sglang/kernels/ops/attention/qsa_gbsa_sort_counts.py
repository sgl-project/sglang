# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused chronological group sorting and counting for A3 GBSA.

Adapted from the vLLM NPU plugin, ops/triton/prepare_indexer_indices.py at 24487f4.
QSA's indexer has already handled visibility. Its GBSA input uses 16777215
padding and implicit residuals. Keep the same reduction order as the SG path.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cann import extension as ext


@triton.jit(do_not_specialize=["rows"])
def _qsa_sort_count_groups(selected, output, counts, rows, block_rows: tl.constexpr):
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    col = tl.arange(0, 512)
    offsets = row[:, None] * 512 + col[None, :]
    ids = tl.load(selected + offsets, row[:, None] < rows, other=-1)
    valid = ids >= 0
    # All group IDs and the existing SG padding value are exact in FP32.
    keys = tl.where(valid, ids, 16777215).to(tl.float32)
    sorted_ids = ext.sort(keys, dim=1, descending=False).to(tl.int32)
    tl.store(output + offsets, sorted_ids, row[:, None] < rows)
    tl.store(counts + row, tl.sum(valid.to(tl.int32), axis=1), row < rows)


def sort_count_groups(indices):
    rows = indices.shape[0]
    packed = torch.empty_like(indices)
    counts = indices.new_empty((rows,))
    # A group of eight counts occupies one 32-byte aligned vector store.
    _qsa_sort_count_groups[(triton.cdiv(rows, 8),)](
        indices, packed, counts, rows, 8, multibuffer=False, unit_flag=False
    )
    return packed.unsqueeze(0), counts.unsqueeze(0)
