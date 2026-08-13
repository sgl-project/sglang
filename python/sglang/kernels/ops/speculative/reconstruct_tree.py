# Copyright 2023-2026 SGLang Team
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
from __future__ import annotations

"""Fused Triton kernel: reconstruct NGRAM verify metadata from a tree mask.

Device-native replacement for the pure-torch fallback
``_reconstruct_indices_from_tree_mask_torch`` (used where the compiled
``sgl_kernel.speculative.reconstruct_indices_from_tree_mask`` op is unavailable,
e.g. Intel XPU). It reproduces that op's contract exactly, mutating
``positions`` / ``retrieve_*`` in place.

Semantics (per batch ``b``, per node ``tid``; ``n = draft_token_num``):
  ``tree_mask[b, i, j]`` (row-major, flat) marks node ``j`` an ancestor of ``i``.
  ``anc[i, j]      = tree_mask[b, i, j] & (j < i)``          (strict ancestors)
  ``positions[b,tid] = (#ancestors of tid) + verified_seq_len[b]``
  ``retrieve_index[b,tid]        = b * n + tid``
  ``parent[tid]                  = max{ j<tid : anc[tid, j] } else -1``
  ``retrieve_next_token[b,tid]   = min{ k>tid : anc[k, tid] } else -1``  (first child)
  ``retrieve_next_sibling[b,tid] = min{ k>tid : parent[k]==parent[tid] }``
                                   else -1, and -1 when parent[tid] < 0.

The whole reduction is over one ``n x n`` bool matrix, so a single program per
batch loads it once into registers and emits all four outputs -- collapsing the
~15-launch, multi-``[bs,n,n]``-HBM-roundtrip torch fallback into one kernel with
zero materialized intermediates.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.utils import next_power_of_2

# Sentinel for "no ancestor / not-yet-found child": any value >= n. Reductions use
# n for min-searches (first child/sibling) and -1 for max-searches (parent).


@triton.jit
def _reconstruct_tree_mask_kernel(
    tree_mask_ptr,  # bool [bs * n * n], row-major (b, i, j)
    verified_seq_len_ptr,  # int [bs]
    positions_ptr,  # int64 [bs * n]      (out)
    retrieve_index_ptr,  # int64 [bs * n]      (out)
    retrieve_next_token_ptr,  # int64 [bs * n]      (out)
    retrieve_next_sibling_ptr,  # int64 [bs * n]      (out)
    n: tl.constexpr,  # draft_token_num
    BLOCK_N: tl.constexpr,  # next_power_of_2(n)
):
    b = tl.program_id(axis=0)

    idx = tl.arange(0, BLOCK_N)
    row = idx[:, None]  # i  -> the node being described
    col = idx[None, :]  # j  -> ancestor candidate
    valid = (row < n) & (col < n)

    # Load this batch's n x n ancestor mask exactly once.
    m = tl.load(tree_mask_ptr + b * (n * n) + row * n + col, mask=valid, other=0).to(
        tl.int1
    )

    # anc[i, j] = mask[i, j] & (j < i): strict lower-triangular ancestors.
    anc = m & (col < row)
    anc_i = anc.to(tl.int32)

    # depth[i] = #ancestors (reduce columns); positions = depth + verified_seq_len.
    depth = tl.sum(anc_i, axis=1)
    seq_len = tl.load(verified_seq_len_ptr + b).to(tl.int64)
    positions = depth.to(tl.int64) + seq_len

    # parent[i] = largest j<i with anc[i, j], else -1 (reduce columns).
    parent = tl.max(tl.where(anc, col, -1), axis=1)  # [BLOCK_N], indexed by i

    # first child of tid = smallest k>tid with anc[k, tid] (reduce rows over the
    # tid-th column). anc already encodes k>tid via (col < row).
    has_child = tl.max(anc_i, axis=0) > 0
    first_child = tl.min(tl.where(anc, row, n), axis=0)  # indexed by column tid
    next_token = tl.where(has_child, first_child, -1)

    # next sibling of tid = smallest k>tid sharing tid's parent; -1 if tid is a
    # root (parent < 0). parent[k] over rows vs parent[tid] over columns.
    parent_k = parent[:, None]
    parent_tid = parent[None, :]
    same_parent = (parent_k == parent_tid) & (row > col) & (row < n)
    first_sibling = tl.min(tl.where(same_parent, row, n), axis=0)  # indexed by tid
    has_parent = parent >= 0
    next_sibling = tl.where(has_parent & (first_sibling < n), first_sibling, -1)

    # Store the n valid entries of row b.
    out = b * n + idx
    store_mask = idx < n
    tl.store(positions_ptr + out, positions, mask=store_mask)
    tl.store(retrieve_index_ptr + out, (b * n + idx).to(tl.int64), mask=store_mask)
    tl.store(retrieve_next_token_ptr + out, next_token.to(tl.int64), mask=store_mask)
    tl.store(
        retrieve_next_sibling_ptr + out, next_sibling.to(tl.int64), mask=store_mask
    )


def _num_warps_for(block_n: int) -> int:
    # One program owns an n x n tile. Small trees (the common decode case) waste
    # EUs with wide warp groups; scale warps with the tile so each subgroup keeps
    # useful lanes. Tuned for Xe2 SIMD16 subgroups.
    if block_n <= 16:
        return 1
    if block_n <= 32:
        return 2
    return 4


def reconstruct_indices_from_tree_mask_triton(
    tree_mask: torch.Tensor,
    verified_seq_len: torch.Tensor,
    positions: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
) -> None:
    """Fused device kernel for NGRAM verify-metadata reconstruction.

    Drop-in for ``reconstruct_indices_from_tree_mask`` on GPU-like devices
    (XPU/CUDA/HIP). Mutates ``positions`` / ``retrieve_*`` in place; identical
    numerical contract to the compiled op and the torch fallback.
    """
    if batch_size == 0:
        return

    n = draft_token_num
    block_n = next_power_of_2(n)

    _reconstruct_tree_mask_kernel[(batch_size,)](
        tree_mask,
        verified_seq_len,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        n,
        block_n,
        num_warps=_num_warps_for(block_n),
    )
