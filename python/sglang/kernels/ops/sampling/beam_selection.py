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
"""CUDA operators for beam selection and graph-replay snapshots."""

import torch
import triton
import triton.language as tl


@triton.jit
def _snapshot_selection_kernel(
    tokens,
    parents,
    scores,
    prefix,
    owned_ids,
    owned_scores,
    owned_prefix,
    width: tl.constexpr,
    prefix_width: tl.constexpr,
    block: tl.constexpr,
):
    rows = tl.program_id(0) * block + tl.arange(0, block)
    valid = rows < width
    tl.store(owned_ids + rows, tl.load(tokens + rows, valid, 0), valid)
    tl.store(owned_ids + width + rows, tl.load(parents + rows, valid, 0), valid)
    tl.store(owned_scores + rows, tl.load(scores + rows, valid, 0), valid)
    for column in tl.static_range(prefix_width):
        offset = rows * prefix_width + column
        tl.store(owned_prefix + offset, tl.load(prefix + offset, valid, 0), valid)


def snapshot_selection(tokens, parents, scores, prefix=None):
    width = tokens.numel()
    ids = torch.empty((2, width), dtype=tokens.dtype, device=tokens.device)
    owned_scores = torch.empty_like(scores)
    owned_prefix = None if prefix is None else torch.empty_like(prefix)
    _snapshot_selection_kernel[(triton.cdiv(width, 256),)](
        tokens,
        parents,
        scores,
        prefix if prefix is not None else ids,
        ids,
        owned_scores,
        owned_prefix if owned_prefix is not None else ids,
        width=width,
        prefix_width=0 if prefix is None else prefix.shape[1],
        block=256,
    )
    return ids, owned_scores, owned_prefix


@triton.jit
def _write_compacted(
    active,
    lanes,
    tokens,
    parents,
    scores,
    out_tokens,
    out_parents,
    out_scores,
    out_count,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Sort positions to preserve score order, with zero-filled unused output slots.
    indices = tl.sort(tl.where(active, lanes, BLOCK))
    valid = indices < BLOCK
    source = tl.minimum(indices, BLOCK - 1)
    tl.store(
        out_tokens + lanes,
        tl.where(valid, tl.gather(tokens, source, axis=0), 0),
        lanes < WIDTH,
    )
    tl.store(
        out_parents + lanes,
        tl.where(valid, tl.gather(parents, source, axis=0), 0),
        lanes < WIDTH,
    )
    tl.store(
        out_scores + lanes,
        tl.where(valid, tl.gather(scores, source, axis=0), 0),
        lanes < WIDTH,
    )
    tl.store(out_count, tl.sum(active.to(tl.int32), 0))


@triton.jit
def _compact_candidates_kernel(
    token_ptr,
    parent_ptr,
    score_ptr,
    stops,
    stop_stride,
    next_tokens,
    next_parents,
    next_scores,
    n_live,
    fin_tokens,
    fin_parents,
    fin_scores,
    n_finished,
    WIDTH: tl.constexpr,
    N: tl.constexpr,
    NUM_STOPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lanes = tl.arange(0, BLOCK)
    valid = lanes < N
    tokens = tl.load(token_ptr + lanes, valid, other=0)
    parents = tl.load(parent_ptr + lanes, valid, other=0)
    scores = tl.load(score_ptr + lanes, valid, other=0)
    stop = tl.full((BLOCK,), False, tl.int1)
    for i in tl.static_range(NUM_STOPS):
        stop |= tokens == tl.load(stops + i * stop_stride)
    live = valid & ~stop
    preceding = tl.cumsum(live.to(tl.int32), 0) - live.to(tl.int32)
    examined = preceding < WIDTH
    _write_compacted(
        live & examined,
        lanes,
        tokens,
        parents,
        scores,
        next_tokens,
        next_parents,
        next_scores,
        n_live,
        WIDTH,
        BLOCK,
    )
    _write_compacted(
        valid & stop & examined,
        lanes,
        tokens,
        parents,
        scores,
        fin_tokens,
        fin_parents,
        fin_scores,
        n_finished,
        N,
        BLOCK,
    )


def compact_candidates(tokens, parents, scores, stop_ids, width):
    size = tokens.numel()

    def allocate(length):
        return (
            tokens.new_empty(length),
            parents.new_empty(length),
            scores.new_empty(length),
            parents.new_empty(()),
        )

    outputs = (*allocate(width), *allocate(size))
    _compact_candidates_kernel[(1,)](
        tokens,
        parents,
        scores,
        stop_ids,
        stop_ids.stride(0),
        *outputs,
        width,
        size,
        stop_ids.numel(),
        triton.next_power_of_2(size),
        num_warps=1,
    )
    return outputs
