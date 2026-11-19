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
"""Request-owned snapshots of graph-replayed beam selections."""

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
