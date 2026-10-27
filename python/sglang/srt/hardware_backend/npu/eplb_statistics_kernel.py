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
"""Exact route histograms for Ascend EPLB's shared per-layer buffers."""

import triton
import triton.language as tl


@triton.jit
def _accumulate_route_counts(
    ids, counts, CAP: tl.constexpr, EXP: tl.constexpr, R: tl.constexpr, E: tl.constexpr
):
    layer = tl.program_id(0)
    expert = tl.program_id(1) * E + tl.arange(0, E)
    route = tl.arange(0, R)
    total = tl.full((E,), 0, tl.int32)
    # Tile both dimensions to bound UB usage, including large configured capacities.
    for block in range(tl.cdiv(CAP, R)):
        offset = block * R + route
        selected = tl.load(ids + layer * CAP + offset, offset < CAP, other=-1)
        # Masked loads produce -1, which cannot match a nonnegative expert ID.
        total += tl.sum((selected[:, None] == expert[None, :]).to(tl.int32), axis=0)
    # Invalid IDs cannot match a valid expert. Preserve earlier direct counts.
    previous = tl.load(counts + layer * EXP + expert, expert < EXP, other=0)
    tl.store(counts + layer * EXP + expert, previous + total, expert < EXP)


def accumulate_route_counts(ids, counts):
    layers, capacity = ids.shape
    experts = counts.shape[1]
    _accumulate_route_counts[(layers, triton.cdiv(experts, 32))](
        ids, counts, capacity, experts, min(triton.next_power_of_2(capacity), 128), 32
    )
