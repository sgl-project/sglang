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
"""Torch fallback for token filter operations (host masks, non-CUDA devices and HIP).

Sets or clears specific bits in an int32 bitmask by token ID.  The token list
is typically tiny (< 10 entries) and the same list is applied over and over:
the strict-thinking filter calls this once per draft-tree node per request per
decode step, on the host-side mask.  The per-list work (bit aggregation and the
reset+apply result) is therefore cached, and a host mask row is written with
numpy in place instead of through tensor indexing.
"""

import ctypes
from typing import Dict, List, Tuple

import numpy as np
import torch

# (token ids, is_allowed, row length) -> (template row, element indices, bitmasks).
# The template is the reset+apply result; the indices/bitmasks apply without reset.
_FILTER_CACHE: Dict[
    Tuple[Tuple[int, ...], bool, int], Tuple[np.ndarray, np.ndarray, np.ndarray]
] = {}
_FILTER_CACHE_MAX = 256


def _filter_entry(token_ids: List[int], is_allowed: bool, row_len: int):
    key = (tuple(token_ids), bool(is_allowed), int(row_len))
    entry = _FILTER_CACHE.get(key)
    if entry is not None:
        return entry
    # Aggregate bit masks per int32 element to handle duplicate indices.
    aggregated: dict[int, int] = {}
    for token_id in token_ids:
        element_idx = token_id // 32
        bit_idx = token_id % 32
        aggregated[element_idx] = aggregated.get(element_idx, 0) | (1 << bit_idx)
    indices = np.fromiter(aggregated.keys(), dtype=np.int64, count=len(aggregated))
    bitmasks = np.fromiter(
        (
            ctypes.c_int32(mask if is_allowed else ~mask).value
            for mask in aggregated.values()
        ),
        dtype=np.int32,
        count=len(aggregated),
    )
    template = np.full(row_len, 0 if is_allowed else -1, dtype=np.int32)
    if is_allowed:
        template[indices] |= bitmasks
    else:
        template[indices] &= bitmasks
    if len(_FILTER_CACHE) >= _FILTER_CACHE_MAX:
        _FILTER_CACHE.clear()
    entry = (template, indices, bitmasks)
    _FILTER_CACHE[key] = entry
    return entry


def set_token_filter_torch(
    vocab_mask: torch.Tensor,
    token_ids: List[int],
    batch_idx: int,
    is_allowed: bool = True,
    reset_vocab_mask: bool = True,
):
    if not token_ids:
        if reset_vocab_mask:
            vocab_mask[batch_idx].fill_(-1 if (not is_allowed) else 0)
        return

    row = vocab_mask[batch_idx]
    template, indices, bitmasks = _filter_entry(token_ids, is_allowed, row.numel())

    if row.device.type == "cpu":
        # A host row (pinned or pageable) shares memory with its numpy view.
        row_np = row.numpy()
        if reset_vocab_mask:
            row_np[:] = template
        elif is_allowed:
            row_np[indices] |= bitmasks
        else:
            row_np[indices] &= bitmasks
        return

    if reset_vocab_mask:
        row.copy_(torch.from_numpy(template), non_blocking=True)
        return
    element_indices = torch.from_numpy(indices).to(row.device)
    bits = torch.from_numpy(bitmasks).to(row.device, dtype=row.dtype)
    if is_allowed:
        row[element_indices] = torch.bitwise_or(row[element_indices], bits)
    else:
        row[element_indices] = torch.bitwise_and(row[element_indices], bits)
