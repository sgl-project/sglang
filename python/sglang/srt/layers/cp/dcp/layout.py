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

"""Pure index math for decode context parallel (DCP): per-rank lengths and
the owner-rule local-index filter."""

import torch

from sglang.srt.distributed.parallel_state import get_dcp_rank, get_dcp_world_size
from sglang.srt.layers.cp.dcp.comm import dcp_enabled


def get_dcp_lens(
    lens: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    start: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-rank visible KV length under the owner rule pos % dcp_size == dcp_rank.

    Superset implementation (PR #25090): supports both start=None and a per-request
    `start` offset. update_local_kv_lens_for_dcp is the start=None special case.
    """
    if dcp_size == 1:
        return lens
    if start is None:
        return lens // dcp_size + (dcp_rank < lens % dcp_size)

    first = start + torch.remainder(dcp_rank - start, dcp_size)
    remaining = start + lens - first
    return torch.clamp((remaining + dcp_size - 1) // dcp_size, min=0)


def filter_dcp_local_kv_indices(kv_indices: torch.Tensor):
    if dcp_enabled():
        kv_indices = (
            kv_indices[kv_indices % get_dcp_world_size() == get_dcp_rank()]
            // get_dcp_world_size()
        )
    return kv_indices


def update_local_kv_lens_for_dcp(kv_len_arr):
    if not dcp_enabled():
        return
    dcp_world_size = get_dcp_world_size()
    dcp_rank = get_dcp_rank()
    offset = dcp_rank + 1
    kv_len_arr.sub_(offset).div_(dcp_world_size, rounding_mode="floor").add_(1)
