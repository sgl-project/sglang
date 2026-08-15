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
from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.npu import get_npu_vector_core_count


@triton.jit(do_not_specialize=["batch_size", "pool_len"])
def _read_cache_locations_kernel(
    req_pool_indices_ptr,
    token_pool_ptr,
    start_offset_ptr,
    end_offset_ptr,
    out_cache_loc_ptr,
    batch_size,
    pool_len,
    NUM_CORES: tl.constexpr,
    BS_UPPER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    for row_idx in tl.range(pid, batch_size, NUM_CORES):
        req_idx = tl.load(req_pool_indices_ptr + row_idx)
        kv_start = tl.load(start_offset_ptr + row_idx)
        kv_end = tl.load(end_offset_ptr + row_idx)
        step = kv_end - kv_start

        prefix_idx = tl.arange(0, BS_UPPER)
        prefix_start = tl.load(
            start_offset_ptr + prefix_idx, mask=prefix_idx < row_idx, other=0
        )
        prefix_end = tl.load(
            end_offset_ptr + prefix_idx, mask=prefix_idx < row_idx, other=0
        )
        out_start = tl.sum(prefix_end - prefix_start, axis=0)
        token_ptr = token_pool_ptr + req_idx * pool_len + kv_start
        out_ptr = out_cache_loc_ptr + out_start

        offsets = tl.arange(0, BLOCK_SIZE)
        for block_idx in tl.range(0, tl.cdiv(step, BLOCK_SIZE)):
            current = offsets + block_idx * BLOCK_SIZE
            mask = current < step
            data = tl.load(token_ptr + current, mask=mask, other=0)
            tl.store(out_ptr + current, data, mask=mask)


def read_cache_locations_npu(
    *,
    req_pool_indices: torch.Tensor,
    token_pool: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
) -> None:
    """Read ragged cache locations on A5 without a fixed maximum step count."""
    batch_size = int(req_pool_indices.shape[0])
    if batch_size == 0:
        return
    num_cores = min(get_npu_vector_core_count(), batch_size)
    _read_cache_locations_kernel[(num_cores,)](
        req_pool_indices,
        token_pool,
        start_offset,
        end_offset,
        out_cache_loc,
        batch_size,
        int(token_pool.shape[1]),
        NUM_CORES=num_cores,
        BS_UPPER=triton.next_power_of_2(batch_size),
        BLOCK_SIZE=32,
    )
