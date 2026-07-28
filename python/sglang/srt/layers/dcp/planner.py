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

"""Decode-CP metadata builders (PR #14194). P2 will wrap these as methods on
DecodeContextParallelStrategy; kept as functions here for behavior-preserving
relocation."""

from typing import Optional

import torch

from sglang.kernels.ops.attention.dcp_kernels import (
    create_dcp_kv_indices,
    update_kv_lens_and_indices,
)
from sglang.srt.layers.dcp.layout import update_local_kv_lens_for_dcp
from sglang.srt.layers.dcp.metadata import DecodeContextParallelMetadata
from sglang.srt.runtime_context import get_parallel, get_server_args


def prepare_decode_context_parallel_metadata(
    seq_lens: torch.Tensor,
    extend_prefix_lens: torch.Tensor,
    extend_prefix_lens_cpu: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens_sum: int,
    kv_buffer_shape: torch.Size,
    kv_cache_dtype,
    kv_cache_device,
    create_chunked_prefix_cache_kv_indices_fn,
) -> Optional[DecodeContextParallelMetadata]:
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return None
    # dcp_kv_buffer tokens' layout
    # [ rank0_r1.prefix_tokens, rank1_r1.prefix_tokens, ..., rank7_r1.prefix_tokens,
    #   ...,
    #   rank0_rn.prefix_tokens, rank1_rn.prefix_tokens, ..., rank7_rn.prefix_tokens,
    #   r1.extend_tokens, r2.extent_tokens, rn.extend_tokens ]
    extend_prefix_starts = torch.zeros(
        len(seq_lens),
        dtype=torch.int32,
        device=get_server_args().device,
    )
    extend_cu_prefix_lens = torch.zeros(
        len(seq_lens) + 1,
        dtype=torch.int32,
        device=get_server_args().device,
    )
    extend_cu_prefix_lens[1:] = torch.cumsum(extend_prefix_lens, dim=0)
    extend_cu_prefix_lens = extend_cu_prefix_lens[:-1]
    extend_prefix_lens_sum = sum([i for i in extend_prefix_lens_cpu])

    dcp_prefix_kv_indices = torch.empty(
        sum(extend_prefix_lens_cpu),
        dtype=torch.int32,
        device=get_server_args().device,
    )
    create_chunked_prefix_cache_kv_indices_fn[(len(seq_lens),)](
        req_to_token,
        req_pool_indices,
        extend_prefix_starts,
        extend_prefix_lens,
        extend_cu_prefix_lens,
        dcp_prefix_kv_indices,
        req_to_token.shape[1],
    )
    dcp_kv_indptr = torch.zeros(
        len(seq_lens) + 1,
        dtype=torch.int32,
        device=get_server_args().device,
    )
    dcp_kv_indptr[1:] = seq_lens.cumsum(dim=0)
    dcp_kv_indptr = dcp_kv_indptr[: (len(seq_lens) + 1)]
    dcp_kv_indices = torch.zeros(
        seq_lens_sum,
        dtype=torch.int32,
        device=get_server_args().device,
    )

    extend_cu_lens = torch.zeros(
        len(seq_lens) + 1,
        dtype=torch.int32,
        device=get_server_args().device,
    )
    extend_cu_lens[1:] = torch.cumsum(extend_seq_lens, dim=0)
    extend_cu_lens = extend_cu_lens[:-1]

    create_dcp_kv_indices[(len(seq_lens),)](
        dcp_kv_indptr,
        extend_seq_lens,
        extend_cu_lens,
        extend_prefix_lens,
        extend_cu_prefix_lens,
        dcp_kv_indices,
        extend_prefix_lens_sum,
        parallel.dcp_size,
    )
    dcp_local_prefix_kv_indices = (
        dcp_prefix_kv_indices[
            dcp_prefix_kv_indices % parallel.dcp_size == parallel.dcp_rank
        ]
        // parallel.dcp_size
    )
    dcp_kv_buffer = torch.empty(
        (
            seq_lens_sum,
            *kv_buffer_shape[1:],
        ),
        dtype=kv_cache_dtype,
        device=kv_cache_device,
    )
    attn_dcp_metadata = DecodeContextParallelMetadata(
        dcp_kv_indptr=dcp_kv_indptr,
        dcp_kv_buffer=dcp_kv_buffer,
        dcp_kv_indices=dcp_kv_indices,
        dcp_local_prefix_kv_indices=dcp_local_prefix_kv_indices,
        dcp_extend_prefix_lens_sum=extend_prefix_lens_sum,
    )
    return attn_dcp_metadata


def plan_dcp_decode_metadata(
    kv_lens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    init_metadata_replay: bool,
    fast_decode_kwargs: dict,
    bs: int,
):
    parallel = get_parallel()
    local_kv_lens = kv_lens.clone()
    update_local_kv_lens_for_dcp(local_kv_lens)
    local_kv_lens.clamp_(min=0)

    if not init_metadata_replay:
        max_local_len = (
            int(local_kv_lens.max().item()) if local_kv_lens.numel() > 0 else 0
        )
        total_local_len = (
            int(local_kv_lens.sum().item()) if local_kv_lens.numel() > 0 else 0
        )
    else:
        max_local_len = (
            int(fast_decode_kwargs["kv_len_arr_cpu"].max().item())
            if fast_decode_kwargs["kv_len_arr_cpu"].numel() > 0
            else 0
        )
        total_local_len = (
            int(fast_decode_kwargs["kv_len_arr_cpu"].sum().item())
            if fast_decode_kwargs["kv_len_arr_cpu"].numel() > 0
            else 0
        )
    local_kv_lens_cumsum = kv_indptr.new_zeros((bs + 1,))
    local_kv_lens_cumsum[1 : bs + 1] = torch.cumsum(local_kv_lens, dim=0)
    local_kv_indices = kv_indices.new_empty(total_local_len)
    BLOCK_SIZE = 128
    num_blocks = (
        (max_local_len + BLOCK_SIZE - 1) // BLOCK_SIZE if max_local_len > 0 else 1
    )
    grid = (bs, num_blocks)
    update_kv_lens_and_indices[grid](
        kv_lens,
        kv_indptr,
        kv_indices,
        local_kv_lens,
        local_kv_lens_cumsum,
        local_kv_indices,
        dcp_rank=parallel.dcp_rank,
        dcp_world_size=parallel.dcp_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    kv_indices[:total_local_len] = local_kv_indices[:total_local_len]
    kv_lens.copy_(local_kv_lens)
    kv_indptr[: bs + 1] = local_kv_lens_cumsum[: bs + 1]
