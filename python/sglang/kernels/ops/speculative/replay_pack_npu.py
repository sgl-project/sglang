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


@triton.jit(
    do_not_specialize=[
        "raw_bs",
        "bs",
        "topk",
        "raw_num_tokens",
        "num_tokens",
        "speculative_num_steps",
    ]
)
def _draft_replay_pack_fused_kernel(
    dst_seq_lens_ptr,
    src_seq_lens_ptr,
    dst_topk_p_ptr,
    src_topk_p_ptr,
    dst_topk_index_ptr,
    src_topk_index_ptr,
    dst_req_pool_indices_ptr,
    src_req_pool_indices_ptr,
    dst_out_cache_loc_ptr,
    src_out_cache_loc_ptr,
    dst_positions_ptr,
    src_positions_ptr,
    raw_bs,
    bs,
    topk,
    seq_len_fill_value,
    raw_num_tokens,
    num_tokens,
    speculative_num_steps,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    topk_offsets = tl.arange(0, BLOCK_TOPK)
    offsets = tl.arange(0, BLOCK_SIZE)

    for row in tl.range(pid, bs, num_programs):
        row_i64 = row.to(tl.int64)
        if row < raw_bs:
            tl.store(dst_seq_lens_ptr + row_i64, tl.load(src_seq_lens_ptr + row_i64))
            tl.store(
                dst_req_pool_indices_ptr + row_i64,
                tl.load(src_req_pool_indices_ptr + row_i64),
            )
            topk_base = row_i64 * topk
            for i in range(tl.cdiv(topk, BLOCK_TOPK)):
                current = topk_offsets + i * BLOCK_TOPK
                mask = current < topk
                tl.store(
                    dst_topk_p_ptr + topk_base + current,
                    tl.load(src_topk_p_ptr + topk_base + current, mask=mask, other=0.0),
                    mask=mask,
                )
                tl.store(
                    dst_topk_index_ptr + topk_base + current,
                    tl.load(
                        src_topk_index_ptr + topk_base + current,
                        mask=mask,
                        other=0,
                    ),
                    mask=mask,
                )
        else:
            tl.store(dst_seq_lens_ptr + row_i64, seq_len_fill_value)

    for block_idx in tl.range(pid, tl.cdiv(num_tokens, BLOCK_SIZE), num_programs):
        token_offset = block_idx * BLOCK_SIZE + offsets
        tl.store(
            dst_positions_ptr + token_offset,
            tl.load(
                src_positions_ptr + token_offset,
                mask=token_offset < raw_num_tokens,
                other=0,
            ),
            mask=token_offset < num_tokens,
        )

    raw_out_len = raw_num_tokens * speculative_num_steps
    out_len = num_tokens * speculative_num_steps
    for block_idx in tl.range(pid, tl.cdiv(out_len, BLOCK_SIZE), num_programs):
        out_offset = block_idx * BLOCK_SIZE + offsets
        tl.store(
            dst_out_cache_loc_ptr + out_offset,
            tl.load(
                src_out_cache_loc_ptr + out_offset,
                mask=out_offset < raw_out_len,
                other=0,
            ),
            mask=out_offset < out_len,
        )


def draft_replay_pack_npu(
    *,
    dst_seq_lens: torch.Tensor,
    src_seq_lens: torch.Tensor,
    dst_out_cache_loc: torch.Tensor,
    src_out_cache_loc: torch.Tensor,
    dst_positions: torch.Tensor,
    src_positions: torch.Tensor,
    dst_topk_p: torch.Tensor,
    src_topk_p: torch.Tensor,
    dst_topk_index: torch.Tensor,
    src_topk_index: torch.Tensor,
    dst_req_pool_indices: torch.Tensor,
    src_req_pool_indices: torch.Tensor,
    raw_bs: int,
    bs: int,
    topk: int,
    speculative_num_steps: int,
    seq_len_fill_value: int,
) -> None:
    """Pack an eager EAGLE draft batch into fixed graph buffers on A5."""
    if bs <= 0:
        return
    num_cores = get_npu_vector_core_count()
    raw_num_tokens = raw_bs * topk
    num_tokens = bs * topk
    _draft_replay_pack_fused_kernel[(num_cores,)](
        dst_seq_lens,
        src_seq_lens,
        dst_topk_p,
        src_topk_p,
        dst_topk_index,
        src_topk_index,
        dst_req_pool_indices,
        src_req_pool_indices,
        dst_out_cache_loc,
        src_out_cache_loc,
        dst_positions,
        src_positions,
        raw_bs,
        bs,
        topk,
        seq_len_fill_value,
        raw_num_tokens,
        num_tokens,
        speculative_num_steps,
        BLOCK_TOPK=16,
        BLOCK_SIZE=256,
    )
