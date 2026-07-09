# Copyright 2023-2024 SGLang Team
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

import triton
import triton.language as tl

from sglang.srt.utils import is_cpu, is_npu

_is_cpu = is_cpu()
_is_npu = is_npu()

if _is_cpu:
    from sgl_kernel import rotate_input_ids_cpu


@triton.jit
def rotate_input_ids_kernel(
    input_ids_ptr,
    extend_start_loc_ptr,
    extend_seq_lens_ptr,
    topk_index_ptr,
    select_index_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    start_loc = tl.load(extend_start_loc_ptr + pid)
    seq_len = tl.load(extend_seq_lens_ptr + pid)
    new_token = tl.load(topk_index_ptr + pid)

    num_elements_to_shift = seq_len - 1

    for off in range(0, num_elements_to_shift, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements_to_shift

        read_ptr = input_ids_ptr + start_loc + offsets + 1
        val = tl.load(read_ptr, mask=mask)
        tl.debug_barrier()

        write_ptr = input_ids_ptr + start_loc + offsets
        tl.store(write_ptr, val, mask=mask)
        tl.debug_barrier()

    if seq_len > 0:
        if select_index_ptr is not None:
            last_pos_ptr = input_ids_ptr + tl.load(select_index_ptr + pid)
        else:
            last_pos_ptr = input_ids_ptr + start_loc + seq_len - 1
        tl.store(last_pos_ptr, new_token)


def rotate_input_ids(
    input_ids, extend_start_loc, extend_seq_lens, topk_index, select_index=None
):
    if _is_cpu:
        rotate_input_ids_cpu(
            input_ids,
            extend_start_loc,
            extend_seq_lens,
            topk_index,
            select_index,
        )
        return input_ids

    batch_size = extend_seq_lens.shape[0]

    # rotate_input_ids_triton skipped: batch_size=0 (empty extend_seq_lens).
    # This is expected when a DP rank has no requests.
    if batch_size == 0 and _is_npu:
        return input_ids

    BLOCK_SIZE = 4096 if select_index is not None else 8
    grid = (batch_size,)

    rotate_input_ids_kernel[grid](
        input_ids,
        extend_start_loc,
        extend_seq_lens,
        topk_index,
        select_index,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return input_ids
