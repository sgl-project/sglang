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

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import torch

from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous


def send_dsa_shared_staged(
    transfer: Callable[[str, list[tuple[int, int, int]]], int],
    mooncake_session_id: str,
    src_buffers: list[torch.Tensor],
    item_lens: list[int],
    src_indices: npt.NDArray[np.int32],
    dst_ptrs: list[int],
    dst_indices: npt.NDArray[np.int32],
    staging_buffer,
) -> int:
    if staging_buffer is None:
        raise RuntimeError("DSA shared PD transfer staging buffer is unavailable")
    if len(src_indices) != len(dst_indices):
        raise ValueError(
            "DSA shared PD transfer requires equal source and destination "
            f"index counts, got {len(src_indices)} and {len(dst_indices)}"
        )
    if len(src_buffers) != len(dst_ptrs) or len(src_buffers) != len(item_lens):
        raise ValueError("DSA shared PD transfer buffer metadata is inconsistent")
    if len(src_indices) == 0:
        return 0

    for src_buffer, dst_ptr, item_len in zip(src_buffers, dst_ptrs, item_lens):
        rows = src_buffer.view(torch.uint8).reshape(-1, item_len)
        rows_per_chunk = staging_buffer.get_size() // item_len
        if rows_per_chunk == 0:
            raise RuntimeError(
                f"DSA shared PD staging buffer is smaller than one item ({item_len} bytes)"
            )

        for start in range(0, len(src_indices), rows_per_chunk):
            end = min(start + rows_per_chunk, len(src_indices))
            chunk_src = src_indices[start:end]
            chunk_dst = dst_indices[start:end]
            index = torch.as_tensor(
                chunk_src, dtype=torch.long, device=src_buffer.device
            )
            packed = staging_buffer.buffer[: (end - start) * item_len].view(
                end - start, item_len
            )
            torch.index_select(rows, 0, index, out=packed)
            if src_buffer.is_cuda:
                torch.cuda.current_stream(src_buffer.device).synchronize()

            src_groups, dst_groups = group_concurrent_contiguous(
                np.arange(end - start, dtype=np.int32), chunk_dst
            )
            transfer_blocks = [
                (
                    staging_buffer.get_ptr() + int(src_group[0]) * item_len,
                    dst_ptr + int(dst_group[0]) * item_len,
                    len(src_group) * item_len,
                )
                for src_group, dst_group in zip(src_groups, dst_groups)
            ]
            status = transfer(mooncake_session_id, transfer_blocks)
            if status != 0:
                return status
    return 0
