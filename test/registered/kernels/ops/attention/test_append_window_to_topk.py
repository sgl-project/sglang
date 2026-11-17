# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
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
import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsa.append_window_to_topk import append_window_to_topk
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE_SIZE = 64


def _reference(topk_slots, seq_lens, page_table, page_rows, window, alignment):
    num_rows, topk = topk_slots.shape
    width = -(-(topk + window) // alignment) * alignment
    out = torch.full((num_rows, width), -1, dtype=torch.int32)
    lengths = torch.zeros(num_rows, dtype=torch.int32)
    for row in range(seq_lens.numel()):
        seq_len = int(seq_lens[row])
        topk_len = min(max(seq_len - window, 0), topk)
        window_len = min(seq_len, window)
        out[row, :topk_len] = topk_slots[row, :topk_len]
        table_row = int(page_rows[row]) if page_rows is not None else row
        for i in range(window_len):
            pos = seq_len - window_len + i
            page = int(page_table[table_row, pos // PAGE_SIZE])
            out[row, topk_len + i] = page * PAGE_SIZE + pos % PAGE_SIZE
        lengths[row] = topk_len + window_len
    return out, lengths


@pytest.mark.parametrize("use_page_rows", [False, True])
@pytest.mark.parametrize("seq_lens", [[1, 5, 128, 129, 700, 3000], [2200, 4096]])
def test_append_window_to_topk(seq_lens, use_page_rows):
    torch.manual_seed(0)
    topk, window, alignment = 2048, 128, 128
    live = len(seq_lens)
    num_rows = live + 2
    num_pages = 4096 // PAGE_SIZE + 1
    if use_page_rows:
        page_table = torch.randperm(4 * num_pages, dtype=torch.int32).view(4, num_pages)
        page_rows = torch.randint(0, 4, (live,), dtype=torch.int32)
    else:
        page_table = torch.randperm(live * num_pages, dtype=torch.int32).view(
            live, num_pages
        )
        page_rows = None
    topk_slots = torch.randint(0, 1 << 20, (num_rows, topk), dtype=torch.int32)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)

    out, lengths = append_window_to_topk(
        topk_slots.cuda(),
        seq_lens_t.cuda(),
        page_table.cuda(),
        window=window,
        page_size=PAGE_SIZE,
        alignment=alignment,
        page_rows=None if page_rows is None else page_rows.cuda(),
    )
    ref_out, ref_lengths = _reference(
        topk_slots, seq_lens_t, page_table, page_rows, window, alignment
    )
    torch.testing.assert_close(out.cpu(), ref_out, rtol=0, atol=0)
    torch.testing.assert_close(lengths.cpu()[:live], ref_lengths[:live], rtol=0, atol=0)
    assert (lengths.cpu()[live:] == 0).all()


def test_append_window_to_topk_ragged():
    torch.manual_seed(0)
    topk, window, alignment = 2048, 128, 128
    seq_lens = torch.tensor([3, 130, 2500], dtype=torch.int32)
    offsets = torch.tensor([0, 3, 133], dtype=torch.int32)
    topk_slots = torch.randint(0, 1 << 20, (4, topk), dtype=torch.int32)
    out, lengths = append_window_to_topk(
        topk_slots.cuda(),
        seq_lens.cuda(),
        torch.zeros(1, 1, dtype=torch.int32, device="cuda"),
        window=window,
        page_size=PAGE_SIZE,
        alignment=alignment,
        ragged_offsets=offsets.cuda(),
    )
    out, lengths = out.cpu(), lengths.cpu()
    for row, (seq_len, offset) in enumerate(zip(seq_lens.tolist(), offsets.tolist())):
        topk_len = min(max(seq_len - window, 0), topk)
        window_len = min(seq_len, window)
        assert lengths[row] == topk_len + window_len
        assert torch.equal(out[row, :topk_len], topk_slots[row, :topk_len])
        expected = torch.arange(seq_len - window_len, seq_len, dtype=torch.int32)
        assert torch.equal(
            out[row, topk_len : topk_len + window_len], expected + offset
        )
    assert lengths[3] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
