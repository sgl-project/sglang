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
"""`generate_draft_decode_kv_indices` over the unified pool's read tables.

Over the plan's physical page table (token ids rebuilt as
`entry * ps + pos % ps`), and over virtual rows translated in the kernel's own
gather, the kernel must emit exactly the translated virtual ids and the same
kv_indptr. The static `req_to_token` path is covered by
test_spec_kv_indices_grid.py.

    python -m pytest test/registered/kernels/ops/speculative/test_draft_decode_kv_indices_read_table.py -v
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_SENTINEL = -7


def _run_kernel(
    *, table, entry_page_size, seq_lens, positions, num_steps, topk, page_size, v2p=None
):
    from sglang.kernels.ops.speculative.cache_locs import (
        generate_draft_decode_kv_indices,
    )
    from sglang.srt.utils import next_power_of_2

    num_seqs = seq_lens.numel()
    bs = num_seqs * topk
    width = int((seq_lens.sum().item() + num_steps * bs) * 2 + 64)
    kv_indices = torch.full(
        (num_steps, width), _SENTINEL, dtype=torch.int64, device="cuda"
    )
    kv_indptr = torch.zeros((num_steps, bs + 1), dtype=torch.int32, device="cuda")
    generate_draft_decode_kv_indices[(num_steps, num_seqs, topk)](
        torch.arange(num_seqs, dtype=torch.int64, device="cuda"),
        table,
        seq_lens,
        kv_indices,
        kv_indptr,
        positions,
        table.stride(0),
        width,
        kv_indptr.shape[1],
        next_power_of_2(num_seqs),
        next_power_of_2(num_steps),
        next_power_of_2(bs),
        page_size,
        ENTRY_PAGE_SIZE=entry_page_size,
        v2p=v2p,
        TRANSLATE=v2p is not None,
    )
    return kv_indices, kv_indptr


class TestDraftDecodeKVIndicesReadTable(CustomTestCase):
    def _check(self, *, page_size, num_steps=3, topk=1):
        torch.manual_seed(7)
        num_seqs, max_context = 3, 64
        num_pages = max_context // page_size
        # Each row holds whole virtual pages in a shuffled order, and the
        # virtual->physical table is a shuffle too, so nothing is identity.
        v2p = torch.randperm(num_pages, device="cuda")
        rows = []
        for _ in range(num_seqs):
            vpages = torch.randperm(num_pages, device="cuda")
            offsets = torch.arange(page_size, device="cuda")
            rows.append((vpages[:, None] * page_size + offsets).reshape(-1))
        req_to_token = torch.stack(rows)
        # The plan's table: one physical page per virtual page of the row.
        table = v2p[req_to_token[:, ::page_size] // page_size].to(torch.int32)
        seq_lens = torch.tensor([5, 1, 9], dtype=torch.int64, device="cuda")
        positions = seq_lens.repeat_interleave(topk)
        common = dict(
            seq_lens=seq_lens,
            positions=positions,
            num_steps=num_steps,
            topk=topk,
            page_size=page_size,
        )
        raw, raw_indptr = _run_kernel(table=req_to_token, entry_page_size=1, **common)
        out, out_indptr = _run_kernel(table=table, entry_page_size=page_size, **common)
        # No shared table: the virtual rows, translated in the kernel's gather.
        gathered, gathered_indptr = _run_kernel(
            table=req_to_token, entry_page_size=1, v2p=v2p, **common
        )

        torch.testing.assert_close(raw_indptr, out_indptr, rtol=0, atol=0)
        written = raw != _SENTINEL
        self.assertTrue(bool((out[~written] == _SENTINEL).all()))
        virt = raw[written]
        expected = v2p[virt // page_size] * page_size + virt % page_size
        torch.testing.assert_close(out[written], expected, rtol=0, atol=0)
        torch.testing.assert_close(gathered_indptr, out_indptr, rtol=0, atol=0)
        torch.testing.assert_close(gathered, out, rtol=0, atol=0)

    def test_page_size_one(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self._check(page_size=1)

    def test_page_size_two(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self._check(page_size=2)


if __name__ == "__main__":
    unittest.main()
