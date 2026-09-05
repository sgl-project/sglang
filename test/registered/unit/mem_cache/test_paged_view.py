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
"""``paged_view`` regroups ``[slots, ...]`` into ``[pages, page_size, ...]``.

The pool's per-layer buffers are strided views: consecutive slots are a whole
entry apart, not one row. So the page dimension the paged backends want cannot
come from ``.view()`` -- that is only legal on a contiguous tensor, and where
it is legal at ``page_size == 1`` it gives the size-1 slot dimension the ROW
stride instead of the SLOT stride. A kernel stepping that dimension then walks
into the next layer's bytes. These tests pin the strides the helper produces,
at both page sizes, against the buffer addresses they must reproduce.

CPU-only.

    python -m pytest test/registered/unit/mem_cache/test_paged_view.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.layout.page_major import paged_row_view, paged_view
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPagedView(unittest.TestCase):
    def test_contiguous_matches_plain_view(self):
        N, ps, H, D = 12, 4, 2, 16
        flat = torch.zeros(N, H, D)
        out = paged_view(flat, ps)
        self.assertEqual(tuple(out.shape), (N // ps, ps, H, D))
        self.assertEqual(out.stride(), flat.view(-1, ps, H, D).stride())
        self.assertEqual(out.data_ptr(), flat.data_ptr())

    def test_strided_slots_keep_their_stride(self):
        # ps == 1 is the case `.view()` gets wrong: the size-1 slot dimension
        # must still step a whole entry, not one row.
        for ps in (1, 4):
            with self.subTest(page_size=ps):
                N, H, D, E = 12, 2, 16, 2 * 16 + 96
                flat = torch.zeros(N * E).as_strided((N, H, D), (E, D, 1))
                out = paged_view(flat, ps)
                self.assertEqual(tuple(out.shape), (N // ps, ps, H, D))
                self.assertEqual(tuple(out.stride()), (ps * E, E, D, 1))
                for t in range(N):
                    self.assertEqual(
                        out[t // ps, t % ps].data_ptr(), flat[t].data_ptr()
                    )
                rows = paged_row_view(flat, ps)
                self.assertEqual(tuple(rows.shape), (N // ps, ps, H * D))
                self.assertEqual(tuple(rows.stride()), (ps * E, E, 1))

    def test_rejects_partial_pages(self):
        with self.assertRaises(AssertionError):
            paged_view(torch.zeros(10, 2, 16), 4)


if __name__ == "__main__":
    unittest.main()
