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
"""Unit test for the DCP draft-pool id-space fix.

Under DCP the draft worker shares the target's widened
``PagedTokenToKVPoolAllocator`` (size T*D, page P*D — max issuable slot id
``T*D + P*D - 1``) while storing UNSHARDED KV. A physical-size draft pool
(bound T + P) trips the ``store_kvcache`` device assert (``kvcache.cuh``,
DFlash bf16 pool) or an unbounded-write IMA (MLA draft pools) as soon as an
allocator id crosses the physical range — the cc16x50K crash class. The fix
widens the draft pool to ``alloc.size + alloc.page_size - draft_page`` rows
(``ModelRunnerKVCacheMixin._init_pools``).

Tests:
  1. CPU: the widening formula covers the allocator's max id for a grid of
     (T, P, D) — fails for any D > 1 with the pre-fix physical sizing.
  2. GPU: a real (tiny) allocator + store_cache write at the TOP allocator id
     into a pool sized by the fix — the exact write that used to assert.

Usage:
    python -m pytest test_dcp_draft_pool_bounds.py -v
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


def _alloc_max_id(T: int, P: int, D: int) -> int:
    # PagedTokenToKVPoolAllocator(size=T*D, page_size=P*D): pages 1..T/P
    # (page 0 reserved), top page covers ids [T*D, T*D + P*D).
    return T * D + P * D - 1


def _widened_pool_size(T: int, P: int, D: int) -> int:
    # The fix: pool.size = alloc.size + alloc.page_size - draft_page.
    return T * D + P * D - P


class TestDraftPoolSizingFormula(unittest.TestCase):
    GRID = [
        (2048, 64, 8),
        (2048, 32, 8),
        (4096, 64, 4),
        (1024, 64, 2),
        (2_000_000, 64, 8),  # bench-scale
    ]

    def test_prefix_sizing_is_oob_for_any_dcp(self):
        for T, P, D in self.GRID:
            if D == 1:
                continue
            physical_bound = T + P  # pre-fix pool.size + page_size
            self.assertGreaterEqual(
                _alloc_max_id(T, P, D),
                physical_bound,
                f"expected OOB exposure at T={T},P={P},D={D}",
            )

    def test_widened_sizing_covers_allocator(self):
        for T, P, D in self.GRID:
            bound = _widened_pool_size(T, P, D) + P  # pool.size + page_size
            self.assertGreater(
                bound,
                _alloc_max_id(T, P, D),
                f"widened pool does not cover allocator at T={T},P={P},D={D}",
            )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA (store_cache jit)")
class TestDraftStoreAtTopAllocatorId(unittest.TestCase):
    """Store at the allocator's top id into a fix-sized pool (tiny shapes)."""

    T, P, D = 2048, 64, 8
    ROW_DIM = 128  # head_num * head_dim, bf16 -> 256B rows

    def test_store_cache_at_top_id(self):
        from sglang.jit_kernel.kvcache import can_use_store_cache, store_cache
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

        T, P, D = self.T, self.P, self.D
        device = torch.device("cuda")
        dtype = torch.bfloat16
        row_bytes = self.ROW_DIM * dtype.itemsize
        if not can_use_store_cache(row_bytes):
            self.skipTest(f"store_cache unsupported for row_bytes={row_bytes}")

        alloc = PagedTokenToKVPoolAllocator(
            T * D,
            page_size=P * D,
            dtype=dtype,
            device="cuda",
            kvcache=None,  # stored, not dereferenced by alloc()
            need_sort=False,
        )
        pool_rows = _widened_pool_size(T, P, D) + P
        k_cache = torch.zeros(pool_rows, self.ROW_DIM, dtype=dtype, device=device)
        v_cache = torch.zeros(pool_rows, self.ROW_DIM, dtype=dtype, device=device)

        # Drain the allocator so the LAST page (top ids) is issued, mimicking
        # a hot allocator handing out the highest virtual ids.
        num_pages = T // P
        ids = None
        for _ in range(num_pages):
            ids = alloc.alloc(P * D)
        self.assertIsNotNone(ids)
        top = int(ids.max())
        self.assertEqual(top, _alloc_max_id(T, P, D), "allocator top id mismatch")

        n = ids.numel()
        k = torch.randn(n, self.ROW_DIM, dtype=dtype, device=device)
        v = torch.randn(n, self.ROW_DIM, dtype=dtype, device=device)
        store_cache(
            k,
            v,
            k_cache.view(-1, self.ROW_DIM),
            v_cache.view(-1, self.ROW_DIM),
            ids,
            row_bytes=row_bytes,
            size_limit=pool_rows,
        )
        torch.cuda.synchronize()  # would raise on the pre-fix device assert
        self.assertTrue(
            torch.equal(k_cache[ids.long()], k), "stored K mismatch at top ids"
        )


if __name__ == "__main__":
    unittest.main()
