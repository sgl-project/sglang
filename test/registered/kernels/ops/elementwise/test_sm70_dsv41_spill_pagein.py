"""SM70 UVA spill page-in copies host rows into landing slots.

Not registered for GPU CI (V100 worktree only).
"""

from __future__ import annotations

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=30,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)


def _sm70() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 7


class TestSm70Dsv41SpillPagein(CustomTestCase):
    @unittest.skipUnless(_sm70(), "SM70 required")
    def test_page_in_copies_host_row_and_remaps(self):
        from sglang.kernels.ops.moe.sm70_dsv41_spill_pagein import spill_page_in

        device = torch.device("cuda")
        n_logical = 8
        n_landing = 2
        n_row = 64  # 256 bytes, 16-byte aligned
        host = (
            torch.arange(4 * n_row, dtype=torch.int32)
            .reshape(4, n_row)
            .contiguous()
            .pin_memory()
        )
        land = torch.full((n_landing, n_row), -7, dtype=torch.int32, device=device)
        # Duplicate miss of logical 5 should reuse landing slot 0.
        topk = torch.tensor([[5, 0, 5, -1]], dtype=torch.int32, device=device)
        land_ids = torch.empty_like(topk)
        slot_host = torch.empty(n_landing, dtype=torch.int32, device=device)
        map_table = torch.full((n_logical,), -1, dtype=torch.int32, device=device)
        map_table[0] = 3  # hit -> GPU slot 3
        host_map = torch.full((n_logical,), -1, dtype=torch.int32, device=device)
        host_map[5] = 1  # miss, host row 1
        src_ptrs = torch.tensor([host.data_ptr()], dtype=torch.int64, device=device)
        dst_ptrs = torch.tensor([land.data_ptr()], dtype=torch.int64, device=device)
        row_bytes = torch.tensor([n_row * 4], dtype=torch.int64, device=device)
        spill_page_in(
            topk, land_ids, slot_host, map_table, host_map, src_ptrs, dst_ptrs, row_bytes
        )
        torch.cuda.synchronize()
        self.assertEqual(int(topk[0, 0]), -1)  # miss -> kept Marlin skips
        self.assertEqual(int(topk[0, 1]), 3)  # hit
        self.assertEqual(int(topk[0, 2]), -1)  # duplicate miss
        self.assertEqual(int(topk[0, 3]), -1)
        self.assertEqual(int(land_ids[0, 0]), 0)
        self.assertEqual(int(land_ids[0, 1]), -1)
        self.assertEqual(int(land_ids[0, 2]), 0)
        self.assertEqual(int(land_ids[0, 3]), -1)
        self.assertTrue(torch.equal(land[0].cpu(), host[1].cpu()))
        # Unused slots early-out; dest stays at the fill value.
        self.assertTrue(torch.equal(land[1].cpu(), torch.full((n_row,), -7, dtype=torch.int32)))

    @unittest.skipUnless(_sm70(), "SM70 required")
    def test_page_in_accepts_dspark_landing_36(self):
        """kMaxLanding was 16; DSPARK sets SGLANG_DSV41_SPILL_LANDING=36."""
        from sglang.kernels.ops.moe.sm70_dsv41_spill_pagein import spill_page_in

        device = torch.device("cuda")
        n_logical = 8
        n_landing = 36
        n_row = 64
        host = (
            torch.arange(4 * n_row, dtype=torch.int32)
            .reshape(4, n_row)
            .contiguous()
            .pin_memory()
        )
        land = torch.zeros((n_landing, n_row), dtype=torch.int32, device=device)
        topk = torch.tensor([[5, 0]], dtype=torch.int32, device=device)
        land_ids = torch.empty_like(topk)
        slot_host = torch.empty(n_landing, dtype=torch.int32, device=device)
        map_table = torch.full((n_logical,), -1, dtype=torch.int32, device=device)
        map_table[0] = 3
        host_map = torch.full((n_logical,), -1, dtype=torch.int32, device=device)
        host_map[5] = 1
        src_ptrs = torch.tensor([host.data_ptr()], dtype=torch.int64, device=device)
        dst_ptrs = torch.tensor([land.data_ptr()], dtype=torch.int64, device=device)
        row_bytes = torch.tensor([n_row * 4], dtype=torch.int64, device=device)
        spill_page_in(
            topk, land_ids, slot_host, map_table, host_map, src_ptrs, dst_ptrs, row_bytes
        )
        torch.cuda.synchronize()
        self.assertEqual(int(topk[0, 0]), -1)
        self.assertEqual(int(topk[0, 1]), 3)
        self.assertEqual(int(land_ids[0, 0]), 0)
        self.assertTrue(torch.equal(land[0].cpu(), host[1].cpu()))
        for s in range(1, n_landing):
            self.assertTrue(torch.equal(land[s].cpu(), torch.zeros(n_row, dtype=torch.int32)), s)


if __name__ == "__main__":
    unittest.main()
