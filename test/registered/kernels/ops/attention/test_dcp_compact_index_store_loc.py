"""Tests for the DCP-sharded indexer's store-path compaction
(dcp_compact_store_loc), covering two separate bugs found and fixed here:

1. CUDA-graph capture crash: the original implementation filtered to owned
   rows via a boolean mask (``tensor[valid_mask]``) guarded by an
   ``if not valid_mask.all():`` host sync -- both a dynamic output shape and
   a stream-capture-illegal host sync, which crashed decode CUDA graph
   capture with ``cudaErrorStreamCaptureUnsupported`` /
   ``cudaErrorStreamCaptureInvalidated``. Fixed by routing non-owned rows to
   a fixed scratch/dummy index instead of filtering them out (fixed-shape,
   capture-safe).

2. Physical-page-contiguity corruption: ownership and the compacted address
   were originally computed at PER-TOKEN granularity (``slot % dcp_size``,
   ``slot // dcp_size``), which does not preserve "every page_size-row
   window is one physical page" -- the invariant the paged-MQA-logits
   kernel's block_tables addressing requires. Fixed by computing ownership
   and the compacted address at PAGE granularity instead (see
   dcp_localize_index_kv's module docstring); this store-path function must
   use the exact same formula as the read-side dcp_localize_page_table.
"""

from __future__ import annotations

import unittest

import torch

from sglang.kernels.ops.attention.dsa.dcp_localize_index_kv import (
    dcp_compact_store_loc,
    dcp_localize_page_table,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


def _reference_compact_loc(
    out_cache_loc: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    page_size: int,
    dummy_loc: int,
) -> torch.Tensor:
    result = torch.full_like(out_cache_loc, dummy_loc)
    for i in range(out_cache_loc.numel()):
        slot = int(out_cache_loc[i])
        page_id = slot // page_size
        if page_id % dcp_size == dcp_rank:
            local_page_id = page_id // dcp_size
            result[i] = local_page_id * page_size + slot % page_size
    return result


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestDCPCompactIndexStoreLoc(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.device = torch.device("cuda")

    def tearDown(self):
        torch.cuda.empty_cache()
        super().tearDown()

    def test_matches_reference_page_level_formula(self) -> None:
        dcp_size, rank, page_size, dummy = 3, 1, 4, 999
        out_cache_loc = torch.arange(50, dtype=torch.int64, device=self.device)
        result = dcp_compact_store_loc(out_cache_loc, dcp_size, rank, page_size, dummy)
        expected = _reference_compact_loc(
            out_cache_loc, dcp_size, rank, page_size, dummy
        )
        torch.testing.assert_close(result, expected)

    def test_matches_read_side_local_physical_slot(self) -> None:
        # The store side and the read side (dcp_localize_page_table) MUST
        # agree on where an owned token's data physically lands -- otherwise
        # the indexer would score whatever garbage happens to already be at
        # the (wrong) address the read side computes.
        dcp_size, rank, page_size, dummy = 4, 2, 64, 100_000
        torch.manual_seed(0)
        num_pages = 6
        page_ids = torch.randperm(1000)[:num_pages]
        flat_slots = torch.cat(
            [
                torch.arange(
                    pid.item() * page_size,
                    pid.item() * page_size + page_size,
                    dtype=torch.int64,
                )
                for pid in page_ids
            ]
        )

        store_side = dcp_compact_store_loc(flat_slots, dcp_size, rank, page_size, dummy)
        owned = store_side != dummy

        page_table_1 = flat_slots.unsqueeze(0).to(torch.int32)
        capacity = num_pages * page_size
        local_page_table, _, _ = dcp_localize_page_table(
            page_table_1, dcp_size, rank, capacity, page_size
        )
        read_side_values = local_page_table[0][local_page_table[0] >= 0]

        # Every value the store path would write for an owned token must
        # appear among the read side's compacted addresses (same set, since
        # both must agree on physical placement), and vice versa.
        self.assertEqual(
            set(store_side[owned].tolist()), set(read_side_values.tolist())
        )

    def test_owned_entries_are_injective_no_collisions(self) -> None:
        # Every owned token's local_loc must be distinct -- a collision here
        # would mean two different global slots silently overwrite the same
        # compacted physical row.
        dcp_size, page_size = 4, 8
        torch.manual_seed(1)
        num_pages = 40
        page_ids = torch.randperm(1000)[:num_pages]
        flat_slots = torch.cat(
            [
                torch.arange(
                    pid.item() * page_size,
                    pid.item() * page_size + page_size,
                    dtype=torch.int64,
                )
                for pid in page_ids
            ]
        ).to(self.device)
        for rank in range(dcp_size):
            result = dcp_compact_store_loc(
                flat_slots, dcp_size, rank, page_size, dummy_loc=1_000_000
            )
            owned_locs = result[result != 1_000_000]
            self.assertEqual(
                owned_locs.numel(), torch.unique(owned_locs).numel(), f"rank {rank}"
            )

    def test_capturable_in_cuda_graph_and_replay_matches_eager(self) -> None:
        """Regression test for the exact crash: the old (filter + .all())
        implementation cannot even be captured -- verify that directly, then
        verify the fixed-shape version captures and replays correctly."""
        dcp_size, rank, page_size, dummy = 2, 1, 64, 500_000
        out_cache_loc = torch.randint(
            0, 4000, (256,), dtype=torch.int64, device=self.device
        )

        def _old_broken(loc: torch.Tensor) -> torch.Tensor:
            valid_mask = loc % dcp_size == rank
            if not valid_mask.all():
                loc = loc[valid_mask]
            return torch.div(loc, dcp_size, rounding_mode="floor")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _old_broken(out_cache_loc)
        torch.cuda.current_stream().wait_stream(s)
        with self.assertRaises(RuntimeError):
            g_broken = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_broken):
                _old_broken(out_cache_loc)

        static_loc = out_cache_loc.clone()
        s2 = torch.cuda.Stream()
        s2.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s2):
            for _ in range(3):
                dcp_compact_store_loc(static_loc, dcp_size, rank, page_size, dummy)
        torch.cuda.current_stream().wait_stream(s2)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_result = dcp_compact_store_loc(
                static_loc, dcp_size, rank, page_size, dummy
            )

        new_loc = torch.randint(0, 4000, (256,), dtype=torch.int64, device=self.device)
        static_loc.copy_(new_loc)
        g.replay()
        torch.cuda.synchronize()

        expected = dcp_compact_store_loc(new_loc, dcp_size, rank, page_size, dummy)
        torch.testing.assert_close(static_result, expected)


if __name__ == "__main__":
    unittest.main()
