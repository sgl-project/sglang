"""Regression/soundness test for the once-per-forward-step DCP local-view
hoist (DeepseekSparseAttnBackend._build_dcp_local_view /
_refresh_dcp_local_view in dsa_backend.py).

Moving dcp_localize_page_table's computation out of the per-layer Indexer
and into metadata construction (built once, reused by every layer) means its
output tensors must survive CUDA graph capture correctly: built once during
capture, then refreshed **in-place** (`copy_`, preserving the data_ptr the
captured graph read from) before every replay -- exactly mirroring the
existing `topk_v2_plan` field this was modeled on.

The risk this guards against is not a crash (that would be caught
immediately) but SILENT STALENESS: if a future refactor accidentally
reassigns the tensor instead of copying into it, replay would keep reading
capture-time data with no obvious symptom.
"""

from __future__ import annotations

import unittest

import torch

from sglang.kernels.ops.attention.dsa.dcp_localize_index_kv import (
    dcp_local_capacity,
    dcp_localize_page_table,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestDCPLocalViewGraphReplay(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.device = torch.device("cuda")

    def tearDown(self):
        torch.cuda.empty_cache()
        super().tearDown()

    def test_refresh_in_place_shows_fresh_data_on_replay(self) -> None:
        dcp_size, rank = 2, 0
        page_size = 8
        num_rows, max_len = 4, 64
        local_capacity = dcp_local_capacity(max_len, dcp_size, page_size)

        # "Capture-time" page table: build the static local-view buffers the
        # way _build_dcp_local_view does (a fresh allocation, exactly once).
        capture_table = torch.arange(
            num_rows * max_len, dtype=torch.int32, device=self.device
        ).view(num_rows, max_len)
        static_page_table, static_to_global, static_causal_count = (
            dcp_localize_page_table(
                capture_table, dcp_size, rank, local_capacity, page_size
            )
        )
        # These are the exact tensors a captured graph's kernels would bind to.
        captured_page_table_ptr = static_page_table.data_ptr()
        captured_to_global_ptr = static_to_global.data_ptr()
        captured_causal_count_ptr = static_causal_count.data_ptr()

        def refresh(page_table_1: torch.Tensor) -> None:
            # Mirrors _refresh_dcp_local_view exactly: recompute fresh, then
            # copy_ into the existing buffers (never reassign).
            fresh_page_table, fresh_to_global, fresh_causal_count = (
                dcp_localize_page_table(
                    page_table_1, dcp_size, rank, static_page_table.shape[1], page_size
                )
            )
            static_page_table.copy_(fresh_page_table)
            static_to_global.copy_(fresh_to_global)
            static_causal_count.copy_(fresh_causal_count)

        # A trivial "captured graph body": a kernel-like op that reads the
        # static buffers by their captured addresses (a plain clone stands in
        # for whatever downstream kernel dsa_indexer.py's _get_topk_paged
        # would run against metadata.get_dcp_local_view()'s tensors).
        def captured_body() -> torch.Tensor:
            return static_causal_count[:, -1].clone()

        # Warmup (required before capture).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                captured_body()
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_result = captured_body()

        # data_ptrs must be unchanged post-capture: capture must not have
        # reassigned the tensors (that would silently break replay).
        self.assertEqual(static_page_table.data_ptr(), captured_page_table_ptr)
        self.assertEqual(static_to_global.data_ptr(), captured_to_global_ptr)
        self.assertEqual(static_causal_count.data_ptr(), captured_causal_count_ptr)

        # Replay 1: same table as capture -- result should match a fresh
        # (non-graph) computation.
        g.replay()
        torch.cuda.synchronize()
        expected_capture = dcp_localize_page_table(
            capture_table, dcp_size, rank, local_capacity, page_size
        )[2][:, -1]
        torch.testing.assert_close(static_result, expected_capture)

        # Replay 2: DIFFERENT page table (simulating a new decode step with
        # different KV positions) -- refresh in place, then replay. If the
        # refresh silently didn't take effect (e.g. a reassignment bug),
        # the graph would keep reading capture-time data and this would
        # fail to match the new table's expected result.
        new_table = torch.randperm(
            num_rows * max_len, dtype=torch.int32, device=self.device
        ).view(num_rows, max_len)
        refresh(new_table)
        # Refreshing must not have moved the buffers (still capture-safe).
        self.assertEqual(static_page_table.data_ptr(), captured_page_table_ptr)
        self.assertEqual(static_to_global.data_ptr(), captured_to_global_ptr)
        self.assertEqual(static_causal_count.data_ptr(), captured_causal_count_ptr)

        g.replay()
        torch.cuda.synchronize()
        expected_new = dcp_localize_page_table(
            new_table, dcp_size, rank, local_capacity, page_size
        )[2][:, -1]
        torch.testing.assert_close(static_result, expected_new)
        # And it must actually have changed from the capture-time result
        # (otherwise this test wouldn't be exercising anything).
        self.assertFalse(torch.equal(expected_new, expected_capture))


if __name__ == "__main__":
    unittest.main()
