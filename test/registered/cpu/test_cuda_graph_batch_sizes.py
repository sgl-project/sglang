# SPDX-License-Identifier: Apache-2.0
"""Batch-size list generation for CUDA graph capture (issue #41923)."""

import unittest

from sglang.srt.arg_groups.cuda_graph_hook import (
    generate_decode_cuda_graph_batch_sizes,
    generate_prefill_cuda_graph_batch_sizes,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu-intel")
register_cpu_ci(est_time=2, suite="base-b-test-cpu-arm64")


class TestPrefillCudaGraphBatchSizes(CustomTestCase):
    def test_off_grid_ceiling_is_captured(self):
        # 3000 sits between grid points; the ceiling itself must be captured
        # instead of silently rounding down to 2816.
        sizes = generate_prefill_cuda_graph_batch_sizes(3000)
        self.assertEqual(sizes[-1], 3000)

    def test_on_grid_ceiling_not_duplicated(self):
        sizes = generate_prefill_cuda_graph_batch_sizes(2048)
        self.assertEqual(sizes[-1], 2048)
        self.assertEqual(len(sizes), len(set(sizes)))

    def test_ceiling_below_first_grid_point(self):
        self.assertEqual(generate_prefill_cuda_graph_batch_sizes(2), [2])

    def test_list_stays_sorted(self):
        for max_bs in (3000, 4500, 4607, 5000, 6000, 10000):
            sizes = generate_prefill_cuda_graph_batch_sizes(max_bs)
            self.assertEqual(sizes, sorted(sizes))
            self.assertEqual(sizes[-1], max_bs)

    def test_decode_ceiling_parity(self):
        # The decode path has always appended the ceiling; pin the parity the
        # prefill fix targets.
        server_args = ServerArgs(model_path="dummy")
        capture_bs = generate_decode_cuda_graph_batch_sizes(server_args, 3000)
        self.assertEqual(capture_bs[-1], 3000)


if __name__ == "__main__":
    unittest.main()
