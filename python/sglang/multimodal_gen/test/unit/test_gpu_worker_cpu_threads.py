# SPDX-License-Identifier: Apache-2.0
"""Contract for the per-worker CPU intra-op thread budget."""

import unittest
from unittest.mock import patch

from sglang.multimodal_gen.runtime.managers.gpu_worker import (
    _worker_cpu_intra_op_threads,
)


class TestWorkerCpuIntraOpThreads(unittest.TestCase):
    def test_divides_host_cores_across_colocated_workers(self):
        with (
            patch.dict("os.environ", {}, clear=False),
            patch("os.cpu_count", return_value=128),
        ):
            import os

            os.environ.pop("OMP_NUM_THREADS", None)
            self.assertEqual(_worker_cpu_intra_op_threads(8), 16)
            self.assertEqual(_worker_cpu_intra_op_threads(4), 16)  # capped
            self.assertEqual(_worker_cpu_intra_op_threads(128), 1)
            self.assertEqual(_worker_cpu_intra_op_threads(256), 1)  # floor

    def test_single_gpu_keeps_cap(self):
        with (
            patch.dict("os.environ", {}, clear=False),
            patch("os.cpu_count", return_value=8),
        ):
            import os

            os.environ.pop("OMP_NUM_THREADS", None)
            self.assertEqual(_worker_cpu_intra_op_threads(1), 8)

    def test_explicit_omp_setting_wins(self):
        with patch.dict("os.environ", {"OMP_NUM_THREADS": "32"}):
            self.assertIsNone(_worker_cpu_intra_op_threads(8))


if __name__ == "__main__":
    unittest.main()
