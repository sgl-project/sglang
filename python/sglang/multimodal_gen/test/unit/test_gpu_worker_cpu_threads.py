# SPDX-License-Identifier: Apache-2.0
"""Worker initialization tests."""

import os
import unittest
from unittest.mock import patch

from sglang.multimodal_gen.runtime.managers import gpu_worker
from sglang.multimodal_gen.runtime.managers.gpu_worker import (
    _worker_cpu_intra_op_threads,
)


class TestWorkerInitialization(unittest.TestCase):
    def test_scheduler_process_loads_plugins(self):
        with (
            patch(
                "sglang.multimodal_gen.plugins.load_plugins",
                side_effect=RuntimeError("plugins loaded"),
            ),
            self.assertRaisesRegex(RuntimeError, "plugins loaded"),
        ):
            gpu_worker.run_scheduler_process(0, 0, 0, None, None, None, None)


class TestWorkerCpuIntraOpThreads(unittest.TestCase):
    def test_divides_host_cores_across_colocated_workers(self):
        with (
            patch.dict("os.environ", {}, clear=False),
            patch("os.cpu_count", return_value=128),
        ):
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
            os.environ.pop("OMP_NUM_THREADS", None)
            self.assertEqual(_worker_cpu_intra_op_threads(1), 8)

    def test_explicit_omp_setting_wins(self):
        with patch.dict("os.environ", {"OMP_NUM_THREADS": "32"}):
            self.assertIsNone(_worker_cpu_intra_op_threads(8))


if __name__ == "__main__":
    unittest.main()
