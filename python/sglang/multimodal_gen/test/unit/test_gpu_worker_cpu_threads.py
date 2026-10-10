# SPDX-License-Identifier: Apache-2.0
"""Contracts for per-worker host CPU setup: intra-op threads and NUMA binding."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.managers.gpu_worker import (
    _bind_worker_to_gpu_numa_node,
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


class TestWorkerNumaBinding(unittest.TestCase):
    def test_explicit_node_is_indexed_by_local_gpu(self):
        with patch("sglang.srt.utils.numa_utils.numa_bind_to_node") as bind:
            node = _bind_worker_to_gpu_numa_node(
                SimpleNamespace(numa_node=[0, 0, 1, 1]), local_rank=2
            )
        self.assertEqual(node, 1)
        bind.assert_called_once_with(1)

    def test_auto_detection_queries_the_worker_gpu(self):
        with (
            patch.dict("os.environ", {"SGLANG_AUTO_NUMA_BIND": "1"}),
            patch("sglang.srt.utils.numa_utils._is_numa_available", return_value=True),
            patch(
                "sglang.srt.utils.numa_utils._query_numa_node_for_gpu",
                return_value=[1],
            ) as query,
            patch("sglang.srt.utils.numa_utils.numa_bind_to_node") as bind,
        ):
            node = _bind_worker_to_gpu_numa_node(
                SimpleNamespace(numa_node=None), local_rank=3
            )
        query.assert_called_once_with(3)
        self.assertEqual(node, 1)
        bind.assert_called_once_with(1)

    def test_disabled_auto_bind_leaves_worker_unbound(self):
        with (
            patch.dict("os.environ", {"SGLANG_AUTO_NUMA_BIND": "0"}),
            patch("sglang.srt.utils.numa_utils.numa_bind_to_node") as bind,
        ):
            node = _bind_worker_to_gpu_numa_node(
                SimpleNamespace(numa_node=None), local_rank=0
            )
        self.assertIsNone(node)
        bind.assert_not_called()


if __name__ == "__main__":
    unittest.main()
