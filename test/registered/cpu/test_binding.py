import os
import re
import unittest
from unittest.mock import patch

import torch

kernel = torch.ops.sgl_kernel

from sglang.srt.utils.numa_utils import init_threads_binding
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="stage-a-tp-test-cpu-intel")


class TestBinding(CustomTestCase):
    def test_binding(self):
        start_id = 1
        n_cpu = 6

        expected_cores = list(map(str, range(start_id, start_id + n_cpu)))
        cpu_ids = ",".join(expected_cores)
        output = kernel.init_cpu_threads_env(cpu_ids)

        bindings = re.findall(r"OMP tid: \d+, core (\d+)", output)
        self.assertEqual(len(bindings), n_cpu)

        self.assertEqual(bindings, expected_cores)


class TestInitThreadsBinding(unittest.TestCase):
    """Tests for init_threads_binding: NUMA/core selection by global rank.

    numa_index is the worker's global device id (gpu_id) across all DP
    replicas; world_size is dp_size * tp_size * pp_size.
    """

    @patch(
        "sglang.srt.utils.numa_utils.get_cpu_ids_by_node",
        return_value=["0,1,2,3", "4,5,6,7", "8,9,10,11", "12,13,14,15"],
    )
    @patch.dict(os.environ, {"SGLANG_CPU_OMP_THREADS_BIND": "all"})
    def test_dp_ranks_get_distinct_numa_nodes(self, _mock_nodes):
        # dp_size=2, tp_size=2 -> world_size=4, numa_index 0..3 global rank.
        results = [init_threads_binding(numa_index=i, world_size=4) for i in range(4)]
        self.assertEqual(results, ["0,1,2,3", "4,5,6,7", "8,9,10,11", "12,13,14,15"])

    @patch(
        "sglang.srt.utils.numa_utils.get_cpu_ids_by_node",
        return_value=["0,1,2,3", "4,5,6,7"],
    )
    @patch.dict(os.environ, {"SGLANG_CPU_OMP_THREADS_BIND": "0-1|4-5"})
    def test_explicit_bind_list_indexed_by_numa_index(self, _mock_nodes):
        self.assertEqual(init_threads_binding(numa_index=0, world_size=2), "0-1")
        self.assertEqual(init_threads_binding(numa_index=1, world_size=2), "4-5")

    @patch(
        "sglang.srt.utils.numa_utils.get_cpu_ids_by_node",
        return_value=["0,1", "2,3", "4,5"],
    )
    @patch.dict(os.environ, {"SGLANG_CPU_OMP_THREADS_BIND": "0-1|2-3|4-5"})
    def test_router_worker_uses_global_numa_index(self, _mock_nodes):
        # Router mode: each worker is an independent dp_size=1 server, so
        # world_size is locally 1 even though the bind string has multiple
        # groups. numa_index is still the global gpu_id and must be able to
        # select any group, not just index 0.
        self.assertEqual(
            init_threads_binding(numa_index=2, world_size=1),
            "4-5",
        )


if __name__ == "__main__":
    unittest.main()
