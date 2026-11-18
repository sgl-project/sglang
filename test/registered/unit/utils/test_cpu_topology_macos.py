"""Unit tests for srt/utils/device.py: CPU topology on hosts without lscpu."""

import unittest
from unittest.mock import patch

from sglang.srt.utils.device import (
    get_cpu_ids_by_node,
    get_physical_cpus_by_numa_or_block,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCpuTopologyOnMacos(CustomTestCase):
    def test_macos_reports_one_node_without_lscpu_or_affinity(self):
        """macOS has no `lscpu` and no `Process.cpu_affinity()`; the helper must
        report a single node of physical cores instead of consulting either."""
        with (
            patch("sglang.srt.utils.device.platform.system", return_value="Darwin"),
            patch("sglang.srt.utils.device.psutil.cpu_count", return_value=4),
            patch(
                "sglang.srt.utils.device.subprocess.check_output",
                side_effect=FileNotFoundError("lscpu"),
            ),
            patch(
                "sglang.srt.utils.device.psutil.Process",
                side_effect=AttributeError("cpu_affinity"),
            ),
        ):
            self.assertEqual(get_physical_cpus_by_numa_or_block(), {0: {0, 1, 2, 3}})
            self.assertEqual(get_cpu_ids_by_node.__wrapped__(), ["0,1,2,3"])

    def test_macos_falls_back_to_one_cpu_when_count_is_unknown(self):
        """`psutil.cpu_count(logical=False)` may return None; the node must
        still be non-empty so the per-node memory split never divides by zero."""
        with (
            patch("sglang.srt.utils.device.platform.system", return_value="Darwin"),
            patch("sglang.srt.utils.device.psutil.cpu_count", return_value=None),
        ):
            self.assertEqual(get_physical_cpus_by_numa_or_block(), {0: {0}})

    def test_linux_still_reads_lscpu_filtered_by_affinity(self):
        """Off macOS the topology must still come from `lscpu`, restricted to
        the CPUs the process may run on; the Darwin branch must not leak."""
        lscpu = "# header\n0,0,0,0,0\n1,1,0,0,0\n2,2,0,1,0\n3,3,0,1,0\n"
        with (
            patch("sglang.srt.utils.device.platform.system", return_value="Linux"),
            patch(
                "sglang.srt.utils.device.subprocess.check_output", return_value=lscpu
            ),
            patch("sglang.srt.utils.device.psutil.Process") as process,
        ):
            process.return_value.cpu_affinity.return_value = [0, 1, 2]
            self.assertEqual(get_physical_cpus_by_numa_or_block(), {0: {0, 1}, 1: {2}})


if __name__ == "__main__":
    unittest.main()
