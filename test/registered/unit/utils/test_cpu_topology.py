"""Unit tests for the CPU topology helpers in srt/utils/device.py."""

import unittest
from unittest import mock

from sglang.srt.utils import device
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMacOSCpuTopology(CustomTestCase):
    def setUp(self):
        device.get_cpu_ids_by_node.cache_clear()
        self.addCleanup(device.get_cpu_ids_by_node.cache_clear)

    def test_macos_uses_one_node_without_lscpu_or_affinity(self):
        """macOS has no `lscpu` and no `Process.cpu_affinity`; every physical
        core is reported on a single node and neither is consulted."""
        with (
            mock.patch.object(device.platform, "system", return_value="Darwin"),
            mock.patch.object(device.psutil, "cpu_count", return_value=4),
            mock.patch.object(
                device.subprocess, "check_output", side_effect=AssertionError
            ),
        ):
            self.assertEqual(
                device.get_physical_cpus_by_numa_or_block(), {0: {0, 1, 2, 3}}
            )
            self.assertEqual(device.get_cpu_ids_by_node(), ["0,1,2,3"])

    def test_macos_falls_back_to_one_cpu_when_count_is_unknown(self):
        """`psutil.cpu_count` can return None; the node still has one CPU."""
        with (
            mock.patch.object(device.platform, "system", return_value="Darwin"),
            mock.patch.object(device.psutil, "cpu_count", return_value=None),
        ):
            self.assertEqual(device.get_physical_cpus_by_numa_or_block(), {0: {0}})

    def test_linux_still_reads_lscpu(self):
        """Off macOS the topology still comes from `lscpu`, filtered by affinity."""
        lscpu = "# header\n0,0,0,0,0\n1,1,0,0,0\n2,2,0,1,0\n3,3,0,1,0\n"
        with (
            mock.patch.object(device.platform, "system", return_value="Linux"),
            mock.patch.object(device.subprocess, "check_output", return_value=lscpu),
            mock.patch.object(device.psutil, "Process") as process,
        ):
            process.return_value.cpu_affinity.return_value = [0, 1, 2, 3]
            self.assertEqual(
                device.get_physical_cpus_by_numa_or_block(), {0: {0, 1}, 1: {2, 3}}
            )


if __name__ == "__main__":
    unittest.main()
