import ctypes
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.hardware_backend.npu.topology import NPUTopology
from sglang.srt.utils.cpu_affinity import (
    allocate_cpu_cores,
    bind_cpu_affinity,
    parse_cpu_list,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCPUAllocation(unittest.TestCase):
    def test_cpu_ranges(self):
        self.assertEqual(parse_cpu_list(" 0-2,8,11-12\n"), {0, 1, 2, 8, 11, 12})

    def test_invalid_cpu_ranges(self):
        for value in ("", "NA", "3-1", "1,,2", "-1", "1-2-3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_cpu_list(value)

    @patch("sglang.srt.utils.cpu_affinity.Path.read_text")
    def test_noncontiguous_smt_and_remainder(self, read):
        # Each call examines the siblings of the requested CPUs, in sorted order.
        siblings = ["2,18", "4,20", "8,24", "2,18", "4,20", "8,24"]
        cpus = {2, 4, 8, 18, 20, 24}
        read.side_effect = siblings * 2
        left = allocate_cpu_cores(cpus, 0, 2)
        right = allocate_cpu_cores(cpus, 1, 2)
        self.assertEqual(left, {2, 18})
        self.assertEqual(right, {4, 8, 20, 24})
        self.assertFalse(left & right)
        self.assertEqual(left | right, cpus)

    @patch("sglang.srt.utils.cpu_affinity.Path.read_text", return_value="2,18")
    def test_disallowed_smt_sibling_is_not_added(self, _read):
        self.assertEqual(allocate_cpu_cores({18}, 0, 1), {18})

    @patch("sglang.srt.utils.cpu_affinity.Path.read_text", return_value="2,18")
    def test_insufficient_physical_cores(self, _read):
        with self.assertRaisesRegex(ValueError, "physical cores"):
            allocate_cpu_cores({2, 18}, 0, 2)


class TestNPUTopology(unittest.TestCase):
    @staticmethod
    def runtime(mapping):
        def lookup(device, output):
            ctypes.cast(output, ctypes.POINTER(ctypes.c_int32))[0] = mapping[device]
            return 0

        return Mock(side_effect=lookup)

    @patch("sglang.srt.hardware_backend.npu.topology.subprocess.run")
    @patch("sglang.srt.hardware_backend.npu.topology.ctypes.CDLL")
    def test_runtime_remapping_and_topology(self, cdll, run):
        # Deliberately differs from any visible-device environment list: the
        # runtime also accounts for container mapping and owns the translation.
        query = self.runtime({0: 5, 1: 3})
        cdll.return_value = SimpleNamespace(aclrtGetPhyDevIdByUserDevId=query)
        run.return_value.stdout = (
            " NPU3 NPU5 CPU Affinity\n"
            "NPU3 X HCCS 96-119\n"
            "NPU5 HCCS X 0-23\n"
            "Legend:\n X = Self\n"
        )
        topology = NPUTopology()
        self.assertEqual(topology.device(0), (5, set(range(24))))
        self.assertEqual(topology.device(1), (3, set(range(96, 120))))
        self.assertEqual([c.args[0] for c in query.call_args_list], [0, 1])
        run.assert_called_once_with(
            ["npu-smi", "info", "-t", "topo"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )

    @patch("sglang.srt.hardware_backend.npu.topology.ctypes.CDLL")
    def test_legacy_api_receives_user_index(self, cdll):
        query = self.runtime({0: 6})
        cdll.return_value = SimpleNamespace(aclrtGetPhyDevIdByLogicDevId=query)
        topology = NPUTopology()
        topology._cpu_topology = {6: {48, 49}}
        self.assertEqual(topology.device(0), (6, {48, 49}))
        self.assertEqual(query.call_args.args[0], 0)

    @patch("sglang.srt.hardware_backend.npu.topology.ctypes.CDLL")
    def test_missing_mapping_api(self, cdll):
        cdll.return_value = SimpleNamespace()
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            NPUTopology().device(0)

    def test_mapping_error(self):
        topology = NPUTopology()
        topology._physical_device = Mock(return_value=100000)
        with self.assertRaisesRegex(RuntimeError, "100000"):
            topology.device(0)

    def test_missing_physical_device_topology(self):
        topology = NPUTopology()
        topology._physical_device = self.runtime({0: 4})
        topology._cpu_topology = {0: {0, 1}}
        with self.assertRaisesRegex(ValueError, "physical NPU 4"):
            topology.device(0)

    @patch("sglang.srt.hardware_backend.npu.topology.subprocess.run")
    def test_unavailable_or_malformed_topology(self, run):
        for output in ("", "NPU0 X NA\n", "NPU0 X 0-1\nNPU0 X 2-3\n"):
            run.return_value.stdout = output
            with self.subTest(output=output), self.assertRaises(ValueError):
                NPUTopology()._cpu_topology

    @patch("sglang.srt.hardware_backend.npu.topology.subprocess.run")
    def test_query_timeout(self, run):
        run.side_effect = subprocess.TimeoutExpired("npu-smi", 5)
        with self.assertRaisesRegex(RuntimeError, "topology query failed"):
            NPUTopology()._cpu_topology


class TestBinding(unittest.TestCase):
    def setUp(self):
        self.provider = Mock(name="provider")
        self.provider.name = "NPU"
        self.provider.numa_nodes.return_value = [0]
        self.provider.device.side_effect = lambda i: [
            (3, {96, 97, 98, 99}),
            (4, {0, 1, 2, 3}),
            (5, {0, 1, 2, 3}),
        ][i]
        self.allowed = {0, 1, 2, 3, 96, 97, 98, 99}
        self.current = set(self.allowed)
        self.get = patch(
            "os.sched_getaffinity", side_effect=lambda _: set(self.current), create=True
        ).start()
        self.set = patch(
            "os.sched_setaffinity", side_effect=self.apply, create=True
        ).start()
        self.read = patch(
            "sglang.srt.utils.cpu_affinity.Path.read_text",
            autospec=True,
            side_effect=lambda p: p.parent.parent.name[3:],
        ).start()
        self.addCleanup(patch.stopall)

    def apply(self, pid, cpus):
        self.current = set(cpus)

    def test_cross_numa_and_shared_numa_devices(self):
        expected = [{96, 97, 98, 99}, {0, 1}, {2, 3}]
        for rank in range(3):
            self.current = set(self.allowed)
            self.assertTrue(bind_cpu_affinity(self.provider, rank, 3))
            self.assertEqual(self.current, expected[rank])

    def test_cpuset_intersection(self):
        self.current = {1, 3, 96}
        self.assertTrue(bind_cpu_affinity(self.provider, 2, 3))
        self.assertEqual(self.current, {3})

    def test_no_local_allowed_cpu_keeps_affinity(self):
        self.current = {96}
        self.assertFalse(bind_cpu_affinity(self.provider, 1, 3))
        self.set.assert_not_called()
        self.assertEqual(self.current, {96})

    def test_discovery_failure_does_not_bind(self):
        self.provider.device.side_effect = RuntimeError("topology unavailable")
        self.assertFalse(bind_cpu_affinity(self.provider, 0, 3))
        self.set.assert_not_called()

    def test_strict_discovery_failure(self):
        self.provider.device.side_effect = RuntimeError("topology unavailable")
        with self.assertRaisesRegex(RuntimeError, "topology unavailable"):
            bind_cpu_affinity(self.provider, 0, 3, strict=True)
        self.set.assert_not_called()

    def test_setaffinity_permission_error_is_nonfatal(self):
        self.set.side_effect = PermissionError("denied")
        self.assertFalse(bind_cpu_affinity(self.provider, 0, 3))
        self.set.assert_called_once()

    def test_readback_mismatch_restores_inherited_affinity(self):
        self.get.side_effect = [self.allowed, {96}]
        self.assertFalse(bind_cpu_affinity(self.provider, 0, 3))
        self.assertEqual(self.current, self.allowed)
        self.assertEqual(self.set.call_count, 2)

    def test_duplicate_devices_do_not_bind(self):
        self.provider.device.side_effect = lambda i: (0, {0, 1})
        self.assertFalse(bind_cpu_affinity(self.provider, 0, 2))
        self.set.assert_not_called()

    def test_overlapping_localities_do_not_bind(self):
        self.provider.device.side_effect = lambda i: [(0, {0, 1}), (1, {1, 2})][i]
        self.assertFalse(bind_cpu_affinity(self.provider, 0, 2))
        self.set.assert_not_called()


if __name__ == "__main__":
    unittest.main()
