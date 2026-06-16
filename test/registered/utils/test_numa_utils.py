import ctypes
import os
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.utils.numa_utils import (
    _handle_numa_bind_failure,
    _is_numa_available,
    _node_cpus,
    _numactl_cpu_mem_args,
    _query_numa_node_for_gpu,
    get_numa_node_if_available,
    numa_bind_to_node,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")
register_cuda_ci(est_time=9, stage="base-c", runner_config="4-gpu-gb300")
register_cuda_ci(est_time=6, stage="base-c", runner_config="4-gpu-b200")


class TestIsNumaAvailable(unittest.TestCase):
    """Tests for _is_numa_available on both NUMA and non-NUMA systems."""

    @patch("sglang.srt.utils.numa_utils._is_cuda", False)
    def test_returns_false_when_not_cuda(self):
        self.assertFalse(_is_numa_available())

    @patch("sglang.srt.utils.numa_utils._is_cuda", True)
    @patch("os.path.isdir", return_value=False)
    def test_returns_false_when_no_numa_nodes(self, _mock_isdir):
        self.assertFalse(_is_numa_available())

    @patch("sglang.srt.utils.numa_utils._can_set_mempolicy", return_value=True)
    @patch("sglang.srt.utils.numa_utils.shutil.which", return_value="/usr/bin/numactl")
    @patch("sglang.srt.utils.numa_utils._is_cuda", True)
    @patch("os.path.isdir", return_value=True)
    def test_returns_true_on_numa_system(
        self, _mock_isdir, _mock_which, _mock_mempolicy
    ):
        self.assertTrue(_is_numa_available())

    @patch("sglang.srt.utils.numa_utils._can_set_mempolicy", return_value=False)
    @patch("sglang.srt.utils.numa_utils.shutil.which", return_value="/usr/bin/numactl")
    @patch("sglang.srt.utils.numa_utils._is_cuda", True)
    @patch("os.path.isdir", return_value=True)
    def test_returns_false_when_mempolicy_not_permitted(
        self, _mock_isdir, _mock_which, _mock_mempolicy
    ):
        self.assertFalse(_is_numa_available())

    @patch("sglang.srt.utils.numa_utils._can_set_mempolicy", return_value=True)
    @patch("sglang.srt.utils.numa_utils.shutil.which", return_value="/usr/bin/numactl")
    @patch("sglang.srt.utils.numa_utils._is_cuda", True)
    @patch("os.path.isdir", return_value=True)
    def test_isdir_called_with_node1_path(
        self, mock_isdir, _mock_which, _mock_mempolicy
    ):
        _is_numa_available()
        mock_isdir.assert_called_with("/sys/devices/system/node/node1")


class TestQueryNumaNodeForGpu(unittest.TestCase):
    """Tests for _query_numa_node_for_gpu with mocked pynvml."""

    @patch(
        "sglang.srt.utils.numa_utils.glob.glob",
        return_value=[
            "/sys/devices/system/node/node0",
            "/sys/devices/system/node/node1",
        ],
    )
    def test_single_node_affinity(self, _mock_glob):
        c_ulong_bits = ctypes.sizeof(ctypes.c_ulong) * 8
        # Bitmask: bit 0 set -> node 0
        node_set = [1]

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetMemoryAffinity.return_value = node_set
        mock_pynvml.NVML_AFFINITY_SCOPE_NODE = 0

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _query_numa_node_for_gpu(0)

        self.assertEqual(result, [0])
        mock_pynvml.nvmlInit.assert_called_once()
        mock_pynvml.nvmlShutdown.assert_called_once()

    @patch(
        "sglang.srt.utils.numa_utils.glob.glob",
        return_value=[
            "/sys/devices/system/node/node0",
            "/sys/devices/system/node/node1",
        ],
    )
    def test_second_node_affinity(self, _mock_glob):
        # Bitmask: bit 1 set -> node 1
        node_set = [2]

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetMemoryAffinity.return_value = node_set
        mock_pynvml.NVML_AFFINITY_SCOPE_NODE = 0

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _query_numa_node_for_gpu(1)

        self.assertEqual(result, [1])

    @patch(
        "sglang.srt.utils.numa_utils.glob.glob",
        return_value=[
            "/sys/devices/system/node/node0",
            "/sys/devices/system/node/node1",
            "/sys/devices/system/node/node2",
            "/sys/devices/system/node/node3",
        ],
    )
    def test_multiple_node_affinity(self, _mock_glob):
        # Bitmask: bits 1 and 3 set -> nodes 1, 3 (binary: ...1010 = 10)
        node_set = [0b1010]

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetMemoryAffinity.return_value = node_set
        mock_pynvml.NVML_AFFINITY_SCOPE_NODE = 0

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _query_numa_node_for_gpu(0)

        self.assertEqual(result, [1, 3])

    @patch(
        "sglang.srt.utils.numa_utils.glob.glob",
        return_value=[
            "/sys/devices/system/node/node0",
            "/sys/devices/system/node/node1",
        ],
    )
    def test_no_affinity(self, _mock_glob):
        node_set = [0]

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetMemoryAffinity.return_value = node_set
        mock_pynvml.NVML_AFFINITY_SCOPE_NODE = 0

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _query_numa_node_for_gpu(0)

        self.assertEqual(result, [])

    @patch(
        "sglang.srt.utils.numa_utils.glob.glob",
        return_value=[
            "/sys/devices/system/node/node0",
            "/sys/devices/system/node/node1",
        ],
    )
    def test_nvml_shutdown_called_on_success(self, _mock_glob):
        node_set = [1]
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetMemoryAffinity.return_value = node_set
        mock_pynvml.NVML_AFFINITY_SCOPE_NODE = 0

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            _query_numa_node_for_gpu(0)

        mock_pynvml.nvmlShutdown.assert_called_once()


class TestGetNumaNodeIfAvailable(unittest.TestCase):
    """Tests for get_numa_node_if_available combining _is_numa_available + _query_numa_node_for_gpu."""

    def _make_server_args(self, numa_node=None):
        args = MagicMock()
        args.numa_node = numa_node
        return args

    def test_returns_explicit_numa_node_from_server_args(self):
        args = self._make_server_args(numa_node=[2, 3, 0, 1])
        self.assertEqual(get_numa_node_if_available(args, 0), 2)
        self.assertEqual(get_numa_node_if_available(args, 1), 3)
        self.assertEqual(get_numa_node_if_available(args, 2), 0)
        self.assertEqual(get_numa_node_if_available(args, 3), 1)

    @patch("sglang.srt.utils.numa_utils._is_numa_available", return_value=False)
    def test_returns_none_when_numa_not_available(self, _mock_avail):
        args = self._make_server_args(numa_node=None)
        self.assertIsNone(get_numa_node_if_available(args, 0))

    @patch("sglang.srt.utils.numa_utils._query_numa_node_for_gpu", return_value=[])
    @patch("sglang.srt.utils.numa_utils._is_numa_available", return_value=True)
    def test_returns_none_when_query_returns_empty(self, _mock_avail, _mock_gpu):
        args = self._make_server_args(numa_node=None)
        self.assertIsNone(get_numa_node_if_available(args, 0))

    @patch("sglang.srt.utils.numa_utils._query_numa_node_for_gpu", return_value=[1])
    @patch("sglang.srt.utils.numa_utils._is_numa_available", return_value=True)
    def test_returns_queried_single_node(self, _mock_avail, _mock_gpu):
        args = self._make_server_args(numa_node=None)
        self.assertEqual(get_numa_node_if_available(args, 0), 1)

    @patch("sglang.srt.utils.numa_utils._query_numa_node_for_gpu", return_value=[0, 2])
    @patch("sglang.srt.utils.numa_utils._is_numa_available", return_value=True)
    def test_returns_first_node_when_multiple_found(self, _mock_avail, _mock_gpu):
        args = self._make_server_args(numa_node=None)
        self.assertEqual(get_numa_node_if_available(args, 0), 0)

    @patch("sglang.srt.utils.numa_utils._query_numa_node_for_gpu", return_value=[0, 2])
    @patch("sglang.srt.utils.numa_utils._is_numa_available", return_value=True)
    def test_logs_warning_when_multiple_nodes(self, _mock_avail, _mock_gpu):
        args = self._make_server_args(numa_node=None)
        with self.assertLogs("sglang.srt.utils.numa_utils", level="WARNING") as cm:
            get_numa_node_if_available(args, 0)
        self.assertTrue(any("Multiple NUMA nodes" in msg for msg in cm.output))

    @patch("sglang.srt.utils.numa_utils._is_numa_available", return_value=True)
    @patch("sglang.srt.utils.numa_utils._query_numa_node_for_gpu", return_value=[1])
    def test_explicit_server_args_takes_precedence(self, _mock_gpu, _mock_avail):
        args = self._make_server_args(numa_node=[5, 6])
        result = get_numa_node_if_available(args, 0)
        self.assertEqual(result, 5)
        _mock_avail.assert_not_called()
        _mock_gpu.assert_not_called()


def _get_gpu_info():
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return name, count
    except Exception:
        return "", 0


_gpu_name, _gpu_count = _get_gpu_info()


def _query_single_numa_node_for_gpu(gpu_id: int):
    nodes = _query_numa_node_for_gpu(gpu_id)
    if len(nodes) != 1:
        raise AssertionError(f"GPU {gpu_id}: expected one NUMA node, got {nodes}")
    return nodes[0]


@unittest.skipUnless(
    ("GB200" in _gpu_name or "GB300" in _gpu_name) and _gpu_count == 4,
    "Requires 4-GPU Grace Blackwell hardware",
)
class TestGraceBlackwellNumaTopology(unittest.TestCase):
    """Hardware test validating expected NUMA topology on 4-GPU GB200/GB300."""

    def test_gpu_numa_mapping(self):
        self.assertEqual(_gpu_count, 4)
        expected = {0: 0, 1: 0, 2: 1, 3: 1}
        for gpu_id, expected_node in expected.items():
            result = _query_single_numa_node_for_gpu(gpu_id)
            self.assertEqual(
                result,
                expected_node,
                f"GPU {gpu_id}: expected NUMA node {expected_node}, got {result}",
            )


@unittest.skipUnless(
    "B200" in _gpu_name and _gpu_count == 4,
    "Requires 4-GPU B200 hardware",
)
class TestB200NumaTopology(unittest.TestCase):
    """Hardware test validating expected NUMA topology on 4-GPU B200."""

    def test_gpu_numa_mapping(self):
        self.assertEqual(_gpu_count, 4)
        numa_nodes = {
            _query_single_numa_node_for_gpu(gpu_id) for gpu_id in range(_gpu_count)
        }
        self.assertEqual(
            len(numa_nodes),
            1,
            f"Expected all visible 4-GPU B200 devices on one NUMA node, got {numa_nodes}",
        )


class TestNumaBindIntersection(unittest.TestCase):
    """Tests for constraint-aware NUMA binding (node CPUs intersected with the
    process's allowed CPUs)."""

    @patch("sglang.srt.utils.numa_utils.get_libnuma", return_value=None)
    def test_node_cpus_no_libnuma_returns_empty(self, _mock_lib):
        self.assertEqual(_node_cpus(0), set())

    @patch("os.sched_getaffinity", return_value=set(range(72)))
    @patch("sglang.srt.utils.numa_utils._node_cpus", return_value=set(range(72)))
    def test_numactl_args_unconstrained_uses_cpunodebind(self, _cpus, _aff):
        self.assertEqual(_numactl_cpu_mem_args(0, 0), "--cpunodebind=0 --membind=0")

    @patch("os.sched_getaffinity", return_value={0} | set(range(21, 144)))
    @patch("sglang.srt.utils.numa_utils._node_cpus", return_value=set(range(72)))
    def test_numactl_args_constrained_uses_physcpubind(self, _cpus, _aff):
        expected_cpus = ",".join(str(c) for c in [0] + list(range(21, 72)))
        self.assertEqual(
            _numactl_cpu_mem_args(0, 0),
            f"--physcpubind={expected_cpus} --membind=0",
        )

    @patch.dict(os.environ, {"SGLANG_CRASH_ON_NUMA_BIND_FAILURE": "0"})
    @patch("os.sched_getaffinity", return_value=set(range(72, 144)))
    @patch("sglang.srt.utils.numa_utils._node_cpus", return_value=set(range(72)))
    def test_numactl_args_empty_intersection_returns_none(self, _cpus, _aff):
        self.assertIsNone(_numactl_cpu_mem_args(0, 0))

    @patch.dict(os.environ, {"SGLANG_CRASH_ON_NUMA_BIND_FAILURE": "1"})
    @patch("os.sched_getaffinity", return_value=set(range(72, 144)))
    @patch("sglang.srt.utils.numa_utils._node_cpus", return_value=set(range(72)))
    def test_numactl_args_empty_intersection_crashes_when_enabled(self, _cpus, _aff):
        with self.assertRaises(RuntimeError):
            _numactl_cpu_mem_args(0, 0)

    @patch("os.sched_setaffinity")
    @patch("os.sched_getaffinity", return_value={0} | set(range(21, 144)))
    @patch("sglang.srt.utils.numa_utils._node_cpus", return_value=set(range(72)))
    @patch("sglang.srt.utils.numa_utils.get_libnuma")
    def test_numa_bind_to_node_constrained_sets_intersection(
        self, mock_libnuma, _cpus, _aff, mock_setaff
    ):
        lib = MagicMock()
        lib.numa_available.return_value = 0
        mock_libnuma.return_value = lib

        numa_bind_to_node(0)

        mock_setaff.assert_called_once_with(0, {0} | set(range(21, 72)))
        lib.numa_set_preferred.assert_called_once()
        lib.numa_run_on_node.assert_not_called()

    @patch.dict(os.environ, {"SGLANG_CRASH_ON_NUMA_BIND_FAILURE": "0"})
    @patch("os.sched_setaffinity")
    @patch("os.sched_getaffinity", return_value=set(range(72, 144)))
    @patch("sglang.srt.utils.numa_utils._node_cpus", return_value=set(range(72)))
    @patch("sglang.srt.utils.numa_utils.get_libnuma")
    def test_numa_bind_to_node_empty_intersection_skips(
        self, mock_libnuma, _cpus, _aff, mock_setaff
    ):
        lib = MagicMock()
        lib.numa_available.return_value = 0
        mock_libnuma.return_value = lib

        numa_bind_to_node(0)

        mock_setaff.assert_not_called()
        lib.numa_set_preferred.assert_not_called()

    @patch.dict(os.environ, {"SGLANG_CRASH_ON_NUMA_BIND_FAILURE": "1"})
    def test_handle_failure_raises_when_enabled(self):
        with self.assertRaises(RuntimeError):
            _handle_numa_bind_failure(0, {72, 73})

    @patch.dict(os.environ, {"SGLANG_CRASH_ON_NUMA_BIND_FAILURE": "0"})
    def test_handle_failure_warns_when_disabled(self):
        with self.assertLogs("sglang.srt.utils.numa_utils", level="WARNING"):
            _handle_numa_bind_failure(0, {72, 73})


if __name__ == "__main__":
    unittest.main()
