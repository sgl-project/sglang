import ctypes
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.utils.numa_utils import (
    _is_numa_available,
    _query_numa_node_for_gpu,
    get_numa_node_if_available,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")
register_cuda_ci(est_time=10, suite="stage-c-test-4-gpu-gb200")
register_cuda_ci(est_time=10, suite="stage-c-test-8-gpu-b200")


class TestIsNumaAvailable(unittest.TestCase):
    """Tests for _is_numa_available on both NUMA and non-NUMA systems."""

    @patch("sglang.srt.utils.numa_utils._is_cuda", False)
    def test_returns_false_when_not_cuda(self):
        self.assertFalse(_is_numa_available())

    @patch("sglang.srt.utils.numa_utils._is_cuda", True)
    @patch("os.path.isdir", return_value=False)
    def test_returns_false_when_no_numa_nodes(self, _mock_isdir):
        self.assertFalse(_is_numa_available())

    @patch("sglang.srt.utils.numa_utils._is_cuda", True)
    @patch("os.path.isdir", return_value=True)
    @patch("sglang.srt.utils.numa_utils.psutil")
    def test_returns_false_when_affinity_constrained(self, mock_psutil, _mock_isdir):
        mock_process = MagicMock()
        mock_process.cpu_affinity.return_value = [0, 1]
        mock_psutil.Process.return_value = mock_process
        mock_psutil.cpu_count.return_value = 128

        self.assertFalse(_is_numa_available())

    @patch("sglang.srt.utils.numa_utils._can_set_mempolicy", return_value=True)
    @patch("sglang.srt.utils.numa_utils.shutil.which", return_value="/usr/bin/numactl")
    @patch("sglang.srt.utils.numa_utils._is_cuda", True)
    @patch("os.path.isdir", return_value=True)
    @patch("sglang.srt.utils.numa_utils.psutil")
    def test_returns_true_on_numa_system_with_full_affinity(
        self, mock_psutil, _mock_isdir, _mock_which, _mock_mempolicy
    ):
        all_cpus = list(range(128))
        mock_process = MagicMock()
        mock_process.cpu_affinity.return_value = all_cpus
        mock_psutil.Process.return_value = mock_process
        mock_psutil.cpu_count.return_value = 128

        self.assertTrue(_is_numa_available())

    @patch("sglang.srt.utils.numa_utils._can_set_mempolicy", return_value=False)
    @patch("sglang.srt.utils.numa_utils.shutil.which", return_value="/usr/bin/numactl")
    @patch("sglang.srt.utils.numa_utils._is_cuda", True)
    @patch("os.path.isdir", return_value=True)
    @patch("sglang.srt.utils.numa_utils.psutil")
    def test_returns_false_when_mempolicy_not_permitted(
        self, mock_psutil, _mock_isdir, _mock_which, _mock_mempolicy
    ):
        all_cpus = list(range(128))
        mock_process = MagicMock()
        mock_process.cpu_affinity.return_value = all_cpus
        mock_psutil.Process.return_value = mock_process
        mock_psutil.cpu_count.return_value = 128

        self.assertFalse(_is_numa_available())

    @patch("sglang.srt.utils.numa_utils._can_set_mempolicy", return_value=True)
    @patch("sglang.srt.utils.numa_utils.shutil.which", return_value="/usr/bin/numactl")
    @patch("sglang.srt.utils.numa_utils._is_cuda", True)
    @patch("os.path.isdir", return_value=True)
    @patch("sglang.srt.utils.numa_utils.psutil")
    def test_isdir_called_with_node1_path(
        self, mock_psutil, mock_isdir, _mock_which, _mock_mempolicy
    ):
        all_cpus = list(range(8))
        mock_process = MagicMock()
        mock_process.cpu_affinity.return_value = all_cpus
        mock_psutil.Process.return_value = mock_process
        mock_psutil.cpu_count.return_value = 8

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


def _get_gpu_name():
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        pynvml.nvmlShutdown()
        return name
    except Exception:
        return ""


_gpu_name = _get_gpu_name()


@unittest.skipUnless("GB200" in _gpu_name, "Requires GB200 hardware")
class TestGB200NumaTopology(unittest.TestCase):
    """Hardware test validating expected NUMA topology on GB200 (2 NUMA nodes, 4 GPUs)."""

    def _make_server_args(self):
        args = MagicMock()
        args.numa_node = None
        return args

    def test_gpu_numa_mapping(self):
        expected = {0: 0, 1: 0, 2: 1, 3: 1}
        args = self._make_server_args()
        for gpu_id, expected_node in expected.items():
            result = get_numa_node_if_available(args, gpu_id)
            self.assertEqual(
                result,
                expected_node,
                f"GPU {gpu_id}: expected NUMA node {expected_node}, got {result}",
            )


@unittest.skipUnless("B200" in _gpu_name, "Requires B200 hardware")
class TestB200NumaTopology(unittest.TestCase):
    """Hardware test validating expected NUMA topology on B200 (2 NUMA nodes, 8 GPUs)."""

    def _make_server_args(self):
        args = MagicMock()
        args.numa_node = None
        return args

    def test_gpu_numa_mapping(self):
        expected = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}
        args = self._make_server_args()
        for gpu_id, expected_node in expected.items():
            result = get_numa_node_if_available(args, gpu_id)
            self.assertEqual(
                result,
                expected_node,
                f"GPU {gpu_id}: expected NUMA node {expected_node}, got {result}",
            )


if __name__ == "__main__":
    unittest.main()
