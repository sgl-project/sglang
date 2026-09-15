import unittest
from array import array
from unittest import mock

import torch

from sglang.srt.utils.common import (
    _read_cgroup_memory_max,
    flatten_arrays_to_int64_tensor,
    get_available_gpu_memory,
    get_device_sm_nvidia_smi,
    get_nvidia_driver_version_str,
    is_musa,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd")


class TestMusaDetection(CustomTestCase):
    def test_is_musa_is_torch_compile_safe(self):
        is_musa.cache_clear()

        @torch.compile(backend="eager", fullgraph=True)
        def add_platform_offset(value):
            return value + 1 if is_musa() else value - 1

        value = torch.zeros(1)
        actual = add_platform_offset(value)
        expected = torch.ones(1) if is_musa() else -torch.ones(1)
        torch.testing.assert_close(actual, expected)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFlattenArraysToInt64Tensor(CustomTestCase):
    """`flatten_arrays_to_int64_tensor` is invoked by `prepare_for_extend`
    to build the per-batch input_ids tensor (pinned, async H2D) from a
    list of array.array('q') per-req get_fill_ids() slices. Tests the
    full matrix of (device, pin) the production code paths through.
    """

    DEVICES = ("cpu", "cuda")
    PIN_OPTIONS = (False, True)

    def _check(self, parts: list, expected: list[int]) -> None:
        for device in self.DEVICES:
            for pin in self.PIN_OPTIONS:
                with self.subTest(device=device, pin=pin):
                    out = flatten_arrays_to_int64_tensor(parts, device, pin)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    self.assertEqual(out.dtype, torch.int64)
                    self.assertEqual(out.device.type, device)
                    self.assertEqual(out.shape, (len(expected),))
                    self.assertEqual(out.cpu().tolist(), expected)

    def test_single_part(self):
        parts = [array("q", [1, 2, 3, 4, 5])]
        self._check(parts, [1, 2, 3, 4, 5])

    def test_multiple_parts(self):
        parts = [
            array("q", [10, 20, 30]),
            array("q", [100, 200]),
            array("q", [1000]),
        ]
        self._check(parts, [10, 20, 30, 100, 200, 1000])


class TestNvidiaDriverVersionStr(CustomTestCase):
    """`get_nvidia_driver_version_str` is typed as `str | None`: it returns
    `None` when nvidia-smi is missing, fails, or emits an empty string. These
    tests exercise both the success and the None-return paths by monkey-
    patching `subprocess.run`, so they don't require a GPU. The function is
    `@lru_cache`d, so the cache is cleared around each test to make the patch
    observable.
    """

    def setUp(self):
        get_nvidia_driver_version_str.cache_clear()

    def tearDown(self):
        get_nvidia_driver_version_str.cache_clear()

    def test_returns_version_string(self):
        import subprocess

        class _R:
            stdout = "595.58.03\n"

        original = subprocess.run
        subprocess.run = lambda *a, **k: _R()
        try:
            self.assertEqual(get_nvidia_driver_version_str(), "595.58.03")
        finally:
            subprocess.run = original

    def test_returns_none_on_empty_output(self):
        import subprocess

        class _R:
            stdout = "\n"

        original = subprocess.run
        subprocess.run = lambda *a, **k: _R()
        try:
            self.assertIsNone(get_nvidia_driver_version_str())
        finally:
            subprocess.run = original

    def test_returns_none_on_called_process_error(self):
        import subprocess

        original = subprocess.run

        def boom(*a, **k):
            raise subprocess.CalledProcessError(1, "nvidia-smi")

        subprocess.run = boom
        try:
            self.assertIsNone(get_nvidia_driver_version_str())
        finally:
            subprocess.run = original

    def test_returns_none_on_file_not_found(self):
        import subprocess

        original = subprocess.run

        def boom(*a, **k):
            raise FileNotFoundError("nvidia-smi")

        subprocess.run = boom
        try:
            self.assertIsNone(get_nvidia_driver_version_str())
        finally:
            subprocess.run = original


class TestGetDeviceSmNvidiaSmi(CustomTestCase):
    """`get_device_sm_nvidia_smi` parses nvidia-smi output into a (major,
    minor) tuple and falls back to (0, 0) -- logging via `logger.error` --
    when nvidia-smi fails. The success path needs a GPU; the fallback path is
    covered here by forcing a failure and asserting the (0, 0) return. The
    fallback path needs no GPU, so this test runs on CPU.
    """

    def test_fallback_on_failure_returns_zero_zero(self):
        import subprocess

        original = subprocess.run

        def boom(*a, **k):
            raise subprocess.CalledProcessError(1, "nvidia-smi")

        subprocess.run = boom
        try:
            self.assertEqual(get_device_sm_nvidia_smi(), (0, 0))
        finally:
            subprocess.run = original


_COMMON = "sglang.srt.utils.common"


class TestReadCgroupMemoryMax(CustomTestCase):
    """`_read_cgroup_memory_max` returns the container's byte limit, or None
    when unlimited/unreadable. The None case is load-bearing: get_available_gpu_
    memory and get_cpu_memory_capacity branch on it to keep bare-metal /
    full-machine containers on the original host-based estimate. A regression
    that made "max" or a missing file parse as a real limit reintroduced the
    over-estimated memory that destabilized the full-machine GNR CI box.
    """

    def _read(self, content):
        m = mock.mock_open(read_data=content)
        with mock.patch("builtins.open", m):
            return _read_cgroup_memory_max()

    def test_numeric_bytes(self):
        self.assertEqual(self._read("236223201280\n"), 236223201280)

    def test_unit_suffix_scaled_to_bytes(self):
        self.assertEqual(self._read("2g\n"), 2 * 1024**3)

    def test_unlimited_returns_none(self):
        # cgroup-v2 writes the literal "max" when there is no limit.
        self.assertIsNone(self._read("max\n"))

    def test_unreadable_returns_none(self):
        with mock.patch("builtins.open", side_effect=FileNotFoundError):
            self.assertIsNone(_read_cgroup_memory_max())


class TestGetAvailableGpuMemoryCpu(CustomTestCase):
    """The CPU branch of get_available_gpu_memory must use the cgroup limit
    only when the process is actually capped. Two guarded regressions:

    * capped: free = (cgroup_limit - cgroup_used) / numa. Using a host-wide
      "used" here drove the estimate negative once a sibling socket-pinned
      container was resident, so KV-cache sizing failed with "no memory".
    * uncapped: free = psutil.available / numa (unchanged host behavior). A
      prior version used cgroup math unconditionally, inflating the estimate on
      the no-limit GNR box.
    """

    def test_capped_uses_cgroup_limit_minus_used(self):
        with (
            mock.patch(
                f"{_COMMON}._read_cgroup_memory_max", return_value=200 * (1 << 30)
            ),
            mock.patch(f"{_COMMON}.get_used_cpu_memory", return_value=50 * (1 << 30)),
            mock.patch(f"{_COMMON}.get_cpu_ids_by_node", return_value=["0"]),
        ):
            # (200 - 50) GB over 1 numa node.
            self.assertAlmostEqual(get_available_gpu_memory("cpu", 0), 150.0, places=1)

    def test_uncapped_uses_psutil_available(self):
        vm = mock.Mock(available=120 * (1 << 30))
        with (
            mock.patch(f"{_COMMON}._read_cgroup_memory_max", return_value=None),
            mock.patch(f"{_COMMON}.psutil.virtual_memory", return_value=vm),
            mock.patch(f"{_COMMON}.get_cpu_ids_by_node", return_value=["0", "1"]),
        ):
            # 120 GB host-available over 2 numa nodes -> 60 GB.
            self.assertAlmostEqual(get_available_gpu_memory("cpu", 0), 60.0, places=1)

    def test_capped_not_divided_by_empty_numa_nodes(self):
        # A socket-pinned container sees one usable node (empty nodes are
        # dropped upstream in get_cpu_ids_by_node). Dividing the 200 GB limit by
        # 1, not 2, is what lets the socket use its full budget.
        with (
            mock.patch(
                f"{_COMMON}._read_cgroup_memory_max", return_value=200 * (1 << 30)
            ),
            mock.patch(f"{_COMMON}.get_used_cpu_memory", return_value=0),
            mock.patch(f"{_COMMON}.get_cpu_ids_by_node", return_value=["0"]),
        ):
            self.assertAlmostEqual(get_available_gpu_memory("cpu", 0), 200.0, places=1)


if __name__ == "__main__":
    unittest.main()
