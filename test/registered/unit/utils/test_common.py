import unittest
from array import array
from unittest.mock import patch

import torch

from sglang.srt.utils.common import (
    flatten_arrays_to_int64_tensor,
    get_device_sm_nvidia_smi,
    get_nvidia_driver_version_str,
    is_sm89_supported,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd")


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


class TestIsSm89Supported(CustomTestCase):
    """`is_sm89_supported` returns True only when CUDA is available, the
    device capability is exactly (8, 9), and the CUDA runtime is >= 11.0.
    The lru_cache means results stick across calls within a process, so each
    test patches the underlying probes and clears the cache in setUp/tearDown.
    Runs on CPU CI by mocking `is_cuda` and `torch.cuda.get_device_capability`.
    """

    def setUp(self):
        is_sm89_supported.cache_clear()

    def tearDown(self):
        is_sm89_supported.cache_clear()

    @patch("sglang.srt.utils.common.torch.version.cuda", "12.3")
    @patch("sglang.srt.utils.common.torch.cuda.get_device_capability", return_value=(8, 9))
    @patch("sglang.srt.utils.common.is_cuda", return_value=True)
    def test_returns_true_on_sm89(self, _mock_is_cuda, _mock_cap):
        self.assertTrue(is_sm89_supported())

    @patch("sglang.srt.utils.common.torch.version.cuda", "12.3")
    @patch("sglang.srt.utils.common.torch.cuda.get_device_capability", return_value=(9, 0))
    @patch("sglang.srt.utils.common.is_cuda", return_value=True)
    def test_returns_false_on_sm90(self, _mock_is_cuda, _mock_cap):
        # SM90 must not be misreported as SM89 — a bare `major==8` check would
        # wrongly accept SM80 (Ampere) too, which is exactly why this helper
        # exists instead of reusing `_check_cuda_device_version`.
        self.assertFalse(is_sm89_supported())

    @patch("sglang.srt.utils.common.torch.version.cuda", "12.3")
    @patch("sglang.srt.utils.common.torch.cuda.get_device_capability", return_value=(8, 0))
    @patch("sglang.srt.utils.common.is_cuda", return_value=True)
    def test_returns_false_on_sm80(self, _mock_is_cuda, _mock_cap):
        # SM80 (Ampere) shares major=8 with SM89 but is not Ada Lovelace and
        # lacks the FP8 throughput this gate ultimately guards.
        self.assertFalse(is_sm89_supported())

    @patch("sglang.srt.utils.common.is_cuda", return_value=False)
    def test_returns_false_without_cuda(self, _mock_is_cuda):
        self.assertFalse(is_sm89_supported())


if __name__ == "__main__":
    unittest.main()
