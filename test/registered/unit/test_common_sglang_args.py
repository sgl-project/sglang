"""Unit tests for --device in the shared CLI parser of sglang.test.test_utils."""

import argparse
import contextlib
import io
import unittest
from unittest.mock import patch

from sglang.srt.configs.device_config import SUPPORTED_DEVICES, DeviceConfig
from sglang.srt.platforms.cpu import CpuSRTPlatform
from sglang.srt.platforms.cuda import CudaSRTPlatform
from sglang.srt.platforms.rocm import RocmSRTPlatform
from sglang.srt.platforms.xpu import XpuSRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    add_common_sglang_args_and_parse,
    auto_config_device,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(argv):
    """Run the shared parser over ``argv`` and return the namespace."""
    parser = argparse.ArgumentParser()
    with patch("sys.argv", ["prog"] + argv):
        return add_common_sglang_args_and_parse(parser)


def _parse_expecting_exit(argv):
    """Parse ``argv`` expecting rejection; return argparse's stderr text."""
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            _parse(argv)
        except SystemExit:
            return stderr.getvalue()
    raise AssertionError(f"parser unexpectedly accepted {argv!r}")


def _detects(device):
    """Simulate a host whose auto-detection resolves to ``device``."""
    return patch("sglang.test.test_utils.get_device", return_value=device)


# ---------------------------------------------------------------------------
# NVIDIA
# ---------------------------------------------------------------------------


class TestNvidiaCudaDevice(CustomTestCase):
    """Tests for --device on NVIDIA GPUs."""

    def test_platform_device_type_is_a_cli_choice(self):
        self.assertEqual(CudaSRTPlatform().device_type, "cuda")
        self.assertEqual(_parse(["--device", "cuda"]).device, "cuda")

    def test_cli_value_builds_a_device_config(self):
        self.assertEqual(DeviceConfig("cuda").device.type, "cuda")

    def test_auto_resolves_to_cuda(self):
        with _detects("cuda"):
            self.assertEqual(auto_config_device(), "cuda")


# ---------------------------------------------------------------------------
# AMD ROCm
# ---------------------------------------------------------------------------


class TestAmdRocmDevice(CustomTestCase):
    """Tests for --device on AMD GPUs, which PyTorch exposes as "cuda"."""

    def test_platform_device_type_is_cuda_not_rocm(self):
        """AMD is driven through the "cuda" CLI value, which NVIDIA also uses."""
        platform = RocmSRTPlatform()
        self.assertTrue(platform.is_rocm())
        self.assertEqual(platform.device_name, "rocm")
        self.assertEqual(platform.device_type, "cuda")
        self.assertEqual(_parse(["--device", "cuda"]).device, "cuda")

    def test_rocm_is_rejected_by_device_config(self):
        with self.assertRaises(RuntimeError):
            DeviceConfig("rocm")

    def test_rocm_is_not_a_cli_choice(self):
        """The parser must not offer a value the server would refuse."""
        self.assertIn("invalid choice", _parse_expecting_exit(["--device", "rocm"]))


# ---------------------------------------------------------------------------
# Intel XPU
# ---------------------------------------------------------------------------


class TestIntelXpuDevice(CustomTestCase):
    """Tests for --device on Intel GPUs."""

    def test_platform_device_type_is_a_cli_choice(self):
        """The reported bug: Intel XPU got "invalid choice: 'xpu'"."""
        self.assertEqual(XpuSRTPlatform().device_type, "xpu")
        self.assertEqual(_parse(["--device", "xpu"]).device, "xpu")

    def test_cli_value_builds_a_device_config(self):
        self.assertEqual(DeviceConfig("xpu").device.type, "xpu")

    def test_auto_resolves_to_xpu(self):
        with _detects("xpu"):
            self.assertEqual(auto_config_device(), "xpu")


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------


class TestCpuDevice(CustomTestCase):
    """Tests for --device on hosts without an accelerator."""

    def test_platform_device_type_is_a_cli_choice(self):
        self.assertEqual(CpuSRTPlatform().device_type, "cpu")
        self.assertEqual(_parse(["--device", "cpu"]).device, "cpu")

    def test_cli_value_builds_a_device_config(self):
        self.assertEqual(DeviceConfig("cpu").device.type, "cpu")

    def test_auto_resolves_to_cpu(self):
        with _detects("cpu"):
            self.assertEqual(auto_config_device(), "cpu")

    def test_auto_falls_back_to_cpu_when_detection_fails(self):
        """A host with no accelerator must degrade to CPU, not raise."""
        for error in (RuntimeError("no accelerator"), ImportError("no driver")):
            with self.subTest(error=type(error).__name__):
                with patch("sglang.test.test_utils.get_device", side_effect=error):
                    self.assertEqual(auto_config_device(), "cpu")


# ---------------------------------------------------------------------------
# Parser-wide contracts
# ---------------------------------------------------------------------------


class TestDeviceChoices(CustomTestCase):
    """Tests that hold for every device, not just one backend."""

    def test_every_supported_device_is_accepted(self):
        """Completeness: extending SUPPORTED_DEVICES must not skip the parser."""
        for device in SUPPORTED_DEVICES:
            with self.subTest(device=device):
                self.assertEqual(_parse(["--device", device]).device, device)

    def test_default_is_auto(self):
        """Omitting --device must defer to auto-detection, not pin a device."""
        self.assertEqual(_parse([]).device, "auto")
        self.assertEqual(_parse(["--device", "auto"]).device, "auto")


if __name__ == "__main__":
    unittest.main()
