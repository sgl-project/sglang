"""Unit tests for --device in the shared CLI parser of sglang.test.test_utils."""

import argparse
import contextlib
import io
import unittest
from unittest.mock import patch

from sglang.srt.configs.device_config import SUPPORTED_DEVICES, DeviceConfig
from sglang.srt.platforms.rocm import RocmSRTPlatform
from sglang.srt.platforms.xpu import XpuSRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    add_common_sglang_args_and_parse,
    auto_config_device,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _parse(argv):
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


class TestIntelXpuDevice(CustomTestCase):
    def test_platform_device_type_is_a_cli_choice(self):
        """Regression: --device xpu was rejected as "invalid choice: 'xpu'".

        The literal is deliberate; test_every_supported_device_is_accepted
        iterates SUPPORTED_DEVICES, so it stays green if "xpu" leaves that list.
        """
        self.assertEqual(XpuSRTPlatform().device_type, "xpu")
        self.assertEqual(_parse(["--device", "xpu"]).device, "xpu")


class TestAmdRocmDevice(CustomTestCase):
    """Tests for --device on AMD GPUs, which PyTorch exposes as "cuda"."""

    def test_platform_device_type_is_cuda_not_rocm(self):
        """AMD is driven through the "cuda" CLI value, which NVIDIA also uses."""
        platform = RocmSRTPlatform()
        self.assertTrue(platform.is_rocm())
        self.assertEqual(platform.device_name, "rocm")
        self.assertEqual(platform.device_type, "cuda")

    def test_rocm_is_rejected_by_cli_and_device_config(self):
        """The parser must not offer a value the server would refuse."""
        with self.assertRaises(RuntimeError):
            DeviceConfig("rocm")
        self.assertIn("invalid choice", _parse_expecting_exit(["--device", "rocm"]))


class TestCpuDevice(CustomTestCase):
    def test_auto_falls_back_to_cpu_when_detection_fails(self):
        """A host with no accelerator must degrade to CPU, not raise."""
        for error in (RuntimeError("no accelerator"), ImportError("no driver")):
            with self.subTest(error=type(error).__name__):
                with patch("sglang.test.test_utils.get_device", side_effect=error):
                    self.assertEqual(auto_config_device(), "cpu")


class TestDeviceChoices(CustomTestCase):
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
