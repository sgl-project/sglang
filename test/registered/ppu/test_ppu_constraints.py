"""Unit tests for PPU-specific behavior guards and config defaults.

Tests that _handle_ppu_backends() correctly sets PPU defaults,
and that PPU-specific constraints are enforced.
All tests use mocks — no PPU hardware required.

NOTE: This file lives in test/registered/ppu/ for organizational clarity,
but registers with register_cpu_ci because it only uses mocks and runs
on any CI runner without PPU hardware.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=10, suite="stage-a-test-cpu-ppu")


class TestPPUDetection(unittest.TestCase):
    def test_device_detection(self):
        from sglang.srt.utils.common import is_ppu

        # Detection uses CUDA availability and the device name, not PPU_SDK.
        for available, name, expected in (
            (True, "ZW810E", True),
            (True, "NVIDIA H100", False),
            (False, "ZW810E", False),
        ):
            with (
                self.subTest(available=available, name=name),
                patch.dict(os.environ, {"PPU_SDK": "/opt/ppu-sdk"}),
                patch("torch.cuda.is_available", return_value=available),
                patch("torch.cuda.get_device_name", return_value=name) as device_name,
            ):
                is_ppu.cache_clear()
                try:
                    self.assertEqual(is_ppu(), expected)
                    if not available:
                        device_name.assert_not_called()
                finally:
                    is_ppu.cache_clear()


class TestPPUBackendDefaults(unittest.TestCase):
    """Verify _handle_ppu_backends() sets correct defaults when is_ppu() is True."""

    def setUp(self):
        self.enterContext(patch.dict(os.environ))

    @patch("sglang.srt.server_args.is_ppu", return_value=True)
    @patch("sglang.srt.server_args.check_acext_version_compatibility")
    def test_custom_allreduce_disabled(self, _mock_acext, _mock_ppu):
        from sglang.srt.server_args import ServerArgs

        args = MagicMock(spec=ServerArgs)
        args.enable_custom_all_reduce = False
        args.disable_custom_all_reduce = False
        ServerArgs._handle_ppu_backends(args)
        self.assertTrue(args.disable_custom_all_reduce)

    @patch("sglang.srt.server_args.is_ppu", return_value=True)
    @patch("sglang.srt.server_args.check_acext_version_compatibility")
    def test_custom_allreduce_not_forced_when_enabled(self, _mock_acext, _mock_ppu):
        from sglang.srt.server_args import ServerArgs

        args = MagicMock(spec=ServerArgs)
        args.enable_custom_all_reduce = True
        args.disable_custom_all_reduce = False
        ServerArgs._handle_ppu_backends(args)
        self.assertFalse(args.disable_custom_all_reduce)

    @patch.dict(os.environ, {}, clear=False)
    @patch("sglang.srt.server_args.is_ppu", return_value=True)
    @patch("sglang.srt.server_args.check_acext_version_compatibility")
    def test_env_defaults_set(self, _mock_acext, _mock_ppu):
        from sglang.srt.environ import envs
        from sglang.srt.server_args import ServerArgs

        for env_name in [
            "SGLANG_SAIL_USE_ACEXT_CUDA",
            "SGLANG_SAIL_DEEPGEMM_DENSE",
            "SGLANG_SAIL_DEEPGEMM_MOE",
            "SGLANG_SAIL_PLA_CUDA",
            "SGLANG_OPT_USE_TOPK_V2",
        ]:
            os.environ.pop(env_name, None)

        args = MagicMock(spec=ServerArgs)
        args.enable_custom_all_reduce = False
        args.disable_custom_all_reduce = False

        ServerArgs._handle_ppu_backends(args)

        self.assertTrue(envs.SGLANG_SAIL_USE_ACEXT_CUDA.get())
        self.assertTrue(envs.SGLANG_SAIL_DEEPGEMM_DENSE.get())
        self.assertTrue(envs.SGLANG_SAIL_DEEPGEMM_MOE.get())
        self.assertTrue(envs.SGLANG_SAIL_PLA_CUDA.get())
        self.assertFalse(envs.SGLANG_OPT_USE_TOPK_V2.get())


if __name__ == "__main__":
    unittest.main(verbosity=3)
