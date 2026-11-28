"""CPU checks of the T-Head PPU monokernel binding: flag selection and the
device-naming answer to mono's standard vendor hook."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.kernel_plugins import mono
from sglang.srt.kernel_plugins.t_head import binding
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class TestTHeadSelected(CustomTestCase):
    def tearDown(self):
        reset_context()

    def test_the_flag_selects_the_t_head_backend(self):
        self.assertEqual(
            binding.selected_monokernel(
                ServerArgs(model_path="dummy", enable_t_head_monokernel=True)
            ),
            ("t_head", "--enable-t-head-monokernel"),
        )

    def test_the_default_configuration_selects_no_monokernel(self):
        self.assertIsNone(binding.selected_monokernel(ServerArgs(model_path="dummy")))

    def test_the_vendor_walk_routes_t_head_through_mono(self):
        publish(
            ServerArgs(model_path="dummy", enable_t_head_monokernel=True), role="test"
        )
        self.assertEqual(mono._backend(), "t_head")
        self.assertEqual(mono._flag(), "--enable-t-head-monokernel")


class TestTHeadDeviceIsa(CustomTestCase):
    def test_the_hook_names_a_sail_device_by_its_reported_part(self):
        with patch("torch.cuda.get_device_name", return_value="PPU-ZW810E"):
            self.assertEqual(binding.device_isa(), "PPU-ZW810E")
            # mono asks the selected binding, not whoever answers first
            self.assertEqual(mono.device_isa(binding), "PPU-ZW810E")

    def test_a_foreign_device_defers_to_the_framework_reading(self):
        with patch("torch.cuda.get_device_name", return_value="NVIDIA L20"):
            self.assertIsNone(binding.device_isa())
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(torch.cuda.current_device())
                expected = f"sm_{props.major}{props.minor}"
            else:
                expected = None
            self.assertEqual(mono.device_isa(binding), expected)


if __name__ == "__main__":
    unittest.main()
