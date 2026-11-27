"""TOP DeepSeek-V4.1 32x32 block-FP8 projection: what it refuses instead of running another program."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.true_on_policy import fp8
from sglang.srt.true_on_policy.fp8.fp8_b32_linear import clamped_swiglu_mlp, linear

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_V41 = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
}


def _operands(n=64, k=64, *, x_dtype=torch.bfloat16, scale_shape=None):
    x = torch.zeros(3, k, dtype=x_dtype)
    weight = torch.zeros(n, k, dtype=torch.float8_e4m3fn)
    scale = torch.ones(scale_shape or (n // 32, k // 32), dtype=torch.float32)
    return x, weight, scale


def _quant_config(**overrides):
    fields = dict(
        get_name=lambda: "fp8",
        weight_block_size=[32, 32],
        activation_scheme="dynamic",
        is_checkpoint_fp8_serialized=True,
        use_mxfp8=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestFP8B32Linear(CustomTestCase):
    def test_refuses_autograd_instead_of_a_silent_detached_forward(self):
        x, weight, scale = _operands()
        with self.assertRaisesRegex(NotImplementedError, "backward"):
            linear(x.requires_grad_(True), weight, scale)

    def test_refuses_128x128_scales(self):
        x, weight, scale = _operands(n=256, k=256, scale_shape=(2, 2))
        with self.assertRaisesRegex(NotImplementedError, "32x32 block scales"):
            linear(x, weight, scale)

    def test_refuses_non_bf16_input(self):
        x, weight, scale = _operands(x_dtype=torch.float16)
        with self.assertRaisesRegex(NotImplementedError, "BF16 input"):
            linear(x, weight, scale)

    def test_refuses_k_off_the_32_grid(self):
        x = torch.zeros(3, 48, dtype=torch.bfloat16)
        weight = torch.zeros(64, 48, dtype=torch.float8_e4m3fn)
        scale = torch.ones(2, 1, dtype=torch.float32)
        with self.assertRaisesRegex(NotImplementedError, "K%32"):
            linear(x, weight, scale)


class TestClampedSwigluB32(CustomTestCase):
    def test_refuses_another_clamp_instead_of_running_it(self):
        x, weight, scale = _operands()
        with self.assertRaisesRegex(NotImplementedError, "swiglu_limit 10.0"):
            clamped_swiglu_mlp(x, (weight, scale), (weight, scale), 7.0)

    def test_refuses_a_reducing_down_projection(self):
        down = SimpleNamespace(
            reduce_results=True, input_size_per_partition=64, input_size=64
        )
        mlp = SimpleNamespace(swiglu_limit=10.0, down_proj=down)
        with patch.object(fp8, "served_block", return_value=fp8.BLOCK32):
            with self.assertRaisesRegex(NotImplementedError, "unreduced"):
                fp8.maybe_clamped_mlp(mlp, torch.zeros(2, 64, dtype=torch.bfloat16))


class TestV41CastExperts(CustomTestCase):
    def test_uncast_fp4_experts_are_refused(self):
        """Without SGLANG_DSV4_FP4_DEQUANT the experts reach the program as packed FP4, another program."""
        runner_input = SimpleNamespace(
            hidden_states=torch.zeros(2, 64, dtype=torch.bfloat16)
        )
        with patch.object(fp8, "served_block", return_value=fp8.BLOCK32):
            with self.assertRaisesRegex(NotImplementedError, "E4M3"):
                fp8.maybe_expert_forward(
                    runner_input,
                    SimpleNamespace(),
                    weight_dtype=torch.int8,
                    activation="silu",
                    swiglu_limit=10.0,
                )


class TestFP8B32Format(CustomTestCase):
    def test_v41_config_is_the_32x32_format_and_not_the_128x128_one(self):
        self.assertEqual(fp8.checkpoint_block(_V41), fp8.BLOCK32)
        self.assertFalse(fp8.is_block_fp8(_V41))

    def test_32x32_without_ue8m0_scales_is_no_top_format(self):
        self.assertIsNone(fp8.checkpoint_block({**_V41, "scale_fmt": None}))

    def test_blackwell_is_refused(self):
        with patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):
            with self.assertRaisesRegex(NotImplementedError, "SM90"):
                fp8.validate_b32_quant_config(_quant_config())

    def test_mxfp8_view_is_refused(self):
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            with self.assertRaisesRegex(NotImplementedError, "32x32"):
                fp8.validate_b32_quant_config(_quant_config(use_mxfp8=True))


if __name__ == "__main__":
    unittest.main()
