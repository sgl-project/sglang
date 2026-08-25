# SPDX-License-Identifier: Apache-2.0
"""Unit tests for step-based W8A8/W8A16 mixed precision on ModelOpt FP8."""

import unittest
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_fp8 import (
    ModelOptFp8Config,
    ModelOptFp8LinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp8Config as HfModelOptFp8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_fp8_step_precision import (
    StepMixedPrecisionController,
    StepMixedPrecisionFp8LinearMethod,
    install_step_mixed_precision,
)


def _make_loaded_fp8_linear(
    in_features: int = 32, out_features: int = 16
) -> tuple[ReplicatedLinear, torch.Tensor, torch.Tensor]:
    """Build a ReplicatedLinear as the loader would leave it post-load.

    Returns the layer plus the raw FP8 weight and per-tensor scale used to
    fill it, for computing the expected W8A16 output.
    """
    layer = ReplicatedLinear(
        in_features,
        out_features,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=ModelOptFp8Config(),
        prefix="gen_layers.0.self_attn.to_qkv",
    )
    w16 = torch.randn(out_features, in_features, dtype=torch.float32) / 8
    scale = (w16.abs().max() / 448.0).reshape(())
    w_fp8 = (w16 / scale).to(torch.float8_e4m3fn)
    layer.weight.data.copy_(w_fp8)
    layer.weight_scale.data.fill_(scale.item())
    layer.input_scale.data.fill_(1.0)
    layer.quant_method.process_weights_after_loading(layer)
    return layer, w_fp8, scale


class TestDefaults(unittest.TestCase):
    def test_step_mixed_precision_defaults_on(self):
        # Cosmos3 ModelOpt FP8 runs step mixed precision unless the user
        # opts out with SGLANG_DIFFUSION_ENABLE_COSMOS3_STEP_MIXED_PRECISION=0.
        import os

        import sglang.multimodal_gen.envs as envs

        with mock.patch.dict(os.environ):
            os.environ.pop("SGLANG_DIFFUSION_ENABLE_COSMOS3_STEP_MIXED_PRECISION", None)
            self.assertTrue(envs.SGLANG_DIFFUSION_ENABLE_COSMOS3_STEP_MIXED_PRECISION)
            self.assertEqual(envs.SGLANG_DIFFUSION_COSMOS3_STEP_MIXED_PRECISION_FIRST_STEPS, 3)
            self.assertEqual(envs.SGLANG_DIFFUSION_COSMOS3_STEP_MIXED_PRECISION_LAST_STEPS, 3)


class TestStepPolicy(unittest.TestCase):
    def test_edge_steps_select_high_precision(self):
        controller = StepMixedPrecisionController(first_steps=3, last_steps=3)
        selected = []
        for step in range(10):
            controller.set_step(step_index=step, num_steps=10)
            selected.append(controller.high_precision)
        self.assertEqual(
            selected,
            [True, True, True, False, False, False, False, True, True, True],
        )

    def test_single_step_schedule_stays_base_precision(self):
        controller = StepMixedPrecisionController(first_steps=3, last_steps=3)
        controller.set_step(step_index=0, num_steps=1)
        self.assertFalse(controller.high_precision)

    def test_reset_returns_to_base_precision(self):
        controller = StepMixedPrecisionController(first_steps=1, last_steps=0)
        controller.set_step(step_index=0, num_steps=4)
        self.assertTrue(controller.high_precision)
        controller.reset()
        self.assertFalse(controller.high_precision)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            StepMixedPrecisionController(first_steps=-1, last_steps=0)
        controller = StepMixedPrecisionController(first_steps=1, last_steps=1)
        with self.assertRaises(ValueError):
            controller.set_step(step_index=0, num_steps=0)
        with self.assertRaises(IndexError):
            controller.set_step(step_index=5, num_steps=5)


class TestInstallAndDispatch(unittest.TestCase):
    def test_install_wraps_only_modelopt_fp8_linears(self):
        fp8_layer, _, _ = _make_loaded_fp8_linear()
        bf16_layer = ReplicatedLinear(
            8, 8, bias=False, params_dtype=torch.bfloat16, prefix="norm_out"
        )
        root = torch.nn.ModuleList([fp8_layer, bf16_layer])
        controller = StepMixedPrecisionController(first_steps=3, last_steps=3)
        wrapped = install_step_mixed_precision(
            module_lists=[root], controller=controller
        )
        self.assertEqual(wrapped, 1)
        self.assertIsInstance(fp8_layer.quant_method, StepMixedPrecisionFp8LinearMethod)
        self.assertNotIsInstance(
            bf16_layer.quant_method, StepMixedPrecisionFp8LinearMethod
        )

    def test_w8a16_matches_dequantized_reference(self):
        layer, w_fp8, scale = _make_loaded_fp8_linear()
        controller = StepMixedPrecisionController(first_steps=1, last_steps=0)
        install_step_mixed_precision(module_lists=[layer], controller=controller)
        controller.set_step(step_index=0, num_steps=4)
        self.assertTrue(controller.high_precision)

        x = torch.randn(5, layer.input_size, dtype=torch.bfloat16)
        out, _ = layer(x)
        expected = torch.nn.functional.linear(
            x, w_fp8.to(torch.bfloat16) * scale.to(torch.bfloat16)
        )
        torch.testing.assert_close(out, expected)

    def test_install_wraps_hf_quant_config_variant(self):
        # The `modelopt_fp8` hf_quant_config path uses a different
        # ModelOptFp8LinearMethod class (modelopt_quant.py); the installer
        # must wrap it too and the shared W8A16 dequant must hold.
        layer = ReplicatedLinear(
            32,
            16,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=HfModelOptFp8Config(is_checkpoint_fp8_serialized=True),
            prefix="gen_layers.0.self_attn.to_qkv",
        )
        w16 = torch.randn(16, 32, dtype=torch.float32) / 8
        scale = (w16.abs().max() / 448.0).reshape(())
        w_fp8 = (w16 / scale).to(torch.float8_e4m3fn)
        layer.weight.data.copy_(w_fp8)
        # Emulate this method's post-load state (its real pass needs CUDA
        # quant kernels): transposed FP8 view plus collapsed scalar scales.
        layer.weight.data = layer.weight.data.t()
        layer.weight_scale.data = scale.clone()
        layer.input_scale.data = torch.ones(())

        controller = StepMixedPrecisionController(first_steps=1, last_steps=0)
        wrapped = install_step_mixed_precision(
            module_lists=[layer], controller=controller
        )
        self.assertEqual(wrapped, 1)
        controller.set_step(step_index=0, num_steps=4)

        x = torch.randn(5, 32, dtype=torch.bfloat16)
        out, _ = layer(x)
        expected = torch.nn.functional.linear(
            x, w_fp8.to(torch.bfloat16) * scale.to(torch.bfloat16)
        )
        torch.testing.assert_close(out, expected)

    def test_base_steps_dispatch_to_w8a8_method(self):
        layer, _, _ = _make_loaded_fp8_linear()
        controller = StepMixedPrecisionController(first_steps=1, last_steps=1)
        install_step_mixed_precision(module_lists=[layer], controller=controller)
        base = layer.quant_method.base_method
        self.assertIsInstance(base, ModelOptFp8LinearMethod)

        x = torch.randn(2, layer.input_size, dtype=torch.bfloat16)
        with mock.patch.object(
            base, "apply", return_value=torch.zeros(2, layer.output_size)
        ) as base_apply:
            controller.set_step(step_index=2, num_steps=6)
            layer(x)
            base_apply.assert_called_once()
            base_apply.reset_mock()
            # Edge step: the W8A16 path runs and the base method is bypassed.
            controller.set_step(step_index=5, num_steps=6)
            layer(x)
            base_apply.assert_not_called()


if __name__ == "__main__":
    unittest.main()
