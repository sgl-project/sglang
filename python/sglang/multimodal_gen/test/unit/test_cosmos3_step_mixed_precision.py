# SPDX-License-Identifier: Apache-2.0
"""Unit tests for step-based W8A8/W8A16 mixed precision on ModelOpt FP8."""

import copy
import os
import unittest
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_fp8 import (
    ModelOptFp8Config,
    ModelOptFp8LinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_fp8_step_precision import (
    GENERATION_PATH,
    REASONER_PATH,
    StepMixedPrecisionController,
    StepMixedPrecisionFp8LinearMethod,
    StepPolicy,
    install_step_mixed_precision,
    read_checkpoint_step_policy,
    resolve_step_policy,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp8Config as HfModelOptFp8Config,
)

# The published checkpoint schema (transformer/config.json,
# quantization_config.runtime.diffusion_step_policy), shared with vLLM-Omni.
CHECKPOINT_POLICY = {
    "schema_version": 1,
    "type": "first_last_n",
    "index_space": "denoising_loop_iteration",
    "scope": ["transformer"],
    "default_mode": "native",
    "first_steps": {"count": 3, "mode": "a16"},
    "last_steps": {"count": 3, "mode": "a16"},
    "overlap": "a16",
    "reasoner": "a16",
}


def _quant_config_with_policy(policy: dict) -> dict:
    return {"quant_algo": "FP8", "runtime": {"diffusion_step_policy": policy}}


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


class TestCheckpointPolicyParsing(unittest.TestCase):
    def test_valid_policy_parses(self):
        policy = read_checkpoint_step_policy(
            _quant_config_with_policy(CHECKPOINT_POLICY)
        )
        self.assertEqual(
            policy, StepPolicy(first_steps=3, last_steps=3, reasoner_a16=True)
        )

    def test_reasoner_native(self):
        raw = copy.deepcopy(CHECKPOINT_POLICY)
        raw["reasoner"] = "native"
        raw["first_steps"]["count"] = 1
        policy = read_checkpoint_step_policy(_quant_config_with_policy(raw))
        self.assertEqual(
            policy, StepPolicy(first_steps=1, last_steps=3, reasoner_a16=False)
        )

    def test_missing_metadata_returns_none(self):
        self.assertIsNone(read_checkpoint_step_policy(None))
        self.assertIsNone(read_checkpoint_step_policy({"quant_algo": "FP8"}))
        self.assertIsNone(
            read_checkpoint_step_policy({"quant_algo": "FP8", "runtime": {}})
        )

    def test_scope_without_transformer_returns_none(self):
        raw = copy.deepcopy(CHECKPOINT_POLICY)
        raw["scope"] = ["vae"]
        self.assertIsNone(read_checkpoint_step_policy(_quant_config_with_policy(raw)))

    def test_malformed_policy_fails_closed(self):
        cases = [
            ("schema_version", 2),
            ("type", "sigmoid"),
            ("index_space", "sigma"),
            ("default_mode", "a16"),
            ("overlap", "native"),
            ("reasoner", "a8"),
            ("first_steps", {"count": -1, "mode": "a16"}),
            ("first_steps", {"count": 3, "mode": "a8"}),
            ("first_steps", {"count": 3}),
            ("scope", []),
        ]
        for field, bad_value in cases:
            raw = copy.deepcopy(CHECKPOINT_POLICY)
            raw[field] = bad_value
            with self.subTest(field=field):
                with self.assertRaises((ValueError, TypeError)):
                    read_checkpoint_step_policy(_quant_config_with_policy(raw))
        raw = copy.deepcopy(CHECKPOINT_POLICY)
        raw["surprise"] = 1
        with self.assertRaises(ValueError):
            read_checkpoint_step_policy(_quant_config_with_policy(raw))
        raw = copy.deepcopy(CHECKPOINT_POLICY)
        del raw["reasoner"]
        with self.assertRaises(ValueError):
            read_checkpoint_step_policy(_quant_config_with_policy(raw))


class TestPolicyResolution(unittest.TestCase):
    def _clean_environ(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in (
            "SGLANG_DIFFUSION_ENABLE_COSMOS3_STEP_MIXED_PRECISION",
            "SGLANG_DIFFUSION_COSMOS3_STEP_MIXED_PRECISION_FIRST_STEPS",
            "SGLANG_DIFFUSION_COSMOS3_STEP_MIXED_PRECISION_LAST_STEPS",
        ):
            os.environ.pop(name, None)

    def test_checkpoint_policy_enables(self):
        self._clean_environ()
        raw = copy.deepcopy(CHECKPOINT_POLICY)
        raw["first_steps"]["count"] = 1
        raw["last_steps"]["count"] = 2
        raw["reasoner"] = "native"
        policy, source = resolve_step_policy(_quant_config_with_policy(raw))
        self.assertEqual(
            policy, StepPolicy(first_steps=1, last_steps=2, reasoner_a16=False)
        )
        self.assertEqual(source, "checkpoint")

    def test_off_without_checkpoint_policy(self):
        # The checkpoint owns the behavior: no diffusion_step_policy means
        # mixed precision must not run.
        self._clean_environ()
        policy, source = resolve_step_policy({"quant_algo": "FP8"})
        self.assertIsNone(policy)
        self.assertEqual(source, "checkpoint carries no diffusion_step_policy")

    def test_explicit_env_force_enables_without_checkpoint_policy(self):
        self._clean_environ()
        os.environ["SGLANG_DIFFUSION_COSMOS3_STEP_MIXED_PRECISION_FIRST_STEPS"] = "2"
        policy, source = resolve_step_policy({"quant_algo": "FP8"})
        self.assertEqual(
            policy, StepPolicy(first_steps=2, last_steps=3, reasoner_a16=True)
        )
        self.assertIn("env vars", source)

    def test_explicit_env_overrides_checkpoint_policy(self):
        self._clean_environ()
        os.environ["SGLANG_DIFFUSION_COSMOS3_STEP_MIXED_PRECISION_FIRST_STEPS"] = "5"
        policy, source = resolve_step_policy(
            _quant_config_with_policy(CHECKPOINT_POLICY)
        )
        self.assertEqual(
            policy, StepPolicy(first_steps=5, last_steps=3, reasoner_a16=True)
        )
        self.assertIn("env override of first_steps", source)

    def test_kill_switch_disables(self):
        self._clean_environ()
        os.environ["SGLANG_DIFFUSION_ENABLE_COSMOS3_STEP_MIXED_PRECISION"] = "0"
        policy, source = resolve_step_policy(
            _quant_config_with_policy(CHECKPOINT_POLICY)
        )
        self.assertIsNone(policy)
        self.assertIn("disabled", source)


class TestStepPolicyDispatch(unittest.TestCase):
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

    def test_reasoner_path_is_static(self):
        controller = StepMixedPrecisionController(
            first_steps=1, last_steps=1, reasoner_a16=False
        )
        controller.set_step(step_index=0, num_steps=10)
        self.assertTrue(controller.use_high_precision(GENERATION_PATH))
        self.assertFalse(controller.use_high_precision(REASONER_PATH))
        controller = StepMixedPrecisionController(
            first_steps=0, last_steps=0, reasoner_a16=True
        )
        controller.set_step(step_index=5, num_steps=10)
        self.assertFalse(controller.use_high_precision(GENERATION_PATH))
        self.assertTrue(controller.use_high_precision(REASONER_PATH))

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
        reasoner_wrapped, generation_wrapped = install_step_mixed_precision(
            reasoner_modules=[], generation_modules=[root], controller=controller
        )
        self.assertEqual((reasoner_wrapped, generation_wrapped), (0, 1))
        self.assertIsInstance(fp8_layer.quant_method, StepMixedPrecisionFp8LinearMethod)
        self.assertEqual(fp8_layer.quant_method.path, GENERATION_PATH)
        self.assertNotIsInstance(
            bf16_layer.quant_method, StepMixedPrecisionFp8LinearMethod
        )

    def test_w8a16_matches_dequantized_reference(self):
        layer, w_fp8, scale = _make_loaded_fp8_linear()
        controller = StepMixedPrecisionController(first_steps=1, last_steps=0)
        install_step_mixed_precision(
            reasoner_modules=[], generation_modules=[layer], controller=controller
        )
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
        _, generation_wrapped = install_step_mixed_precision(
            reasoner_modules=[], generation_modules=[layer], controller=controller
        )
        self.assertEqual(generation_wrapped, 1)
        controller.set_step(step_index=0, num_steps=4)

        x = torch.randn(5, 32, dtype=torch.bfloat16)
        out, _ = layer(x)
        expected = torch.nn.functional.linear(
            x, w_fp8.to(torch.bfloat16) * scale.to(torch.bfloat16)
        )
        torch.testing.assert_close(out, expected)

    def test_reasoner_native_dispatches_to_w8a8_on_edge_steps(self):
        layer, _, _ = _make_loaded_fp8_linear()
        controller = StepMixedPrecisionController(
            first_steps=1, last_steps=1, reasoner_a16=False
        )
        install_step_mixed_precision(
            reasoner_modules=[layer], generation_modules=[], controller=controller
        )
        self.assertEqual(layer.quant_method.path, REASONER_PATH)
        base = layer.quant_method.base_method

        x = torch.randn(2, layer.input_size, dtype=torch.bfloat16)
        with mock.patch.object(
            base, "apply", return_value=torch.zeros(2, layer.output_size)
        ) as base_apply:
            controller.set_step(step_index=0, num_steps=6)
            layer(x)
            base_apply.assert_called_once()

    def test_base_steps_dispatch_to_w8a8_method(self):
        layer, _, _ = _make_loaded_fp8_linear()
        controller = StepMixedPrecisionController(first_steps=1, last_steps=1)
        install_step_mixed_precision(
            reasoner_modules=[], generation_modules=[layer], controller=controller
        )
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
