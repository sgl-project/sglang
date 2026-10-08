# SPDX-License-Identifier: Apache-2.0
"""Checkpoint contract: fused scales and FP32 stability modules must survive FP8."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import GemmaConfig

from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.models.vlas.pi05_core import (
    PiGemmaMLP,
    linear_forward,
)
from sglang.multimodal_gen.runtime.vla.pi05_quantization import (
    DEFAULT_COMPONENTS,
    finalize_fp8_weights,
    projection_names,
    replace_projections,
    validate_quantization_config,
)
from sglang.multimodal_gen.tools.quantize_pi05_modelopt_fp8 import (
    dummy_observations,
    export_state,
    load_observations,
    run_actions,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def tiny_model():
    config = GemmaConfig(
        hidden_size=32,
        intermediate_size=64,
        hidden_activation="gelu_pytorch_tanh",
        dtype="bfloat16",
    )
    model = nn.Module()
    model.paligemma_with_expert = nn.Module()
    model.paligemma_with_expert.gemma_expert = nn.Module()
    model.paligemma_with_expert.gemma_expert.model = nn.Module()
    layer = nn.Module()
    layer.mlp = PiGemmaMLP(config).to(torch.bfloat16)
    layer.input_layernorm = nn.Linear(32, 32).float()
    model.paligemma_with_expert.gemma_expert.model.layers = nn.ModuleList([layer])
    model.action_out_proj = nn.Linear(32, 7).float()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1)
    return model


class TestPi05ModelOptFp8(CustomTestCase):
    def test_fused_conversion_preserves_forward_and_stability_modules(self):
        model = tiny_model()
        layer = model.paligemma_with_expert.gemma_expert.model.layers[0]
        x = torch.randn(1, 3, 32, dtype=torch.bfloat16)
        before = linear_forward(layer.mlp.gate_up_proj, x)
        original_state = {
            name: value.clone() for name, value in model.state_dict().items()
        }
        names = replace_projections(model, ["action_expert"], quantized=False)
        torch.testing.assert_close(
            linear_forward(layer.mlp.gate_up_proj, x), before, atol=0, rtol=0
        )
        self.assertEqual(len(names), 2)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original_state[name], atol=0, rtol=0)
        replace_projections(model, ["action_expert"], quantized=True)
        self.assertEqual(layer.mlp.projection_dtype, torch.bfloat16)
        self.assertEqual(layer.mlp.gate_up_proj.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.mlp.gate_up_proj.weight_scale.numel(), 1)
        self.assertEqual(layer.input_layernorm.weight.dtype, torch.float32)
        self.assertEqual(model.action_out_proj.weight.dtype, torch.float32)
        with self.assertRaisesRegex(ValueError, "Invalid per-tensor"):
            finalize_fp8_weights(model, names)

    def test_extended_coverage_preserves_bias_interfaces_and_exclusions(self):
        # A missed component would silently export an incomplete FP8 checkpoint;
        # tuple-returning vision projections must still feed their existing callers.
        model = tiny_model()
        paligemma = model.paligemma_with_expert
        paligemma.paligemma = nn.Module()
        paligemma.paligemma.model = nn.Module()
        prefix = paligemma.paligemma.model
        prefix.language_model = nn.Module()
        prefix.language_model.layers = nn.ModuleList([])
        prefix.vision_tower = nn.Module()
        prefix.vision_tower.encoder = nn.Module()
        vision_layer = nn.Module()
        vision_layer.mlp = nn.Module()
        vision_layer.mlp.fc1 = layer = ReplicatedLinear(
            32, 64, bias=True, params_dtype=torch.bfloat16
        )
        prefix.vision_tower.encoder.layers = nn.ModuleList([vision_layer])
        prefix.vision_tower.patch_embedding = nn.Conv2d(3, 32, 2)
        prefix.multi_modal_projector = nn.Module()
        prefix.multi_modal_projector.linear = nn.Linear(64, 32).bfloat16()
        model.action_in_proj = nn.Linear(32, 32).float()
        model.time_mlp_in = nn.Linear(32, 32).float()
        expert_layer = paligemma.gemma_expert.model.layers[0]
        expert_layer.input_layernorm.dense = nn.Linear(32, 96).float()
        components = [c for c in DEFAULT_COMPONENTS if c != "paligemma"]
        x = torch.randn(1, 3, 32, dtype=torch.bfloat16)
        with torch.no_grad():
            layer.weight.uniform_(-0.1, 0.1)
            layer.bias.uniform_(-0.1, 0.1)
        before, _ = layer(x)
        bias = layer.bias.detach().clone()
        names = replace_projections(model, components, quantized=False)
        after, unused_bias = vision_layer.mlp.fc1(x)
        torch.testing.assert_close(after, before, atol=0, rtol=0)
        self.assertIsNone(unused_bias)
        torch.testing.assert_close(vision_layer.mlp.fc1.bias, bias)
        self.assertEqual(len(names), 6)
        replace_projections(model, components, quantized=True)
        self.assertEqual(model.action_in_proj.out_features, 32)
        self.assertEqual(model.action_out_proj.output_dtype, torch.float32)
        self.assertEqual(
            prefix.multi_modal_projector.linear.weight.dtype, torch.float8_e4m3fn
        )
        self.assertEqual(model.time_mlp_in.weight.dtype, torch.float32)
        self.assertEqual(expert_layer.input_layernorm.dense.weight.dtype, torch.float32)
        self.assertEqual(
            prefix.vision_tower.patch_embedding.weight.dtype, torch.float32
        )

    def test_export_uses_modelopt_amax_and_preserves_unquantized_weights(self):
        model = tiny_model()
        names = projection_names(model, ["action_expert"])
        for name in names:
            layer = model.get_submodule(name)
            # Boundary fixture: ModelOpt's calibrated quantizer state.
            for key, value in (("weight_quantizer", 2.0), ("input_quantizer", 3.0)):
                quantizer = nn.Module()
                quantizer.register_buffer("amax", torch.tensor(value))
                setattr(layer, key, quantizer)
        state = export_state(model, names)
        for name in names:
            actual = state[f"{name}.weight"].float() * state[f"{name}.weight_scale"]
            expected = model.get_submodule(name).weight.float()
            torch.testing.assert_close(actual, expected, atol=0.007, rtol=0.05)
        self.assertEqual(state["action_out_proj.weight"].dtype, torch.float32)
        self.assertFalse(any("_quantizer." in name for name in state))
        model.get_submodule(names[0]).input_quantizer.amax.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "invalid calibrated"):
            export_state(model, names)

    def test_calibration_integrates_every_step_without_mutating_fixed_noise(self):
        # A linear velocity field has an analytic two-step Euler result:
        # x=2 -> -0.5 at t=1 -> -0.25 at t=0.5, with dt=-0.5.
        class LinearFlow:
            def encode_prefix(self, *args):
                return None, None, True

            def prepare_denoise_layout(self, *args):
                return None

            def denoise_step(self, masks, kv, actions, timestep, full, **kwargs):
                return 2 * actions + timestep[:, None, None]

        sample = {
            "images": [],
            "image_masks": [],
            "tokens": None,
            "token_masks": None,
            "noise": torch.full((1, 2, 3), 2.0),
        }
        actual = run_actions(LinearFlow(), sample, num_steps=2)
        torch.testing.assert_close(
            actual, torch.full_like(actual, -0.25), atol=0, rtol=0
        )
        torch.testing.assert_close(
            sample["noise"], torch.full_like(actual, 2.0), atol=0, rtol=0
        )

    def test_dummy_inputs_are_reproducible_without_changing_global_rng(self):
        config = SimpleNamespace(
            image_size=(4, 4),
            image_keys=("image", "empty_camera_0"),
            max_token_len=8,
            action_horizon=3,
            action_dim=2,
        )
        rng_state = torch.random.get_rng_state()
        first = dummy_observations(config, 1, 123, torch.device("cpu"))[0]
        second = dummy_observations(config, 1, 123, torch.device("cpu"))[0]
        held_out = dummy_observations(config, 1, 124, torch.device("cpu"))[0]
        self.assertTrue(torch.equal(rng_state, torch.random.get_rng_state()))
        torch.testing.assert_close(first["noise"], second["noise"], atol=0, rtol=0)
        torch.testing.assert_close(first["tokens"], second["tokens"], atol=0, rtol=0)
        self.assertFalse(torch.equal(first["noise"], held_out["noise"]))
        self.assertEqual(first["noise"].shape, (1, 3, 2))
        # The tutorial's dummy path marks every camera present, including empties.
        self.assertTrue(all(mask.all() for mask in first["image_masks"]))
        with self.assertRaisesRegex(ValueError, "positive"):
            dummy_observations(config, 0, 123, torch.device("cpu"))

    def test_rejects_unfused_checkpoint_and_empty_calibration(self):
        with self.assertRaisesRegex(ValueError, "fused-projection"):
            validate_quantization_config(
                dict(quant_method="modelopt", quant_algo="FP8")
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.pt"
            torch.save([], path)
            with self.assertRaisesRegex(ValueError, "nonempty"):
                load_observations(str(path), None, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
