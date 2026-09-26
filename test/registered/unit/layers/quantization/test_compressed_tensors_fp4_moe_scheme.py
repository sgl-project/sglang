"""CPU tests for FP4 MoE scheme selection in compressed-tensors.

granite-5.0-20b targets bare `Linear`, so its experts are quantized too and the
linear schemes alone cannot load it. Two failure modes are guarded. NVFP4 MoE
had no weight-only scheme, so an SM90 host either raised or -- for the a16
variant, which has no input_activations at all -- dereferenced None in the w8a8
predicates and killed the scheduler with an AttributeError. Separately, the
weight-only fallback must not lower the SM100 gate on the native w4a4 path.

Scheme selection is pure config parsing, so it runs on CPU with the reported
device capability mocked.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Nvfp4MoE,
    CompressedTensorsW4A16Nvfp4MoE,
    CompressedTensorsW8A8Fp8MoE,
)
from sglang.srt.runtime_context import override_platform
from sglang.test.test_utils import CustomTestCase

MOE_LAYER = "model.layers.0.block_sparse_moe.experts"

NVFP4_WEIGHTS = {
    "num_bits": 4,
    "type": "float",
    "symmetric": True,
    "strategy": "tensor_group",
    "group_size": 16,
    "dynamic": False,
}
INT4_WEIGHTS = {
    "num_bits": 4,
    "type": "int",
    "symmetric": True,
    "strategy": "group",
    "group_size": 128,
    "dynamic": False,
}
FP8_DYNAMIC_ACT = {
    "num_bits": 8,
    "type": "float",
    "symmetric": True,
    "strategy": "token",
    "dynamic": True,
}


def _make_config(quant_format, weights, input_activations=None):
    return {
        "quant_method": "compressed-tensors",
        "format": quant_format,
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": weights,
                "input_activations": input_activations,
            }
        },
        "ignore": ["lm_head", "re:.*block_sparse_moe.router"],
    }


class FusedMoE(torch.nn.Module):
    """find_matched_target matches on the module's class name, and
    _add_fused_moe_to_target_scheme_map aliases a bare `Linear` target onto
    `FusedMoE`; the name is what makes the lookup resolve."""


def _get_moe_scheme(config_dict, capability):
    """`capability` drives _check_scheme_supported; the w4a4 scheme's __init__
    reads get_platform().is_blackwell instead, so both must agree."""
    quant_config = CompressedTensorsConfig.from_config(config_dict)
    with (
        override_platform(is_blackwell=capability >= (10, 0)),
        mock.patch("torch.cuda.get_device_capability", return_value=capability),
    ):
        return quant_config.get_moe_scheme(FusedMoE(), layer_name=MOE_LAYER)


class TestFp4MoeSchemeSelection(CustomTestCase):
    def test_nvfp4a16_selects_weight_only_on_hopper(self):
        scheme = _get_moe_scheme(
            _make_config("nvfp4-pack-quantized", NVFP4_WEIGHTS), (9, 0)
        )
        self.assertIsInstance(scheme, CompressedTensorsW4A16Nvfp4MoE)
        self.assertEqual(scheme.group_size, 16)
        self.assertFalse(scheme.has_input_global_scale)

    def test_nvfp4_w4a4_downgrades_to_weight_only_on_hopper(self):
        scheme = _get_moe_scheme(
            _make_config(
                "nvfp4-pack-quantized", NVFP4_WEIGHTS, dict(NVFP4_WEIGHTS, dynamic=True)
            ),
            (9, 0),
        )
        self.assertIsInstance(scheme, CompressedTensorsW4A16Nvfp4MoE)
        # The checkpoint still ships input_global_scale tensors; the expert
        # loader raises KeyError on any tensor with no registered destination.
        self.assertTrue(scheme.has_input_global_scale)

    def test_nvfp4_w4a4_keeps_native_scheme_on_blackwell(self):
        """The weight-only fallback must not lower the SM100 gate on the native
        w4a4 MoE path -- the reason the two schemes are separate classes."""
        scheme = _get_moe_scheme(
            _make_config(
                "nvfp4-pack-quantized", NVFP4_WEIGHTS, dict(NVFP4_WEIGHTS, dynamic=True)
            ),
            (10, 0),
        )
        self.assertIsInstance(scheme, CompressedTensorsW4A4Nvfp4MoE)

    def test_weight_only_config_does_not_dereference_missing_input_quant(self):
        """A weight-only checkpoint has no input_activations. The w8a8 predicates
        read input_quant.num_bits before any weight-only branch runs, so without
        a None guard this raised AttributeError and killed the scheduler."""
        for name, weights in (("nvfp4a16", NVFP4_WEIGHTS), ("int4", INT4_WEIGHTS)):
            with self.subTest(variant=name):
                quant_format = (
                    "nvfp4-pack-quantized" if name == "nvfp4a16" else "pack-quantized"
                )
                # Must not raise; each variant has its own weight-only scheme.
                scheme = _get_moe_scheme(_make_config(quant_format, weights), (9, 0))
                self.assertIsNotNone(scheme)

    def test_fp8_w8a8_moe_still_selects_fp8(self):
        """Guards the None-guard additions to the w8a8 predicates against
        regressing the path they were written for."""
        fp8_weights = {
            "num_bits": 8,
            "type": "float",
            "symmetric": True,
            "strategy": "channel",
            "dynamic": False,
        }
        scheme = _get_moe_scheme(
            _make_config("float-quantized", fp8_weights, FP8_DYNAMIC_ACT), (9, 0)
        )
        self.assertIsInstance(scheme, CompressedTensorsW8A8Fp8MoE)


class TestWeightOnlyNvfp4MoeWeightCreation(CustomTestCase):
    """The registered parameters are a contract with the checkpoint's tensor
    names, shapes and dtypes; a mismatch surfaces only as a load-time failure."""

    def _create(self, has_input_global_scale=False):
        layer = torch.nn.Module()
        scheme = CompressedTensorsW4A16Nvfp4MoE(
            has_input_global_scale=has_input_global_scale
        )
        scheme.create_weights(
            layer=layer,
            num_experts=8,
            hidden_size=2048,
            intermediate_size_per_partition=1536,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *args, **kwargs: None,
        )
        return layer, scheme

    def test_registered_parameters_match_checkpoint_layout(self):
        layer, _ = self._create()
        # Two fp4 items per byte along the input dimension.
        self.assertEqual(layer.w13_weight_packed.shape, (8, 3072, 1024))
        self.assertEqual(layer.w13_weight_packed.dtype, torch.uint8)
        self.assertEqual(layer.w2_weight_packed.shape, (8, 2048, 768))
        # One E4M3 scale per 16 input elements.
        self.assertEqual(layer.w13_weight_scale.shape, (8, 3072, 128))
        self.assertEqual(layer.w13_weight_scale.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.w2_weight_scale.shape, (8, 2048, 96))

    def test_global_scales_have_one_entry_per_shard(self):
        """w13 carries a gate and an up scale per expert; w2 one per expert."""
        layer, _ = self._create()
        self.assertEqual(layer.w13_weight_global_scale.shape, (8, 2))
        self.assertEqual(layer.w2_weight_global_scale.shape, (8,))

    def test_a16_checkpoint_registers_no_input_scale(self):
        """An nvfp4a16 checkpoint ships none; registering one would claim a
        tensor that does not exist."""
        layer, _ = self._create()
        self.assertFalse(hasattr(layer, "w13_input_global_scale"))
        self.assertFalse(hasattr(layer, "w2_input_global_scale"))

    def test_w4a4_checkpoint_gets_an_input_scale_destination(self):
        """A downgraded w4a4 checkpoint still ships input_global_scale. The
        Marlin kernel never reads it, but granitemoe_load_split_experts raises
        KeyError on a tensor with no registered param, so it needs a home."""
        layer, _ = self._create(has_input_global_scale=True)
        self.assertEqual(layer.w13_input_global_scale.shape, (8, 2))
        self.assertEqual(layer.w2_input_global_scale.shape, (8,))


if __name__ == "__main__":
    unittest.main()
