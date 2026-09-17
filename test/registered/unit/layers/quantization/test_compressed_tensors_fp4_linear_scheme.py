"""CPU tests for FP4 linear scheme selection in compressed-tensors.

Two failure modes are guarded here. NVFP4 checkpoints used to be unservable on
pre-Blackwell GPUs: the w4a4 scheme is the only FP4 linear scheme and it reports
a min capability of 100, so an SM90 host raised NotImplementedError instead of
falling back to the weight-only FP4 Marlin kernel. Separately, weight-only FP4
configs were diverted into the WNA16 branch -- which only checked strategy, not
weight type -- and died with a misleading error naming W4A16Sparse24.

Scheme selection is pure config parsing, so it runs on CPU with the reported
device capability mocked.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import logging
import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization.compressed_tensors import compressed_tensors
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Fp4,
    CompressedTensorsW4A16Fp4,
    CompressedTensorsWNA16,
)
from sglang.test.test_utils import CustomTestCase

LINEAR_LAYER = "model.layers.0.self_attn.q_proj"
# The module logger the downgrade warning is emitted on.
SCHEME_LOGGER = compressed_tensors.logger

NVFP4_WEIGHTS = {
    "num_bits": 4,
    "type": "float",
    "symmetric": True,
    "strategy": "tensor_group",
    "group_size": 16,
    "dynamic": False,
}
MXFP4_WEIGHTS = {
    "num_bits": 4,
    "type": "float",
    "symmetric": True,
    "strategy": "group",
    "group_size": 32,
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
        "ignore": ["lm_head"],
    }


def _get_scheme(config_dict, capability):
    quant_config = CompressedTensorsConfig.from_config(config_dict)
    with mock.patch("torch.cuda.get_device_capability", return_value=capability):
        return quant_config.get_linear_scheme(
            torch.nn.Linear(16, 16), layer_name=LINEAR_LAYER
        )


class TestFp4LinearSchemeSelection(CustomTestCase):
    def test_nvfp4a16_selects_weight_only_on_hopper(self):
        scheme = _get_scheme(
            _make_config("nvfp4-pack-quantized", NVFP4_WEIGHTS), (9, 0)
        )
        self.assertIsInstance(scheme, CompressedTensorsW4A16Fp4)
        self.assertFalse(scheme.has_input_global_scale)

    def test_nvfp4_w4a4_downgrades_to_weight_only_on_hopper(self):
        scheme = _get_scheme(
            _make_config(
                "nvfp4-pack-quantized", NVFP4_WEIGHTS, dict(NVFP4_WEIGHTS, dynamic=True)
            ),
            (9, 0),
        )
        self.assertIsInstance(scheme, CompressedTensorsW4A16Fp4)
        # The checkpoint still carries input_global_scale, which must be
        # registered so the weight loader finds a destination for it.
        self.assertTrue(scheme.has_input_global_scale)

    def test_nvfp4_w4a4_keeps_native_scheme_on_blackwell(self):
        """The weight-only fallback must not lower the SM100 gate on the native
        w4a4 path -- the reason the two schemes are separate classes."""
        scheme = _get_scheme(
            _make_config(
                "nvfp4-pack-quantized", NVFP4_WEIGHTS, dict(NVFP4_WEIGHTS, dynamic=True)
            ),
            (10, 0),
        )
        self.assertIsInstance(scheme, CompressedTensorsW4A4Fp4)

    def test_nvfp4a16_is_weight_only_on_blackwell_too(self):
        scheme = _get_scheme(
            _make_config("nvfp4-pack-quantized", NVFP4_WEIGHTS), (10, 0)
        )
        self.assertIsInstance(scheme, CompressedTensorsW4A16Fp4)

    def test_hopper_downgrade_warns_and_a16_does_not(self):
        """The silent-precision-downgrade warning is user-facing behavior.

        warning_once is lru_cache-wrapped, so the message is emitted once per
        process and an earlier test that triggered it would leave this one
        asserting against a cache hit. Clear the cache to stay order-independent.
        """
        logging.Logger.warning_once.cache_clear()
        self.addCleanup(logging.Logger.warning_once.cache_clear)

        w4a4_config = _make_config(
            "nvfp4-pack-quantized", NVFP4_WEIGHTS, dict(NVFP4_WEIGHTS, dynamic=True)
        )
        with self.assertLogs(SCHEME_LOGGER, level="WARNING") as captured:
            _get_scheme(w4a4_config, (9, 0))
        self.assertIn("weight-only", "\n".join(captured.output))

        # nvfp4a16 drops no activation quantization, so it must stay silent.
        with mock.patch.object(SCHEME_LOGGER, "warning_once") as warn:
            _get_scheme(_make_config("nvfp4-pack-quantized", NVFP4_WEIGHTS), (9, 0))
        warn.assert_not_called()

    def test_int4_pack_quantized_still_selects_wna16(self):
        """Guards the `type == INT` check added to _is_wNa16_group_channel."""
        scheme = _get_scheme(_make_config("pack-quantized", INT4_WEIGHTS), (9, 0))
        self.assertIsInstance(scheme, CompressedTensorsWNA16)
        self.assertEqual(scheme.group_size, 128)

    def test_mxfp4_reports_no_compatible_scheme(self):
        """MXFP4 linears have no kernel yet (the dense FP4 Marlin kernel is only
        instantiated for group_size 16 and cannot decode E8M0 scales). The error
        must say that, not name an unrelated sparse-24 scheme."""
        for name, input_activations in (
            ("mxfp4a16", None),
            ("mxfp4", dict(MXFP4_WEIGHTS, dynamic=True)),
        ):
            with self.subTest(variant=name):
                with self.assertRaises(NotImplementedError) as ctx:
                    _get_scheme(
                        _make_config(
                            "mxfp4-pack-quantized", MXFP4_WEIGHTS, input_activations
                        ),
                        (9, 0),
                    )
                self.assertNotIn("Sparse24", str(ctx.exception))

    def test_fp4_weights_with_non_nvfp4_activations_raise(self):
        """A float4 weight config with activation quantization that is not the
        NVFP4 layout must be rejected rather than silently served weight-only."""
        with self.assertRaises(NotImplementedError) as ctx:
            _get_scheme(
                _make_config(
                    "nvfp4-pack-quantized",
                    NVFP4_WEIGHTS,
                    dict(NVFP4_WEIGHTS, group_size=32, dynamic=True),
                ),
                (9, 0),
            )
        self.assertIn("NVFP4", str(ctx.exception))


class TestWeightOnlyFp4WeightCreation(CustomTestCase):
    """The registered parameters are a contract with the checkpoint's tensor
    names, shapes and dtypes; a mismatch surfaces only as a load-time failure."""

    def _create(self, has_input_global_scale):
        layer = torch.nn.Module()
        scheme = CompressedTensorsW4A16Fp4(
            has_input_global_scale=has_input_global_scale
        )
        scheme.create_weights(
            layer=layer,
            output_partition_sizes=[512],
            input_size_per_partition=2048,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *args, **kwargs: None,
        )
        return layer, scheme

    def test_registered_parameters_match_checkpoint_layout(self):
        layer, _ = self._create(has_input_global_scale=False)
        self.assertEqual(layer.weight_packed.shape, (512, 1024))
        self.assertEqual(layer.weight_packed.dtype, torch.uint8)
        # One E4M3 scale per 16 input elements.
        self.assertEqual(layer.weight_scale.shape, (512, 128))
        self.assertEqual(layer.weight_scale.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.weight_global_scale.shape, (1,))
        self.assertFalse(hasattr(layer, "input_global_scale"))

    def test_w4a4_checkpoint_gets_an_input_scale_destination(self):
        layer, _ = self._create(has_input_global_scale=True)
        self.assertEqual(layer.input_global_scale.shape, (1,))

    def test_marlin_prep_attributes_are_set(self):
        """prepare_nvfp4_layer_for_marlin reads both off the layer; a missing
        quant_config silently skips its group_size validation, and a missing
        params_dtype makes it raise on a None activation dtype."""
        layer, _ = self._create(has_input_global_scale=False)
        self.assertEqual(layer.params_dtype, torch.bfloat16)
        self.assertEqual(layer.quant_config.group_size, 16)


if __name__ == "__main__":
    unittest.main()
