"""CPU tests for the weight-only FP4 compressed-tensors schemes.

Scheme selection and weight creation are pure config parsing and parameter
registration, so they run on CPU with the device capability mocked.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import logging
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.quantization.compressed_tensors import compressed_tensors
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Fp4,
    CompressedTensorsW4A4Nvfp4MoE,
    CompressedTensorsW4A16Fp4,
    CompressedTensorsW4A16Mxfp4,
    CompressedTensorsW4A16Nvfp4MoE,
    CompressedTensorsW8A8Fp8MoE,
    CompressedTensorsWNA16,
)
from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
from sglang.srt.runtime_context import override_platform
from sglang.test.test_utils import CustomTestCase

LINEAR_LAYER = "model.layers.0.self_attn.q_proj"
MOE_LAYER = "model.layers.0.block_sparse_moe.experts"
SM90, SM100 = (9, 0), (10, 0)
# The module logger the weight-only downgrade warning is emitted on.
SCHEME_LOGGER = compressed_tensors.logger

NVFP4_WEIGHTS = {
    "num_bits": 4,
    "type": "float",
    "symmetric": True,
    "strategy": "tensor_group",
    "group_size": 16,
    "dynamic": False,
}
MXFP4_WEIGHTS = dict(NVFP4_WEIGHTS, strategy="group", group_size=32)
INT4_WEIGHTS = dict(NVFP4_WEIGHTS, type="int", strategy="group", group_size=128)
FP8_WEIGHTS = {
    "num_bits": 8,
    "type": "float",
    "symmetric": True,
    "strategy": "channel",
    "dynamic": False,
}
FP8_DYNAMIC_ACT = dict(FP8_WEIGHTS, strategy="token", dynamic=True)

# variant -> (format, weights, input_activations). The w4a4 variants quantize
# activations dynamically with the weights' own layout.
VARIANTS = {
    "nvfp4a16": ("nvfp4-pack-quantized", NVFP4_WEIGHTS, None),
    "nvfp4": ("nvfp4-pack-quantized", NVFP4_WEIGHTS, dict(NVFP4_WEIGHTS, dynamic=True)),
    "mxfp4a16": ("mxfp4-pack-quantized", MXFP4_WEIGHTS, None),
    "mxfp4": ("mxfp4-pack-quantized", MXFP4_WEIGHTS, dict(MXFP4_WEIGHTS, dynamic=True)),
    "int4": ("pack-quantized", INT4_WEIGHTS, None),
    "fp8": ("float-quantized", FP8_WEIGHTS, FP8_DYNAMIC_ACT),
}


def _make_config(variant, input_activations=...):
    quant_format, weights, default_act = VARIANTS[variant]
    return {
        "quant_method": "compressed-tensors",
        "format": quant_format,
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": weights,
                "input_activations": (
                    default_act if input_activations is ... else input_activations
                ),
            }
        },
        "ignore": ["lm_head", "re:.*block_sparse_moe.router"],
    }


class FusedMoE(torch.nn.Module):
    """find_matched_target matches on the class name, and a bare `Linear` target
    is aliased onto `FusedMoE`; the name is what makes the MoE lookup resolve."""


def _get_scheme(kind, config_dict, capability):
    """`capability` drives _check_scheme_supported; the native w4a4 schemes read
    get_platform().is_blackwell instead, so both are set."""
    quant_config = CompressedTensorsConfig.from_config(config_dict)
    with (
        override_platform(is_blackwell=capability >= SM100),
        mock.patch("torch.cuda.get_device_capability", return_value=capability),
    ):
        if kind == "linear":
            return quant_config.get_linear_scheme(
                torch.nn.Linear(16, 16), layer_name=LINEAR_LAYER
            )
        return quant_config.get_moe_scheme(FusedMoE(), layer_name=MOE_LAYER)


# (kind, variant, capability, expected scheme or None for "any", expected attrs)
# fmt: off
SELECTION_CASES = [
    # NVFP4 w4a4 keeps its native scheme on Blackwell and is served weight-only
    # before it. A downgraded w4a4 checkpoint still ships input_global_scale,
    # which needs a registered destination.
    ("linear", "nvfp4a16", SM90, CompressedTensorsW4A16Fp4, {"has_input_global_scale": False}),
    ("linear", "nvfp4", SM90, CompressedTensorsW4A16Fp4, {"has_input_global_scale": True}),
    ("linear", "nvfp4a16", SM100, CompressedTensorsW4A16Fp4, {}),
    ("linear", "nvfp4", SM100, CompressedTensorsW4A4Fp4, {}),
    ("moe", "nvfp4a16", SM90, CompressedTensorsW4A16Nvfp4MoE, {"has_input_global_scale": False, "group_size": 16}),
    ("moe", "nvfp4", SM90, CompressedTensorsW4A16Nvfp4MoE, {"has_input_global_scale": True}),
    ("moe", "nvfp4", SM100, CompressedTensorsW4A4Nvfp4MoE, {}),
    # No native MXFP4 linear kernel exists, so both variants are weight-only on
    # every capability.
    ("linear", "mxfp4a16", SM90, CompressedTensorsW4A16Mxfp4, {"has_input_activations": False}),
    ("linear", "mxfp4", SM90, CompressedTensorsW4A16Mxfp4, {"has_input_activations": True}),
    ("linear", "mxfp4", SM100, CompressedTensorsW4A16Mxfp4, {"has_input_activations": True}),
    # Neighbours of the fp4 predicates. The int4 type check keeps fp4 out of
    # WNA16; a weight-only MoE config has no input_activations, which the w8a8
    # predicates used to dereference.
    ("linear", "int4", SM90, CompressedTensorsWNA16, {"group_size": 128}),
    ("moe", "int4", SM90, None, {}),
    ("moe", "fp8", SM90, CompressedTensorsW8A8Fp8MoE, {}),
]
# fmt: on


class TestFp4SchemeSelection(CustomTestCase):
    def test_selects_expected_scheme(self):
        """Each case asserts the exact class, so it also catches one format's
        predicate capturing another's checkpoints."""
        for kind, variant, capability, expected, attrs in SELECTION_CASES:
            with self.subTest(kind=kind, variant=variant, capability=capability):
                scheme = _get_scheme(kind, _make_config(variant), capability)
                if expected is None:
                    self.assertIsNotNone(scheme)
                else:
                    self.assertIsInstance(scheme, expected)
                for name, value in attrs.items():
                    self.assertEqual(getattr(scheme, name), value, name)

    def test_weight_only_downgrade_warns_and_a16_does_not(self):
        """warning_once is lru_cache-wrapped; clear it to stay order-independent."""
        logging.Logger.warning_once.cache_clear()
        self.addCleanup(logging.Logger.warning_once.cache_clear)
        for w4a4, a16 in (("nvfp4", "nvfp4a16"), ("mxfp4", "mxfp4a16")):
            with self.subTest(variant=w4a4):
                with self.assertLogs(SCHEME_LOGGER, level="WARNING") as captured:
                    _get_scheme("linear", _make_config(w4a4), SM90)
                self.assertIn("weight-only", "\n".join(captured.output))
                with mock.patch.object(SCHEME_LOGGER, "warning_once") as warn:
                    _get_scheme("linear", _make_config(a16), SM90)
                warn.assert_not_called()

    def test_fp4_weights_with_non_nvfp4_activations_raise(self):
        """Mismatched activation quantization must not be served weight-only."""
        config = _make_config(
            "nvfp4", input_activations=dict(NVFP4_WEIGHTS, group_size=32, dynamic=True)
        )
        with self.assertRaisesRegex(NotImplementedError, "NVFP4"):
            _get_scheme("linear", config, SM90)


def _noop_loader(*args, **kwargs):
    pass


def _linear_layer(scheme, output_partition_sizes=(512,)):
    layer = torch.nn.Module()
    scheme.create_weights(
        layer=layer,
        output_partition_sizes=list(output_partition_sizes),
        input_size_per_partition=2048,
        params_dtype=torch.bfloat16,
        weight_loader=_noop_loader,
    )
    return layer


def _moe_layer(scheme, is_gated=True):
    layer = torch.nn.Module()
    # FusedMoE sets moe_runner_config before create_weights; w13's shard count
    # comes from it.
    layer.moe_runner_config = MoeRunnerConfig(is_gated=is_gated)
    scheme.create_weights(
        layer=layer,
        num_experts=8,
        hidden_size=2048,
        intermediate_size_per_partition=1536,
        params_dtype=torch.bfloat16,
        weight_loader=_noop_loader,
    )
    return layer


# (name, scheme class, group size, scale dtype, has a weight global scale)
LINEAR_FORMATS = [
    ("nvfp4", CompressedTensorsW4A16Fp4, 16, torch.float8_e4m3fn, True),
    # E8M0 scales arrive as raw uint8 exponent bytes; no outer global scale.
    ("mxfp4", CompressedTensorsW4A16Mxfp4, 32, torch.uint8, False),
]
MOE_FORMATS = [
    ("nvfp4", CompressedTensorsW4A16Nvfp4MoE, 16, torch.float8_e4m3fn, True),
]


class TestWeightOnlyFp4WeightCreation(CustomTestCase):
    """Registered parameters are a contract with the checkpoint's tensor names,
    shapes and dtypes; a mismatch surfaces only as a load-time failure."""

    def test_linear_parameters_match_checkpoint_layout(self):
        for name, cls, group_size, scale_dtype, has_global in LINEAR_FORMATS:
            with self.subTest(fmt=name):
                layer = _linear_layer(cls())
                self.assertEqual(layer.weight_packed.shape, (512, 1024))
                self.assertEqual(layer.weight_packed.dtype, torch.uint8)
                self.assertEqual(layer.weight_scale.shape, (512, 2048 // group_size))
                self.assertEqual(layer.weight_scale.dtype, scale_dtype)
                self.assertEqual(hasattr(layer, "weight_global_scale"), has_global)
                # Read by the Marlin prep helper: a missing quant_config skips its
                # group_size check, a missing params_dtype makes it raise.
                self.assertEqual(layer.params_dtype, torch.bfloat16)
                self.assertEqual(layer.quant_config.group_size, group_size)

    def test_linear_w4a4_input_scale_destination(self):
        """A downgraded NVFP4 w4a4 checkpoint still ships input_global_scale."""
        self.assertFalse(
            hasattr(_linear_layer(CompressedTensorsW4A16Fp4()), "input_global_scale")
        )
        layer = _linear_layer(CompressedTensorsW4A16Fp4(has_input_global_scale=True))
        self.assertEqual(layer.input_global_scale.shape, (1,))
        # MXFP4 folds its outer scale into the E8M0 group scales.
        layer = _linear_layer(CompressedTensorsW4A16Mxfp4(has_input_activations=True))
        self.assertFalse(hasattr(layer, "input_global_scale"))

    def test_mismatched_fused_global_scales_are_rejected(self):
        """Both kernels take one global scale per layer, so fused projections
        (q/k/v) whose scales differ cannot be served without changing weights."""
        scheme = CompressedTensorsW4A16Fp4()
        layer = _linear_layer(scheme, output_partition_sizes=(256, 128, 128))
        layer.weight_global_scale.data.copy_(torch.tensor([2.0, 2.0, 4.0]))
        with self.assertRaisesRegex(ValueError, "share one weight_global_scale"):
            scheme.process_weights_after_loading(layer)

    def test_moe_parameters_match_checkpoint_layout(self):
        for name, cls, group_size, scale_dtype, has_global in MOE_FORMATS:
            with self.subTest(fmt=name):
                layer = _moe_layer(cls())
                # Two fp4 items per byte along the input dimension.
                self.assertEqual(layer.w13_weight_packed.shape, (8, 3072, 1024))
                self.assertEqual(layer.w13_weight_packed.dtype, torch.uint8)
                self.assertEqual(layer.w2_weight_packed.shape, (8, 2048, 768))
                self.assertEqual(
                    layer.w13_weight_scale.shape, (8, 3072, 2048 // group_size)
                )
                self.assertEqual(
                    layer.w2_weight_scale.shape, (8, 2048, 1536 // group_size)
                )
                self.assertEqual(layer.w13_weight_scale.dtype, scale_dtype)
                self.assertEqual(layer.w2_weight_scale.dtype, scale_dtype)
                if has_global:
                    # A gate and an up scale per expert for w13, one for w2.
                    self.assertEqual(layer.w13_weight_global_scale.shape, (8, 2))
                    self.assertEqual(layer.w2_weight_global_scale.shape, (8,))
                else:
                    self.assertFalse(hasattr(layer, "w13_weight_global_scale"))
                    self.assertFalse(hasattr(layer, "w2_weight_global_scale"))

    def test_moe_input_scale_registered_only_for_w4a4(self):
        """The Marlin kernel never reads it, but the split-expert loader raises
        KeyError on a checkpoint tensor with no registered destination."""
        for name, cls, *_ in MOE_FORMATS:
            with self.subTest(fmt=name):
                layer = _moe_layer(cls())
                self.assertFalse(hasattr(layer, "w13_input_global_scale"))
                self.assertFalse(hasattr(layer, "w2_input_global_scale"))
                layer = _moe_layer(cls(has_input_global_scale=True))
                self.assertEqual(layer.w13_input_global_scale.shape, (8, 2))
                self.assertEqual(layer.w2_input_global_scale.shape, (8,))

    def test_non_gated_experts_register_one_w13_shard(self):
        """Non-gated experts have only an up projection; the Marlin repack sizes
        w13 from the same is_gated flag."""
        for name, cls, group_size, _, has_global in MOE_FORMATS:
            with self.subTest(fmt=name):
                layer = _moe_layer(cls(has_input_global_scale=True), is_gated=False)
                self.assertEqual(layer.w13_weight_packed.shape, (8, 1536, 1024))
                self.assertEqual(
                    layer.w13_weight_scale.shape, (8, 1536, 2048 // group_size)
                )
                self.assertEqual(layer.w13_input_global_scale.shape, (8, 1))
                if has_global:
                    self.assertEqual(layer.w13_weight_global_scale.shape, (8, 1))

    def test_mismatched_gate_up_global_scales_are_rejected(self):
        """Marlin takes one NVFP4 w13 global scale per expert, so differing gate
        and up scales must fail at load rather than change the up weights."""
        scheme = CompressedTensorsW4A16Nvfp4MoE()
        layer = _moe_layer(scheme)
        layer.w13_weight_global_scale.data.fill_(2.0)
        layer.w13_weight_global_scale.data[3, 1] = 4.0
        layer.w2_weight_global_scale.data.fill_(2.0)
        with self.assertRaisesRegex(ValueError, r"experts \[3\] differ"):
            scheme.process_weights_after_loading(layer)


class TestWeightOnlyFp4BackendSelection(CustomTestCase):
    """The NVFP4 weight-only linear follows --fp4-gemm-backend; only the cuDNN
    and CuTe-DSL FlashInfer backends implement bf16 x fp4."""

    def _use_flashinfer(
        self, backend, is_blackwell=True, dtype=torch.bfloat16, cudnn_version=92301
    ):
        """cudnn_version=None simulates cuDNN not being installed."""
        from sglang.srt.layers.quantization import fp4_utils
        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_w4a16_nvfp4 as scheme_module,
        )

        cudnn = (
            None
            if cudnn_version is None
            else SimpleNamespace(backend_version=lambda: cudnn_version)
        )
        with (
            mock.patch.object(fp4_utils, "FP4_GEMM_RUNNER_BACKEND", backend),
            override_platform(is_blackwell=is_blackwell),
            mock.patch.dict(sys.modules, {"cudnn": cudnn}),
        ):
            return scheme_module._use_flashinfer_bf16_fp4(dtype)

    def test_backend_choice(self):
        """`auto` is resolved at startup (cutedsl on SM100, marlin before), so an
        unresolved AUTO here means Marlin."""
        B = Fp4GemmRunnerBackend
        cases = [
            (B.AUTO, False),
            (B.MARLIN, False),
            (B.FLASHINFER_CUTEDSL, True),
            (B.FLASHINFER_CUDNN, True),
            # No bf16 x fp4 GEMM: fall back to Marlin.
            (B.FLASHINFER_CUTLASS, False),
            (B.FLASHINFER_TRTLLM, False),
        ]
        for backend, expected in cases:
            with self.subTest(backend=backend.value):
                self.assertEqual(self._use_flashinfer(backend), expected)

    def test_fp16_activations_fall_back_to_marlin(self):
        """mm_bf16_fp4 supports only bf16 activations."""
        self.assertFalse(
            self._use_flashinfer(
                Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL, dtype=torch.float16
            )
        )

    def test_old_or_missing_cudnn_is_rejected_at_load(self):
        """FlashInfer checks the cuDNN version only on the first GEMM, and torch
        may pin a cuDNN older than the 9.23.1 it needs."""
        for version, found in ((92000, "found 92000"), (None, "not installed")):
            with self.subTest(cudnn_version=version):
                with self.assertRaisesRegex(ValueError, found):
                    self._use_flashinfer(
                        Fp4GemmRunnerBackend.FLASHINFER_CUDNN, cudnn_version=version
                    )
        # CuTe-DSL does not use cuDNN.
        self.assertTrue(
            self._use_flashinfer(
                Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL, cudnn_version=None
            )
        )

    def test_flashinfer_backend_before_blackwell_raises(self):
        with self.assertRaisesRegex(ValueError, "requires SM100"):
            self._use_flashinfer(
                Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL, is_blackwell=False
            )


if __name__ == "__main__":
    unittest.main()
