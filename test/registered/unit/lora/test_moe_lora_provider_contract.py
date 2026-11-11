"""Real provider methods test admission, geometry, and finalization with GPU boundaries mocked."""

from __future__ import annotations

import ast
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

ROOT = Path(__file__).resolve().parents[4]
PROVIDER = ROOT / "python/sglang/srt/lora/moe/base_gemm_provider"


def _function(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"function {name!r} not found")


register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNvFp4PreparedQuantInfo(CustomTestCase):
    def test_non_bf16_activations_fail_without_mutation(self):
        from sglang.srt.lora.moe.quant_info import MoeLoraNvFp4MarlinQuantInfo

        # FP16 is the silent case: Marlin accepts half activations while the
        # runner allocates BF16 stage buffers; FP32 fails loudly in the JIT.
        layer = SimpleNamespace(
            params_dtype=torch.float16,
            w13_weight=torch.zeros((2, 4, 512), dtype=torch.int32),
            w2_weight=torch.zeros((2, 8, 128), dtype=torch.int32),
            w13_weight_scale=torch.zeros((2, 4, 256), dtype=torch.float8_e4m3fn),
            w2_weight_scale=torch.zeros((2, 8, 64), dtype=torch.float8_e4m3fn),
            w13_weight_scale_2=torch.ones(2, dtype=torch.float16),
            w2_weight_scale_2=torch.zeros(2, dtype=torch.float16),
            moe_runner_config=SimpleNamespace(is_gated=True),
            num_local_experts=2,
            intermediate_size_per_partition=128,
            hidden_size=64,
        )
        original = vars(layer).copy()
        with self.assertRaisesRegex(NotImplementedError, "BF16 activations"):
            MoeLoraNvFp4MarlinQuantInfo.from_layer(layer)
        for name, value in original.items():
            self.assertIs(getattr(layer, name), value)


class TestMxfp4PreparedQuantInfo(CustomTestCase):
    @staticmethod
    def _layer(**overrides):
        def e8m0(*shape):
            return torch.zeros(shape, dtype=torch.uint8).view(torch.float8_e8m0fnu)

        layer = SimpleNamespace(
            params_dtype=torch.bfloat16,
            w13_weight=torch.zeros((2, 16, 512), dtype=torch.int32),
            w2_weight=torch.zeros((2, 8, 512), dtype=torch.int32),
            w13_weight_scale=e8m0(2, 8, 256),
            w2_weight_scale=e8m0(2, 4, 256),
            moe_runner_config=SimpleNamespace(is_gated=True),
            num_local_experts=2,
            intermediate_size_per_partition=128,
            hidden_size=256,
        )
        vars(layer).update(overrides)
        return layer

    def test_non_bf16_activations_fail_without_mutation(self):
        from sglang.srt.lora.moe.quant_info import MoeLoraMxfp4MarlinQuantInfo

        layer = self._layer(params_dtype=torch.float16)
        original = vars(layer).copy()
        with self.assertRaisesRegex(NotImplementedError, "BF16 activations"):
            MoeLoraMxfp4MarlinQuantInfo.from_layer(layer)
        for name, value in original.items():
            self.assertIs(getattr(layer, name), value)

    def test_binds_logical_sizes_from_prepared_tiles(self):
        from sglang.srt.lora.moe.quant_info import MoeLoraMxfp4MarlinQuantInfo

        quant_info = MoeLoraMxfp4MarlinQuantInfo.from_layer(self._layer())
        self.assertEqual(
            (
                quant_info.hidden_size,
                quant_info.intermediate_size,
                quant_info.gate_up_slices,
            ),
            (256, 128, 2),
        )

    def test_loader_padded_hidden_width_is_rejected(self):
        from sglang.srt.lora.moe.quant_info import MoeLoraMxfp4MarlinQuantInfo

        # The loader rounds the hidden width to 256; stages address real columns.
        with self.assertRaisesRegex(NotImplementedError, "hidden_size % 256"):
            MoeLoraMxfp4MarlinQuantInfo.from_layer(self._layer(hidden_size=200))


class TestFp8PreparedQuantInfo(CustomTestCase):
    """Marlin FP8 tiles bind with the K group size the scale rows imply."""

    @staticmethod
    def _layer(group_size=128, **overrides):
        # Both K extents hold at least two 128-wide groups (the repack's rule).
        hidden, inter = 512, 256

        def groups(k):
            return 1 if group_size == -1 else k // group_size

        layer = SimpleNamespace(
            params_dtype=torch.bfloat16,
            w13_weight=torch.zeros((2, hidden // 16, 2 * inter * 4), dtype=torch.int32),
            w2_weight=torch.zeros((2, inter // 16, hidden * 4), dtype=torch.int32),
            w13_weight_scale=torch.zeros(
                (2, groups(hidden), 2 * inter), dtype=torch.bfloat16
            ),
            w2_weight_scale=torch.zeros(
                (2, groups(inter), hidden), dtype=torch.bfloat16
            ),
            moe_runner_config=SimpleNamespace(is_gated=True),
            num_local_experts=2,
            intermediate_size_per_partition=inter,
            hidden_size=hidden,
        )
        vars(layer).update(overrides)
        return layer

    def test_binds_the_group_size_from_the_scale_rows(self):
        from sglang.srt.lora.moe.quant_info import MoeLoraFp8MarlinQuantInfo

        for group_size in (32, 128):
            with self.subTest(group_size=group_size):
                info = MoeLoraFp8MarlinQuantInfo.from_layer(self._layer(group_size))
                self.assertEqual(info.group_size, group_size)
                self.assertEqual(
                    (info.hidden_size, info.intermediate_size, info.gate_up_slices),
                    (512, 256, 2),
                )

    def test_rejects_layers_the_repack_did_not_prepare(self):
        from sglang.srt.lora.moe.quant_info import MoeLoraFp8MarlinQuantInfo

        cases = {
            # The FP8 repack pads no widths; a padded tile count is another vendor's.
            "padded_tiles": (
                dict(w13_weight=torch.zeros((2, 17, 2048), dtype=torch.int32)),
                ValueError,
                "keep the layer's hidden",
            ),
            # The MoE kernel has no FP8 instantiation for 64-wide groups.
            "group_64": (dict(group_size=64), NotImplementedError, "32- or 128-wide"),
            # One group over K is the kernel's channel-wise path, which scales
            # after the K accumulation; the repack never hands it over.
            "single_group": (
                dict(group_size=-1),
                NotImplementedError,
                "at least two K groups",
            ),
            "fp32_scales": (
                dict(w13_weight_scale=torch.zeros((2, 4, 512), dtype=torch.float32)),
                ValueError,
                "activation dtype",
            ),
            "stages_disagree": (
                dict(w2_weight_scale=torch.zeros((2, 4, 512), dtype=torch.bfloat16)),
                ValueError,
                "share one K group size",
            ),
            "fp16_activations": (
                dict(params_dtype=torch.float16),
                NotImplementedError,
                "BF16 activations",
            ),
        }
        for name, (overrides, error, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(error, message):
                MoeLoraFp8MarlinQuantInfo.from_layer(self._layer(**overrides))


class TestFp8MarlinFamilyBinding(CustomTestCase):
    """Under marlin / lora_marlin the FP8 methods repack their experts, so the
    runner binds the family from Marlin tiles instead of e4m3 weights."""

    @staticmethod
    def _fp8_method(backend, **config):
        from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
        from sglang.srt.runtime_context import get_flags

        with get_flags().moe.override(runner_backend=backend):
            return Fp8MoEMethod(Fp8Config(is_checkpoint_fp8_serialized=True, **config))

    def test_fp8_method_binds_marlin_tiles_under_lora_marlin(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.lora.moe.families import family_of_layer

        tiles = torch.zeros((2, 16, 1024), dtype=torch.int32)
        for config, family in (
            (dict(weight_block_size=[128, 128]), "fp8"),
            (dict(use_mxfp8=True), "mxfp8"),
        ):
            with self.subTest(family):
                method = self._fp8_method(MoeRunnerBackend.LORA_MARLIN, **config)
                self.assertTrue(method.use_marlin)
                layer = SimpleNamespace(
                    quant_method=method, params_dtype=torch.bfloat16, w13_weight=tiles
                )
                self.assertEqual(family_of_layer(layer), family)
        # Without Marlin the fp8 family still needs the e4m3 weights resident.
        method = self._fp8_method(MoeRunnerBackend.AUTO, weight_block_size=[128, 128])
        self.assertFalse(method.use_marlin)
        layer = SimpleNamespace(
            quant_method=method, params_dtype=torch.bfloat16, w13_weight=tiles
        )
        with self.assertRaisesRegex(NotImplementedError, "float8_e4m3fn"):
            family_of_layer(layer)

    def test_fp8_method_per_tensor_experts_have_no_marlin_path(self):
        """The fp8 quant method's per-tensor experts (no weight blocks) stay
        outside Marlin under both backends; the refusal names the layout."""
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        for backend in (MoeRunnerBackend.MARLIN, MoeRunnerBackend.LORA_MARLIN):
            with self.subTest(backend.value):
                with self.assertRaisesRegex(NotImplementedError, "per-tensor"):
                    self._fp8_method(backend)

    def test_compressed_tensors_schemes_repack_under_marlin_backends(self):
        """Per-channel and per-tensor compressed-tensors experts, with static or
        dynamic activation metadata, repack under marlin / lora_marlin and under
        nothing else; the sglang w8a8_fp8 method follows the same rule."""
        from compressed_tensors.quantization import QuantizationArgs

        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
            CompressedTensorsW8A8Fp8MoE,
        )
        from sglang.srt.layers.quantization.w8a8_fp8 import W8A8FP8MoEMethod
        from sglang.srt.runtime_context import get_flags

        layouts = {
            "channel": (dict(strategy="channel"), dict(strategy="token", dynamic=True)),
            "tensor_dynamic": (
                dict(strategy="tensor"),
                dict(strategy="tensor", dynamic=True),
            ),
            "tensor_static": (
                dict(strategy="tensor"),
                dict(strategy="tensor", dynamic=False),
            ),
        }
        for backend, expected in (
            (MoeRunnerBackend.LORA_MARLIN, True),
            (MoeRunnerBackend.MARLIN, True),
            (MoeRunnerBackend.LORA_CUTEDSL, False),
            (MoeRunnerBackend.AUTO, False),
        ):
            with get_flags().moe.override(runner_backend=backend):
                for name, (weights, activations) in layouts.items():
                    with self.subTest(backend.value, layout=name):
                        scheme = CompressedTensorsW8A8Fp8MoE(
                            QuantizationArgs(num_bits=8, type="float", **weights),
                            QuantizationArgs(num_bits=8, type="float", **activations),
                        )
                        self.assertEqual(scheme.use_marlin, expected)
                with self.subTest(backend.value, layout="w8a8_fp8"):
                    self.assertEqual(W8A8FP8MoEMethod(None).use_marlin, expected)


class TestMxfp4ScaleStorage(CustomTestCase):
    """Under a lora_* backend the FP8 method stores FP4-expert scales as E8M0 bytes."""

    def _scales(self, backend):
        from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
        from sglang.srt.runtime_context import get_flags

        layer = torch.nn.Module()
        layer.moe_tp_size = 1
        layer.moe_runner_config = SimpleNamespace(is_gated=True)
        with get_flags().moe.override(runner_backend=backend):
            Fp8MoEMethod(
                Fp8Config(
                    is_checkpoint_fp8_serialized=True,
                    weight_block_size=[128, 128],
                    is_fp4_experts=True,
                )
            ).create_weights(
                layer=layer,
                num_experts=2,
                hidden_size=256,
                intermediate_size_per_partition=128,
                params_dtype=torch.bfloat16,
            )
        return layer.w13_weight_scale_inv, layer.w2_weight_scale_inv

    def test_lora_backends_store_e8m0_scales(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        for backend, dtype in (
            (MoeRunnerBackend.LORA_CUTEDSL, torch.float8_e8m0fnu),
            (MoeRunnerBackend.AUTO, torch.float32),
        ):
            for scale in self._scales(backend):
                self.assertEqual(scale.dtype, dtype, backend)


class TestCompressedTensorsFp8Admission(CustomTestCase):
    """compressed-tensors FP8 experts bind through the scheme the layer carries;
    the fp8 families read the scales the scheme registers."""

    _BLOCK = dict(strategy="block", block_structure=[128, 128])
    _GROUP_ACT = dict(strategy="group", group_size=128, dynamic=True)
    _CHANNEL = dict(strategy="channel")
    _TOKEN_ACT = dict(strategy="token", dynamic=True)

    @staticmethod
    def _layer(weights, activations, params_dtype=torch.bfloat16):
        from compressed_tensors.quantization import QuantizationArgs

        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsFusedMoEMethod,
        )
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
            CompressedTensorsW8A8Fp8MoE,
        )

        scheme = CompressedTensorsW8A8Fp8MoE(
            QuantizationArgs(num_bits=8, type="float", **weights),
            QuantizationArgs(num_bits=8, type="float", **activations),
        )
        layer = torch.nn.Module()
        layer.quant_method = object.__new__(CompressedTensorsFusedMoEMethod)
        layer.scheme = scheme
        layer.moe_runner_config = SimpleNamespace(is_gated=True)
        layer.moe_tp_size = 1
        layer.params_dtype = params_dtype
        layer.num_local_experts = 2
        scheme.create_weights(
            layer=layer,
            num_experts=2,
            hidden_size=256,
            intermediate_size_per_partition=128,
            params_dtype=params_dtype,
        )
        return layer

    def test_block_scheme_binds_as_the_fp8_family(self):
        from sglang.srt.lora.moe.families import family_of_layer
        from sglang.srt.lora.moe.quant_info import MoeLoraFp8QuantInfo

        layer = self._layer(self._BLOCK, self._GROUP_ACT)
        self.assertEqual(family_of_layer(layer), "fp8")
        info = MoeLoraFp8QuantInfo.from_layer(layer)
        self.assertIs(info.w13_scale, layer.w13_weight_scale)
        self.assertIs(info.w2_scale, layer.w2_weight_scale)
        self.assertEqual(info.block_shape, (128, 128))
        self.assertEqual(tuple(info.w13_scale.shape), (2, 2, 2))
        self.assertEqual(tuple(info.w2_scale.shape), (2, 2, 1))
        self.assertEqual((info.hidden_size, info.intermediate_size), (256, 128))

    def test_channel_scheme_binds_as_the_fp8_channel_family(self):
        from sglang.srt.lora.moe.families import family_of_layer
        from sglang.srt.lora.moe.quant_info import MoeLoraFp8ChannelQuantInfo

        layer = self._layer(self._CHANNEL, self._TOKEN_ACT)
        self.assertEqual(family_of_layer(layer), "fp8_channel")
        info = MoeLoraFp8ChannelQuantInfo.from_layer(layer)
        self.assertIs(info.w13_scale, layer.w13_weight_scale)
        self.assertIs(info.w2_scale, layer.w2_weight_scale)
        self.assertEqual(tuple(info.w13_scale.shape), (2, 256, 1))
        self.assertEqual(tuple(info.w2_scale.shape), (2, 256, 1))
        self.assertEqual((info.hidden_size, info.intermediate_size), (256, 128))

    def test_w8a8_fp8_method_binds_as_the_fp8_channel_family(self):
        from sglang.srt.layers.quantization.w8a8_fp8 import W8A8FP8MoEMethod
        from sglang.srt.lora.moe.families import family_of_layer
        from sglang.srt.lora.moe.quant_info import MoeLoraFp8ChannelQuantInfo

        layer = torch.nn.Module()
        layer.quant_method = object.__new__(W8A8FP8MoEMethod)
        layer.params_dtype = torch.bfloat16
        layer.num_local_experts = 2
        layer.quant_method.create_weights(
            layer=layer,
            num_experts=2,
            hidden_size=256,
            intermediate_size_per_partition=128,
            params_dtype=torch.bfloat16,
        )
        self.assertEqual(family_of_layer(layer), "fp8_channel")
        info = MoeLoraFp8ChannelQuantInfo.from_layer(layer)
        self.assertIs(info.w13_scale, layer.w13_weight_scale)
        self.assertEqual(tuple(info.w2_scale.shape), (2, 256, 1))
        layer.params_dtype = torch.float16
        with self.assertRaisesRegex(NotImplementedError, "BF16 activations"):
            family_of_layer(layer)

    def test_tensor_scheme_binds_as_the_fp8_tensor_family(self):
        from sglang.srt.lora.moe.families import family_of_layer
        from sglang.srt.lora.moe.quant_info import MoeLoraFp8TensorQuantInfo

        for dynamic in (False, True):
            with self.subTest(dynamic_activations=dynamic):
                layer = self._layer(
                    dict(strategy="tensor"), dict(strategy="tensor", dynamic=dynamic)
                )
                self.assertEqual(family_of_layer(layer), "fp8_tensor")
                # The runner binds after process_weights_after_loading folded the
                # gate/up scales and the input scales to one value each.
                layer.w13_weight_scale = torch.nn.Parameter(
                    layer.w13_weight_scale.max(dim=1).values, requires_grad=False
                )
                if not dynamic:
                    layer.w13_input_scale = torch.nn.Parameter(
                        layer.w13_input_scale.max(), requires_grad=False
                    )
                    layer.w2_input_scale = torch.nn.Parameter(
                        layer.w2_input_scale.max(), requires_grad=False
                    )
                info = MoeLoraFp8TensorQuantInfo.from_layer(layer)
                self.assertEqual(tuple(info.w13_scale.shape), (2,))
                self.assertIs(info.w2_scale, layer.w2_weight_scale)
                self.assertEqual(
                    (info.w13_input_scale is None, info.w2_input_scale is None),
                    (dynamic, dynamic),
                )

    def test_other_layouts_and_schemes_are_refused(self):
        from sglang.srt.lora.moe.families import family_of_layer
        from sglang.srt.lora.moe.quant_info import (
            MoeLoraFp8QuantInfo,
            MoeLoraFp8TensorQuantInfo,
        )

        with self.assertRaisesRegex(NotImplementedError, r"\[128, 128\]"):
            MoeLoraFp8QuantInfo.from_layer(
                self._layer(
                    dict(strategy="block", block_structure=[64, 64]), self._GROUP_ACT
                )
            )
        # Unfolded gate/up scales mean the weights were not processed yet.
        with self.assertRaisesRegex(ValueError, r"w13_scale must be fp32 \[2\]"):
            MoeLoraFp8TensorQuantInfo.from_layer(
                self._layer(
                    dict(strategy="tensor"), dict(strategy="tensor", dynamic=True)
                )
            )
        cases = {
            "fp16_activations": (
                self._BLOCK,
                self._GROUP_ACT,
                "BF16 activations",
                torch.float16,
            ),
        }
        for name, (weights, activations, message, *dtype) in cases.items():
            with (
                self.subTest(name),
                self.assertRaisesRegex(NotImplementedError, message),
            ):
                family_of_layer(self._layer(weights, activations, *dtype))
        layer = self._layer(self._BLOCK, self._GROUP_ACT)
        layer.scheme = object()
        with self.assertRaisesRegex(NotImplementedError, "FP8 scheme only"):
            family_of_layer(layer)


class TestVendorResolution(CustomTestCase):
    def test_vendor_resolution_keeps_the_weight_layout(self):
        """Every quantized family defaults to Marlin. A runner whose vendor has
        no provider for a family is served only by a vendor that reads the
        layout the loader left for that runner, never by Marlin: only
        lora_marlin prepares its tiles."""
        from sglang.srt.lora.moe.families import (
            _SAME_LAYOUT_VENDOR,
            VENDORS,
            default_vendor,
            resolve_vendor,
        )

        self.assertEqual(default_vendor("bf16"), "cutedsl")
        for family in ("fp8", "fp8_channel", "fp8_tensor", "mxfp8", "nvfp4", "mxfp4"):
            self.assertEqual(default_vendor(family), "marlin")
            self.assertEqual(resolve_vendor(family, "marlin"), "marlin")
        # An explicit vendor with a provider serves as asked.
        for family, vendor in (
            ("fp8", "cutedsl"),
            ("fp8", "triton"),
            ("fp8_channel", "cutedsl"),
            ("fp8_channel", "triton"),
            ("fp8_tensor", "triton"),
            ("mxfp8", "cutedsl"),
            ("mxfp4", "cutedsl"),
            ("nvfp4", "cutedsl"),
        ):
            self.assertEqual(resolve_vendor(family, vendor), vendor)
        # A vendor without a provider: the vendor reading the same layout.
        self.assertEqual(resolve_vendor("bf16", "marlin"), "cutedsl")
        self.assertEqual(resolve_vendor("fp8_tensor", "cutedsl"), "triton")
        self.assertEqual(resolve_vendor("mxfp8", "triton"), "cutedsl")
        for family, fallback in _SAME_LAYOUT_VENDOR.items():
            self.assertNotEqual(fallback, "marlin")
            self.assertIn(fallback, VENDORS[family])
        for family in ("mxfp4", "nvfp4"):
            with self.assertRaisesRegex(
                NotImplementedError, "lora_triton; use lora_marlin or lora_cutedsl$"
            ):
                resolve_vendor(family, "triton")

    def test_explicit_w8a8_runners_keep_their_providers(self):
        """The Marlin default leaves the W8A8 providers in place: an explicit
        lora_cutedsl or lora_triton runner binds the provider it named, or the
        one that reads the same checkpoint layout, for either row order."""
        from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
        from sglang.srt.lora.moe.families import resolve_vendor

        fp8 = ("CuteDslFp8MaskedProvider", "CuteDslFp8ContiguousProvider")
        channel = (
            "CuteDslFp8ChannelMaskedProvider",
            "CuteDslFp8ChannelContiguousProvider",
        )
        mxfp8 = ("CuteDslMxfp8MaskedProvider", "CuteDslMxfp8ContiguousProvider")
        cases = {
            ("fp8", "cutedsl"): fp8,
            ("fp8", "triton"): ("TritonFp8ContiguousProvider",) * 2,
            ("fp8_channel", "cutedsl"): channel,
            ("fp8_channel", "triton"): ("TritonFp8ChannelContiguousProvider",) * 2,
            ("fp8_tensor", "triton"): ("TritonFp8TensorContiguousProvider",) * 2,
            # No CuTeDSL per-tensor provider: Triton reads the same layout.
            ("fp8_tensor", "cutedsl"): ("TritonFp8TensorContiguousProvider",) * 2,
            ("mxfp8", "cutedsl"): mxfp8,
            # No Triton MXFP8 provider: CuTeDSL reads the same layout.
            ("mxfp8", "triton"): mxfp8,
        }
        for (family, runner_vendor), (masked, contiguous) in cases.items():
            with self.subTest(family=family, runner=f"lora_{runner_vendor}"):
                vendor = resolve_vendor(family, runner_vendor)
                self.assertNotEqual(vendor, "marlin")
                self.assertEqual(
                    select_provider_cls("expert_major", family, vendor).__name__,
                    masked,
                )
                self.assertEqual(
                    select_provider_cls("route_major", family, vendor).__name__,
                    contiguous,
                )

    def test_lora_runners_never_fold_routed_scaling(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            _fuses_routed_scaling_factor_in_topk,
        )
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.layers.quantization.mxfp4_marlin_moe import (
            Mxfp4MarlinMoEMethod,
        )

        # The method folds for its fused runner; the LoRA finalize owns it.
        method = object.__new__(Mxfp4MarlinMoEMethod)
        self.assertTrue(
            _fuses_routed_scaling_factor_in_topk(method, MoeRunnerBackend.MARLIN)
        )
        for backend in (
            MoeRunnerBackend.LORA_MARLIN,
            MoeRunnerBackend.LORA_CUTEDSL,
            MoeRunnerBackend.LORA_TRITON,
        ):
            self.assertFalse(_fuses_routed_scaling_factor_in_topk(method, backend))


class TestFamilyRecognizersAgree(CustomTestCase):
    """family_of_config is provisional, family_of_layer is authoritative. They
    name the same family wherever the config settles the experts' layout; the
    two ways they may differ are pinned by the second test."""

    @staticmethod
    def _config(quantization=None, is_fp4_experts=False, **hf_quant_config):
        return SimpleNamespace(
            quantization=quantization,
            is_fp4_experts=is_fp4_experts,
            hf_quant_config=hf_quant_config,
        )

    @staticmethod
    def _compressed_tensors(*groups):
        return {
            f"group_{index}": {
                "weights": {"type": "float", "num_bits": 8, **weights},
                "input_activations": {"type": "float", "num_bits": 8, **activations},
            }
            for index, (weights, activations) in enumerate(groups)
        }

    def _cases(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptNvFp4FusedMoEMethod,
        )
        from sglang.srt.layers.quantization.mxfp4_marlin_moe import (
            Mxfp4MarlinMoEMethod,
        )
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
        from sglang.srt.layers.quantization.w8a8_fp8 import W8A8FP8MoEMethod

        bf16 = torch.zeros(1, dtype=torch.bfloat16)
        e4m3 = torch.zeros(1, dtype=torch.float8_e4m3fn)
        fp8_method = TestFp8MarlinFamilyBinding._fp8_method
        ct_layer = TestCompressedTensorsFp8Admission._layer
        block = (
            dict(strategy="block", block_structure=[128, 128]),
            dict(strategy="group", group_size=128, dynamic=True),
        )
        channel = (dict(strategy="channel"), dict(strategy="token", dynamic=True))
        tensor = (dict(strategy="tensor"), dict(strategy="tensor", dynamic=False))
        layer = lambda method, **attrs: SimpleNamespace(
            quant_method=method, params_dtype=torch.bfloat16, **attrs
        )
        return {
            "bf16": (
                self._config(),
                layer(
                    object.__new__(UnquantizedFusedMoEMethod),
                    w13_weight=bf16,
                    w2_weight=bf16,
                ),
            ),
            "fp8": (
                self._config("fp8", weight_block_size=[128, 128]),
                layer(
                    fp8_method(MoeRunnerBackend.AUTO, weight_block_size=[128, 128]),
                    w13_weight=e4m3,
                ),
            ),
            "mxfp8": (
                self._config("mxfp8"),
                layer(fp8_method(MoeRunnerBackend.AUTO, use_mxfp8=True)),
            ),
            "mxfp4": (
                self._config("fp8", is_fp4_experts=True),
                layer(object.__new__(Mxfp4MarlinMoEMethod)),
            ),
            "nvfp4": (
                self._config("modelopt_fp4"),
                layer(object.__new__(ModelOptNvFp4FusedMoEMethod)),
            ),
            "fp8_channel": (
                self._config("w8a8_fp8"),
                layer(object.__new__(W8A8FP8MoEMethod)),
            ),
            "fp8 (compressed-tensors block)": (
                self._config(
                    "compressed-tensors", config_groups=self._compressed_tensors(block)
                ),
                ct_layer(*block),
            ),
            "fp8_channel (compressed-tensors channel)": (
                self._config(
                    "compressed_tensors",
                    config_groups=self._compressed_tensors(channel),
                ),
                ct_layer(*channel),
            ),
            "fp8_tensor (compressed-tensors tensor)": (
                self._config(
                    "compressed-tensors", config_groups=self._compressed_tensors(tensor)
                ),
                ct_layer(*tensor),
            ),
        }

    def test_a_config_that_settles_the_layout_names_the_layers_family(self):
        from sglang.srt.lora.moe.families import (
            CONFIG_QUANTIZATIONS,
            VENDORS,
            family_of_config,
            family_of_layer,
        )

        cases = self._cases()
        for name, (config, layer) in cases.items():
            family = name.split(" ")[0]
            with self.subTest(name):
                self.assertEqual(family_of_config(config), family)
                self.assertEqual(family_of_layer(layer), family)
        # Every family and every accepted --quantization has a case above.
        self.assertEqual({name.split(" ")[0] for name in cases}, set(VENDORS))
        self.assertEqual(
            {config.quantization for config, _ in cases.values()},
            set(CONFIG_QUANTIZATIONS),
        )

    def test_the_config_may_defer_or_be_overruled_by_the_layer(self):
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
        from sglang.srt.lora.moe.families import family_of_config, family_of_layer

        block = (
            dict(strategy="block", block_structure=[128, 128]),
            dict(strategy="group", group_size=128, dynamic=True),
        )
        channel = (dict(strategy="channel"), dict(strategy="token", dynamic=True))
        # Groups that disagree: the config defers, each layer still has a family.
        mixed = self._config(
            "compressed-tensors", config_groups=self._compressed_tensors(block, channel)
        )
        self.assertIsNone(family_of_config(mixed))
        self.assertEqual(
            family_of_layer(TestCompressedTensorsFp8Admission._layer(*channel)),
            "fp8_channel",
        )
        # A quantization the engine has no family for: the config defers.
        self.assertIsNone(family_of_config(self._config("awq")))
        # An FP8 config that excludes the experts: the config says fp8, the
        # resident BF16 layer decides.
        excluded = self._config(
            "fp8", weight_block_size=[128, 128], ignored_layers=["mlp.experts"]
        )
        self.assertEqual(family_of_config(excluded), "fp8")
        bf16 = torch.zeros(1, dtype=torch.bfloat16)
        layer = SimpleNamespace(
            quant_method=object.__new__(UnquantizedFusedMoEMethod),
            w13_weight=bf16,
            w2_weight=bf16,
        )
        self.assertEqual(family_of_layer(layer), "bf16")


class TestProviderGeometry(CustomTestCase):
    def test_triton_fp8_k_tiles_stay_within_scale_groups(self):
        from sglang.srt.lora.moe.base_gemm_provider.triton_bf16 import (
            TritonBf16ContiguousProvider,
        )

        # The FP8 provider shares the method; the stand-in below carries its
        # block shape and dtype tag.
        config_for = TritonBf16ContiguousProvider._config_for
        module = types.ModuleType(
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config"
        )
        # The FP8 GEMM loads one scale per K block: a 256-wide block over
        # [128, 128] scale groups would apply the first group's scale to its
        # second half, and upstream never clamps override or table configs.
        original = {"BLOCK_SIZE_K": 256, "USE_TMA": True}
        upstream = (dict(original), dict(original))
        module.try_get_optimal_moe_config = mock.Mock(
            return_value=(upstream[0], (upstream[1], None))
        )
        provider = SimpleNamespace(
            quant_info=SimpleNamespace(
                w13_weight=SimpleNamespace(shape=(2, 256, 128)),
                w2_weight=SimpleNamespace(shape=(2, 128, 128)),
            ),
            _config_dtype_tag="fp8_w8a8",
            _config_block_shape=[128, 128],
            _config_per_channel_quant=False,
            _configs={},
        )
        with mock.patch.dict(sys.modules, {module.__name__: module}):
            resolved = config_for(provider, 32, 2)
            self.assertIs(config_for(provider, 32, 2), resolved)
        for actual, cached in zip(resolved, upstream):
            self.assertEqual(actual, {"BLOCK_SIZE_K": 128})
            self.assertEqual(cached, original)
            self.assertIsNot(actual, cached)
        module.try_get_optimal_moe_config.assert_called_once()


class TestDerivedBaseWeights(CustomTestCase):
    def test_runner_names_what_its_providers_derive(self):
        from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProvider

        self.assertIsNone(MoeBaseProvider.derived_base_weights)
        source = _function(
            (ROOT / "python/sglang/srt/lora/moe/runner.py").read_text(),
            "derived_base_weights",
        )
        namespace = {}
        exec("from __future__ import annotations\n" + source, namespace)
        derived = namespace["derived_base_weights"]

        def runner(*reasons):
            return SimpleNamespace(
                providers={
                    str(i): SimpleNamespace(derived_base_weights=reason)
                    for i, reason in enumerate(reasons)
                }
            )

        self.assertIsNone(derived(runner(None)))
        self.assertEqual(derived(runner("atoms", None, "atoms")), "atoms")
        self.assertEqual(derived(runner("atoms", "tiles")), "atoms and tiles")

    def test_prepared_weight_providers_declare_derived_base_weights(self):
        # Packed scale atoms, NVFP4 alpha products, and Marlin repacks are not
        # refreshed by an in-place base-weight update; BF16 and FP8 bind views.
        for name in ("cutedsl_blockscaled.py", "marlin_common.py"):
            self.assertIn(
                "derived_base_weights = ", (PROVIDER / name).read_text(), name
            )
        for name in ("cutedsl_bf16.py", "cutedsl_fp8.py", "triton_bf16.py"):
            self.assertNotIn(
                "derived_base_weights", (PROVIDER / name).read_text(), name
            )


class TestLayerScaleStorage(CustomTestCase):
    """A layer's block-scaled providers bind one quant info and the tensors it
    derives; another layer binds its own; the checkpoint-form tensors stay."""

    @staticmethod
    def _nvfp4_layer(mode="static", seed=0):
        g = torch.Generator().manual_seed(seed)
        experts, inter, hidden = 2, 128, 128

        def block_scales(rows, k):
            return torch.rand((experts, rows, k // 16), generator=g).to(
                torch.float8_e4m3fn
            )

        def codes(rows, k):
            return torch.randint(
                0, 256, (experts, rows, k // 2), generator=g, dtype=torch.uint8
            )

        return SimpleNamespace(
            params_dtype=torch.bfloat16,
            w13_weight=codes(2 * inter, hidden),
            w2_weight=codes(hidden, inter),
            w13_weight_scale=block_scales(2 * inter, hidden),
            w2_weight_scale=block_scales(hidden, inter),
            w13_weight_scale_2=torch.rand((experts, 2), generator=g) + 0.5,
            w2_weight_scale_2=torch.rand(experts, generator=g) + 0.5,
            w13_input_scale=torch.tensor([0.5]),
            w2_input_scale=torch.tensor([0.25]),
            quant_method=SimpleNamespace(
                quant_config=SimpleNamespace(
                    use_per_token_activation=mode == "per_token"
                )
            ),
            moe_runner_config=SimpleNamespace(is_gated=True),
            num_local_experts=experts,
            intermediate_size_per_partition=inter,
            hidden_size=hidden,
        )

    @staticmethod
    def _packing():
        """The real host-side packing without importing its Triton module,
        and the scales it was asked to pack."""
        source = _function(
            (
                ROOT / "python/sglang/kernels/ops/lora/moe/quant_blockscaled.py"
            ).read_text(),
            "pack_scale_atoms",
        )
        namespace = {"torch": torch}
        exec("from __future__ import annotations\n" + source, namespace)
        packed = []

        def pack_scale_atoms(scales):
            packed.append(scales)
            return namespace["pack_scale_atoms"](scales)

        module = types.ModuleType("sglang.kernels.ops.lora.moe.quant_blockscaled")
        module.pack_scale_atoms = pack_scale_atoms
        return mock.patch.dict(sys.modules, {module.__name__: module}), packed

    def test_a_layers_providers_bind_one_quant_info(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.lora.moe import runner as runner_module
        from sglang.srt.lora.moe.quant_info import MoeLoraNvFp4QuantInfo
        from sglang.srt.lora.moe.runner import MoeLoraRunner
        from sglang.srt.runtime_context import get_context, get_flags

        built = {}

        def provider_cls(rows, family, vendor):
            # Binds what the block-scaled providers bind in _bind_weights.
            class Provider:
                contract = SimpleNamespace(
                    quant_info_cls=MoeLoraNvFp4QuantInfo,
                    lora_delta_dtype=torch.bfloat16,
                )
                hidden_size = intermediate_size = 128
                num_local_experts = 2
                gate_up_slices = 1  # refused after the providers are built

                def __init__(self, quant_info):
                    self.quant_info = quant_info
                    self.sf_w13, self.sf_w2 = quant_info.packed_scales
                    self.alpha = quant_info.alpha
                    built[rows] = self

            return Provider

        def layer_providers(layer):
            """The providers from_layer builds for the layer, by row layout.
            from_layer admits CUDA layers only, so it is handed the layer's
            attributes with the down weight's device reported as CUDA; the
            quant info is built from the layer itself."""
            view = SimpleNamespace(**vars(layer), source=layer)
            view.w2_weight = SimpleNamespace(device=torch.device("cuda"))
            view.moe_runner_config = SimpleNamespace(
                activation="silu", top_k=2, routed_scaling_factor=1.0, is_gated=True
            )
            built.clear()
            with self.assertRaisesRegex(ValueError, "gate/up slices"):
                MoeLoraRunner.from_layer(
                    view, workspace=object(), is_shared_outer=False, physical_rank=64
                )
            return built["expert_major"], built["route_major"]

        patch_packing, packed = self._packing()
        layer = self._nvfp4_layer()
        quant_info_of = MoeLoraNvFp4QuantInfo.from_layer
        with (
            patch_packing,
            mock.patch.object(MoeLoraRunner, "_admit", return_value="nvfp4"),
            mock.patch.object(runner_module, "select_provider_cls", provider_cls),
            mock.patch.object(
                MoeLoraNvFp4QuantInfo,
                "from_layer",
                side_effect=lambda view: quant_info_of(view.source),
            ) as from_layer,
            mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(10, 0)
            ),
            get_context().override_server_args(),
            get_flags().moe.override(runner_backend=MoeRunnerBackend.LORA_CUTEDSL),
        ):
            masked, contiguous = layer_providers(layer)
            self.assertEqual(from_layer.call_count, 1)
            self.assertIs(masked.quant_info, contiguous.quant_info)
            # Derived once for the layer, whichever provider asks first or again.
            self.assertIs(masked.sf_w13, contiguous.sf_w13)
            self.assertIs(masked.sf_w2, contiguous.sf_w2)
            for mine, other in zip(masked.alpha, contiguous.alpha):
                self.assertIs(mine, other)
            self.assertEqual(len(packed), 2)
            self.assertIs(packed[0], masked.quant_info.w13_scale)
            self.assertIs(packed[1], masked.quant_info.w2_scale)

            # A second layer, even with the same contents, binds its own.
            other, _ = layer_providers(self._nvfp4_layer())
            self.assertEqual(from_layer.call_count, 2)
            self.assertIsNot(other.quant_info, masked.quant_info)
            for mine, theirs in (
                (other.sf_w13, masked.sf_w13),
                (other.sf_w2, masked.sf_w2),
                (other.alpha[0], masked.alpha[0]),
                (other.alpha[1], masked.alpha[1]),
            ):
                self.assertTrue(torch.equal(mine, theirs))
                self.assertNotEqual(mine.data_ptr(), theirs.data_ptr())
            # Writing one layer's derived copy leaves the other's alone.
            reference = masked.sf_w13.clone()
            other.sf_w13.fill_(0)
            self.assertTrue(torch.equal(masked.sf_w13, reference))

    def test_from_layer_gives_each_layer_its_own_quant_infos(self):
        from dataclasses import replace

        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.lora.moe import runner as runner_module
        from sglang.srt.lora.moe.plan import resolve_plans
        from sglang.srt.lora.moe.runner import MoeLoraRunner
        from sglang.srt.runtime_context import get_context, get_flags

        layer = SimpleNamespace(
            w2_weight=SimpleNamespace(
                device=torch.device("cuda"), shape=(256, 2048, 768)
            ),
            num_local_experts=256,
            hidden_size=2048,
            moe_runner_config=SimpleNamespace(
                activation="silu", top_k=8, routed_scaling_factor=1.0, is_gated=True
            ),
        )

        def on_one_row_layout(**kwargs):
            return {
                phase: replace(selected, base_gemm_rows="route_major")
                for phase, selected in resolve_plans(**kwargs).items()
            }

        cases = {
            # One quant-info class for both row layouts: the provider tables' case.
            "one_class": (True, resolve_plans),
            # One class per row layout: the layer's cache is keyed by class.
            "class_per_row_layout": (False, resolve_plans),
            # Both phases on one row layout: one provider serves both.
            "one_row_layout": (True, on_one_row_layout),
        }
        for name, (shared, plans) in cases.items():
            with self.subTest(name):
                built = []  # (row layout, quant-info class, quant info) per provider
                classes = {}

                def quant_info_cls(key):
                    class QuantInfo:
                        made = []

                        @classmethod
                        def from_layer(cls, base_layer):
                            cls.made.append(base_layer)
                            return object()

                    return classes.setdefault(key, QuantInfo)

                def provider_cls(rows, family, vendor):
                    info_cls = quant_info_cls("both" if shared else rows)

                    class Provider:
                        contract = SimpleNamespace(
                            quant_info_cls=info_cls, lora_delta_dtype=torch.bfloat16
                        )
                        hidden_size = 2048
                        intermediate_size = 768
                        num_local_experts = 256
                        gate_up_slices = 1  # refused after the providers are built

                        def __init__(self, quant_info):
                            built.append((rows, info_cls, quant_info))

                    return Provider

                layers = []
                for _ in range(2):
                    start = len(built)
                    with (
                        mock.patch.object(MoeLoraRunner, "_admit", return_value="bf16"),
                        mock.patch.object(
                            runner_module, "select_provider_cls", provider_cls
                        ),
                        mock.patch.object(runner_module, "resolve_plans", plans),
                        mock.patch.object(
                            torch.cuda, "get_device_capability", return_value=(9, 0)
                        ),
                        get_context().override_server_args(),
                        get_flags().moe.override(
                            runner_backend=MoeRunnerBackend.LORA_CUTEDSL
                        ),
                        self.assertRaisesRegex(ValueError, "gate/up slices"),
                    ):
                        MoeLoraRunner.from_layer(
                            layer,
                            workspace=object(),
                            is_shared_outer=False,
                            physical_rank=64,
                        )
                    layers.append(built[start:])
                first, second = layers
                row_layouts = [rows for rows, _, _ in first]
                # One provider per row layout the plans use, in the plans' order.
                self.assertEqual(
                    row_layouts,
                    ["route_major"]
                    if plans is on_one_row_layout
                    else ["expert_major", "route_major"],
                )
                self.assertEqual([rows for rows, _, _ in second], row_layouts)
                self.assertEqual(len(classes), 1 if shared else len(row_layouts))
                for info_cls in classes.values():
                    # Built once per layer and class, from that layer.
                    self.assertEqual(info_cls.made, [layer, layer])
                for layer_built in layers:
                    # Every provider of a layer binds its class's one quant info.
                    by_class = {}
                    for _, info_cls, quant_info in layer_built:
                        by_class.setdefault(info_cls, set()).add(id(quant_info))
                    self.assertEqual(
                        {len(ids) for ids in by_class.values()}, {1}, by_class
                    )
                # A second layer shares none of the first layer's.
                self.assertFalse(
                    {id(info) for _, _, info in first}
                    & {id(info) for _, _, info in second}
                )

    def test_derived_scales_are_kept_and_the_checkpoint_form_stays(self):
        from sglang.srt.lora.moe.quant_info import MoeLoraNvFp4QuantInfo

        for mode in ("static", "per_token"):
            with self.subTest(mode):
                layer = self._nvfp4_layer(mode)
                attributes = vars(layer).copy()
                snapshot = {
                    name: value.clone()
                    for name, value in attributes.items()
                    if isinstance(value, torch.Tensor)
                }
                patch_packing, packed = self._packing()
                with patch_packing:
                    quant_info = MoeLoraNvFp4QuantInfo.from_layer(layer)
                    packed_scales, alpha = quant_info.packed_scales, quant_info.alpha
                    self.assertIs(quant_info.packed_scales, packed_scales)
                    self.assertIs(quant_info.alpha, alpha)
                    self.assertEqual(len(packed), 2)
                # The layer keeps every canonical tensor, bit for bit, and the
                # quant info reads the checkpoint-form scales in place.
                for name, value in attributes.items():
                    self.assertIs(getattr(layer, name), value, name)
                for name, value in snapshot.items():
                    self.assertTrue(
                        torch.equal(
                            getattr(layer, name).reshape(-1).view(torch.uint8),
                            value.reshape(-1).view(torch.uint8),
                        ),
                        name,
                    )
                for derived, canonical in (
                    (quant_info.w13_scale, layer.w13_weight_scale),
                    (quant_info.w2_scale, layer.w2_weight_scale),
                ):
                    self.assertEqual(derived.data_ptr(), canonical.data_ptr())
                for packed_scale, canonical in zip(
                    packed_scales, (layer.w13_weight_scale, layer.w2_weight_scale)
                ):
                    self.assertNotEqual(packed_scale.data_ptr(), canonical.data_ptr())
                # alpha: the weight global scales, times the activation scale
                # only when it is static.
                w13_global = layer.w13_weight_scale_2.float().reshape(2, -1)
                w2_global = layer.w2_weight_scale_2.float().reshape(-1, 1)
                if mode == "static":
                    w13_global, w2_global = w13_global * 0.5, w2_global * 0.25
                self.assertTrue(torch.equal(alpha[0], w13_global))
                self.assertTrue(torch.equal(alpha[1], w2_global))


class TestActQuantReadiness(CustomTestCase):
    """``act_quant_ready`` says whether the activation that just ran wrote the
    down GEMM's input. Serving prepares a fresh row state per forward; a row
    state that is activated again (materialized, then fused) must not keep the
    first answer, or down() feeds its GEMM the earlier activation's rows."""

    @staticmethod
    def _provider(cls, **attributes):
        """The real class without its GPU setup: only what the activation
        wrappers and down() read."""
        provider = object.__new__(cls)
        provider.quant_info = SimpleNamespace(
            num_local_experts=2, intermediate_size=128, hidden_size=128
        )
        provider._launch = mock.Mock()
        for name, value in attributes.items():
            setattr(provider, name, value)
        return provider

    @staticmethod
    def _row_state(cls, **fields):
        import msgspec

        # Required fields: those before the defaults and the gaps among them.
        names, defaults = cls.__struct_fields__, cls.__struct_defaults__
        tail = len(names) - len(defaults)
        values = dict.fromkeys(names[:tail])
        values.update(
            (name, None)
            for name, default in zip(names[tail:], defaults)
            if default is msgspec.NODEFAULT
        )
        values.update(fields)
        return cls(**values)

    @staticmethod
    def _activate(provider, row_state, *, fused, tile_width=64):
        """One activation through the shared wrapper with its kernel mocked;
        returns the down input the kernel was told to write (None: BF16 only)."""
        from sglang.kernels.ops.lora.moe import activation_delta, fused_act

        with (
            mock.patch.object(activation_delta, "act_delta") as act_delta,
            mock.patch.object(fused_act, "fused_b_act") as fused_b_act,
        ):
            if not fused:
                provider.act_with_delta(row_state, None, None, None, None, None)
                return act_delta.call_args.kwargs["act_quant"]
            provider.fused_act(
                row_state,
                activation="silu",
                base_gateup=None,
                act_rows=None,
                act_pairs=None,
                routing=None,
                config={"BLOCK_SIZE_W": tile_width},
            )
            return fused_b_act.call_args.kwargs["act_quant"]

    def test_block_scaled_down_quantizes_after_a_fused_activation_that_did_not(self):
        from sglang.srt.lora.moe.base_gemm_provider import cutedsl_blockscaled as bs

        masked, contiguous = (
            bs.CuteDslBlockScaledMaskedRowState,
            bs.CuteDslBlockScaledContiguousRowState,
        )
        cases = (
            (bs.CuteDslMxfp8MaskedProvider, masked, False),
            (bs.CuteDslMxfp4ContiguousProvider, contiguous, False),
            (bs.CuteDslNvFp4MaskedProvider, masked, False),
            (bs.CuteDslNvFp4ContiguousProvider, contiguous, True),
        )
        for provider_cls, state_cls, per_token in cases:
            with self.subTest(provider=provider_cls.__name__, per_token=per_token):
                provider = self._provider(
                    provider_cls,
                    _per_row=per_token,
                    _rows_spec=lambda stage, data, scales, rows, row_scales=None: (
                        stage,
                        data,
                        scales,
                        rows,
                        row_scales,
                    ),
                    act_out_shape=lambda row_state: (8, 128),
                    _quantize_rows=mock.Mock(),
                )
                state = self._row_state(
                    state_cls,
                    act_q="act_q",
                    act_sf="act_sf",
                    act_rs="act_rs" if per_token else None,
                    pair_to_row="pair_to_row",
                )
                self.assertFalse(state.act_quant_ready)
                # The activation's BF16 rows: two expert slabs or one flat run.
                slabs = (2,) if state_cls is masked else ()
                act_out = torch.zeros((*slabs, 8, 128), dtype=torch.bfloat16)

                def down_quantizes():
                    provider._quantize_rows.reset_mock()
                    provider.down(state, act_out, "out")
                    if not provider._quantize_rows.called:
                        return False
                    # The quantizer takes the rows flattened, the down input's
                    # buffers as rows of one slab of the activation as it came,
                    # and the pair map.
                    rows, spec, pairs = provider._quantize_rows.call_args.args
                    self.assertEqual(provider._quantize_rows.call_args.kwargs, {})
                    self.assertEqual(tuple(rows.shape), (act_out.numel() // 128, 128))
                    self.assertEqual(rows.data_ptr(), act_out.data_ptr())
                    self.assertEqual(
                        spec, ("gemm2", "act_q", "act_sf", 8, state.act_rs)
                    )
                    self.assertEqual(pairs, "pair_to_row")
                    return True

                # A fresh state, fused: nothing written, down() quantizes.
                group = provider.SF_VEC_SIZE
                split = group + group // 2  # a column tile that splits a scale group
                self.assertIsNone(
                    self._activate(provider, state, fused=True, tile_width=split)
                )
                self.assertTrue(down_quantizes())
                # Materialized: the activation writes the rows in its second pass.
                self.assertIsNotNone(self._activate(provider, state, fused=False))
                self.assertTrue(state.act_quant_ready)
                self.assertFalse(down_quantizes())
                # Fused again on the same state: BF16 rows only, so the rows the
                # materialized activation wrote are stale and down() must not
                # take them.
                self.assertIsNone(
                    self._activate(provider, state, fused=True, tile_width=split)
                )
                self.assertFalse(state.act_quant_ready)
                self.assertTrue(down_quantizes())
                # Whole scale groups per tile: the static fused activation writes
                # them; a per-token scale needs the whole row and never does.
                wrote = self._activate(
                    provider, state, fused=True, tile_width=2 * group
                )
                self.assertEqual(wrote is None, per_token)
                self.assertEqual(state.act_quant_ready, not per_token)
                self.assertEqual(down_quantizes(), per_token)

    def test_fp8_down_quantizes_after_a_fused_activation(self):
        from sglang.srt.lora.moe.base_gemm_provider import (
            cutedsl_fp8,
            cutedsl_fp8_channel,
        )

        group, per_row = (
            "sglang_per_token_group_quant_fp8",
            "quantize_rows_per_token_fp8",
        )
        cases = (
            (cutedsl_fp8, "CuteDslFp8Masked", group),
            (cutedsl_fp8_channel, "CuteDslFp8ChannelMasked", per_row),
            (cutedsl_fp8_channel, "CuteDslFp8ChannelContiguous", per_row),
        )
        for module, name, quantizer in cases:
            with self.subTest(provider=name):
                provider = self._provider(getattr(module, f"{name}Provider"))
                state_cls = getattr(module, f"{name}RowState")
                fields = dict(act_fp8="act_fp8", act_scale="act_scale")
                if "Contiguous" in name:
                    fields["seg_offsets"] = torch.zeros(3, dtype=torch.int32)
                state = self._row_state(state_cls, **fields)
                with mock.patch.object(
                    module, quantizer, return_value=("requantized", "rescaled")
                ) as quantize:

                    def down_quantizes():
                        quantize.reset_mock()
                        provider.down(state, "act_out", "out")
                        return quantize.called

                    # Only the materialized activation writes FP8 rows.
                    self.assertIsNotNone(self._activate(provider, state, fused=False))
                    self.assertTrue(state.act_quant_ready)
                    self.assertFalse(down_quantizes())
                    self.assertIsNone(self._activate(provider, state, fused=True))
                    self.assertFalse(state.act_quant_ready)
                    self.assertTrue(down_quantizes())

    def test_triton_fp8_down_takes_rows_only_from_the_activation_that_just_ran(self):
        from sglang.srt.lora.moe.base_gemm_provider import triton_bf16, triton_fp8

        provider = self._provider(
            triton_fp8.TritonFp8ContiguousProvider,
            act_out_shape=lambda row_state: (4, 128),
            _invoke_fp8=mock.Mock(),
        )
        provider.quant_info.w2_weight = provider.quant_info.w2_scale = None
        state = self._row_state(
            triton_bf16.TritonRowState, pair_to_row=torch.zeros(4, dtype=torch.int32)
        )

        def down_input():
            provider._invoke_fp8.reset_mock()
            provider.down(state, "act_out", "out")
            call = provider._invoke_fp8.call_args
            return call.args[0], call.kwargs["a_scale"]

        self.assertIsNotNone(self._activate(provider, state, fused=False))
        rows, scales = state.act_quant
        self.assertTrue(state.act_quant_ready)
        self.assertEqual([id(x) for x in down_input()], [id(rows), id(scales)])
        # After a fused activation the launcher quantizes the BF16 rows itself;
        # the buffers the earlier activation filled are still bound but stale.
        self.assertIsNone(self._activate(provider, state, fused=True))
        self.assertIs(state.act_quant[0], rows)
        self.assertFalse(state.act_quant_ready)
        self.assertEqual(down_input(), ("act_out", None))

    def test_row_states_without_the_flag_are_left_alone(self):
        from sglang.srt.lora.moe.base_gemm_provider import cutedsl_bf16

        provider = self._provider(cutedsl_bf16.CuteDslBf16MaskedProvider)
        state = self._row_state(cutedsl_bf16.CuteDslBf16MaskedRowState)
        for fused in (False, True):
            self.assertIsNone(self._activate(provider, state, fused=fused))
        self.assertFalse(hasattr(state, "act_quant_ready"))


class TestCompileCache(CustomTestCase):
    def test_stage_constants_separate_compiled_code(self):
        """NVFP4 gate/up and down stages of one geometry differ only in their
        epilogue constants; the cache must not hand one the other's code."""
        from sglang.srt.lora.moe.base_gemm_provider import cutedsl_common

        class Provider(cutedsl_common.CuteDslTileMixin):
            _ROW_MODE = "masked"
            _DTYPE_TAG = "nvfp4"

            def __init__(self):
                self.quant_info = SimpleNamespace(num_local_experts=2)
                self._compiled = {64: {}}
                self._as_dynamic_cute_tensor = lambda tensor, leading_dim: tensor
                self.compiles = 0

            def _stage_weight(self, stage):
                return torch.zeros((2, 256, 128), dtype=torch.uint8)

            def _stage_constexprs(self, stage):
                return (16, True, 128 if stage == "gemm1" else 1 << 30)

            def _prepare_dummy(self, stage, config):
                self.compiles += 1
                return SimpleNamespace(launch=lambda: None, compiled_fn=object())

        with mock.patch.dict(cutedsl_common._COMPILE_CACHE, clear=True):
            provider = Provider()
            provider._compile_stage(64, "gemm1", config="tile")
            provider._compile_stage(64, "gemm2", config="tile")
            self.assertEqual(provider.compiles, 2)
            calls = provider._compiled[64]
            self.assertIsNot(calls["gemm1"].compiled_fn, calls["gemm2"].compiled_fn)
            # The same geometry and constants reuse the compiled code.
            provider._compile_stage(64, "gemm1", config="tile")
            self.assertEqual(provider.compiles, 2)


class TestBlockScaledBuffers(CustomTestCase):
    def test_masked_buffers_keep_the_expert_dimension_for_one_expert(self):
        from sglang.srt.lora.moe.base_gemm_provider.base import prepare_buffer

        source = _function(
            (PROVIDER / "cutedsl_blockscaled.py").read_text(), "_quant_buffers"
        )
        namespace = {"prepare_buffer": prepare_buffer, "torch": torch}
        exec("from __future__ import annotations\n" + source, namespace)
        quant_buffers = namespace["_quant_buffers"]
        provider = SimpleNamespace(
            _pack=2,
            _per_row=False,
            _w13=torch.zeros(1, dtype=torch.uint8),
            SF_VEC_SIZE=32,
            _scale_atom_bytes=lambda rows, k, vec, slabs=1: slabs * rows * k // vec,
        )
        # The masked kernels address [experts, rows, K] slabs for one expert too.
        data, scales, row_scales = quant_buffers(
            provider, None, "masked:rows", (1, 64), 256, "cpu"
        )
        self.assertIsNone(row_scales)  # static mode: no stored row scales
        self.assertEqual(tuple(data.shape), (1, 64, 128))
        self.assertEqual(scales.numel(), 64 * 256 // 32)
        data, _, _ = quant_buffers(provider, None, "flat:rows", (64,), 256, "cpu")
        self.assertEqual(tuple(data.shape), (64, 128))


class TestMaterializedFinalize(CustomTestCase):
    """Execute the host dispatch with real CPU buffers and mocked GPU launches."""

    def test_materialized_finalize_preserves_strided_weights(self):
        from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProvider

        finalize = types.ModuleType("sglang.kernels.ops.lora.moe.finalize")
        finalize.invoke_small_finalize = mock.Mock()
        reorder = types.ModuleType("sglang.kernels.ops.moe.ep_moe_kernels")
        reorder.post_reorder_deepgemm = mock.Mock()
        # Production hands over strided top-k weight views; post_reorder_deepgemm
        # addresses them contiguously, so even a small batch must not take the
        # gather path and must receive a contiguous copy.
        tokens = 64
        weights = torch.arange(tokens * 6).view(tokens, 6)[:, ::2]
        ids = torch.zeros((tokens, 3), dtype=torch.int64)
        output = torch.empty((tokens, 5))
        delta = torch.empty((tokens, 3, 5))
        with mock.patch.dict(
            sys.modules, {finalize.__name__: finalize, reorder.__name__: reorder}
        ):
            MoeBaseProvider().finalize(
                SimpleNamespace(pair_to_row=torch.arange(tokens * 3)),
                torch.empty((tokens * 3, 5)),
                ids,
                weights,
                2.5,
                output,
                lora_delta=delta,
            )
        finalize.invoke_small_finalize.assert_not_called()
        call = reorder.post_reorder_deepgemm.call_args
        actual = call.args[4]
        self.assertTrue(actual.is_contiguous())
        torch.testing.assert_close(actual, weights)
        self.assertIs(call.kwargs["lora_delta"], delta)


class TestRunnerAdmission(CustomTestCase):
    """Layers the runner would otherwise serve with silently wrong numerics,
    and the plan check binding must surface. The device capability is the
    only GPU boundary, pinned to SM90."""

    def setUp(self):
        patcher = mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _base_layer(**overrides):
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        quant_method = object.__new__(UnquantizedFusedMoEMethod)
        dispatcher = object.__new__(StandardDispatcher)
        dispatcher.skip_local_expert_mapping = overrides.get("skip_local", False)
        return SimpleNamespace(
            quant_method=quant_method,
            dispatcher=dispatcher,
            w13_weight=torch.zeros(2, 8, 4, dtype=torch.bfloat16),
            w2_weight=torch.zeros(2, 4, 4, dtype=torch.bfloat16),
            num_local_experts=2,
            should_fuse_routed_scaling_factor_in_topk=overrides.get(
                "fused_scaling", False
            ),
            with_bias=overrides.get("with_bias", False),
            moe_runner_config=SimpleNamespace(
                activation="silu",
                is_gated=True,
                gemm1_alpha=None,
                gemm1_clamp_limit=None,
                swiglu_limit=None,
                apply_router_weight_on_input=overrides.get(
                    "apply_router_weight_on_input", False
                ),
                no_combine=overrides.get("no_combine", False),
                num_fused_shared_experts=overrides.get("num_fused_shared_experts", 0),
                top_k=2,
                routed_scaling_factor=1.0,
            ),
        )

    def test_rejects_layers_it_would_serve_wrongly(self):
        from sglang.srt.lora.moe.runner import MoeLoraRunner

        cases = {
            # Global expert IDs would index the wrong rank's experts.
            "global_expert_ids": (dict(skip_local=True), "EP-local"),
            # Scaling folded into top-k would be applied twice.
            "prefolded_routed_scaling": (dict(fused_scaling=True), "routed scaling"),
            # Providers add no bias term; a biased layer would lose it.
            "expert_biases": (dict(with_bias=True), "expert biases"),
            # The runner weights routes in finalize, never on the input rows.
            "router_weight_on_input": (
                dict(apply_router_weight_on_input=True),
                "owned by finalize",
            ),
            # The runner's finalize always combines the top-k rows.
            "no_combine": (dict(no_combine=True), "owned by finalize"),
            "fused_shared_experts": (
                dict(num_fused_shared_experts=1),
                "fused shared experts",
            ),
        }
        for name, (overrides, message) in cases.items():
            with (
                self.subTest(name),
                self.assertRaisesRegex(NotImplementedError, message),
            ):
                MoeLoraRunner._admit(self._base_layer(**overrides))

    def test_binding_surfaces_plan_validation(self):
        """A provider exposing one gate/up slice cannot serve a gated layer."""
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.lora.moe import runner as runner_module
        from sglang.srt.lora.moe.runner import MoeLoraRunner
        from sglang.srt.runtime_context import get_context, get_flags

        class QuantInfo:
            @staticmethod
            def from_layer(base_layer):
                return None

        class Provider:
            contract = SimpleNamespace(
                quant_info_cls=QuantInfo, lora_delta_dtype=torch.bfloat16
            )
            hidden_size = 2048
            intermediate_size = 768
            num_local_experts = 256
            gate_up_slices = 1

            def __init__(self, quant_info):
                pass

        layer = SimpleNamespace(
            w2_weight=SimpleNamespace(
                device=torch.device("cuda"), shape=(256, 2048, 768)
            ),
            num_local_experts=256,
            hidden_size=2048,
            moe_runner_config=SimpleNamespace(
                activation="silu", top_k=8, routed_scaling_factor=1.0, is_gated=True
            ),
        )
        # from_layer reads the published MoE runner backend for its vendor.
        with (
            mock.patch.object(MoeLoraRunner, "_admit", return_value="bf16"),
            mock.patch.object(
                runner_module, "select_provider_cls", return_value=Provider
            ),
            get_context().override_server_args(),
            get_flags().moe.override(runner_backend=MoeRunnerBackend.LORA_CUTEDSL),
            self.assertRaisesRegex(ValueError, "gate/up slices"),
        ):
            MoeLoraRunner.from_layer(
                layer, workspace=object(), is_shared_outer=False, physical_rank=64
            )


class TestLayerRunnerBackendResolution(CustomTestCase):
    """The selected lora runner backend binds the new MoE runner and rebinds
    run_moe_core."""

    def test_v2_accepts_each_new_moe_runner(self):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.lora.layers import FusedMoEWithLoRA
        from sglang.srt.lora.moe.runner import MoeLoraRunner

        base_layer = SimpleNamespace(
            quant_method=mock.MagicMock(),
            moe_runner_config=MoeRunnerConfig(
                num_experts=8,
                num_local_experts=8,
                hidden_size=64,
                intermediate_size_per_partition=32,
                layer_id=0,
                top_k=2,
                num_fused_shared_experts=0,
                params_dtype=torch.bfloat16,
                activation="silu",
                inplace=True,
            ),
            dispatcher=None,
            num_local_experts=8,
            should_fuse_routed_scaling_factor_in_topk=False,
            moe_tp_size=1,
            moe_tp_rank=0,
            intermediate_size_per_partition=32,
            runner=SimpleNamespace(runner_backend=MoeRunnerBackend.DEEP_GEMM),
            run_moe_core=mock.Mock(),
        )
        backend = SimpleNamespace(name="triton_v2", lora_workspace=object())
        original_run_moe_core = base_layer.run_moe_core
        with (
            mock.patch(
                "sglang.srt.layers.moe.utils.get_moe_runner_backend",
                return_value=MoeRunnerBackend.LORA_CUTEDSL,
            ),
            mock.patch.object(MoeLoraRunner, "from_layer") as bind,
        ):
            layer = FusedMoEWithLoRA(
                base_layer,
                backend,
                experts_shared_outer_loras=True,
                max_lora_rank=16,
            )
        self.assertIs(layer._lora_runner_backend, MoeRunnerBackend.LORA_CUTEDSL)
        bind.assert_called_once_with(
            base_layer,
            workspace=backend.lora_workspace,
            is_shared_outer=True,
            physical_rank=16,
        )
        self.assertIs(layer._moe_lora_runner, bind.return_value)
        self.assertIs(layer._base_run_moe_core, original_run_moe_core)
        self.assertEqual(base_layer.run_moe_core, layer._run_moe_core_with_lora)


if __name__ == "__main__":
    unittest.main()
