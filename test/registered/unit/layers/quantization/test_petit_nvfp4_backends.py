import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from sglang.kernels.ops.gemm import rdna4_nvfp4
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.petit import PetitNvFp4Config
from sglang.srt.layers.quantization.petit_utils import (
    PETIT_NVFP4_BACKEND,
    RDNA4_NVFP4_BACKEND,
    apply_petit_nvfp4_linear,
    prepare_nvfp4_layer_for_petit,
    select_nvfp4_backend,
)
from sglang.srt.utils import is_gfx1201_supported
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRdna4Nvfp4BackendSelection(CustomTestCase):
    def setUp(self):
        super().setUp()
        # is_gfx1201_supported caches per device; the mocked properties below
        # would otherwise leak between subtests.
        is_gfx1201_supported.cache_clear()
        self.addCleanup(is_gfx1201_supported.cache_clear)

    @staticmethod
    def _mock_device(gcn_arch_name, hip="7.0"):
        return (
            mock.patch.object(torch.version, "hip", hip),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(gcnArchName=gcn_arch_name),
            ),
        )

    def test_exact_gfx1201_device_detection(self):
        cases = (
            # A feature suffix must not defeat the match; a near neighbour must.
            ("gfx1201:sramecc+:xnack-", True),
            ("GFX1201", True),
            ("gfx1200", False),
            ("gfx942", False),
        )
        for gcn_arch_name, expected in cases:
            with self.subTest(gcn_arch_name=gcn_arch_name):
                is_gfx1201_supported.cache_clear()
                with contextlib.ExitStack() as stack:
                    for patcher in self._mock_device(gcn_arch_name):
                        stack.enter_context(patcher)
                    self.assertEqual(
                        rdna4_nvfp4.is_rdna4_nvfp4_device("cuda:0"), expected
                    )

    def test_non_hip_and_non_gpu_devices_are_rejected(self):
        with contextlib.ExitStack() as stack:
            for patcher in self._mock_device("gfx1201", hip=None):
                stack.enter_context(patcher)
            self.assertFalse(rdna4_nvfp4.is_rdna4_nvfp4_device("cuda:0"))

        is_gfx1201_supported.cache_clear()
        with contextlib.ExitStack() as stack:
            for patcher in self._mock_device("gfx1201"):
                stack.enter_context(patcher)
            self.assertFalse(rdna4_nvfp4.is_rdna4_nvfp4_device("cpu"))

    def test_selector_preserves_petit_for_non_rdna4(self):
        with mock.patch(
            "sglang.srt.layers.quantization.petit_utils.is_rdna4_nvfp4_device",
            return_value=False,
        ):
            self.assertEqual(select_nvfp4_backend(), PETIT_NVFP4_BACKEND)

    def test_rdna4_preparation_keeps_canonical_layout_without_petit(self):
        layer = nn.Module()
        layer.input_size_per_partition = 16
        layer.output_size_per_partition = 3
        layer.weight = nn.Parameter(
            torch.arange(24, dtype=torch.uint8).reshape(3, 8),
            requires_grad=False,
        )
        layer.weight_scale = nn.Parameter(
            torch.ones(3, 1, dtype=torch.float32).to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        expected_weight = layer.weight.clone()
        expected_scale = layer.weight_scale.clone()

        with (
            mock.patch(
                "sglang.srt.layers.quantization.petit_utils.select_nvfp4_backend",
                return_value=RDNA4_NVFP4_BACKEND,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.petit_utils._load_petit_ops",
                side_effect=AssertionError("Petit must not be imported"),
            ),
            mock.patch(
                "sglang.srt.layers.quantization.petit_utils.try_warmup_rdna4_nvfp4"
            ) as warmup,
        ):
            prepare_nvfp4_layer_for_petit(layer)

        warmup.assert_called_once()

        self.assertEqual(layer.nvfp4_backend, RDNA4_NVFP4_BACKEND)
        torch.testing.assert_close(layer.weight, expected_weight)
        torch.testing.assert_close(layer.weight_scale, expected_scale)

    def test_non_rdna4_preparation_preserves_petit_repack(self):
        layer = nn.Module()
        layer.input_size_per_partition = 16
        layer.output_size_per_partition = 3
        layer.weight = nn.Parameter(
            torch.arange(24, dtype=torch.uint8).reshape(3, 8),
            requires_grad=False,
        )
        layer.weight_scale = nn.Parameter(
            torch.ones(3, 1, dtype=torch.float32).to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        repacked = torch.arange(6, dtype=torch.int32).reshape(3, 2)
        processed_scale = torch.arange(3, dtype=torch.float32).reshape(3, 1)
        petit_ops = SimpleNamespace(
            repack_nvfp4=mock.Mock(return_value=repacked),
            process_nvfp4_scales=mock.Mock(return_value=processed_scale),
        )

        with (
            mock.patch(
                "sglang.srt.layers.quantization.petit_utils.select_nvfp4_backend",
                return_value=PETIT_NVFP4_BACKEND,
            ),
            mock.patch(
                "sglang.srt.layers.quantization.petit_utils._load_petit_ops",
                return_value=petit_ops,
            ),
        ):
            prepare_nvfp4_layer_for_petit(layer)

        self.assertEqual(layer.nvfp4_backend, PETIT_NVFP4_BACKEND)
        torch.testing.assert_close(layer.weight, repacked)
        torch.testing.assert_close(layer.weight_scale, processed_scale)
        petit_ops.repack_nvfp4.assert_called_once_with(
            mock.ANY,
            size_n=3,
            size_k=16,
        )
        petit_ops.process_nvfp4_scales.assert_called_once_with(
            scales=mock.ANY,
            size_k=16,
            size_n=3,
        )

    def test_apply_uses_cached_rdna4_backend(self):
        input_tensor = torch.randn(2, 16, dtype=torch.bfloat16)
        expected = torch.randn(2, 3, dtype=torch.bfloat16)
        with mock.patch(
            "sglang.srt.layers.quantization.petit_utils.rdna4_nvfp4_linear",
            return_value=expected,
        ) as rdna4_linear:
            output = apply_petit_nvfp4_linear(
                input=input_tensor,
                weight=torch.empty(3, 8, dtype=torch.uint8),
                weight_scale=torch.empty(3, 1, dtype=torch.float8_e4m3fn),
                weight_scale_2=torch.ones(1),
                size_n=3,
                size_k=16,
                backend=RDNA4_NVFP4_BACKEND,
            )

        self.assertIs(output, expected)
        rdna4_linear.assert_called_once()

    def test_apply_preserves_petit_shape_and_bias(self):
        input_tensor = torch.randn(1, 2, 16, dtype=torch.bfloat16)
        petit_output = torch.zeros(2, 3, dtype=torch.bfloat16)
        petit_ops = SimpleNamespace(mul_nvfp4_a16=mock.Mock(return_value=petit_output))
        with mock.patch(
            "sglang.srt.layers.quantization.petit_utils._load_petit_ops",
            return_value=petit_ops,
        ):
            output = apply_petit_nvfp4_linear(
                input=input_tensor,
                weight=torch.empty(3, 2, dtype=torch.int32),
                weight_scale=torch.empty(3, 1),
                weight_scale_2=torch.ones(1),
                size_n=3,
                size_k=16,
                bias=torch.ones(3, dtype=torch.bfloat16),
                backend=PETIT_NVFP4_BACKEND,
            )

        self.assertEqual(output.shape, (1, 2, 3))
        torch.testing.assert_close(output, torch.ones_like(output))
        petit_ops.mul_nvfp4_a16.assert_called_once_with(
            a=mock.ANY,
            b=mock.ANY,
            s=mock.ANY,
            global_scale=mock.ANY,
            size_m=2,
            size_n=3,
            size_k=16,
            solution_id=-1,
        )

    def test_rdna4_wrapper_rejects_unsupported_contracts_before_launch(self):
        valid_input = torch.randn(2, 16, dtype=torch.bfloat16)
        valid_weight = torch.empty(3, 8, dtype=torch.uint8)
        valid_scale = torch.empty(3, 1, dtype=torch.float8_e4m3fn)
        valid_global_scale = torch.ones(1, dtype=torch.float32)
        cases = (
            (
                {"input": valid_input.float()},
                TypeError,
                "supports BF16 and FP16",
            ),
            (
                {"input": torch.randn(2, 15, dtype=torch.bfloat16)},
                ValueError,
                "K divisible by 16",
            ),
            (
                {"input": torch.empty(0, 16, dtype=torch.bfloat16)},
                ValueError,
                "at least one row",
            ),
            (
                {"weight": valid_weight.to(torch.int8)},
                TypeError,
                "2D uint8 packed tensor",
            ),
            (
                {"weight": torch.empty(0, 8, dtype=torch.uint8)},
                ValueError,
                "at least one output row",
            ),
            (
                {"weight_scale": valid_scale.float()},
                TypeError,
                "2D float8_e4m3fn tensor",
            ),
            (
                {"weight_global_scale": valid_global_scale.bfloat16()},
                TypeError,
                "one float32 value",
            ),
        )

        with mock.patch.object(rdna4_nvfp4, "is_rdna4_nvfp4_device", return_value=True):
            for override, error_type, message in cases:
                arguments = {
                    "input": valid_input,
                    "weight": valid_weight,
                    "weight_scale": valid_scale,
                    "weight_global_scale": valid_global_scale,
                    **override,
                }
                with self.subTest(override=override):
                    with self.assertRaisesRegex(error_type, message):
                        rdna4_nvfp4.rdna4_nvfp4_linear(**arguments)

    def test_modelopt_declines_dense_nvfp4_on_rocm(self):
        """The ModelOpt configs must step aside so Petit can claim the checkpoint."""
        config = {"quant_method": "modelopt_fp4", "quant_algo": "NVFP4"}
        with mock.patch.object(torch.version, "hip", "7.0"):
            for requested_method in (None, "modelopt", "modelopt_fp4", "petit_nvfp4"):
                with self.subTest(requested_method=requested_method):
                    self.assertIsNone(
                        QuantizationConfig._modelopt_override_quantization_method(
                            config, requested_method
                        )
                    )

    def test_petit_claims_both_modelopt_method_spellings_on_rocm(self):
        configs = (
            {"quant_method": "modelopt", "quant_algo": "NVFP4"},
            {"quant_method": "modelopt_fp4", "quant_algo": "NVFP4"},
        )
        with mock.patch(
            "sglang.srt.layers.quantization.petit._is_hip",
            True,
        ):
            for config in configs:
                with self.subTest(quant_method=config["quant_method"]):
                    self.assertEqual(
                        PetitNvFp4Config.override_quantization_method(config, None),
                        "petit_nvfp4",
                    )

    def test_petit_declines_algorithms_it_cannot_serve(self):
        # NVFP4_AWQ / W4A16_NVFP4 keep resolving to modelopt_fp4 and its
        # explicit "not supported in ROCm" error rather than silently landing
        # on a backend that only implements group-16 dense NVFP4.
        with mock.patch("sglang.srt.layers.quantization.petit._is_hip", True):
            for quant_algo in ("NVFP4_AWQ", "W4A16_NVFP4"):
                with self.subTest(quant_algo=quant_algo):
                    config = {
                        "quant_method": "modelopt_fp4",
                        "quant_algo": quant_algo,
                    }
                    self.assertIsNone(
                        PetitNvFp4Config.override_quantization_method(config, None)
                    )
        with mock.patch.object(torch.version, "hip", "7.0"):
            self.assertEqual(
                QuantizationConfig._modelopt_override_quantization_method(
                    {"quant_method": "modelopt_fp4", "quant_algo": "NVFP4_AWQ"}, None
                ),
                "modelopt_fp4",
            )

    def test_petit_declines_everything_off_rocm(self):
        with mock.patch("sglang.srt.layers.quantization.petit._is_hip", False):
            self.assertIsNone(
                PetitNvFp4Config.override_quantization_method(
                    {"quant_method": "modelopt_fp4", "quant_algo": "NVFP4"}, None
                )
            )

    def test_quantized_moe_layers_fail_fast(self):
        """A quantized expert must not silently fall through to unquantized MoE."""

        class _FakeFusedMoE(nn.Module):
            pass

        config = PetitNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo="FP8",
            group_size=16,
            exclude_modules=["lm_head"],
        )
        with mock.patch(
            "sglang.srt.layers.moe.fused_moe_triton.FusedMoE",
            _FakeFusedMoE,
        ):
            with self.assertRaisesRegex(NotImplementedError, "does not support"):
                config.get_quant_method(_FakeFusedMoE(), "model.layers.0.mlp.experts")
            # An excluded expert stays unquantized instead of raising.
            self.assertIsNone(config.get_quant_method(_FakeFusedMoE(), "lm_head"))

    def test_cuda_flagless_modelopt_nvfp4_routing_is_unchanged(self):
        config = {"quant_method": "modelopt_fp4", "quant_algo": "NVFP4"}
        with mock.patch.object(torch.version, "hip", None):
            override = QuantizationConfig._modelopt_override_quantization_method(
                config, None
            )
        self.assertEqual(override, "modelopt_fp4")

    def test_config_json_style_flat_quant_block_is_supported(self):
        config = PetitNvFp4Config.from_config(
            {
                "quant_algo": "NVFP4",
                "config_groups": {
                    "group_0": {
                        "weights": {"group_size": 16},
                    }
                },
                "ignore": ["lm_head"],
                "kv_cache_scheme": {"type": "float", "num_bits": 8},
            }
        )
        self.assertTrue(config.is_checkpoint_nvfp4_serialized)
        self.assertEqual(config.group_size, 16)
        self.assertEqual(config.kv_cache_quant_algo, "FP8")
        self.assertEqual(config.exclude_modules, ["lm_head"])


if __name__ == "__main__":
    unittest.main()
