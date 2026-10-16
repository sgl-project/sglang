import builtins
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMiniMaxM3Mxfp8FlyDsl(CustomTestCase):
    def test_tp4_weight_shapes_match_dense_contract(self):
        from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
            MXFP8_FLYDSL_WEIGHT_SHAPES,
        )

        self.assertEqual(
            MXFP8_FLYDSL_WEIGHT_SHAPES,
            frozenset(
                {
                    (2304, 6144),
                    (2560, 6144),
                    (6144, 2048),
                    (6144, 6144),
                    (6144, 3072),
                    (1536, 6144),
                    (6144, 768),
                }
            ),
        )

    def test_decode_buckets_through_64_are_tuned(self):
        from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
            MXFP8_FLYDSL_M_VALUES,
        )

        self.assertEqual(
            MXFP8_FLYDSL_M_VALUES[:12],
            (1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64),
        )

    def test_missing_aiter_flydsl_is_an_error(self):
        from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
            get_flydsl_mxfp8_config,
        )

        real_import = builtins.__import__

        def reject_aiter_flydsl(name, *args, **kwargs):
            if name == "aiter.ops.flydsl":
                raise ImportError("missing FlyDSL GEMM")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", reject_aiter_flydsl),
            self.assertRaisesRegex(
                RuntimeError, "MX-scale preshuffle GEMM is unavailable"
            ),
        ):
            get_flydsl_mxfp8_config(4, 2304, 6144)

    def test_gfx95_default_stays_on_dot_scaled(self):
        from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
            dot_scaled_mxfp8_blockscaled_linear,
        )
        from sglang.srt.layers.quantization import fp8_utils

        with (
            patch.object(fp8_utils, "_is_hip", True),
            patch.object(fp8_utils, "_is_gfx95_supported", True),
            patch.object(
                fp8_utils,
                "FP8_GEMM_RUNNER_BACKEND",
                fp8_utils.Fp8GemmRunnerBackend.AUTO,
            ),
        ):
            self.assertTrue(
                fp8_utils.resolve_mxfp8_dense_gemm_backend().is_gfx95_dot_scaled()
            )
            self.assertIs(
                fp8_utils.dispatch_w8a8_mxfp8_linear(),
                dot_scaled_mxfp8_blockscaled_linear,
            )

    def test_explicit_aiter_backend_dispatches_to_flydsl(self):
        from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
            flydsl_mxfp8_blockscaled_linear,
        )
        from sglang.srt.layers.quantization import fp8_utils

        with (
            patch.object(fp8_utils, "_use_aiter", True),
            patch.object(fp8_utils, "_is_hip", True),
            patch.object(fp8_utils, "_is_gfx95_supported", True),
            patch.object(
                fp8_utils,
                "FP8_GEMM_RUNNER_BACKEND",
                fp8_utils.Fp8GemmRunnerBackend.AITER,
            ),
        ):
            self.assertTrue(
                fp8_utils.resolve_mxfp8_dense_gemm_backend().is_gfx95_mxfp8_flydsl()
            )
            self.assertTrue(
                fp8_utils.resolve_block_fp8_mxfp8_backend().is_gfx95_mxfp8_aiter()
            )
            self.assertIs(
                fp8_utils.dispatch_w8a8_mxfp8_linear(),
                flydsl_mxfp8_blockscaled_linear,
            )

    def test_unknown_topology_keeps_canonical_weights(self):
        from sglang.kernels.ops.quantization import mxfp8_amd_gfx95
        from sglang.srt.layers.quantization import fp8 as fp8_module
        from sglang.srt.layers.quantization import fp8_utils

        fake_aiter = ModuleType("aiter")
        fake_aiter.__path__ = []
        fake_ops = ModuleType("aiter.ops")
        fake_ops.__path__ = []
        fake_shuffle = ModuleType("aiter.ops.shuffle")
        fake_shuffle.shuffle_scale_a16w4 = lambda *args, **kwargs: args[0]
        fake_shuffle.shuffle_weight = lambda value, **kwargs: value

        with (
            patch.dict(
                sys.modules,
                {
                    "aiter": fake_aiter,
                    "aiter.ops": fake_ops,
                    "aiter.ops.shuffle": fake_shuffle,
                },
            ),
            patch.object(
                mxfp8_amd_gfx95, "MXFP8_FLYDSL_WEIGHT_SHAPES", frozenset({(32, 256)})
            ),
            patch.object(mxfp8_amd_gfx95, "MXFP8_FLYDSL_M_VALUES", (4,)),
            patch.object(
                mxfp8_amd_gfx95, "get_flydsl_mxfp8_config", lambda *args: None
            ),
        ):
            layer = torch.nn.Module()
            layer.register_parameter(
                "weight",
                torch.nn.Parameter(
                    torch.empty((32, 256), dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                ),
            )
            layer.register_parameter(
                "weight_scale_inv",
                torch.nn.Parameter(
                    torch.empty((32, 8), dtype=torch.uint8), requires_grad=False
                ),
            )
            weight_ptr = layer.weight.data_ptr()
            scale_ptr = layer.weight_scale_inv.data_ptr()
            method = fp8_module.Fp8LinearMethod.__new__(fp8_module.Fp8LinearMethod)
            method.use_mxfp8 = True
            method.mxfp8_dense_backend = (
                fp8_utils.Mxfp8DenseGemmBackend.GFX95_MXFP8_FLYDSL
            )
            method._process_mxfp8_linear_weight_scale(layer)

        self.assertEqual(layer.weight.data_ptr(), weight_ptr)
        self.assertEqual(layer.weight_scale_inv.data_ptr(), scale_ptr)
        self.assertFalse(hasattr(layer, "weight_mxfp8_flydsl"))
        self.assertFalse(hasattr(layer, "weight_scale_inv_mxfp8_flydsl"))

    def test_decode_requires_a_captured_graph_bucket(self):
        from sglang.kernels.ops.quantization import mxfp8_amd_gfx95
        from sglang.srt import runtime_context
        from sglang.srt.layers.quantization import fp8 as fp8_module
        from sglang.srt.layers.quantization import fp8_utils
        from sglang.srt.model_executor.cuda_graph_config import Backend, CudaGraphConfig

        config = {"kernelName": "unit_test", "splitK": 1}
        fallback_calls = []
        fallback_output = torch.empty((4, 32), dtype=torch.bfloat16)

        def fallback(**kwargs):
            fallback_calls.append(kwargs)
            return fallback_output

        layer = torch.nn.Module()
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty((32, 256), dtype=torch.float8_e4m3fn),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "weight_scale_inv",
            torch.nn.Parameter(
                torch.empty((32, 8), dtype=torch.uint8), requires_grad=False
            ),
        )
        layer.register_parameter(
            "weight_mxfp8_flydsl",
            torch.nn.Parameter(
                torch.empty((32, 256), dtype=torch.float8_e4m3fn),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "weight_scale_inv_mxfp8_flydsl",
            torch.nn.Parameter(
                torch.empty((32, 8), dtype=torch.uint8), requires_grad=False
            ),
        )
        method = fp8_module.Fp8LinearMethod.__new__(fp8_module.Fp8LinearMethod)
        method.use_marlin = False
        method.use_mxfp8 = True
        method.block_fp8_as_mxfp8 = False
        method.mxfp8_dense_backend = fp8_utils.Mxfp8DenseGemmBackend.GFX95_MXFP8_FLYDSL
        flydsl_calls = []
        flydsl_output = torch.empty((4, 32), dtype=torch.bfloat16)

        def flydsl(**kwargs):
            flydsl_calls.append(kwargs)
            return flydsl_output

        method.w8a8_mxfp8_linear = flydsl
        x = torch.empty((4, 256), dtype=torch.bfloat16)
        disabled = CudaGraphConfig()
        disabled.decode.backend = Backend.DISABLED
        enabled = CudaGraphConfig()
        enabled.decode.backend = Backend.FULL
        enabled.decode.bs = [4]
        enabled.decode.max_bs = 64

        with (
            patch.object(
                mxfp8_amd_gfx95, "get_flydsl_mxfp8_config", lambda *args: config
            ),
            patch.object(
                mxfp8_amd_gfx95, "dot_scaled_mxfp8_blockscaled_linear", fallback
            ),
        ):
            with patch.object(
                runtime_context,
                "get_server_args",
                lambda: SimpleNamespace(cuda_graph_config=disabled),
            ):
                self.assertIs(method.apply(layer, x), fallback_output)
                self.assertIs(fallback_calls[-1]["weight"], layer.weight)

            with patch.object(
                runtime_context,
                "get_server_args",
                lambda: SimpleNamespace(cuda_graph_config=enabled),
            ):
                self.assertIs(method.apply(layer, x), flydsl_output)
                self.assertIs(flydsl_calls[-1]["weight"], layer.weight_mxfp8_flydsl)
                flydsl_count = len(flydsl_calls)
                fallback_count = len(fallback_calls)
                self.assertIs(method.apply(layer, x.to(torch.float16)), fallback_output)
                self.assertEqual(len(flydsl_calls), flydsl_count)
                self.assertEqual(len(fallback_calls), fallback_count + 1)

    def test_decode_buffers_are_pooled_across_buckets(self):
        from sglang.kernels.ops.quantization import mxfp8_amd_gfx95
        from sglang.srt.layers.quantization import fp8 as fp8_module

        layer = torch.nn.Module()
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty((32, 256), dtype=torch.float8_e4m3fn),
                requires_grad=False,
            ),
        )
        method = fp8_module.Fp8LinearMethod.__new__(fp8_module.Fp8LinearMethod)
        with (
            patch.object(mxfp8_amd_gfx95, "MXFP8_FLYDSL_M_VALUES", (4, 64)),
            patch.object(
                mxfp8_amd_gfx95,
                "get_flydsl_mxfp8_config",
                lambda m, *_: {"splitK": 2 if m == 4 else 3},
            ),
        ):
            small = method._get_mxfp8_flydsl_runtime_buffers(
                layer,
                m=4,
                n=32,
                k=256,
                output_dtype=torch.bfloat16,
                split_k=2,
            )
            large = method._get_mxfp8_flydsl_runtime_buffers(
                layer,
                m=64,
                n=32,
                k=256,
                output_dtype=torch.bfloat16,
                split_k=3,
            )

        for name in (
            "activation_scale_padded",
            "activation_scale_shuffled",
            "output_buffer",
            "splitk_workspace",
        ):
            self.assertEqual(small[name].data_ptr(), large[name].data_ptr())
            self.assertTrue(small[name].is_contiguous())
            self.assertTrue(large[name].is_contiguous())
        self.assertEqual(tuple(small["output_buffer"].shape), (4, 32))
        self.assertEqual(tuple(large["output_buffer"].shape), (64, 32))
        self.assertEqual(tuple(small["splitk_workspace"].shape), (2, 4, 32))


if __name__ == "__main__":
    unittest.main()
