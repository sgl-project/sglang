"""Native MXFP8 dense linears (MiniMax-M3) on gfx950 with --fp8-gemm-backend=aiter: backend
selection, the load-time weight layout and what apply() hands aiter's MXFP8 GEMM. aiter is stubbed,
so this runs on CPU; test/registered/kernels/ops/gemm/test_mxfp8_amd_gfx95.py runs the GEMM."""

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_AITER_ROUTE = "sglang.kernels.ops.gemm.mxfp8_aiter_gfx95"


def _gfx95_runner(fp8_utils, runner_backend):
    return (
        patch.object(fp8_utils, "_is_hip", True),
        patch.object(fp8_utils, "_is_gfx95_supported", True),
        patch.object(fp8_utils, "FP8_GEMM_RUNNER_BACKEND", runner_backend),
    )


def _native_layer(n: int, k: int) -> torch.nn.Module:
    layer = torch.nn.Module()
    weight = torch.randint(0, 120, (n, k), dtype=torch.uint8)
    layer.weight = torch.nn.Parameter(
        weight.view(torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale_inv = torch.nn.Parameter(
        torch.randint(118, 136, (n, k // 32), dtype=torch.uint8), requires_grad=False
    )
    return layer


class TestMiniMaxM3Mxfp8Aiter(CustomTestCase):
    def test_gfx95_default_stays_on_dot_scaled(self):
        from sglang.kernels.ops.gemm.mxfp8_amd_gfx95 import (
            dot_scaled_mxfp8_blockscaled_linear,
        )
        from sglang.srt.layers.quantization import fp8_utils

        p0, p1, p2 = _gfx95_runner(fp8_utils, fp8_utils.Fp8GemmRunnerBackend.AUTO)
        with p0, p1, p2:
            self.assertTrue(
                fp8_utils.resolve_mxfp8_dense_gemm_backend().is_gfx95_dot_scaled()
            )
            self.assertIs(
                fp8_utils.dispatch_w8a8_mxfp8_linear(),
                dot_scaled_mxfp8_blockscaled_linear,
            )

    def test_aiter_backend_runs_aiter_mxfp8_gemm(self):
        from sglang.srt.layers.quantization import fp8_utils

        route = ModuleType(_AITER_ROUTE)
        route.aiter_mxfp8_blockscaled_linear = object()
        p0, p1, p2 = _gfx95_runner(fp8_utils, fp8_utils.Fp8GemmRunnerBackend.AITER)
        with p0, p1, p2, patch.dict(sys.modules, {_AITER_ROUTE: route}):
            native = fp8_utils.resolve_mxfp8_dense_gemm_backend()
            self.assertTrue(native.is_gfx95_mxfp8_aiter_1x32())
            # the gfx950 checks (weight preparation, draft-model stacking) still see it
            self.assertTrue(native.is_gfx95())
            self.assertTrue(
                fp8_utils.resolve_block_fp8_mxfp8_backend().is_gfx95_mxfp8_aiter()
            )
            self.assertIs(
                fp8_utils.dispatch_w8a8_mxfp8_linear(),
                route.aiter_mxfp8_blockscaled_linear,
            )

    def _process(self, layer, preshuffle_fits):
        from sglang.srt.layers.quantization import fp8 as fp8_module
        from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend

        layouts = []

        def shuffle_weight(weight, layout):
            layouts.append(layout)
            return weight.view(torch.uint8).flip(0).view(weight.dtype)

        aiter = ModuleType("aiter")
        aiter.__path__ = []
        aiter_ops = ModuleType("aiter.ops")
        aiter_ops.__path__ = []
        aiter_shuffle = ModuleType("aiter.ops.shuffle")
        aiter_shuffle.shuffle_weight = shuffle_weight
        route = ModuleType(_AITER_ROUTE)
        route.aiter_preshuffle_fits = lambda n, k: preshuffle_fits
        modules = {
            "aiter": aiter,
            "aiter.ops": aiter_ops,
            "aiter.ops.shuffle": aiter_shuffle,
            _AITER_ROUTE: route,
        }
        method = fp8_module.Fp8LinearMethod.__new__(fp8_module.Fp8LinearMethod)
        method.use_mxfp8 = True
        method.mxfp8_dense_backend = Mxfp8DenseGemmBackend.GFX95_MXFP8_AITER_1X32
        with patch.dict(sys.modules, modules):
            method._process_mxfp8_linear_weight_scale(layer)
        return layouts

    def test_weight_is_preshuffled_in_place_and_scale_kept(self):
        layer = _native_layer(32, 256)
        weight, scale = layer.weight, layer.weight_scale_inv
        plain = weight.data.view(torch.uint8).clone()
        loaded_scale = scale.data.clone()

        self.assertEqual(self._process(layer, preshuffle_fits=True), [(16, 16)])

        self.assertTrue(layer.mxfp8_aiter_preshuffled)
        self.assertIs(layer.weight, weight)
        self.assertTrue(torch.equal(layer.weight.data.view(torch.uint8), plain.flip(0)))
        self.assertIs(layer.weight_scale_inv, scale)
        self.assertTrue(torch.equal(layer.weight_scale_inv.data, loaded_scale))

    def test_weight_aiter_cannot_preshuffle_stays_plain(self):
        layer = _native_layer(40, 256)
        plain = layer.weight.data.view(torch.uint8).clone()

        self.assertEqual(self._process(layer, preshuffle_fits=False), [])

        self.assertFalse(layer.mxfp8_aiter_preshuffled)
        self.assertTrue(torch.equal(layer.weight.data.view(torch.uint8), plain))

    def test_apply_hands_aiter_the_loaded_scale(self):
        from sglang.srt.layers.quantization import fp8 as fp8_module
        from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend

        calls = []
        output = torch.empty((4, 32), dtype=torch.bfloat16)

        def linear(**kwargs):
            calls.append(kwargs)
            return output

        method = fp8_module.Fp8LinearMethod.__new__(fp8_module.Fp8LinearMethod)
        method.use_marlin = False
        method.use_mxfp8 = True
        method.block_fp8_as_mxfp8 = False
        method.mxfp8_dense_backend = Mxfp8DenseGemmBackend.GFX95_MXFP8_AITER_1X32
        method.w8a8_mxfp8_linear = linear
        layer = _native_layer(32, 256)
        layer.mxfp8_aiter_preshuffled = True

        x = torch.empty((4, 256), dtype=torch.bfloat16)
        self.assertIs(method.apply(layer, x), output)
        self.assertIs(calls[-1]["input"], x)
        self.assertIsNone(calls[-1]["input_scale"])
        self.assertIs(calls[-1]["weight"], layer.weight)
        self.assertIs(calls[-1]["weight_scale_ue8m0"], layer.weight_scale_inv)
        self.assertTrue(calls[-1]["weight_preshuffled"])
        self.assertIsNone(calls[-1]["bias"])

        xq = torch.empty((4, 256), dtype=torch.float8_e4m3fn)
        xs = torch.empty((4, 8), dtype=torch.uint8)
        self.assertIs(method.apply(layer, (xq, xs)), output)
        self.assertIs(calls[-1]["input"], xq)
        self.assertIs(calls[-1]["input_scale"], xs)


if __name__ == "__main__":
    unittest.main()
