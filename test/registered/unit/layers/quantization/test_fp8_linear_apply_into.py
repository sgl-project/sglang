"""Fp8LinearMethod.apply_into writes a block-FP8 linear into caller-owned storage.

The DeepGEMM path writes the caller's tensor directly; the Triton fallback
computes and copies. Either way the result must match ``apply`` exactly and
the returned tensor must alias the one passed in.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_utils import Fp8GemmRunnerBackend
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import (
    init_single_process_dist,
    load_linear_weights,
    make_tp1_column_parallel_linear,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-small")

FP8_MAX = 448.0
# (M, N, K), N and K multiples of the (128, 128) weight block.
SHAPES = [(1, 512, 512), (5, 384, 896), (128, 1024, 1024)]


def _backends():
    sm = get_device_sm()
    if 100 <= sm < 110:
        return ["triton", "deep_gemm"]
    if sm == 90:
        return ["triton", "deep_gemm"]
    if sm >= 120:
        return ["triton"]
    return []


def _quantize_fp8_blockwise(w: torch.Tensor, block: int = 128):
    n, k = w.shape
    tiles = w.float().reshape(n // block, block, k // block, block)
    amax = tiles.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = amax / FP8_MAX
    w_fp8 = (tiles / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return w_fp8.reshape(n, k), scale


def _build_layer(n: int, k: int):
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    layer = make_tp1_column_parallel_linear(
        quant_config, n, k, skip_block_quant_check=True
    )
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
    w_fp8, scale_inv = _quantize_fp8_blockwise(w)
    load_linear_weights(layer, weight=w_fp8, weight_scale_inv=scale_inv)
    layer.quant_method.process_weights_after_loading(layer)
    return layer


@unittest.skipIf(get_device_sm() < 90, "FP8 GEMM backends require SM90+")
class TestFp8LinearApplyInto(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    def _check(self, backend: str):
        if backend not in _backends():
            self.skipTest(f"{backend} not in SM{get_device_sm()} backend set")
        torch.manual_seed(7)
        with mock.patch.object(
            fp8_utils, "FP8_GEMM_RUNNER_BACKEND", Fp8GemmRunnerBackend(backend)
        ):
            for m, n, k in SHAPES:
                with self.subTest(backend=backend, shape=(m, n, k)):
                    layer = _build_layer(n, k)
                    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
                    with torch.inference_mode():
                        ref = layer.quant_method.apply(layer, x)
                        out = torch.full(
                            (m, n), float("nan"), device="cuda", dtype=torch.bfloat16
                        )
                        ret = layer.quant_method.apply_into(layer, x, out)
                    self.assertIs(ret, out)
                    torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_triton(self):
        self._check("triton")

    def test_deep_gemm(self):
        self._check("deep_gemm")

    def test_rejects_wrong_shape(self):
        layer = _build_layer(512, 512)
        x = torch.randn((4, 512), device="cuda", dtype=torch.bfloat16)
        bad = torch.empty((4, 256), device="cuda", dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            layer.quant_method.apply_into(layer, x, bad)


if __name__ == "__main__":
    unittest.main()
