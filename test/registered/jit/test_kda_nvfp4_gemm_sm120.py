import unittest

import torch
from flashinfer import fp4_quantize, mm_fp4

from sglang.kernels.ops.gemm import (
    can_dispatch_kda_nvfp4_gemm,
    can_use_kda_nvfp4_gemm,
    kda_nvfp4_gemm,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-small")

_QWEN35_DECODE_SHAPES = (
    (2560, 18432),
    (9216, 2560),
    (4096, 24576),
    (12288, 4096),
)
_QWEN38_DECODE_SHAPES = (
    (5120, 34816),
    (17408, 5120),
    (5120, 248320),
)


def _make_inputs(m: int, k: int, n: int):
    x_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    input_global_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    x, x_sf = fp4_quantize(x_bf16, input_global_scale)
    weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda").T
    weight_sf = torch.ones((n, k // 16), dtype=torch.float8_e4m3fn, device="cuda").T
    alpha = torch.tensor(0.03125, dtype=torch.float32, device="cuda")
    return x, weight, x_sf, weight_sf, alpha


def _reference(args):
    return mm_fp4(*args, torch.bfloat16, backend="auto")


class TestKdaNvfp4GemmSm120(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        if torch.cuda.get_device_capability() != (12, 0):
            raise unittest.SkipTest("requires an SM120 GPU")

    @classmethod
    def tearDownClass(cls):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_decode_matches_flashinfer(self):
        torch.manual_seed(0)
        for k, n in _QWEN35_DECODE_SHAPES:
            for m in (1, 2, 4, 8, 9):
                args = _make_inputs(m, k, n)
                expected = _reference(args)
                actual = kda_nvfp4_gemm(*args, torch.bfloat16, n)
                torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
                del args, expected, actual
            torch.cuda.empty_cache()

        for k, n in _QWEN38_DECODE_SHAPES:
            for m in (1, 9):
                args = _make_inputs(m, k, n)
                expected = _reference(args)
                actual = kda_nvfp4_gemm(*args, torch.bfloat16, n)
                torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
                del args, expected, actual
            torch.cuda.empty_cache()

    def test_large_down_matches_flashinfer(self):
        args = _make_inputs(4369, 17408, 5120)
        expected = _reference(args)
        actual = kda_nvfp4_gemm(*args, torch.bfloat16, 5120)
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)

    def test_cuda_graph_replay(self):
        args = _make_inputs(9, 17408, 5120)
        expected = _reference(args)
        kda_nvfp4_gemm(*args, torch.bfloat16, 5120)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = kda_nvfp4_gemm(*args, torch.bfloat16, 5120)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)

    def test_modelopt_dispatch_and_shape_guard(self):
        from sglang.srt.layers.quantization.modelopt_quant import fp4_gemm

        args = _make_inputs(9, 17408, 5120)
        self.assertTrue(can_use_kda_nvfp4_gemm(*args, torch.bfloat16, 5120))
        self.assertTrue(can_dispatch_kda_nvfp4_gemm(*args, torch.bfloat16, 5120))
        args_m8 = (args[0][:8], args[1], args[2], args[3], args[4])
        self.assertFalse(can_use_kda_nvfp4_gemm(*args_m8, torch.bfloat16, 5120))
        args_m1 = (args[0][:1], args[1], args[2], args[3], args[4])
        self.assertTrue(can_use_kda_nvfp4_gemm(*args_m1, torch.bfloat16, 5120))
        self.assertFalse(can_dispatch_kda_nvfp4_gemm(*args_m1, torch.bfloat16, 5120))

        expected = _reference(args)
        with envs.SGLANG_ENABLE_KDA_NVFP4_GEMM.override(True):
            actual = fp4_gemm(*args, torch.bfloat16, 5120)
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)

        for k, n in _QWEN35_DECODE_SHAPES:
            for m in (1, 2, 4, 8):
                qwen35_args = _make_inputs(m, k, n)
                self.assertTrue(
                    can_dispatch_kda_nvfp4_gemm(*qwen35_args, torch.bfloat16, n)
                )
            qwen35_m9_args = _make_inputs(9, k, n)
            self.assertTrue(can_use_kda_nvfp4_gemm(*qwen35_m9_args, torch.bfloat16, n))
            self.assertFalse(
                can_dispatch_kda_nvfp4_gemm(*qwen35_m9_args, torch.bfloat16, n)
            )

        unsupported_prefill = _make_inputs(4369, 9216, 2560)
        self.assertFalse(
            can_use_kda_nvfp4_gemm(*unsupported_prefill, torch.bfloat16, 2560)
        )


if __name__ == "__main__":
    unittest.main()
