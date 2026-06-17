"""Arm64 MoE test.

Tests fused_experts_cpu with W8A8 INT8 quantization, which is supported
on Arm64 via aarch64/moe.cpp (PR #16045). Additional quantization paths
(BF16, INT4) will be added here as Arm kernels land.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="per-commit-cpu-arm64")

import itertools
import math
import os
import platform
import sys
import unittest

import torch

# Add parent dir (test/srt/cpu/) to path for utils import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sglang.srt.layers.amx_utils import CPUQuantMethod
from sglang.test.test_utils import CustomTestCase

kernel = torch.ops.sgl_kernel
IS_ARM64 = platform.machine().lower() in ("aarch64", "arm64")

torch.manual_seed(128)

from utils import (
    precision,
    torch_w8a8_per_column_fused_moe,
)


class TestFusedExpertsInt8(CustomTestCase):
    M = [1, 6, 32, 64]
    N = [256, 512]
    K = [256, 512]
    E = [8]
    topk = [4]

    def _int8_moe(self, M, N, K, E, topk):
        dtype = torch.bfloat16
        # Arm64 INT8 MoE currently uses the unpacked, out-of-place path.
        prepack = not IS_ARM64

        int8_factor_for_scale = 1e-2
        int8_max = 127
        int8_min = -128

        a = torch.randn((M, K), dtype=dtype) / math.sqrt(K)

        w1_fp32 = (torch.rand((E, 2 * N, K), dtype=torch.float32) - 0.5) * 2
        w1 = (w1_fp32 * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)

        w2_fp32 = (torch.rand((E, K, N), dtype=torch.float32) - 0.5) * 2
        w2 = (w2_fp32 * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)

        w1_s = torch.rand(E, 2 * N, device=w1_fp32.device) * int8_factor_for_scale
        w2_s = torch.rand(E, K, device=w2_fp32.device) * int8_factor_for_scale

        score = torch.randn((M, E), dtype=dtype)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)

        ref_out = torch_w8a8_per_column_fused_moe(
            a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, topk
        )

        inplace = not IS_ARM64
        packed_w1 = kernel.convert_weight_packed(w1) if prepack else w1
        packed_w2 = kernel.convert_weight_packed(w2) if prepack else w2
        out = kernel.fused_experts_cpu(
            a,
            packed_w1,
            packed_w2,
            topk_weight,
            topk_ids.to(torch.int32),
            inplace,
            CPUQuantMethod.INT8_W8A8,
            w1_s,
            w2_s,
            None,
            None,
            None,
            prepack,
        )

        atol = rtol = precision[ref_out.dtype]
        if IS_ARM64:
            atol = rtol = 0.03
        elif M > 35:
            atol = rtol = 0.02
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    def test_int8_moe(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.E,
            self.topk,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                E=params[3],
                topk=params[4],
            ):
                self._int8_moe(*params)


if __name__ == "__main__":
    unittest.main()
