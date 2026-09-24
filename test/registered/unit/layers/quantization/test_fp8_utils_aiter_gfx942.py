"""Which AITER blockscale GEMM aiter_w8a8_block_fp8_linear picks on gfx942 — CPU-only."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import fp8_utils
from sglang.test.test_utils import CustomTestCase


class TestAiterGfx942BlockscaleGemm(CustomTestCase):
    def _gemm_used(self, *, gfx942: bool) -> list[str]:
        m, n, k = 4, 256, 256
        used = []

        def gemm(name):
            def op(q_input, weight, x_scale, weight_scale, dtype):
                used.append(name)
                return torch.zeros(q_input.shape[0], weight.shape[0], dtype=dtype)

            return op

        def quant(x, quant_dtype, transpose_scale):
            return x, torch.ones(x.shape[0], x.shape[1] // 128)

        with mock.patch.multiple(
            fp8_utils,
            create=True,
            _use_aiter_bpreshuffle_gfx95=False,
            _use_aiter_gfx95=False,
            _use_aiter_gfx942=gfx942,
            aiter=SimpleNamespace(dtypes=SimpleNamespace(fp8=None)),
            aiter_per1x128_quant=quant,
            ck_gemm_a8w8_blockscale=gemm("ck"),
            triton_gemm_a8w8_blockscale=gemm("triton"),
        ):
            out = fp8_utils.aiter_w8a8_block_fp8_linear(
                torch.randn(m, k, dtype=torch.bfloat16),
                torch.empty(n, k),
                [128, 128],
                torch.ones(n // 128, k // 128),
            )
        self.assertEqual(tuple(out.shape), (m, n))
        return used

    def test_gfx942_uses_ck(self):
        self.assertEqual(self._gemm_used(gfx942=True), ["ck"])

    def test_gfx942_env_selects_triton(self):
        with envs.SGLANG_AITER_GFX942_BLOCKSCALE_USE_TRITON.override(True):
            self.assertEqual(self._gemm_used(gfx942=True), ["triton"])

    def test_other_non_gfx95_arch_keeps_triton(self):
        self.assertEqual(self._gemm_used(gfx942=False), ["triton"])


if __name__ == "__main__":
    unittest.main()
