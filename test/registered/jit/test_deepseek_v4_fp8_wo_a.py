"""DeepSeek-V4 wo_a FP8 activation quant for DeepGEMM fp8_einsum.

Covers the dedicated DSV4 wo_a quant helper: bit-exact FP8/scales against the
ordinary flat UE8M0 quant values, group-major scale storage, large / DSV4-shaped
token axes, and the DeepGEMM fp8_einsum consumer contract.
"""

import unittest

import torch

import sglang.jit_kernel.dsv4.fp8_wo_a as fp8_wo_a_module
from sglang.jit_kernel.dsv4 import sglang_per_token_group_quant_fp8_dsv4_woa
from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    quant_weight_ue8m0,
    transform_scale_ue8m0,
)
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_GROUP_SIZE = 128


class TestDeepSeekV4FP8WoA(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("Test requires CUDA SM 100 or higher")

        try:
            import deep_gemm
        except ImportError as exc:
            raise unittest.SkipTest("deep_gemm is required") from exc

        cls.deep_gemm = deep_gemm

    def _flat_reference(self, o):
        T, G, D = o.shape
        q_ref, s_ref = sglang_per_token_group_quant_fp8(
            o.contiguous().view(T * G, D),
            _GROUP_SIZE,
            scale_ue8m0=True,
        )
        return q_ref.view(T, G, D), s_ref.view(T, G, D // _GROUP_SIZE)

    def _strided_tgd(self, T, G, D, dtype, device):
        storage = (
            torch.randn(T, G + 1, D, device=device, dtype=torch.float32) * 0.25
        ).to(dtype)
        o = storage[:, 1:, :]
        self.assertFalse(o.is_contiguous())
        self.assertEqual(o.stride(-1), 1)
        return o

    def _assert_matches_flat_reference(self, o, o_fp8, o_s):
        T, G, D = o.shape
        q_ref, s_ref = self._flat_reference(o)
        torch.cuda.synchronize()

        self.assertEqual(o_fp8.shape, (T, G, D))
        self.assertEqual(o_fp8.dtype, fp8_dtype)
        self.assertEqual(o_s.shape, (T, G, D // _GROUP_SIZE))
        self.assertEqual(o_s.dtype, torch.float32)
        self.assertEqual(o_s.stride(), (D // _GROUP_SIZE, T * (D // _GROUP_SIZE), 1))
        self.assertTrue(o_s[:, 0, :].is_contiguous())
        self.assertTrue(
            torch.equal(o_fp8.view(torch.int8), q_ref.view(torch.int8)),
            "fp8 codes differ",
        )
        self.assertTrue(torch.equal(o_s, s_ref), "scales differ")

    def test_dsv4_woa_quant_matches_flat_reference(self):
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)

        device = torch.device("cuda")
        for dtype, T, G, D in [
            (torch.bfloat16, 9, 5, 384),
            (torch.float16, 7, 3, 512),
        ]:
            with self.subTest(dtype=dtype, T=T, G=G, D=D):
                o = (
                    torch.randn(T, G, D, device=device, dtype=torch.float32) * 0.25
                ).to(dtype)
                o_fp8, o_s = sglang_per_token_group_quant_fp8_dsv4_woa(o)
                self._assert_matches_flat_reference(o, o_fp8, o_s)

                o = self._strided_tgd(T, G, D, dtype, device)
                o_fp8, o_s = sglang_per_token_group_quant_fp8_dsv4_woa(o)
                self._assert_matches_flat_reference(o, o_fp8, o_s)

    def test_dsv4_woa_quant_large_token_dimension(self):
        torch.manual_seed(2)
        torch.cuda.manual_seed_all(2)

        cases = [
            (10_001, 1, 128, False),
            (10_001, 8, 4096, False),
            (300_000, 1, 128, True),
        ]

        device = torch.device("cuda")
        for T, G, D, strided in cases:
            with self.subTest(T=T, G=G, D=D, strided=strided):
                if strided:
                    o = self._strided_tgd(T, G, D, torch.bfloat16, device)
                else:
                    o = (
                        torch.randn(T, G, D, device=device, dtype=torch.float32) * 0.25
                    ).to(torch.bfloat16)
                o_fp8, o_s = sglang_per_token_group_quant_fp8_dsv4_woa(o)
                self._assert_matches_flat_reference(o, o_fp8, o_s)

    def test_dsv4_woa_quant_uses_dedicated_jit(self):
        torch.manual_seed(3)
        torch.cuda.manual_seed_all(3)

        original_jit_module = fp8_wo_a_module._jit_module
        jit_module_calls = 0

        def wrapped_jit_module(*args, **kwargs):
            nonlocal jit_module_calls
            jit_module_calls += 1
            return original_jit_module(*args, **kwargs)

        fp8_wo_a_module._jit_module = wrapped_jit_module
        try:
            o = self._strided_tgd(3, 2, 256, torch.bfloat16, "cuda")
            sglang_per_token_group_quant_fp8_dsv4_woa(o)
            torch.cuda.synchronize()
            self.assertGreater(jit_module_calls, 0)
        finally:
            fp8_wo_a_module._jit_module = original_jit_module

    def test_fp8_wo_a_einsum_uses_group_major_activation_scales(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        cases = [
            (5, 2, 256, 256),
            (7, 8, 4096, 1024),
        ]
        device = torch.device("cuda")
        for T, G, D, R in cases:
            with self.subTest(T=T, G=G, D=D, R=R):
                token_scale = torch.linspace(0.5, 1.5, T, device=device).view(T, 1, 1)
                group_scale = torch.pow(
                    2.0, (torch.arange(G, device=device).float() % 5) - 2.0
                ).view(1, G, 1)
                o = (
                    torch.randn(T, G, D, device=device, dtype=torch.float32)
                    * token_scale
                    * group_scale
                    * 0.2
                ).to(torch.bfloat16)
                weight = (
                    torch.randn(G, R, D, device=device, dtype=torch.float32) * 0.2
                ).to(torch.bfloat16)

                weight_fp8, weight_s_raw = quant_weight_ue8m0(
                    weight, weight_block_size=[_GROUP_SIZE, _GROUP_SIZE]
                )
                weight_s = transform_scale_ue8m0(weight_s_raw, mn=R)

                q_dsv4, s_dsv4 = sglang_per_token_group_quant_fp8_dsv4_woa(o)
                out = torch.empty(T, G, R, device=device, dtype=torch.bfloat16)
                self.deep_gemm.fp8_einsum(
                    "bhr,hdr->bhd",
                    (q_dsv4, s_dsv4),
                    (weight_fp8, weight_s),
                    out,
                    recipe=(1, 1, _GROUP_SIZE),
                )
                torch.cuda.synchronize()

                o_dequant = q_dsv4.float().view(
                    T, G, D // _GROUP_SIZE, _GROUP_SIZE
                ) * s_dsv4.unsqueeze(-1)
                weight_dequant = block_quant_dequant(
                    weight_fp8,
                    weight_s_raw,
                    block_size=[_GROUP_SIZE, _GROUP_SIZE],
                    dtype=torch.float32,
                )
                ref = torch.einsum(
                    "tgd,grd->tgr", o_dequant.view(T, G, D), weight_dequant.float()
                ).to(torch.bfloat16)
                torch.testing.assert_close(out, ref, atol=1e-1, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
