"""DeepSeek-V4 wo_a FP8 activation quant for DeepGEMM fp8_einsum.

Covers the wo_a consumer contract of the packed outer-major UE8M0 scale
layout (scale_outer_major + scale_tma_aligned): bit-exactness against the 2D
packed path run per outer slice, large / DSV4-shaped token axes, and the
deep_gemm.fp8_einsum end-to-end check. Generic layout unit tests live in
test_per_token_group_quant_8bit_v2.py.
"""

import importlib
import unittest

import torch

from sglang.jit_kernel.utils import should_run_full_tests
from sglang.srt.utils import ceil_align, get_device_sm
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
            cls.deep_gemm = importlib.import_module("deep_gemm")
        except ImportError as exc:
            raise unittest.SkipTest("deep_gemm is required") from exc

        fp8_kernel = importlib.import_module(
            "sglang.srt.layers.quantization.fp8_kernel"
        )
        fp8_utils = importlib.import_module("sglang.srt.layers.quantization.fp8_utils")
        jit_v2 = importlib.import_module(
            "sglang.jit_kernel.per_token_group_quant_8bit_v2"
        )

        cls.quant = staticmethod(fp8_kernel.sglang_per_token_group_quant_fp8)
        cls.create_output_scale = staticmethod(
            fp8_kernel.create_per_token_group_quant_fp8_output_scale
        )
        cls.fp8_dtype = fp8_kernel.fp8_dtype
        cls.fp8_min = float(fp8_kernel.fp8_min)
        cls.fp8_max = float(fp8_kernel.fp8_max)
        cls.jit_quant_v2 = staticmethod(jit_v2.per_token_group_quant_8bit_v2)
        cls.quant_weight_ue8m0 = staticmethod(fp8_utils.quant_weight_ue8m0)
        cls.transform_scale_ue8m0 = staticmethod(fp8_utils.transform_scale_ue8m0)
        cls.block_quant_dequant = staticmethod(fp8_utils.block_quant_dequant)

    def _quant_packed_outer_major(self, o):
        return self.quant(
            o.contiguous(),
            _GROUP_SIZE,
            scale_ue8m0=True,
            scale_outer_major=True,
            scale_tma_aligned=True,
        )

    def _assert_matches_per_slice_reference(self, o, o_fp8, o_s):
        """Reference: the 2D packed column-major path (the slab layout
        DeepGEMM consumes), run once per outer slice."""
        T, G, D = o.shape
        for g in range(G):
            q_ref = torch.empty((T, D), device=o.device, dtype=self.fp8_dtype)
            s_ref = self.create_output_scale(
                x_shape=(T, D),
                device=o.device,
                group_size=_GROUP_SIZE,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
            self.jit_quant_v2(
                o[:, g].contiguous(),
                q_ref,
                s_ref,
                _GROUP_SIZE,
                1e-10,
                self.fp8_min,
                self.fp8_max,
                scale_ue8m0=True,
            )
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(o_fp8[:, g].view(torch.int8), q_ref.view(torch.int8)),
                f"fp8 codes differ for outer slice {g}",
            )
            self.assertTrue(
                torch.equal(o_s[:, g, :], s_ref), f"scales differ for outer slice {g}"
            )

    def test_packed_outer_major_quant_matches_per_slice_reference(self):
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)

        T, G, D = 9, 5, 384  # odd T (TMA tail) and K=3 (packed-word tail)
        device = torch.device("cuda")
        o = (torch.randn(T, G, D, device=device, dtype=torch.float32) * 0.25).to(
            torch.bfloat16
        )

        o_fp8, o_s = self._quant_packed_outer_major(o)
        scale_inner = ceil_align(D // _GROUP_SIZE, 4) // 4
        aligned_t = ceil_align(T, 4)
        self.assertEqual(o_fp8.shape, (T, G, D))
        self.assertEqual(o_s.shape, (T, G, scale_inner))
        self.assertEqual(o_s.dtype, torch.int32)
        self.assertEqual(o_s.stride(), (1, scale_inner * aligned_t, aligned_t))
        self.assertFalse(o_s.is_contiguous())
        self._assert_matches_per_slice_reference(o, o_fp8, o_s)

    def test_packed_outer_major_quant_large_token_dimension(self):
        torch.manual_seed(2)
        torch.cuda.manual_seed_all(2)

        cases = [
            # >10k live tokens / large decode batch without using much memory.
            (10_001, 1, 128),
            # >10k live tokens with the DeepSeek-V4 wo_a shape: D = 64 heads
            # * 512 head_dim / 8 o_groups = 4096, G = o_groups (attn TP1).
            (10_001, 8, 4096),
        ]
        if should_run_full_tests():
            # Exercises 64-bit token indexing for long-context-sized token
            # axes and per-slice slab offsets with outer index > 0.
            cases.append((900_001, 2, 128))

        device = torch.device("cuda")
        for T, G, D in cases:
            with self.subTest(T=T, G=G, D=D):
                o = (
                    torch.randn(T, G, D, device=device, dtype=torch.float32) * 0.25
                ).to(torch.bfloat16)
                o_fp8, o_s = self._quant_packed_outer_major(o)
                self._assert_matches_per_slice_reference(o, o_fp8, o_s)

    def test_fp8_wo_a_einsum_uses_packed_outer_major_activation_scales(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        cases = [
            (5, 2, 256, 256),  # tiny adversarial case (K-pack tail, odd T)
            # DeepSeek-V4 wo_a decode at attn TP1: D = 4096, R = o_lora_rank.
            (7, 8, 4096, 1024),
        ]
        device = torch.device("cuda")
        for T, G, D, R in cases:
            with self.subTest(T=T, G=G, D=D, R=R):
                token_scale = torch.linspace(0.5, 1.5, T, device=device).view(T, 1, 1)
                # Distinct per-outer-slice magnitudes (2^-2 .. 2^2) so a
                # scale/slice misassociation blows far past the tolerance.
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

                weight_fp8, weight_s_raw = self.quant_weight_ue8m0(
                    weight, weight_block_size=[_GROUP_SIZE, _GROUP_SIZE]
                )
                weight_s = self.transform_scale_ue8m0(weight_s_raw, mn=R)

                def run_einsum(o_fp8, o_s):
                    out = torch.empty(T, G, R, device=device, dtype=torch.bfloat16)
                    self.deep_gemm.fp8_einsum(
                        "bhr,hdr->bhd",
                        (o_fp8, o_s),
                        (weight_fp8, weight_s),
                        out,
                        recipe=(1, 1, _GROUP_SIZE),
                    )
                    return out

                q_packed, s_packed = self._quant_packed_outer_major(o)
                q_fp32, s_fp32 = self.quant(
                    o.contiguous(),
                    _GROUP_SIZE,
                    scale_ue8m0=True,
                    scale_outer_major=True,
                )
                out_packed = run_einsum(q_packed, s_packed)
                out_fp32 = run_einsum(q_fp32, s_fp32)
                torch.cuda.synchronize()

                self.assertTrue(
                    torch.equal(q_packed.view(torch.int8), q_fp32.view(torch.int8)),
                    "fp8 codes differ between packed and fp32 outer-major modes",
                )
                self.assertTrue(
                    torch.equal(out_packed, out_fp32),
                    "einsum outputs differ between scale modes",
                )

                o_dequant = q_packed.float().view(
                    T, G, D // _GROUP_SIZE, _GROUP_SIZE
                ) * s_fp32.unsqueeze(-1)
                weight_dequant = self.block_quant_dequant(
                    weight_fp8,
                    weight_s_raw,
                    block_size=[_GROUP_SIZE, _GROUP_SIZE],
                    dtype=torch.float32,
                )
                ref = torch.einsum(
                    "tgd,grd->tgr", o_dequant.view(T, G, D), weight_dequant.float()
                ).to(torch.bfloat16)
                # rtol accommodates bf16-output accumulation differences over
                # D=4096; a scale misassociation would be off by 2x-16x.
                torch.testing.assert_close(out_packed, ref, atol=1e-1, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
