"""Correctness tests for the Blackwell SM100 CuTe DSL fused NVFP4 grouped GEMM."""

import unittest

import torch

from sglang.kernels.jit.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.kernels.ops.gemm import nvfp4_grouped_gemm
from sglang.kernels.ops.gemm.cutedsl_nvfp4_group_gemm import (
    _ceil_div,
    permute_nvfp4_scale,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = input_matrix.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    if rows != n_row_blocks * 128 or cols != n_col_blocks * 4:
        padded = torch.zeros(
            (n_row_blocks * 128, n_col_blocks * 4),
            dtype=input_matrix.dtype,
            device=input_matrix.device,
        )
        padded[:rows, :cols] = input_matrix
        input_matrix = padded
    blocks = input_matrix.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def _make_group_inputs(problem_sizes, out_dtype=torch.float16, seed=1234):
    torch.manual_seed(seed)
    abc_tensors = []
    sfasfb_ref = []
    sfasfb_perm = []
    expected = []
    for m, n, k, l in problem_sizes:
        raw_a = (
            torch.randint(255, size=(l, m, k // 2), dtype=torch.uint8, device="cuda")
            & 0b1011_1011
        )
        raw_b = (
            torch.randint(255, size=(l, n, k // 2), dtype=torch.uint8, device="cuda")
            & 0b1011_1011
        )
        a = raw_a.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
        b = raw_b.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
        c = torch.zeros((l, m, n), dtype=out_dtype, device="cuda").permute(1, 2, 0)
        sf_k = _ceil_div(k, 16)
        sfa = (
            torch.rand((l, m, sf_k), dtype=torch.float32, device="cuda")
            .to(torch.float8_e4m3fn)
            .permute(1, 2, 0)
        )
        sfb = (
            torch.rand((l, n, sf_k), dtype=torch.float32, device="cuda")
            .to(torch.float8_e4m3fn)
            .permute(1, 2, 0)
        )
        abc_tensors.append((a, b, c))
        sfasfb_ref.append((sfa, sfb))
        sfasfb_perm.append((permute_nvfp4_scale(sfa), permute_nvfp4_scale(sfb)))
        ref_c = torch._scaled_mm(
            a[:, :, 0],
            b[:, :, 0].transpose(0, 1),
            _to_blocked(sfa[:, :, 0]),
            _to_blocked(sfb[:, :, 0]),
            bias=None,
            out_dtype=torch.float32,
        )
        expected.append(ref_c)
    return abc_tensors, sfasfb_ref, sfasfb_perm, expected


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCuteDSLNvfp4GroupGemm(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if is_hip_runtime() or get_jit_cuda_arch().major != 10:
            raise unittest.SkipTest("NVIDIA SM10x required")

    def test_1cta_and_2cta_groups_fp16_and_bf16(self):
        cases = [
            [(128, 256, 256, 1), (256, 512, 256, 1)],
            [(256, 4096, 7168, 1), (512, 4096, 7168, 1), (384, 3072, 4096, 1)],
            [
                (320, 4096, 512, 1),
                (608, 4096, 512, 1),
                (512, 4096, 512, 1),
                (704, 4096, 512, 1),
            ],
            [
                (640, 4096, 512, 1),
                (1024, 4096, 512, 1),
                (768, 4096, 512, 1),
                (1152, 4096, 512, 1),
            ],
            [(128, 64, 256, 1), (256, 192, 512, 1)],
        ]
        for problem_sizes in cases:
            for out_dtype in (torch.float16, torch.bfloat16):
                with self.subTest(problem_sizes=problem_sizes, out_dtype=out_dtype):
                    abc_tensors, _, sfasfb_perm, expected_fp32 = _make_group_inputs(
                        problem_sizes, out_dtype=out_dtype, seed=555
                    )
                    outputs = nvfp4_grouped_gemm(
                        abc_tensors,
                        sfasfb_perm,
                        problem_sizes,
                        out_dtype=out_dtype,
                    )
                    tol = 1e-2 if out_dtype == torch.bfloat16 else 1e-3
                    for actual, exp_f32 in zip(outputs, expected_fp32):
                        torch.testing.assert_close(
                            actual.squeeze(-1),
                            exp_f32.to(out_dtype),
                            rtol=tol,
                            atol=tol,
                        )

    def test_2d_with_alphas_and_multi_chunk_groups(self):
        problem_sizes = [(128, 256, 256, 1)] * 10
        abc_tensors, sfasfb_ref, _, expected_fp32 = _make_group_inputs(
            problem_sizes, out_dtype=torch.bfloat16, seed=777
        )
        alphas = [
            torch.tensor([0.125 * (i + 1)], dtype=torch.float32, device="cuda")
            for i in range(len(problem_sizes))
        ]
        ab_2d = [(a.squeeze(-1), b.squeeze(-1)) for a, b, _ in abc_tensors]
        sf_2d = [(sfa.squeeze(-1), sfb.squeeze(-1)) for sfa, sfb in sfasfb_ref]

        outputs = nvfp4_grouped_gemm(
            ab_2d,
            sf_2d,
            alphas=alphas,
            out_dtype=torch.bfloat16,
        )
        self.assertEqual(len(outputs), 10)
        for i, (actual, exp_f32) in enumerate(zip(outputs, expected_fp32)):
            exp = (exp_f32 * (0.125 * (i + 1))).to(torch.bfloat16)
            torch.testing.assert_close(actual, exp, rtol=1e-2, atol=1e-2)

    def test_cuda_stream_and_cuda_graph(self):
        problem_sizes = [(256, 512, 256, 1), (256, 256, 512, 1)]
        abc_tensors, _, sfasfb_perm, expected_fp32 = _make_group_inputs(
            problem_sizes, out_dtype=torch.float16, seed=999
        )
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            nvfp4_grouped_gemm(abc_tensors, sfasfb_perm, problem_sizes)
        stream.synchronize()
        for (_, _, c), exp_f32 in zip(abc_tensors, expected_fp32):
            torch.testing.assert_close(
                c.squeeze(-1), exp_f32.to(torch.float16), rtol=1e-3, atol=1e-3
            )

        for _, _, c in abc_tensors:
            c.zero_()
        with torch.cuda.graph(graph, stream=stream):
            nvfp4_grouped_gemm(abc_tensors, sfasfb_perm, problem_sizes)
        stream.synchronize()
        for _, _, c in abc_tensors:
            self.assertEqual(c.abs().sum().item(), 0.0)
        graph.replay()
        stream.synchronize()
        for (_, _, c), exp_f32 in zip(abc_tensors, expected_fp32):
            torch.testing.assert_close(
                c.squeeze(-1), exp_f32.to(torch.float16), rtol=1e-3, atol=1e-3
            )

    def test_persistent_masked_grouped_gemm_1cta_and_2cta(self):
        from flashinfer import scaled_fp4_grouped_quantize
        from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked

        from sglang.kernels.ops.gemm import nvfp4_masked_grouped_gemm

        for m_max, mm_vals in [
            (256, [64, 0, 176, 128]),
            (512, [160, 352, 0, 480]),
            (256, [0, 0, 0, 0]),
        ]:
            with self.subTest(m_max=m_max, mm_vals=mm_vals):
                G, n, k = 4, 256, 512
                torch.manual_seed(42 + m_max)
                x = (
                    torch.randn((G, m_max, k), device="cuda", dtype=torch.bfloat16)
                    * 0.25
                )
                w_bf16 = (
                    torch.randn((G, n, k), device="cuda", dtype=torch.bfloat16) * 0.25
                )
                masked_m = torch.tensor(mm_vals, device="cuda", dtype=torch.int32)
                in_gs = torch.ones((G,), device="cuda", dtype=torch.float32)
                alpha = torch.full((G,), 0.25, device="cuda", dtype=torch.float32)

                aq, aq_sf = scaled_fp4_grouped_quantize(x, masked_m, in_gs)
                wq, w_sf = scaled_fp4_grouped_quantize(
                    w_bf16,
                    torch.full((G,), n, device="cuda", dtype=torch.int32),
                    torch.ones((G,), device="cuda", dtype=torch.float32),
                )
                ref = torch.zeros(
                    (G, m_max, n), device="cuda", dtype=torch.bfloat16
                ).permute(1, 2, 0)
                grouped_gemm_nt_masked(
                    (aq, aq_sf),
                    (wq, w_sf),
                    ref,
                    masked_m,
                    ab_dtype="float4_e2m1fn",
                    sf_dtype="float8_e4m3fn",
                    c_dtype="bfloat16",
                    sf_vec_size=16,
                    alpha=alpha.view(1, 1, G),
                    alpha_dtype="float32",
                )
                actual = nvfp4_masked_grouped_gemm(aq, aq_sf, wq, w_sf, masked_m, alpha)
                for g in range(G):
                    mg = mm_vals[g]
                    if mg > 0:
                        torch.testing.assert_close(
                            actual[:mg, :, g], ref[:mg, :, g], rtol=1e-3, atol=1e-3
                        )

    def test_empty_and_validation_errors(self):
        self.assertEqual(nvfp4_grouped_gemm([], []), [])

        problem_sizes = [(128, 256, 256, 1), (256, 256, 256, 1)]
        abc_tensors, _, sfasfb_perm, expected_fp32 = _make_group_inputs(
            problem_sizes, out_dtype=torch.float16, seed=888
        )
        outputs_fast = nvfp4_grouped_gemm(
            abc_tensors,
            sfasfb_perm,
            problem_sizes,
            swizzled_scales=True,
            pdl_safe=False,
        )
        for actual, exp_f32 in zip(outputs_fast, expected_fp32):
            torch.testing.assert_close(
                actual.squeeze(-1), exp_f32.to(torch.float16), rtol=1e-3, atol=1e-3
            )

        with self.assertRaisesRegex(ValueError, "Expected sfasfb_tensors length"):
            nvfp4_grouped_gemm(abc_tensors, sfasfb_perm[:1])

        with self.assertRaisesRegex(ValueError, "Expected problem_sizes length"):
            nvfp4_grouped_gemm(abc_tensors, sfasfb_perm, problem_sizes[:1])

        with self.assertRaisesRegex(ValueError, "Expected alphas length"):
            nvfp4_grouped_gemm(
                abc_tensors,
                sfasfb_perm,
                alphas=[torch.tensor([1.0], dtype=torch.float32, device="cuda")],
            )

        with self.assertRaisesRegex(ValueError, "Unsupported output dtype"):
            nvfp4_grouped_gemm(abc_tensors, sfasfb_perm, out_dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "does not match tensor shapes"):
            nvfp4_grouped_gemm(
                abc_tensors, sfasfb_perm, [(128, 128, 256, 1), (256, 256, 256, 1)]
            )

        a0, b0, c0 = abc_tensors[0]
        unaligned_out = nvfp4_grouped_gemm([(a0[:80], b0)], [sfasfb_perm[0]])
        torch.testing.assert_close(
            unaligned_out[0].squeeze(-1),
            expected_fp32[0][:80].to(torch.float16),
            rtol=1e-3,
            atol=1e-3,
        )
        empty_out = nvfp4_grouped_gemm([(a0[:0], b0)], [sfasfb_perm[0]])
        self.assertEqual(empty_out[0].shape, (0, 256, 1))

        with self.assertRaisesRegex(ValueError, "unsupported NVFP4 grouped GEMM shape"):
            nvfp4_grouped_gemm([(a0, b0[:100])], [sfasfb_perm[0]])

        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            nvfp4_grouped_gemm([(a0, b0[:, :64, :], c0)], [sfasfb_perm[0]])

        from sglang.kernels.ops.gemm import nvfp4_masked_grouped_gemm

        zero_m_out = nvfp4_masked_grouped_gemm(
            torch.empty((2, 0, 128), device="cuda", dtype=torch.uint8),
            torch.empty((2, 0, 4), device="cuda", dtype=torch.int32),
            torch.empty((2, 128, 128), device="cuda", dtype=torch.uint8),
            torch.empty((2, 128, 4), device="cuda", dtype=torch.int32),
            torch.zeros((2,), device="cuda", dtype=torch.int32),
            torch.ones((2,), device="cuda", dtype=torch.float32),
        )
        self.assertEqual(zero_m_out.shape, (2, 0, 128))
        with self.assertRaisesRegex(
            ValueError, "Unsupported nvfp4_masked_grouped_gemm shape"
        ):
            nvfp4_masked_grouped_gemm(
                torch.empty((2, 128, 64), device="cuda", dtype=torch.uint8),
                torch.empty((2, 128, 2), device="cuda", dtype=torch.int32),
                torch.empty((2, 128, 64), device="cuda", dtype=torch.uint8),
                torch.empty((2, 128, 2), device="cuda", dtype=torch.int32),
                torch.ones((2,), device="cuda", dtype=torch.int32),
                torch.ones((2,), device="cuda", dtype=torch.float32),
            )


if __name__ == "__main__":
    unittest.main()
