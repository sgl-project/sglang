"""Correctness tests for the Blackwell SM100 CuTe DSL PDL-safe NVFP4 GEMM."""

import os
import unittest

import torch

from sglang.kernels.jit.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.kernels.ops.gemm import nvfp4_gemm_pdl_safe, try_cutedsl_nvfp4_gemm
from sglang.kernels.ops.gemm.cutedsl_nvfp4_gemm import (
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


def _make_nvfp4_gemm_inputs(m: int, n: int, k: int, l: int, seed: int):
    torch.manual_seed(seed)

    def create_fp4(mn: int) -> torch.Tensor:
        raw = torch.randint(255, size=(l, mn, k // 2), dtype=torch.uint8, device="cuda")
        raw = raw & 0b1011_1011
        return raw.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)

    a = create_fp4(m)
    b = create_fp4(n)
    sf_k = _ceil_div(k, 16)

    def create_sf(mn: int):
        ref = (
            torch.rand((l, mn, sf_k), dtype=torch.float32, device="cuda")
            .to(torch.float8_e4m3fn)
            .permute(1, 2, 0)
        )
        return ref, permute_nvfp4_scale(ref)

    sfa_ref, sfa_perm = create_sf(m)
    sfb_ref, sfb_perm = create_sf(n)
    return a, b, sfa_ref, sfb_ref, sfa_perm, sfb_perm


def _reference_nvfp4_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa_ref: torch.Tensor,
    sfb_ref: torch.Tensor,
    alpha_val: float = 1.0,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    m, _, l = a.shape
    n = b.shape[0]
    ref = torch.empty((l, m, n), dtype=torch.float32, device="cuda").permute(1, 2, 0)
    for l_idx in range(l):
        sa = _to_blocked(sfa_ref[:, :, l_idx])
        sb = _to_blocked(sfb_ref[:, :, l_idx])
        ref[:, :, l_idx] = (
            torch._scaled_mm(
                a[:, :, l_idx],
                b[:, :, l_idx].transpose(0, 1),
                sa,
                sb,
                bias=None,
                out_dtype=torch.float32,
            )
            * alpha_val
        )
    return ref.to(out_dtype)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCuteDSLNvfp4Gemm(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if is_hip_runtime() or get_jit_cuda_arch().major != 10:
            raise unittest.SkipTest("NVIDIA SM10x required")

    def test_1cta_and_2cta_shapes_bf16_and_fp16(self):
        shapes = [
            (128, 64, 256, 1),
            (128, 512, 7168, 1),
            (128, 4096, 7168, 1),
            (128, 7168, 16384, 1),
            (256, 512, 256, 1),
            (256, 4096, 7168, 1),
            (384, 704, 512, 1),
            (384, 512, 512, 2),
            (512, 3072, 4096, 1),
            (512, 5120, 2048, 1),
            (1024, 4096, 2048, 1),
        ]
        for m, n, k, l in shapes:
            for out_dtype in (torch.float16, torch.bfloat16):
                with self.subTest(m=m, n=n, k=k, l=l, out_dtype=out_dtype):
                    a, b, sfa_ref, sfb_ref, sfa_perm, sfb_perm = (
                        _make_nvfp4_gemm_inputs(m, n, k, l, seed=101)
                    )
                    expected = _reference_nvfp4_gemm(
                        a, b, sfa_ref, sfb_ref, alpha_val=1.0, out_dtype=out_dtype
                    )
                    actual = nvfp4_gemm_pdl_safe(
                        a, b, sfa_perm, sfb_perm, out_dtype=out_dtype
                    )
                    tol = 1e-2 if out_dtype == torch.bfloat16 else 1e-3
                    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)

    def test_alpha_2d_and_try_cutedsl_nvfp4_gemm(self):
        m, n, k, l = 256, 1024, 512, 1
        a, b, sfa_ref, sfb_ref, _, _ = _make_nvfp4_gemm_inputs(m, n, k, l, seed=202)
        alpha = torch.tensor([0.25], dtype=torch.float32, device="cuda")
        expected = _reference_nvfp4_gemm(
            a, b, sfa_ref, sfb_ref, alpha_val=0.25, out_dtype=torch.bfloat16
        ).squeeze(-1)

        actual = nvfp4_gemm_pdl_safe(
            a.squeeze(-1),
            b.squeeze(-1),
            sfa_ref.squeeze(-1),
            sfb_ref.squeeze(-1),
            alpha=alpha,
            out_dtype=torch.bfloat16,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

        # Test try_cutedsl_nvfp4_gemm with ModelOpt swizzled 2D scales and M-padding (m=200 -> pad_m=256)
        m_unaligned = 200
        a_u8 = a.squeeze(-1).view(torch.uint8)[:m_u8] if (m_u8 := m_unaligned) else None
        b_u8 = b.squeeze(-1).view(torch.uint8)
        sfa_zero_tail = sfa_ref.clone()
        sfa_zero_tail[m_unaligned:, :, :] = 0
        sfa_swz_2d = (
            sfa_zero_tail.view(2, 4, 32, k // 64, 4, 1)
            .permute(5, 0, 3, 2, 1, 4)
            .contiguous()
            .view(m, k // 16)
        )
        sfb_swz_2d = (
            sfb_ref.view(n // 128, 4, 32, k // 64, 4, 1)
            .permute(5, 0, 3, 2, 1, 4)
            .contiguous()
            .view(n, k // 16)
        )
        old_env = os.environ.get("SGLANG_USE_CUTEDSL_NVFP4_DOWN_GEMM")
        try:
            os.environ["SGLANG_USE_CUTEDSL_NVFP4_DOWN_GEMM"] = "1"
            out_try = try_cutedsl_nvfp4_gemm(
                a_u8,
                b_u8.T,
                sfa_swz_2d,
                sfb_swz_2d.T,
                alpha,
                torch.bfloat16,
                n,
            )
            self.assertIsNotNone(out_try)
            self.assertEqual(tuple(out_try.shape), (m_unaligned, n))
            torch.testing.assert_close(
                out_try, expected[:m_unaligned], rtol=1e-2, atol=1e-2
            )
        finally:
            if old_env is None:
                os.environ.pop("SGLANG_USE_CUTEDSL_NVFP4_DOWN_GEMM", None)
            else:
                os.environ["SGLANG_USE_CUTEDSL_NVFP4_DOWN_GEMM"] = old_env

    def test_cuda_stream_and_cuda_graph(self):
        for m, n, k, l in [
            (128, 512, 256, 1),
            (512, 5120, 512, 1),
            (1024, 4096, 512, 1),
        ]:
            with self.subTest(m=m, n=n, k=k, l=l):
                a, b, sfa_ref, sfb_ref, sfa_perm, sfb_perm = _make_nvfp4_gemm_inputs(
                    m, n, k, l, seed=303
                )
                expected = _reference_nvfp4_gemm(
                    a, b, sfa_ref, sfb_ref, alpha_val=1.0, out_dtype=torch.float16
                )
                out = torch.zeros(
                    (l, m, n), dtype=torch.float16, device="cuda"
                ).permute(1, 2, 0)
                stream = torch.cuda.Stream()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream):
                    nvfp4_gemm_pdl_safe(a, b, sfa_perm, sfb_perm, out)
                stream.synchronize()
                torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)

                out.zero_()
                with torch.cuda.graph(graph, stream=stream):
                    nvfp4_gemm_pdl_safe(a, b, sfa_perm, sfb_perm, out)
                stream.synchronize()
                self.assertEqual(out.abs().sum().item(), 0.0)
                graph.replay()
                stream.synchronize()
                torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)

    def test_input_validation_and_fallback(self):
        a, b, sfa_ref, _, sfa_perm, sfb_perm = _make_nvfp4_gemm_inputs(
            128, 512, 256, 1, seed=404
        )
        with self.assertRaisesRegex(ValueError, "Unsupported output dtype"):
            nvfp4_gemm_pdl_safe(a, b, sfa_perm, sfb_perm, out_dtype=torch.float32)

        bad_out = torch.empty(
            (1, 128, 512), dtype=torch.bfloat16, device="cuda"
        ).permute(1, 2, 0)
        with self.assertRaisesRegex(ValueError, "does not match out_dtype"):
            nvfp4_gemm_pdl_safe(
                a, b, sfa_perm, sfb_perm, out=bad_out, out_dtype=torch.float16
            )

        wrong_shape_out = torch.empty(
            (1, 128, 256), dtype=torch.float16, device="cuda"
        ).permute(1, 2, 0)
        with self.assertRaisesRegex(ValueError, "Expected out shape"):
            nvfp4_gemm_pdl_safe(a, b, sfa_perm, sfb_perm, out=wrong_shape_out)

        non_row_major_out = torch.empty(
            (512, 128, 1), dtype=torch.float16, device="cuda"
        ).transpose(0, 1)
        with self.assertRaisesRegex(ValueError, "row-major"):
            nvfp4_gemm_pdl_safe(a, b, sfa_perm, sfb_perm, out=non_row_major_out)

        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            nvfp4_gemm_pdl_safe(a, b[:, :64, :], sfa_perm, sfb_perm)

        with self.assertRaisesRegex(ValueError, "Expected 2D or 3D input tensor"):
            nvfp4_gemm_pdl_safe(a.unsqueeze(0), b.unsqueeze(0), sfa_perm, sfb_perm)

        non_k_major_a = (
            torch.zeros((128, 128, 1), dtype=torch.uint8, device="cuda")
            .transpose(0, 1)
            .view(torch.float4_e2m1fn_x2)
        )
        with self.assertRaisesRegex(ValueError, "K-major"):
            nvfp4_gemm_pdl_safe(non_k_major_a, b, sfa_perm, sfb_perm)

        with self.assertRaisesRegex(ValueError, "multiple of 128"):
            nvfp4_gemm_pdl_safe(a[:100], b, sfa_perm, sfb_perm)

        with self.assertRaisesRegex(ValueError, "multiple of 64"):
            nvfp4_gemm_pdl_safe(a, b[:100], sfa_perm, sfb_perm)

        with self.assertRaisesRegex(ValueError, "Expected 2D or 3D scale tensor"):
            permute_nvfp4_scale(sfa_ref.unsqueeze(0))

        # Verify try_cutedsl_nvfp4_gemm fallback conditions
        a_2d = a.squeeze(-1).view(torch.uint8)
        b_2d = b.squeeze(-1).view(torch.uint8)
        sfa_2d = sfa_ref.squeeze(-1)
        alpha = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        old_down = os.environ.get("SGLANG_USE_CUTEDSL_NVFP4_DOWN_GEMM")
        old_gemm = os.environ.get("SGLANG_USE_CUTEDSL_NVFP4_GEMM")
        try:
            os.environ["SGLANG_USE_CUTEDSL_NVFP4_DOWN_GEMM"] = "0"
            os.environ["SGLANG_USE_CUTEDSL_NVFP4_GEMM"] = "0"
            self.assertIsNone(
                try_cutedsl_nvfp4_gemm(
                    a_2d, b_2d.T, sfa_2d, sfa_2d.T, alpha, torch.bfloat16, 512
                )
            )
            os.environ["SGLANG_USE_CUTEDSL_NVFP4_GEMM"] = "1"
            self.assertIsNone(
                try_cutedsl_nvfp4_gemm(
                    a_2d[:64], b_2d.T, sfa_2d, sfa_2d.T, alpha, torch.bfloat16, 512
                )
            )
            self.assertIsNone(
                try_cutedsl_nvfp4_gemm(
                    a_2d, b_2d.T, None, sfa_2d.T, alpha, torch.bfloat16, 512
                )
            )
            self.assertIsNone(
                try_cutedsl_nvfp4_gemm(
                    a_2d, b_2d.T, sfa_2d, sfa_2d.T, alpha, torch.float32, 512
                )
            )
        finally:
            if old_down is None:
                os.environ.pop("SGLANG_USE_CUTEDSL_NVFP4_DOWN_GEMM", None)
            else:
                os.environ["SGLANG_USE_CUTEDSL_NVFP4_DOWN_GEMM"] = old_down
            if old_gemm is None:
                os.environ.pop("SGLANG_USE_CUTEDSL_NVFP4_GEMM", None)
            else:
                os.environ["SGLANG_USE_CUTEDSL_NVFP4_GEMM"] = old_gemm

    def test_same_process_grouped_gemm_then_persistent_2cta_layer(self):
        import torch.nn.functional as F

        from sglang.kernels.ops.gemm import (
            nvfp4_dual_gemm_swiglu,
            nvfp4_grouped_gemm,
        )
        from sglang.kernels.ops.gemm.cutedsl_nvfp4_dual_gemm import (
            permute_nvfp4_scale as permute_dual_scale,
        )
        from sglang.kernels.ops.gemm.cutedsl_nvfp4_group_gemm import (
            permute_nvfp4_scale as permute_group_scale,
        )

        def make_fp4(rows: int, cols: int) -> torch.Tensor:
            raw = (
                torch.randint(
                    255, size=(rows, cols // 2), dtype=torch.uint8, device="cuda"
                )
                & 0xBB
            )
            return raw.view(torch.float4_e2m1fn_x2)

        def make_sf(rows: int, cols: int, perm_fn):
            raw = (
                torch.randint(
                    1, 3, (rows, _ceil_div(cols, 16)), dtype=torch.int8, device="cuda"
                ).float()
                * 0.125
            ).to(torch.float8_e4m3fn)
            return perm_fn(raw)

        d_model, d_ff = 7168, 4096
        for m in (256, 1024):
            torch.manual_seed(42 + m)
            ms_up = [
                max(32, ((m0 * m) // 256 // 8) * 8)
                for m0 in (80, 176, 128, 72, 64, 248, 96, 160)
            ]
            abc_up, sf_up = [], []
            for mg in ms_up:
                m_pad = _ceil_div(mg, 128) * 128
                a_pad = make_fp4(m_pad, d_model)
                b_grp = make_fp4(d_ff, d_model)
                c_grp = torch.empty((mg, d_ff), dtype=torch.float16, device="cuda")
                sfa_pad = make_sf(m_pad, d_model, lambda t: t)
                abc_up.append((a_pad[:mg], b_grp, c_grp))
                sf_up.append(
                    (
                        permute_group_scale(sfa_pad[:mg].contiguous()),
                        make_sf(d_ff, d_model, permute_group_scale),
                    )
                )
            ps_up = [(mg, d_ff, d_model, 1) for mg in ms_up]
            for _ in range(5):
                nvfp4_grouped_gemm(abc_up, sf_up, ps_up, swizzled_scales=True)
            torch.cuda.synchronize()

            x = torch.randn((m, d_model), dtype=torch.float16, device="cuda")
            w = torch.ones((d_model,), dtype=torch.float16, device="cuda")
            q = torch.randn((1, 32, m, 128), dtype=torch.float16, device="cuda")
            k = torch.randn((1, 8, m, 128), dtype=torch.float16, device="cuda")
            v = torch.randn_like(k)
            a_qkv, b_qkv = make_fp4(m, d_model), make_fp4(d_ff, d_model)
            sa_qkv = make_sf(m, d_model, permute_nvfp4_scale)
            sb_qkv = make_sf(d_ff, d_model, permute_nvfp4_scale)
            a_o, b_o = make_fp4(m, d_ff), make_fp4(d_model, d_ff)
            sa_o = make_sf(m, d_ff, permute_nvfp4_scale)
            sb_o = make_sf(d_model, d_ff, permute_nvfp4_scale)
            a_mlp = make_fp4(m, d_model)
            b1, b2 = make_fp4(d_ff, d_model), make_fp4(d_ff, d_model)
            sa_mlp = make_sf(m, d_model, permute_dual_scale)
            sb1 = make_sf(d_ff, d_model, permute_dual_scale)
            sb2 = make_sf(d_ff, d_model, permute_dual_scale)
            c_qkv = torch.empty((m, d_ff), dtype=torch.float16, device="cuda")
            c_o = torch.empty((m, d_model), dtype=torch.float16, device="cuda")
            c_mlp = torch.empty((m, d_ff), dtype=torch.float16, device="cuda")
            c_dn = torch.empty((m, d_model), dtype=torch.float16, device="cuda")

            def run_layer():
                F.rms_norm(x, (d_model,), w, 1e-6)
                nvfp4_gemm_pdl_safe(
                    a_qkv, b_qkv, sa_qkv, sb_qkv, out=c_qkv, swizzled_scales=True
                )
                F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
                nvfp4_gemm_pdl_safe(a_o, b_o, sa_o, sb_o, out=c_o, swizzled_scales=True)
                F.rms_norm(x, (d_model,), w, 1e-6)
                nvfp4_dual_gemm_swiglu(
                    a_mlp,
                    b1,
                    b2,
                    sa_mlp,
                    sb1,
                    sb2,
                    out=c_mlp,
                    swizzled_scales=True,
                    enable_pdl=True,
                    pdl_safe=True,
                )
                nvfp4_gemm_pdl_safe(
                    a_o, b_o, sa_o, sb_o, out=c_dn, swizzled_scales=True
                )

            for _ in range(10):
                run_layer()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                run_layer()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                for _ in range(10):
                    run_layer()
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            for out_t in (c_qkv, c_o, c_mlp, c_dn):
                self.assertTrue(torch.isfinite(out_t.float()).all().item())
                self.assertGreater(out_t.float().abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
