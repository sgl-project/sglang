"""Correctness tests for the Blackwell SM100 CuTe DSL NVFP4 dual GEMM + SwiGLU."""

import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.jit.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.kernels.ops.gemm import nvfp4_dual_gemm_swiglu
from sglang.kernels.ops.gemm.cutedsl_nvfp4_dual_gemm import (
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


def _make_nvfp4_dual_gemm_inputs(m: int, n: int, k: int, l: int, seed: int):
    torch.manual_seed(seed)

    def create_fp4(mn: int) -> torch.Tensor:
        raw = torch.randint(255, size=(l, mn, k // 2), dtype=torch.uint8, device="cuda")
        raw = raw & 0b1011_1011
        return raw.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)

    a = create_fp4(m)
    b1 = create_fp4(n)
    b2 = create_fp4(n)

    sf_k = _ceil_div(k, 16)

    def create_sf(mn: int):
        ref = (
            torch.rand((l, mn, sf_k), dtype=torch.float32, device="cuda")
            .to(torch.float8_e4m3fn)
            .permute(1, 2, 0)
        )
        return ref, permute_nvfp4_scale(ref)

    sfa_ref, sfa_perm = create_sf(m)
    sfb1_ref, sfb1_perm = create_sf(n)
    sfb2_ref, sfb2_perm = create_sf(n)
    return a, b1, b2, sfa_ref, sfb1_ref, sfb2_ref, sfa_perm, sfb1_perm, sfb2_perm


def _reference_nvfp4_dual_gemm(
    a: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    sfa_ref: torch.Tensor,
    sfb1_ref: torch.Tensor,
    sfb2_ref: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    m, _, l = a.shape
    n = b1.shape[0]
    ref1 = torch.empty((l, m, n), dtype=torch.float32, device="cuda").permute(1, 2, 0)
    ref2 = torch.empty((l, m, n), dtype=torch.float32, device="cuda").permute(1, 2, 0)
    for l_idx in range(l):
        sa = _to_blocked(sfa_ref[:, :, l_idx])
        sb1 = _to_blocked(sfb1_ref[:, :, l_idx])
        sb2 = _to_blocked(sfb2_ref[:, :, l_idx])
        ref1[:, :, l_idx] = torch._scaled_mm(
            a[:, :, l_idx],
            b1[:, :, l_idx].transpose(0, 1),
            sa,
            sb1,
            bias=None,
            out_dtype=torch.float32,
        )
        ref2[:, :, l_idx] = torch._scaled_mm(
            a[:, :, l_idx],
            b2[:, :, l_idx].transpose(0, 1),
            sa,
            sb2,
            bias=None,
            out_dtype=torch.float32,
        )
    return (F.silu(ref1) * ref2).to(out_dtype)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCuteDSLNvfp4DualGemm(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if is_hip_runtime() or get_jit_cuda_arch().major != 10:
            raise unittest.SkipTest("NVIDIA SM10x required")

    def test_2cta_and_1cta_shapes_fp16(self):
        shapes = [
            (128, 64, 256, 1),
            (128, 512, 256, 1),
            (256, 192, 512, 1),
            (256, 512, 256, 1),
            (256, 512, 256, 2),
            (1536, 512, 7168, 1),
            (3072, 1024, 1536, 1),
            (4608, 384, 7168, 1),
            (256, 4096, 7168, 1),
            (512, 4096, 7168, 1),
            (256, 3072, 4096, 1),
            (512, 3072, 7168, 1),
        ]
        for m, n, k, l in shapes:
            with self.subTest(m=m, n=n, k=k, l=l):
                (
                    a,
                    b1,
                    b2,
                    sfa_ref,
                    sfb1_ref,
                    sfb2_ref,
                    sfa_perm,
                    sfb1_perm,
                    sfb2_perm,
                ) = _make_nvfp4_dual_gemm_inputs(m, n, k, l, seed=1111)
                expected = _reference_nvfp4_dual_gemm(
                    a, b1, b2, sfa_ref, sfb1_ref, sfb2_ref, torch.float16
                )
                actual = nvfp4_dual_gemm_swiglu(
                    a, b1, b2, sfa_perm, sfb1_perm, sfb2_perm
                )
                torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_2d_unpermuted_scales_and_bf16(self):
        m, n, k, l = 256, 1024, 512, 1
        a, b1, b2, sfa_ref, sfb1_ref, sfb2_ref, _, _, _ = _make_nvfp4_dual_gemm_inputs(
            m, n, k, l, seed=2026
        )
        expected = _reference_nvfp4_dual_gemm(
            a, b1, b2, sfa_ref, sfb1_ref, sfb2_ref, torch.bfloat16
        ).squeeze(-1)
        actual = nvfp4_dual_gemm_swiglu(
            a.squeeze(-1),
            b1.squeeze(-1),
            b2.squeeze(-1),
            sfa_ref.squeeze(-1),
            sfb1_ref.squeeze(-1),
            sfb2_ref.squeeze(-1),
            out_dtype=torch.bfloat16,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

        prealloc_out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        ret = nvfp4_dual_gemm_swiglu(
            a.squeeze(-1),
            b1.squeeze(-1),
            b2.squeeze(-1),
            sfa_ref.squeeze(-1),
            sfb1_ref.squeeze(-1),
            sfb2_ref.squeeze(-1),
            out=prealloc_out,
        )
        self.assertIs(ret, prealloc_out)
        torch.testing.assert_close(prealloc_out, expected, rtol=1e-2, atol=1e-2)

        actual_no_pdl = nvfp4_dual_gemm_swiglu(
            a,
            b1,
            b2,
            sfa_ref,
            sfb1_ref,
            sfb2_ref,
            out_dtype=torch.bfloat16,
            enable_pdl=False,
        )
        torch.testing.assert_close(
            actual_no_pdl.squeeze(-1), expected, rtol=1e-2, atol=1e-2
        )

        actual_pdl_safe = nvfp4_dual_gemm_swiglu(
            a,
            b1,
            b2,
            sfa_ref,
            sfb1_ref,
            sfb2_ref,
            out_dtype=torch.bfloat16,
            enable_pdl=True,
            pdl_safe=True,
        )
        torch.testing.assert_close(
            actual_pdl_safe.squeeze(-1), expected, rtol=1e-2, atol=1e-2
        )

        alpha = torch.tensor([0.375], dtype=torch.float32, device="cuda")
        ref1 = (
            torch._scaled_mm(
                a[:, :, 0],
                b1[:, :, 0].transpose(0, 1),
                _to_blocked(sfa_ref[:, :, 0]),
                _to_blocked(sfb1_ref[:, :, 0]),
                out_dtype=torch.float32,
            )
            * alpha.item()
        )
        ref2 = (
            torch._scaled_mm(
                a[:, :, 0],
                b2[:, :, 0].transpose(0, 1),
                _to_blocked(sfa_ref[:, :, 0]),
                _to_blocked(sfb2_ref[:, :, 0]),
                out_dtype=torch.float32,
            )
            * alpha.item()
        )
        expected_alpha = (F.silu(ref1) * ref2).to(torch.bfloat16)
        actual_alpha = nvfp4_dual_gemm_swiglu(
            a.squeeze(-1),
            b1.squeeze(-1),
            b2.squeeze(-1),
            sfa_ref.squeeze(-1),
            sfb1_ref.squeeze(-1),
            sfb2_ref.squeeze(-1),
            alpha=alpha,
            out_dtype=torch.bfloat16,
            enable_pdl=True,
            pdl_safe=True,
            hybrid_rcp_mod_override=0,
            nr_steps_override=2,
        )
        torch.testing.assert_close(actual_alpha, expected_alpha, rtol=1e-2, atol=1e-2)

    def test_cuda_stream_and_cuda_graph_capture(self):
        m, n, k, l = 256, 512, 256, 1
        (
            a,
            b1,
            b2,
            sfa_ref,
            sfb1_ref,
            sfb2_ref,
            sfa_perm,
            sfb1_perm,
            sfb2_perm,
        ) = _make_nvfp4_dual_gemm_inputs(m, n, k, l, seed=4242)
        expected = _reference_nvfp4_dual_gemm(
            a, b1, b2, sfa_ref, sfb1_ref, sfb2_ref, torch.float16
        )
        out = torch.zeros((l, m, n), dtype=torch.float16, device="cuda").permute(
            1, 2, 0
        )

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            nvfp4_dual_gemm_swiglu(a, b1, b2, sfa_perm, sfb1_perm, sfb2_perm, out)
        stream.synchronize()
        torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)

        out.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            nvfp4_dual_gemm_swiglu(a, b1, b2, sfa_perm, sfb1_perm, sfb2_perm, out)
        stream.synchronize()
        self.assertEqual(out.abs().sum().item(), 0.0)
        graph.replay()
        stream.synchronize()
        torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)

    def test_input_validation_errors(self):
        a, b1, b2, sfa_ref, _, _, sfa_perm, sfb1_perm, sfb2_perm = (
            _make_nvfp4_dual_gemm_inputs(256, 512, 256, 1, seed=7)
        )
        with self.assertRaisesRegex(ValueError, "Unsupported output dtype"):
            nvfp4_dual_gemm_swiglu(
                a, b1, b2, sfa_perm, sfb1_perm, sfb2_perm, out_dtype=torch.float32
            )

        bad_out = torch.empty(
            (1, 256, 512), dtype=torch.bfloat16, device="cuda"
        ).permute(1, 2, 0)
        with self.assertRaisesRegex(ValueError, "does not match out_dtype"):
            nvfp4_dual_gemm_swiglu(
                a,
                b1,
                b2,
                sfa_perm,
                sfb1_perm,
                sfb2_perm,
                out=bad_out,
                out_dtype=torch.float16,
            )

        wrong_shape_out = torch.empty(
            (1, 256, 256), dtype=torch.float16, device="cuda"
        ).permute(1, 2, 0)
        with self.assertRaisesRegex(ValueError, "Expected out shape"):
            nvfp4_dual_gemm_swiglu(
                a, b1, b2, sfa_perm, sfb1_perm, sfb2_perm, out=wrong_shape_out
            )

        non_row_major_out = torch.empty(
            (512, 256, 1), dtype=torch.float16, device="cuda"
        ).transpose(0, 1)
        with self.assertRaisesRegex(ValueError, "row-major"):
            nvfp4_dual_gemm_swiglu(
                a, b1, b2, sfa_perm, sfb1_perm, sfb2_perm, out=non_row_major_out
            )

        with self.assertRaisesRegex(ValueError, "matching shapes"):
            nvfp4_dual_gemm_swiglu(a, b1[:256], b2, sfa_perm, sfb1_perm, sfb2_perm)

        with self.assertRaisesRegex(ValueError, "K/L dimension mismatch"):
            nvfp4_dual_gemm_swiglu(
                a, b1[:, :64, :], b2[:, :64, :], sfa_perm, sfb1_perm, sfb2_perm
            )

        with self.assertRaisesRegex(ValueError, "Expected 2D or 3D input tensor"):
            nvfp4_dual_gemm_swiglu(
                a.unsqueeze(0),
                b1.unsqueeze(0),
                b2.unsqueeze(0),
                sfa_perm,
                sfb1_perm,
                sfb2_perm,
            )

        non_k_major_a = (
            torch.zeros((128, 256, 1), dtype=torch.uint8, device="cuda")
            .transpose(0, 1)
            .view(torch.float4_e2m1fn_x2)
        )
        with self.assertRaisesRegex(ValueError, "K-major"):
            nvfp4_dual_gemm_swiglu(
                non_k_major_a, b1, b2, sfa_perm, sfb1_perm, sfb2_perm
            )

        with self.assertRaisesRegex(ValueError, "multiple of 128"):
            nvfp4_dual_gemm_swiglu(a[:100], b1, b2, sfa_perm, sfb1_perm, sfb2_perm)

        with self.assertRaisesRegex(ValueError, "multiple of 64"):
            nvfp4_dual_gemm_swiglu(
                a, b1[:100], b2[:100], sfa_perm, sfb1_perm, sfb2_perm
            )

        with self.assertRaisesRegex(ValueError, "Expected 2D or 3D scale tensor"):
            permute_nvfp4_scale(sfa_ref.unsqueeze(0))

    def test_dual_gemm_nvfp4_layer_integration(self):
        import os

        from flashinfer import fp4_quantize, mm_fp4

        from sglang.kernels.ops.gemm import try_cutedsl_nvfp4_gemm
        from sglang.srt.layers.dual_gemm import DualGemm

        hidden_size = 512
        intermediate_size = 1024

        class ModelOptFp4LinearMethod:
            quant_mode = "w4a4"

        class _QuantCfg:
            is_awq = False

        class _DummyLinear(torch.nn.Module):
            def __init__(self, in_features, out_partition_sizes, seed):
                super().__init__()
                torch.manual_seed(seed)
                out_features = sum(out_partition_sizes)
                self.output_partition_sizes = out_partition_sizes
                self.params_dtype = torch.bfloat16
                self.quant_config = _QuantCfg()
                self.quant_method = ModelOptFp4LinearMethod()
                self.weights_padding_cols = 0
                self.weight = torch.nn.Parameter(
                    torch.randint(
                        0,
                        255,
                        (out_features, in_features // 2),
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    & 0b1011_1011,
                    requires_grad=False,
                )
                raw_scale = torch.rand(
                    (out_features, in_features // 16),
                    dtype=torch.float32,
                    device="cuda",
                ).to(torch.float8_e4m3fn)
                swz = (
                    raw_scale.view(
                        out_features // 128, 4, 32, (in_features // 16) // 4, 4
                    )
                    .permute(0, 3, 2, 1, 4)
                    .contiguous()
                    .view(out_features, in_features // 16)
                )
                self.weight_scale_interleaved = torch.nn.Parameter(
                    swz, requires_grad=False
                )
                self.input_scale_inv = torch.nn.Parameter(
                    torch.tensor(2.0, dtype=torch.float32, device="cuda"),
                    requires_grad=False,
                )
                self.alpha = torch.nn.Parameter(
                    torch.tensor(0.125, dtype=torch.float32, device="cuda"),
                    requires_grad=False,
                )

            def forward_ref(self, x: torch.Tensor) -> torch.Tensor:
                x_fp4, x_sf = fp4_quantize(x, self.input_scale_inv, backend="cute-dsl")
                return mm_fp4(
                    x_fp4,
                    self.weight.T,
                    x_sf,
                    self.weight_scale_interleaved.T,
                    self.alpha,
                    x.dtype,
                    backend="cute-dsl",
                )

            def forward_cutedsl(self, x: torch.Tensor) -> torch.Tensor:
                x_fp4, x_sf = fp4_quantize(x, self.input_scale_inv, backend="cute-dsl")
                out = try_cutedsl_nvfp4_gemm(
                    x_fp4,
                    self.weight.T,
                    x_sf,
                    self.weight_scale_interleaved.T,
                    self.alpha,
                    x.dtype,
                    self.weight.shape[0],
                )
                assert out is not None
                return out

        gate_up = _DummyLinear(
            hidden_size, [intermediate_size, intermediate_size], seed=2026
        )
        down = _DummyLinear(intermediate_size, [hidden_size], seed=2027)
        dg = DualGemm(gate_up, down, hidden_size)
        self.assertTrue(dg.nvfp4_eligible)

        old_dual = os.environ.get("SGLANG_USE_CUTEDSL_NVFP4_DUAL_GEMM")
        old_down = os.environ.get("SGLANG_USE_CUTEDSL_NVFP4_DOWN_GEMM")
        try:
            for m in (256, 300):
                x = (
                    torch.randn((m, hidden_size), dtype=torch.bfloat16, device="cuda")
                    * 0.1
                )
                os.environ["SGLANG_USE_CUTEDSL_NVFP4_DUAL_GEMM"] = "0"
                os.environ["SGLANG_USE_CUTEDSL_NVFP4_DOWN_GEMM"] = "0"
                self.assertFalse(dg.can_run(x, gate_up))
                gu_ref = gate_up.forward_ref(x)
                g_ref, u_ref = gu_ref.chunk(2, dim=-1)
                mid_ref = (F.silu(g_ref.float()) * u_ref.float()).to(torch.bfloat16)
                out_ref = down.forward_ref(mid_ref)

                os.environ["SGLANG_USE_CUTEDSL_NVFP4_DUAL_GEMM"] = "1"
                os.environ["SGLANG_USE_CUTEDSL_NVFP4_DOWN_GEMM"] = "1"
                self.assertTrue(dg.can_run(x, gate_up))
                mid_cute = dg(x, gate_up)
                out_down_same_in = down.forward_cutedsl(mid_ref)
                out_cute = down.forward_cutedsl(mid_cute)
                cos_mid = F.cosine_similarity(
                    mid_ref.float().flatten(), mid_cute.float().flatten(), dim=0
                ).item()
                cos_down_same = F.cosine_similarity(
                    out_ref.float().flatten(), out_down_same_in.float().flatten(), dim=0
                ).item()
                cos_out = F.cosine_similarity(
                    out_ref.float().flatten(), out_cute.float().flatten(), dim=0
                ).item()
                self.assertGreater(cos_mid, 0.9999)
                self.assertGreater(cos_down_same, 0.99999)
                self.assertGreater(cos_out, 0.9995)
        finally:
            if old_dual is None:
                os.environ.pop("SGLANG_USE_CUTEDSL_NVFP4_DUAL_GEMM", None)
            else:
                os.environ["SGLANG_USE_CUTEDSL_NVFP4_DUAL_GEMM"] = old_dual
            if old_down is None:
                os.environ.pop("SGLANG_USE_CUTEDSL_NVFP4_DOWN_GEMM", None)
            else:
                os.environ["SGLANG_USE_CUTEDSL_NVFP4_DOWN_GEMM"] = old_down


if __name__ == "__main__":
    unittest.main()
