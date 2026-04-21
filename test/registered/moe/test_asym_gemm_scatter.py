"""
Tests for the AsymGEMM Option B scatter epilogue.

Compares three paths on the same inputs:
    R  = fp32 PyTorch reference (ground truth)
    D  = deep_gemm masked FP8 GEMM  (baseline; dense [E, max_m, N] output)
    A  = asym_gemm fp8 + Option B scatter  (compact [T, N] output)

Validations:
    |D - R|_max ≤ tol_deep_gemm   — DeepGEMM uses TMEM FP32 accumulation
    |A - R|_max ≤ tol_asym_gemm   — AsymGEMM has a slightly larger bias due
                                    to HBM bf16 accumulation staging
    |A - D|_max ≤ tol_cross       — the AsymGEMM-vs-DeepGEMM gap must stay
                                    bounded; anything beyond this indicates
                                    a scatter epilogue bug
"""

import unittest

import torch

try:
    import deep_gemm

    HAS_DEEP_GEMM = True
except ImportError:
    HAS_DEEP_GEMM = False

try:
    import asym_gemm

    HAS_ASYM_GEMM = hasattr(asym_gemm, "m_grouped_fp8_asym_gemm_nt_masked_scatter")
except ImportError:
    HAS_ASYM_GEMM = False

from sglang.test.test_utils import CustomTestCase


def _quantize_fp8_per_token_group(x_bf16, group_size=128):
    """Per-token-group fp8 quant matching what the production MoE path uses."""
    from sglang.srt.layers.asym_gemm_wrapper.configurer import ASYMGEMM_SCALE_UE8M0
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    return sglang_per_token_group_quant_fp8(
        x_bf16.contiguous(),
        group_size,
        column_major_scales=ASYMGEMM_SCALE_UE8M0,
        scale_tma_aligned=ASYMGEMM_SCALE_UE8M0,
        scale_ue8m0=ASYMGEMM_SCALE_UE8M0,
    )


def _make_random_masked_input(
    num_experts, max_m, k, *, mean_fill, device="cuda", seed=0
):
    """
    Create realistic masked-GEMM inputs.

    Each expert gets a random token count in [1, max_m], nudged toward
    `mean_fill * max_m`. Padding rows are filled with NaN so any stray
    read will blow up immediately.
    """
    gen = torch.Generator(device=device).manual_seed(seed)

    masked_m = torch.randint(
        1, max_m + 1, (num_experts,), generator=gen, device=device, dtype=torch.int32
    )
    masked_m = torch.clamp(
        (masked_m.float() * mean_fill).to(torch.int32), min=1, max=max_m
    )

    x = torch.randn(
        num_experts, max_m, k, generator=gen, device=device, dtype=torch.bfloat16
    )
    for e in range(num_experts):
        x[e, int(masked_m[e]) :].fill_(float("nan"))
    return x, masked_m


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
@unittest.skipIf(not HAS_ASYM_GEMM, "asym_gemm scatter API not built")
class TestAsymGemmScatterVsDeepGemm(CustomTestCase):
    NUM_EXPERTS = 32
    MAX_M = 128
    K = 2048
    N = 2048

    # Empirically calibrated on H100. Bigger atol for asym_gemm because
    # the HBM bf16 accumulation stage adds one extra rounding vs DeepGEMM.
    TOL_DEEP_GEMM = dict(atol=2e-2, rtol=1e-2)
    TOL_ASYM_GEMM = dict(atol=5e-2, rtol=2e-2)
    TOL_CROSS = dict(atol=4e-2, rtol=1.5e-2)

    def setUp(self):
        torch.manual_seed(42)
        self.device = "cuda"

    # -------------------------------------------------------------------
    # path helpers
    # -------------------------------------------------------------------

    def _fp32_reference(self, x_bf16, w_bf16, masked_m):
        """Dense bf16 × bf16 matmul in fp32 — ground truth."""
        E = x_bf16.shape[0]
        N = w_bf16.shape[1]
        out = torch.zeros(
            E, x_bf16.shape[1], N, device=self.device, dtype=torch.float32
        )
        x32 = x_bf16.float()
        w32 = w_bf16.float()
        for e in range(E):
            m = int(masked_m[e])
            if m == 0:
                continue
            out[e, :m] = x32[e, :m] @ w32[e].t()
        return out

    def _run_deep_gemm(self, x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m):
        E, max_m, _ = x_fp8.shape
        N = w_fp8.shape[1]
        out = torch.empty(
            (E, max_m, N), device=self.device, dtype=torch.bfloat16
        )
        deep_gemm.m_grouped_fp8_gemm_nt_masked(
            (x_fp8, x_scale),
            (w_fp8, w_scale),
            out,
            masked_m,
            expected_m=expected_m,
        )
        return out

    def _run_asym_gemm_scatter(
        self, x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m
    ):
        """AsymGEMM + Option B scatter epilogue."""
        from sglang.srt.layers.asym_gemm_wrapper.entrypoint import (
            grouped_gemm_nt_f8f8bf16_masked_scatter,
        )
        from sglang.srt.layers.moe.ep_moe.kernels import (
            build_scatter_map_from_masked_m,
        )
        from sglang.srt.layers.moe.moe_runner.asym_gemm import (
            build_offsets_experts_from_masked_m,
            compute_contiguous_offsets_from_masked_m,
        )

        E, max_m, _ = x_fp8.shape
        N = w_fp8.shape[1]

        contiguous_offsets, total_active = (
            compute_contiguous_offsets_from_masked_m(masked_m)
        )
        scatter_map = build_scatter_map_from_masked_m(
            masked_m, contiguous_offsets, max_m
        )
        offsets, experts, list_size = build_offsets_experts_from_masked_m(
            masked_m, E
        )

        out_compact = torch.empty(
            (max(total_active, 1), N),
            device=self.device,
            dtype=torch.bfloat16,
        )
        if total_active > 0:
            grouped_gemm_nt_f8f8bf16_masked_scatter(
                (x_fp8, x_scale),
                (w_fp8, w_scale),
                out_compact,
                offsets,
                experts,
                scatter_map,
                list_size,
                expected_m,
                max_m,
            )
        return out_compact, contiguous_offsets, total_active

    def _compact_to_dense(self, out_compact, contiguous_offsets, masked_m, ref_shape):
        """Re-expand compact output to dense for element-wise comparison."""
        out_dense = torch.full(
            ref_shape, float("nan"), device=self.device, dtype=torch.bfloat16
        )
        for e in range(ref_shape[0]):
            m = int(masked_m[e])
            if m == 0:
                continue
            start = int(contiguous_offsets[e])
            out_dense[e, :m] = out_compact[start : start + m]
        return out_dense

    # -------------------------------------------------------------------
    # tests
    # -------------------------------------------------------------------

    @unittest.skipIf(not HAS_DEEP_GEMM, "deep_gemm not installed")
    def test_scatter_matches_deep_gemm_within_bias(self):
        E, max_m, K, N = self.NUM_EXPERTS, self.MAX_M, self.K, self.N

        x_bf16, masked_m = _make_random_masked_input(
            E, max_m, K, mean_fill=0.6, device=self.device
        )
        w_bf16 = (
            torch.randn(E, N, K, device=self.device, dtype=torch.bfloat16) * 0.05
        )
        expected_m = int(masked_m.max().item())

        ref_fp32 = self._fp32_reference(x_bf16, w_bf16, masked_m)

        # Quantise once; feed the same fp8 inputs to both backends.
        x_clean = torch.nan_to_num(x_bf16, nan=0.0)
        x_fp8, x_scale = _quantize_fp8_per_token_group(
            x_clean.reshape(E * max_m, K)
        )
        x_fp8 = x_fp8.reshape(E, max_m, K)
        x_scale = x_scale.reshape(E, max_m, -1)

        w_fp8, w_scale = _quantize_fp8_per_token_group(w_bf16.reshape(E * N, K))
        w_fp8 = w_fp8.reshape(E, N, K)
        w_scale = w_scale.reshape(E, N, -1)

        out_deep = self._run_deep_gemm(
            x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m
        )
        out_asym_compact, contig_offsets, total_active = (
            self._run_asym_gemm_scatter(
                x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m
            )
        )
        out_asym_dense = self._compact_to_dense(
            out_asym_compact, contig_offsets, masked_m, out_deep.shape
        )

        def _check(name, got_bf16, tol):
            got = got_bf16.float()
            finite = ~torch.isnan(got)
            diff = (got[finite] - ref_fp32[finite]).abs()
            ref_mag = ref_fp32[finite].abs().clamp_min(1e-3)
            max_abs = diff.max().item()
            max_rel = (diff / ref_mag).max().item()
            print(
                f"[{name}]  max|Δ|={max_abs:.3e}  max|Δ/ref|={max_rel:.3e}"
            )
            self.assertLessEqual(max_abs, tol["atol"], f"{name} atol")
            self.assertLessEqual(max_rel, tol["rtol"], f"{name} rtol")

        _check("deep_gemm  vs fp32", out_deep, self.TOL_DEEP_GEMM)
        _check("asym_gemm  vs fp32", out_asym_dense, self.TOL_ASYM_GEMM)

        # Cross comparison (cancels the shared fp8 quant noise)
        valid = ~torch.isnan(out_asym_dense)
        cross = (
            out_asym_dense[valid].float() - out_deep[valid].float()
        ).abs()
        print(
            f"[asym_gemm vs deep_gemm]  "
            f"max|Δ|={cross.max().item():.3e}  "
            f"mean|Δ|={cross.mean().item():.3e}"
        )
        self.assertLessEqual(cross.max().item(), self.TOL_CROSS["atol"])

    def test_scatter_memory_footprint(self):
        """
        Regression test: the scatter path must never allocate an
        [E, max_m, N] tensor of bf16. We use the CUDA caching allocator
        high-water mark as a proxy.
        """
        E, max_m, K, N = self.NUM_EXPERTS, self.MAX_M, self.K, self.N
        dense_bytes = E * max_m * N * 2  # bf16 [E, max_m, N] would cost this

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        x_bf16, masked_m = _make_random_masked_input(
            E, max_m, K, mean_fill=0.5, device=self.device
        )
        w_bf16 = (
            torch.randn(E, N, K, device=self.device, dtype=torch.bfloat16) * 0.05
        )
        x_clean = torch.nan_to_num(x_bf16, nan=0.0)
        x_fp8, x_scale = _quantize_fp8_per_token_group(
            x_clean.reshape(E * max_m, K)
        )
        x_fp8 = x_fp8.reshape(E, max_m, K)
        x_scale = x_scale.reshape(E, max_m, -1)
        w_fp8, w_scale = _quantize_fp8_per_token_group(w_bf16.reshape(E * N, K))
        w_fp8 = w_fp8.reshape(E, N, K)
        w_scale = w_scale.reshape(E, N, -1)
        expected_m = int(masked_m.max().item())

        baseline_peak = torch.cuda.max_memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

        out_compact, _, total_active = self._run_asym_gemm_scatter(
            x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m
        )
        torch.cuda.synchronize()

        scatter_peak = torch.cuda.max_memory_allocated(self.device) - baseline_peak
        compact_bytes = max(total_active, 1) * N * 2  # bf16

        # The scatter path's incremental allocation must be dominated by
        # the compact output plus small scratch (<5 MB for scatter_map
        # and offsets). It must not approach the dense allocation.
        self.assertLess(
            scatter_peak,
            compact_bytes + dense_bytes // 2,
            f"scatter peak {scatter_peak / 1e6:.1f} MB is too close to "
            f"dense baseline {dense_bytes / 1e6:.1f} MB",
        )

    def test_scatter_handles_all_padding(self):
        """Expert with masked_m == 0 must produce no writes and no crashes."""
        E, max_m, K, N = self.NUM_EXPERTS, self.MAX_M, self.K, self.N

        x_bf16, masked_m = _make_random_masked_input(
            E, max_m, K, mean_fill=0.5, device=self.device
        )
        # Zero out a few experts
        masked_m[0] = 0
        masked_m[5] = 0
        masked_m[-1] = 0
        expected_m = int(masked_m.max().item())

        w_bf16 = (
            torch.randn(E, N, K, device=self.device, dtype=torch.bfloat16) * 0.05
        )

        x_clean = torch.nan_to_num(x_bf16, nan=0.0)
        x_fp8, x_scale = _quantize_fp8_per_token_group(
            x_clean.reshape(E * max_m, K)
        )
        x_fp8 = x_fp8.reshape(E, max_m, K)
        x_scale = x_scale.reshape(E, max_m, -1)
        w_fp8, w_scale = _quantize_fp8_per_token_group(w_bf16.reshape(E * N, K))
        w_fp8 = w_fp8.reshape(E, N, K)
        w_scale = w_scale.reshape(E, N, -1)

        out_compact, _, total_active = self._run_asym_gemm_scatter(
            x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m
        )
        torch.cuda.synchronize()

        # Just check it runs to completion with a sensible compact size.
        expected_total = int(masked_m.sum().item())
        # total_active is block-128 padded but must be ≥ actual sum.
        self.assertGreaterEqual(total_active, expected_total)


if __name__ == "__main__":
    unittest.main()
