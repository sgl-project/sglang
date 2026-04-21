"""
Option A — Chunked Expert Processing integration + consistency tests.

Three ground truths are stacked for every numerical test:

    fp32_ref  — dense bf16 matmul accumulated in fp32 (exact ground truth)
    deep_gemm — masked grouped FP8 GEMM via deep_gemm library (TMEM fp32 acc.)
    chunked   — Option A path: _run_masked_gemm_down_chunked  (AsymGEMM bf16)

Tolerance hierarchy (from most to least precise):
    |deep_gemm  - fp32|  ≤  TOL_DEEP  (2e-2)
    |chunked    - fp32|  ≤  TOL_ASYM  (5e-2)   AsymGEMM uses bf16 HBM staging
    |chunked - deep_gemm| ≤  TOL_CROSS (4e-2)  shared fp8 quant noise cancels

Test matrix
-----------
test_chunk32_matches_baseline              compact.md §4 Step 4 requirement
test_chunk32_consistent_with_deep_gemm    cross-compare vs DeepGEMM (primary)
test_chunk32_vs_fp32_reference            absolute error vs fp32 ground truth
test_chunk32_output_shape_and_dtype       shape / dtype contract
test_chunk32_various_sizes_vs_deep_gemm   chunk_size ∈ {1, 8, 32, E} all match
test_chunk32_empty_experts                masked_m[e] == 0 handled correctly
test_chunk32_all_full_experts             all experts at max capacity
"""

import unittest

import torch

try:
    import asym_gemm as _asym_lib  # noqa: F401

    HAS_ASYM_GEMM = True
except ImportError:
    HAS_ASYM_GEMM = False

try:
    import deep_gemm

    HAS_DEEP_GEMM = True
except ImportError:
    HAS_DEEP_GEMM = False

try:
    from sglang.srt.layers.asym_gemm_wrapper.configurer import ASYMGEMM_SCALE_UE8M0

    HAS_SGLANG = True
except ImportError:
    HAS_SGLANG = False
    ASYMGEMM_SCALE_UE8M0 = False

from sglang.test.test_utils import CustomTestCase


# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------


def _quant_fp8(x_bf16: torch.Tensor, group_size: int = 128):
    """
    Per-token-group FP8 quantisation, matching the production masked-GEMM path.

    Args:
        x_bf16: [..., K]  bfloat16 — any leading shape, quantised along last dim
    Returns:
        x_fp8:  same shape, float8_e4m3fn
        scale:  [..., K // group_size]  in UE8M0 or float32 depending on config
    """
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    leading = x_bf16.shape[:-1]
    K = x_bf16.shape[-1]
    x_fp8_flat, scale_flat = sglang_per_token_group_quant_fp8(
        x_bf16.reshape(-1, K).contiguous(),
        group_size,
        column_major_scales=ASYMGEMM_SCALE_UE8M0,
        scale_tma_aligned=ASYMGEMM_SCALE_UE8M0,
        scale_ue8m0=ASYMGEMM_SCALE_UE8M0,
    )
    return x_fp8_flat.reshape(*leading, K), scale_flat.reshape(*leading, -1)


def _make_raw_inputs(num_experts, max_m, n_int, k_hidden,
                     mean_fill=0.6, seed=0, device="cuda"):
    """
    Create bf16 activations and weights before quantisation.

    Returns:
        act_bf16  [E, max_m, n_int]  bfloat16  (padding rows zeroed)
        w_bf16    [E, k_hidden, n_int]  bfloat16
        masked_m  [E]  int32
        expected_m  int
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    masked_m = torch.randint(
        1, max_m + 1, (num_experts,), generator=gen, device=device, dtype=torch.int32
    )
    masked_m = torch.clamp(
        (masked_m.float() * mean_fill).to(torch.int32), min=1, max=max_m
    )
    expected_m = int(masked_m.max().item())

    act = torch.randn(
        num_experts, max_m, n_int, generator=gen, device=device, dtype=torch.bfloat16
    )
    for e in range(num_experts):
        act[e, int(masked_m[e]):].zero_()  # zero padding rows

    w = (
        torch.randn(
            num_experts, k_hidden, n_int, generator=gen, device=device, dtype=torch.bfloat16
        )
        * 0.05
    )
    return act, w, masked_m, expected_m


# ---------------------------------------------------------------------------
# Reference paths
# ---------------------------------------------------------------------------


def _fp32_reference(act_bf16, w_bf16, masked_m):
    """
    Dense bf16 × bf16 matmul in fp32 accumulation — absolute ground truth.

    Returns dense [E, max_m, K] float32 (padding rows are zero).
    """
    E, max_m, N = act_bf16.shape
    K = w_bf16.shape[1]
    out = torch.zeros(E, max_m, K, device=act_bf16.device, dtype=torch.float32)
    a32 = act_bf16.float()
    w32 = w_bf16.float()
    for e in range(E):
        m = int(masked_m[e])
        if m > 0:
            out[e, :m] = a32[e, :m] @ w32[e].t()
    return out


def _run_deep_gemm(x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m):
    """
    DeepGEMM masked grouped FP8 GEMM.

    Uses TMEM fp32 accumulation — more accurate than AsymGEMM's bf16 staging.
    Returns dense [E, max_m, K] bfloat16.
    """
    E, max_m, _ = x_fp8.shape
    K = w_fp8.shape[1]
    out = torch.empty((E, max_m, K), device=x_fp8.device, dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        (x_fp8, x_scale),
        (w_fp8, w_scale),
        out,
        masked_m,
        expected_m=expected_m,
    )
    return out


def _run_asym_baseline(x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m,
                       device="cuda"):
    """
    Standard AsymGEMM path (CHUNK_SIZE=0): dense GEMM + post-GEMM compaction.

    Returns compact [total_active, K] bfloat16.
    """
    from sglang.srt.layers import asym_gemm_wrapper
    from sglang.srt.layers.moe.ep_moe.kernels import compact_masked_to_contiguous
    from sglang.srt.layers.moe.moe_runner.asym_gemm import (
        build_offsets_experts_from_masked_m,
        compute_contiguous_offsets_from_masked_m,
    )

    E, max_m, _ = x_fp8.shape
    K = w_fp8.shape[1]

    offsets, experts, list_size = build_offsets_experts_from_masked_m(masked_m, E)
    dense = torch.empty((E, max_m, K), device=device, dtype=torch.bfloat16)
    asym_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
        (x_fp8, x_scale), (w_fp8, w_scale),
        dense, masked_m, expected_m, offsets, experts, list_size,
    )

    cont_offsets, total_active = compute_contiguous_offsets_from_masked_m(masked_m)
    compact = torch.empty((max(total_active, 1), K), device=device, dtype=torch.bfloat16)
    if total_active > 0:
        compact_masked_to_contiguous(dense, compact, masked_m, cont_offsets, max_m)
    return compact[:total_active], cont_offsets, total_active


def _run_chunked(x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m,
                 chunk_size, device="cuda"):
    """
    Option A chunked path via _run_masked_gemm_down_chunked.

    Allocates one small [chunk_size, max_m, K] buffer, reuses it per iteration,
    compacts each chunk's results into the global output.
    Returns compact [total_active, K] bfloat16.
    """
    from sglang.srt.layers.moe.moe_runner.asym_gemm import AsymGemmRunnerCore
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

    E, max_m, _ = x_fp8.shape
    total = int(masked_m.sum().item())

    # Build minimal src2dst / topk_ids matching the standard dispatch layout.
    topk_ids = torch.zeros(total, device=device, dtype=torch.int32)
    src2dst = torch.zeros(total, device=device, dtype=torch.int32)
    pos = 0
    for e in range(E):
        for t in range(int(masked_m[e])):
            topk_ids[pos] = e
            src2dst[pos] = e * max_m + t
            pos += 1

    running_state = {
        "hidden_states_device": device,
        "src2dst": src2dst,
        "topk_ids": topk_ids,
        "down_gemm_overlap_args": None,
    }

    cfg = MoeRunnerConfig(
        activation="silu", is_gated=True, top_k=1,
        num_local_experts=E, routed_scaling_factor=None,
    )
    runner = AsymGemmRunnerCore.__new__(AsymGemmRunnerCore)
    runner.config = cfg

    return runner._run_masked_gemm_down_chunked(
        x_fp8, x_scale, w_fp8, w_scale,
        masked_m, expected_m, max_m, running_state, chunk_size=chunk_size,
    )


def _compact_dense(dense, masked_m, device="cuda"):
    """
    Compact a dense [E, max_m, K] tensor to [total_active, K] using the
    same block-aligned offsets as the production path.  Used to bring
    DeepGEMM dense output into the same layout as the Option A compact output.
    """
    from sglang.srt.layers.moe.ep_moe.kernels import compact_masked_to_contiguous
    from sglang.srt.layers.moe.moe_runner.asym_gemm import (
        compute_contiguous_offsets_from_masked_m,
    )

    E, max_m, K = dense.shape
    cont_offsets, total_active = compute_contiguous_offsets_from_masked_m(masked_m)
    out = torch.empty((total_active, K), device=device, dtype=dense.dtype)
    if total_active > 0:
        compact_masked_to_contiguous(dense, out, masked_m, cont_offsets, max_m)
    return out, cont_offsets, total_active


# ---------------------------------------------------------------------------
# Numerical reporting helper
# ---------------------------------------------------------------------------


def _report(tag, got, ref, tol_atol, tol_rtol=None):
    """
    Print per-test numerical statistics and return (max_abs, max_rel).
    Raises AssertionError if atol or rtol is exceeded.
    """
    diff = (got.float() - ref.float()).abs()
    ref_mag = ref.float().abs().clamp_min(1e-4)
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = (diff / ref_mag).max().item()
    print(
        f"  [{tag}]  max|Δ|={max_abs:.3e}  mean|Δ|={mean_abs:.3e}"
        + (f"  max|Δ/ref|={max_rel:.3e}" if tol_rtol else "")
    )
    assert max_abs <= tol_atol, (
        f"[{tag}] atol exceeded: {max_abs:.3e} > {tol_atol}"
    )
    if tol_rtol is not None:
        assert max_rel <= tol_rtol, (
            f"[{tag}] rtol exceeded: {max_rel:.3e} > {tol_rtol}"
        )
    return max_abs, max_rel


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

_SKIP_NO_CUDA = unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
_SKIP_NO_ASYM = unittest.skipIf(not HAS_ASYM_GEMM, "asym_gemm library not installed")
_SKIP_NO_SGLANG = unittest.skipIf(not HAS_SGLANG, "sglang not importable")
_SKIP_NO_UE8M0 = unittest.skipIf(
    not ASYMGEMM_SCALE_UE8M0, "Chunked path requires ASYMGEMM_SCALE_UE8M0=True"
)
_SKIP_NO_DEEP = unittest.skipIf(not HAS_DEEP_GEMM, "deep_gemm library not installed")


@_SKIP_NO_CUDA
@_SKIP_NO_ASYM
@_SKIP_NO_SGLANG
@_SKIP_NO_UE8M0
class TestAsymGemmMoeOptionA(CustomTestCase):
    """
    Integration and consistency tests for Option A — Chunked Expert Processing.

    Numerical chain:   fp32_ref  ←  deep_gemm  ←  Option-A-chunked
    """

    # Dimensions — small enough to run on any GPU, large enough for statistical
    # significance.  N_INT=512 ≫ group_size=128 so scale values are well-sampled.
    NUM_EXPERTS = 32
    MAX_M = 128
    N_INT = 512
    K_HIDDEN = 512

    # Tolerances (calibrated on H100 against known DeepGEMM/AsymGEMM bias)
    TOL_DEEP = 2e-2   # DeepGEMM vs fp32  (TMEM fp32 acc → tight)
    TOL_ASYM = 5e-2   # AsymGEMM vs fp32  (bf16 HBM staging → looser)
    TOL_CROSS = 4e-2  # AsymGEMM vs DeepGEMM (shared fp8 noise cancels → mid)
    TOL_BASELINE = 1e-2  # chunked vs same-kernel baseline (should be exact)

    def setUp(self):
        self.device = "cuda"
        torch.manual_seed(42)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _inputs(self, seed=0, mean_fill=0.6):
        act_bf16, w_bf16, masked_m, expected_m = _make_raw_inputs(
            self.NUM_EXPERTS, self.MAX_M, self.N_INT, self.K_HIDDEN,
            mean_fill=mean_fill, seed=seed, device=self.device,
        )
        x_fp8, x_scale = _quant_fp8(act_bf16)
        w_fp8, w_scale = _quant_fp8(w_bf16)
        return act_bf16, w_bf16, x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m

    # ------------------------------------------------------------------
    # 1. compact.md §4 Step 4 — chunked vs same-kernel baseline
    # ------------------------------------------------------------------

    def test_chunk32_matches_baseline(self):
        """
        Option A (chunk_size=32) matches the standard single-GEMM + compact
        baseline produced by the same AsymGEMM kernel.

        This is the primary assertion from compact.md Step 4:
            torch.allclose(output_chunked, output_baseline, atol=1e-2)

        Because both paths call grouped_gemm_nt_f8f8bf16_masked with identical
        fp8 inputs, the only difference is the compaction order; results must
        be bit-identical or very close (rounding in compact kernel is the only
        source of divergence).
        """
        _, _, x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m = self._inputs(seed=0)

        baseline, _, _ = _run_asym_baseline(
            x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m, device=self.device,
        )
        chunked = _run_chunked(
            x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m,
            chunk_size=32, device=self.device,
        )

        self.assertEqual(chunked.shape, baseline.shape, "shape mismatch")
        print("\n  [chunk32 vs baseline]")
        _report("chunked vs baseline", chunked, baseline,
                tol_atol=self.TOL_BASELINE)

    # ------------------------------------------------------------------
    # 2. Consistency with DeepGEMM (independent kernel, same fp8 inputs)
    # ------------------------------------------------------------------

    @_SKIP_NO_DEEP
    def test_chunk32_consistent_with_deep_gemm(self):
        """
        Option A chunked output is consistent with DeepGEMM running on the
        same fp8-quantised inputs.

        DeepGEMM uses TMEM fp32 accumulation; AsymGEMM uses bf16 HBM staging.
        The cross-tolerance (TOL_CROSS=4e-2) is wider than the DeepGEMM-vs-fp32
        tolerance but tighter than the sum of both individual tolerances — shared
        fp8 quantisation noise largely cancels in the difference.

        Layout bridge: DeepGEMM returns dense [E, max_m, K]; we compact it with
        the same block-aligned offsets to obtain [total_active, K] for a
        row-for-row comparison against the chunked compact output.
        """
        _, _, x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m = self._inputs(seed=1)

        deep_dense = _run_deep_gemm(
            x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m
        )
        deep_compact, _, total_active = _compact_dense(deep_dense, masked_m, self.device)

        chunked = _run_chunked(
            x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m,
            chunk_size=32, device=self.device,
        )

        self.assertEqual(chunked.shape[0], total_active, "total_active mismatch")
        self.assertEqual(chunked.shape, deep_compact.shape, "shape mismatch")
        print("\n  [chunk32 vs DeepGEMM]")
        _report("chunked vs deep_gemm", chunked, deep_compact, tol_atol=self.TOL_CROSS)

    # ------------------------------------------------------------------
    # 3. Absolute error vs fp32 ground truth
    # ------------------------------------------------------------------

    @_SKIP_NO_DEEP
    def test_chunk32_vs_fp32_reference(self):
        """
        Both DeepGEMM and Option A chunked must stay within their respective
        tolerances against the fp32 matmul ground truth.

        Prints the full numerical picture to simplify tolerance calibration:

            [deep_gemm  vs fp32]  max|Δ|=…  mean|Δ|=…  max|Δ/ref|=…
            [chunked    vs fp32]  max|Δ|=…  mean|Δ|=…  max|Δ/ref|=…
            [chunked vs deep_gemm] max|Δ|=…  mean|Δ|=…
        """
        act_bf16, w_bf16, x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m = (
            self._inputs(seed=2)
        )

        # fp32 ground truth — valid rows only (padding is zero, skip in comparison)
        fp32_dense = _fp32_reference(act_bf16, w_bf16, masked_m)

        # DeepGEMM dense output → compact
        deep_dense = _run_deep_gemm(
            x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m
        )
        deep_compact, cont_offsets, total_active = _compact_dense(
            deep_dense, masked_m, self.device
        )

        # fp32 reference compacted with the same offsets
        fp32_compact, _, _ = _compact_dense(
            fp32_dense.bfloat16(), masked_m, self.device
        )

        # Option A chunked
        chunked = _run_chunked(
            x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m,
            chunk_size=32, device=self.device,
        )

        print("\n  [fp32 reference comparison]")
        _report("deep_gemm  vs fp32", deep_compact, fp32_compact,
                tol_atol=self.TOL_DEEP, tol_rtol=1e-2)
        _report("chunked    vs fp32", chunked, fp32_compact,
                tol_atol=self.TOL_ASYM, tol_rtol=2e-2)
        _report("chunked vs deep_gemm", chunked, deep_compact,
                tol_atol=self.TOL_CROSS)

    # ------------------------------------------------------------------
    # 4. Shape and dtype contract
    # ------------------------------------------------------------------

    def test_chunk32_output_shape_and_dtype(self):
        """Chunked output is [total_active, K] bfloat16."""
        from sglang.srt.layers.moe.moe_runner.asym_gemm import (
            compute_contiguous_offsets_from_masked_m,
        )

        _, _, x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m = self._inputs(seed=3)
        _, total_active = compute_contiguous_offsets_from_masked_m(masked_m)

        out = _run_chunked(
            x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m,
            chunk_size=32, device=self.device,
        )

        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[0], total_active)
        self.assertEqual(out.shape[1], self.K_HIDDEN)

    # ------------------------------------------------------------------
    # 5. Various chunk sizes all match DeepGEMM
    # ------------------------------------------------------------------

    @_SKIP_NO_DEEP
    def test_chunk32_various_sizes_vs_deep_gemm(self):
        """
        chunk_size ∈ {1, 8, 32, NUM_EXPERTS} all produce output that is
        consistent with DeepGEMM, regardless of how the experts are partitioned.
        """
        _, _, x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m = self._inputs(seed=4)

        deep_dense = _run_deep_gemm(
            x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m
        )
        deep_compact, _, _ = _compact_dense(deep_dense, masked_m, self.device)

        print("\n  [various chunk sizes vs DeepGEMM]")
        for cs in [1, 8, 32, self.NUM_EXPERTS]:
            out = _run_chunked(
                x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m,
                chunk_size=cs, device=self.device,
            )
            self.assertEqual(out.shape, deep_compact.shape,
                             f"chunk_size={cs}: shape mismatch")
            _report(f"chunked({cs}) vs deep_gemm", out, deep_compact,
                    tol_atol=self.TOL_CROSS)

    # ------------------------------------------------------------------
    # 6. Empty experts
    # ------------------------------------------------------------------

    @_SKIP_NO_DEEP
    def test_chunk32_empty_experts(self):
        """
        Experts with masked_m == 0 produce no output rows and no NaN values.
        Remaining valid rows are consistent with DeepGEMM.
        """
        _, _, x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m = self._inputs(seed=5)
        masked_m[0] = 0
        masked_m[7] = 0
        masked_m[-1] = 0
        expected_m = int(masked_m.max().item())

        deep_dense = _run_deep_gemm(
            x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m
        )
        deep_compact, _, total_active = _compact_dense(deep_dense, masked_m, self.device)

        out = _run_chunked(
            x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m,
            chunk_size=32, device=self.device,
        )

        self.assertEqual(out.shape[0], total_active, "total_active mismatch")
        self.assertFalse(torch.isnan(out).any(), "NaN in chunked output")
        print("\n  [empty experts vs DeepGEMM]")
        _report("chunked(empty) vs deep_gemm", out, deep_compact, tol_atol=self.TOL_CROSS)

    # ------------------------------------------------------------------
    # 7. All experts at maximum capacity
    # ------------------------------------------------------------------

    @_SKIP_NO_DEEP
    def test_chunk32_all_full_experts(self):
        """
        All experts at max_m tokens (no padding at all).
        Verifies the corner case where every chunk is fully packed.
        """
        E, max_m = self.NUM_EXPERTS, self.MAX_M
        device = self.device

        masked_m = torch.full((E,), max_m, device=device, dtype=torch.int32)
        expected_m = max_m

        gen = torch.Generator(device=device).manual_seed(6)
        act = torch.randn(E, max_m, self.N_INT, generator=gen, device=device, dtype=torch.bfloat16) * 0.1
        w = torch.randn(E, self.K_HIDDEN, self.N_INT, generator=gen, device=device, dtype=torch.bfloat16) * 0.05

        x_fp8, x_scale = _quant_fp8(act)
        w_fp8, w_scale = _quant_fp8(w)

        deep_dense = _run_deep_gemm(x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m)
        deep_compact, _, _ = _compact_dense(deep_dense, masked_m, device)

        out = _run_chunked(
            x_fp8, x_scale, w_fp8, w_scale, masked_m, expected_m,
            chunk_size=32, device=device,
        )

        self.assertEqual(out.shape, deep_compact.shape)
        print("\n  [all-full experts vs DeepGEMM]")
        _report("chunked(full) vs deep_gemm", out, deep_compact, tol_atol=self.TOL_CROSS)


if __name__ == "__main__":
    unittest.main()
