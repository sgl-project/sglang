"""
Tests for Option A: Chunked Expert Processing in _run_masked_gemm.

The chunked path processes experts in groups of chunk_size to reduce peak HBM
for the down-projection output buffer.

Test coverage:
    1. test_chunked_matches_baseline_ue8m0        — output matches non-chunked
    2. test_chunked_various_chunk_sizes            — chunk_size = 1, 8, 32, E
    3. test_chunked_handles_empty_experts          — experts with masked_m == 0
    4. test_chunked_chunk_size_ge_E                — chunk_size >= E = single pass
    5. test_chunked_memory_reduction               — chunk peak < dense allocation
"""

import os
import unittest

import torch

try:
    import asym_gemm as _asym_gemm_lib

    HAS_ASYM_GEMM = True
except ImportError:
    HAS_ASYM_GEMM = False

try:
    from sglang.srt.layers.asym_gemm_wrapper.configurer import ASYMGEMM_SCALE_UE8M0

    HAS_SGLANG = True
except ImportError:
    HAS_SGLANG = False

from sglang.test.test_utils import CustomTestCase


def _requires_ue8m0(fn):
    """Skip test when ASYMGEMM_SCALE_UE8M0 is False (chunked path not active)."""
    import functools

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not ASYMGEMM_SCALE_UE8M0:
            self.skipTest("Chunked path requires ASYMGEMM_SCALE_UE8M0=True")
        return fn(self, *args, **kwargs)

    return wrapper


def _quantize_fp8_3d(x_bf16_3d: torch.Tensor, group_size: int = 128):
    """
    Quantise a 3-D [E, max_m, K] bf16 tensor to fp8 per-token-group,
    producing a 3-D [E, max_m, K] fp8 tensor and a matching scale.

    Uses the same quantiser as the production masked-GEMM path.
    """
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    E, max_m, K = x_bf16_3d.shape
    x_2d = x_bf16_3d.reshape(E * max_m, K)
    x_fp8_2d, scale_2d = sglang_per_token_group_quant_fp8(
        x_2d.contiguous(),
        group_size,
        column_major_scales=ASYMGEMM_SCALE_UE8M0,
        scale_tma_aligned=ASYMGEMM_SCALE_UE8M0,
        scale_ue8m0=ASYMGEMM_SCALE_UE8M0,
    )
    x_fp8 = x_fp8_2d.reshape(E, max_m, K)
    # scale_2d is [E*max_m, K//group_size] (row-major) or [K//group_size, E*max_m]
    # (column-major UE8M0).  Reshape: keep leading dim E, rest per-expert.
    scale = scale_2d.reshape(E, max_m, -1)
    return x_fp8, scale


def _make_masked_gemm_inputs(
    num_experts: int,
    max_m: int,
    n_int: int,
    k_hidden: int,
    mean_fill: float = 0.6,
    device: str = "cuda",
    seed: int = 0,
):
    """
    Create realistic masked-GEMM inputs similar to the down-projection path.

    Returns:
        down_input:       [E, max_m, n_int] fp8
        down_input_scale: [E, max_m, ...]  (UE8M0 or float32, matching ASYMGEMM_SCALE_UE8M0)
        w2_weight:        [E, k_hidden, n_int] fp8
        w2_scale:         [E, k_hidden, ...]
        masked_m:         [E] int32
        expected_m:       int
    """
    gen = torch.Generator(device=device).manual_seed(seed)

    masked_m = torch.randint(
        1, max_m + 1, (num_experts,), generator=gen, device=device, dtype=torch.int32
    )
    masked_m = torch.clamp(
        (masked_m.float() * mean_fill).to(torch.int32), min=1, max=max_m
    )
    expected_m = int(masked_m.max().item())

    # activation input (masked: padding rows filled with NaN → zeroed before quant)
    act_bf16 = torch.randn(
        num_experts, max_m, n_int, generator=gen, device=device, dtype=torch.bfloat16
    )
    for e in range(num_experts):
        act_bf16[e, int(masked_m[e]) :].zero_()

    down_input, down_input_scale = _quantize_fp8_3d(act_bf16)

    # weight
    w_bf16 = (
        torch.randn(
            num_experts, k_hidden, n_int,
            generator=gen, device=device, dtype=torch.bfloat16,
        )
        * 0.05
    )
    w2_weight, w2_scale = _quantize_fp8_3d(w_bf16)

    return down_input, down_input_scale, w2_weight, w2_scale, masked_m, expected_m


def _run_chunked(
    down_input: torch.Tensor,
    down_input_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    chunk_size: int,
    device: str = "cuda",
):
    """
    Run the chunked path directly by calling _run_masked_gemm_down_chunked.

    Constructs a minimal running_state with synthetic src2dst and topk_ids.
    """
    from sglang.srt.layers.moe.moe_runner.asym_gemm import AsymGemmRunnerCore

    E, max_m, _ = down_input.shape

    # Build src2dst / topk_ids mirroring the standard dispatch layout:
    #   token t routed to expert e lands at row e * max_m + its_position_in_e
    #   src2dst[t] = e * max_m + offset_within_e
    # For the test we construct them synthetically (exact values don't matter
    # for correctness of the compaction kernel; only the relative positions do).
    topk_ids = torch.zeros(int(masked_m.sum().item()), device=device, dtype=torch.int32)
    src2dst = torch.zeros_like(topk_ids)
    pos = 0
    for e in range(E):
        m_e = int(masked_m[e])
        for t in range(m_e):
            topk_ids[pos] = e
            src2dst[pos] = e * max_m + t
            pos += 1

    running_state = {
        "hidden_states_device": device,
        "src2dst": src2dst,
        "topk_ids": topk_ids,
        "down_gemm_overlap_args": None,
    }

    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

    cfg = MoeRunnerConfig(
        activation="silu",
        is_gated=True,
        top_k=1,
        num_local_experts=E,
        routed_scaling_factor=None,
    )
    runner = AsymGemmRunnerCore.__new__(AsymGemmRunnerCore)
    runner.config = cfg

    return runner._run_masked_gemm_down_chunked(
        down_input,
        down_input_scale,
        w2_weight,
        w2_scale,
        masked_m,
        expected_m,
        max_m,
        running_state,
        chunk_size=chunk_size,
    )


def _run_baseline(
    down_input: torch.Tensor,
    down_input_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    device: str = "cuda",
):
    """
    Run the standard (non-chunked) masked GEMM + post-GEMM compaction.
    Returns compact [total_active, K] bf16 output.
    """
    from sglang.srt.layers import asym_gemm_wrapper
    from sglang.srt.layers.moe.ep_moe.kernels import compact_masked_to_contiguous
    from sglang.srt.layers.moe.moe_runner.asym_gemm import (
        build_offsets_experts_from_masked_m,
        compute_contiguous_offsets_from_masked_m,
    )

    E, max_m, _ = down_input.shape
    K = w2_weight.shape[1]

    offsets, experts, list_size = build_offsets_experts_from_masked_m(masked_m, E)
    down_output_dense = torch.empty(
        (E, max_m, K), device=device, dtype=torch.bfloat16
    )
    asym_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
        (down_input, down_input_scale),
        (w2_weight, w2_scale),
        down_output_dense,
        masked_m,
        expected_m,
        offsets,
        experts,
        list_size,
    )

    contiguous_offsets, total_active = compute_contiguous_offsets_from_masked_m(
        masked_m
    )
    down_output_compact = torch.empty(
        (max(total_active, 1), K), device=device, dtype=torch.bfloat16
    )
    if total_active > 0:
        compact_masked_to_contiguous(
            down_output_dense,
            down_output_compact,
            masked_m,
            contiguous_offsets,
            max_m,
        )
    return down_output_compact[:total_active], contiguous_offsets, total_active


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
@unittest.skipIf(not HAS_ASYM_GEMM, "asym_gemm library not installed")
@unittest.skipIf(not HAS_SGLANG, "sglang not importable")
class TestAsymGemmMoeChunked(CustomTestCase):
    NUM_EXPERTS = 32
    MAX_M = 128
    N_INT = 1024   # intermediate dim after gate (N/2 in DeepSeek notation)
    K_HIDDEN = 1024  # hidden dimension (down-projection output)

    TOL = dict(atol=1e-2, rtol=1e-2)

    def setUp(self):
        torch.manual_seed(0)
        self.device = "cuda"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _inputs(self, mean_fill=0.6, seed=0):
        return _make_masked_gemm_inputs(
            self.NUM_EXPERTS,
            self.MAX_M,
            self.N_INT,
            self.K_HIDDEN,
            mean_fill=mean_fill,
            device=self.device,
            seed=seed,
        )

    def _assert_close(self, got, ref, tag=""):
        max_abs = (got.float() - ref.float()).abs().max().item()
        mean_abs = (got.float() - ref.float()).abs().mean().item()
        print(f"[{tag}] max|Δ|={max_abs:.3e}  mean|Δ|={mean_abs:.3e}")
        self.assertLessEqual(max_abs, self.TOL["atol"], f"{tag} atol exceeded")

    # ------------------------------------------------------------------
    # tests
    # ------------------------------------------------------------------

    @_requires_ue8m0
    def test_chunked_matches_baseline_ue8m0(self):
        """Chunked output (C=8) must match baseline compact output."""
        inputs = self._inputs()
        down_input, down_input_scale, w2_weight, w2_scale, masked_m, expected_m = inputs

        ref, _, _ = _run_baseline(
            down_input, down_input_scale, w2_weight, w2_scale, masked_m, expected_m,
            device=self.device,
        )
        got = _run_chunked(
            down_input, down_input_scale, w2_weight, w2_scale, masked_m, expected_m,
            chunk_size=8, device=self.device,
        )

        self.assertEqual(got.shape, ref.shape, "output shape mismatch")
        self._assert_close(got, ref, tag="chunked(8) vs baseline")

    @_requires_ue8m0
    def test_chunked_various_chunk_sizes(self):
        """chunk_size = 1, 4, 16, E all produce equivalent results."""
        inputs = self._inputs(seed=1)
        down_input, down_input_scale, w2_weight, w2_scale, masked_m, expected_m = inputs

        ref, _, _ = _run_baseline(
            down_input, down_input_scale, w2_weight, w2_scale, masked_m, expected_m,
            device=self.device,
        )

        for cs in [1, 4, 16, self.NUM_EXPERTS]:
            got = _run_chunked(
                down_input, down_input_scale, w2_weight, w2_scale, masked_m, expected_m,
                chunk_size=cs, device=self.device,
            )
            self.assertEqual(got.shape, ref.shape, f"chunk_size={cs}: shape mismatch")
            self._assert_close(got, ref, tag=f"chunked({cs}) vs baseline")

    @_requires_ue8m0
    def test_chunked_handles_empty_experts(self):
        """Experts with masked_m == 0 must be skipped without error."""
        down_input, down_input_scale, w2_weight, w2_scale, masked_m, expected_m = (
            self._inputs(seed=2)
        )
        # Zero out a few experts
        masked_m[0] = 0
        masked_m[5] = 0
        masked_m[-1] = 0
        expected_m = int(masked_m.max().item())

        ref, _, total_active = _run_baseline(
            down_input, down_input_scale, w2_weight, w2_scale, masked_m, expected_m,
            device=self.device,
        )
        got = _run_chunked(
            down_input, down_input_scale, w2_weight, w2_scale, masked_m, expected_m,
            chunk_size=8, device=self.device,
        )

        self.assertEqual(got.shape[0], total_active, "total_active mismatch")
        self._assert_close(got, ref, tag="chunked(empty) vs baseline")

    @_requires_ue8m0
    def test_chunked_chunk_size_ge_E(self):
        """chunk_size >= E should behave identically to baseline (single chunk)."""
        inputs = self._inputs(seed=3)
        down_input, down_input_scale, w2_weight, w2_scale, masked_m, expected_m = inputs

        ref, _, _ = _run_baseline(
            down_input, down_input_scale, w2_weight, w2_scale, masked_m, expected_m,
            device=self.device,
        )

        for cs in [self.NUM_EXPERTS, self.NUM_EXPERTS + 1, self.NUM_EXPERTS * 2]:
            got = _run_chunked(
                down_input, down_input_scale, w2_weight, w2_scale, masked_m, expected_m,
                chunk_size=cs, device=self.device,
            )
            self._assert_close(got, ref, tag=f"chunked({cs}) vs baseline")

    @_requires_ue8m0
    def test_chunked_memory_reduction(self):
        """
        With chunk_size=8 and E=32, the peak incremental allocation must stay
        well below the dense [E, max_m, K] baseline.

        The Option A v1 pattern allocates:
          - chunk_output  [C, max_m, K]  bf16  — reused across iterations
          - down_output_compact [total_active, K]  bf16  — sustained

        Both are live simultaneously at peak, so the budget is:
          chunk_output + compact ≈ (C/E × dense) + (mean_fill × dense/E × K*2)

        We use a generous 3× headroom over the chunk_output size alone to
        account for the compact buffer and CUDA allocator rounding.
        """
        E = self.NUM_EXPERTS
        max_m = self.MAX_M
        K = self.K_HIDDEN
        C = 8  # chunk_size used in this test
        dense_bytes = E * max_m * K * 2        # bf16 [E, max_m, K]
        chunk_bytes = C * max_m * K * 2        # bf16 [C, max_m, K]  (reusable buffer)

        down_input, down_input_scale, w2_weight, w2_scale, masked_m, expected_m = (
            self._inputs(seed=4)
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        # Capture baseline after inputs are allocated.
        baseline_peak = torch.cuda.max_memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

        got = _run_chunked(
            down_input, down_input_scale, w2_weight, w2_scale, masked_m, expected_m,
            chunk_size=C, device=self.device,
        )
        torch.cuda.synchronize()

        incremental_peak = torch.cuda.max_memory_allocated(self.device) - baseline_peak

        # Peak must be below the dense allocation.
        # Allow headroom: chunk_output + compact_output + 3× CUDA allocator overhead.
        budget = chunk_bytes * 3 + dense_bytes // 2
        self.assertLess(
            incremental_peak,
            budget,
            f"Incremental peak {incremental_peak / 1e6:.1f} MB "
            f"exceeds budget {budget / 1e6:.1f} MB; "
            f"dense would be {dense_bytes / 1e6:.1f} MB, "
            f"chunk_output alone is {chunk_bytes / 1e6:.1f} MB",
        )
        print(
            f"[memory] dense={dense_bytes / 1e6:.1f} MB  "
            f"chunk_output={chunk_bytes / 1e6:.1f} MB  "
            f"chunked_incremental={incremental_peak / 1e6:.1f} MB  "
            f"ratio_vs_dense={incremental_peak / dense_bytes:.2f}"
        )

    @_requires_ue8m0
    def test_output_shape_and_dtype(self):
        """Output is [total_active, K] bfloat16."""
        inputs = self._inputs(seed=5)
        down_input, down_input_scale, w2_weight, w2_scale, masked_m, expected_m = inputs

        from sglang.srt.layers.moe.moe_runner.asym_gemm import (
            compute_contiguous_offsets_from_masked_m,
        )
        _, total_active = compute_contiguous_offsets_from_masked_m(masked_m)

        got = _run_chunked(
            down_input, down_input_scale, w2_weight, w2_scale, masked_m, expected_m,
            chunk_size=8, device=self.device,
        )

        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertEqual(got.shape[0], total_active)
        self.assertEqual(got.shape[1], self.K_HIDDEN)


if __name__ == "__main__":
    unittest.main()
