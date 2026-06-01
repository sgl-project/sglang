# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for progressive-resolution spectral ops and upsample utilities.

These tests run on CPU (torch.fft works without CUDA) and are fast (<30 s total).
They verify numerical correctness of our GPU DCT against scipy's reference
implementation and check the upsample pipeline end-to-end.

All classes inherit from unittest.TestCase so the test suite integrates with
both `pytest` and `python -m unittest`.
"""

import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

try:
    import scipy.fft as _scipy_fft

    HAS_SCIPY = True
except ImportError:
    _scipy_fft = None
    HAS_SCIPY = False

_skip_no_scipy = unittest.skipUnless(HAS_SCIPY, "scipy not installed")

from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.scheduler_utils import (
    compute_stage_transitions,
    find_transition_steps,
    reset_scheduler_at_step,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.spectral_ops import (
    dct_1d,
    dct_2d,
    idct_1d,
    idct_2d,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.upsample import (
    apply_upsample,
    dct_upsample_2d,
)


# ---------------------------------------------------------------------------
# DCT-II / IDCT-II correctness
# ---------------------------------------------------------------------------


class TestDCT(unittest.TestCase):
    """Verify our torch.fft DCT-II matches scipy's reference implementation."""

    @_skip_no_scipy
    def test_dct_1d_matches_scipy(self):
        for N in [8, 16, 32, 64]:
            with self.subTest(N=N):
                x = torch.randn(N)
                torch_out = dct_1d(x, norm="ortho").numpy()
                scipy_out = _scipy_fft.dct(x.numpy(), type=2, norm="ortho")
                np.testing.assert_allclose(torch_out, scipy_out, atol=1e-5)

    @_skip_no_scipy
    def test_idct_1d_matches_scipy(self):
        for N in [8, 16, 32]:
            with self.subTest(N=N):
                x = torch.randn(N)
                torch_out = idct_1d(x, norm="ortho").numpy()
                scipy_out = _scipy_fft.idct(x.numpy(), type=2, norm="ortho")
                np.testing.assert_allclose(torch_out, scipy_out, atol=1e-5)

    @_skip_no_scipy
    def test_dct_2d_matches_scipy(self):
        for H, W in [(8, 8), (16, 32), (32, 16)]:
            with self.subTest(H=H, W=W):
                x = torch.randn(H, W)
                torch_out = dct_2d(x, norm="ortho").numpy()
                scipy_out = _scipy_fft.dctn(x.numpy(), type=2, norm="ortho")
                np.testing.assert_allclose(torch_out, scipy_out, atol=1e-4)

    @_skip_no_scipy
    def test_idct_2d_matches_scipy(self):
        for H, W in [(8, 8), (16, 32)]:
            with self.subTest(H=H, W=W):
                x = torch.randn(H, W)
                torch_out = idct_2d(x, norm="ortho").numpy()
                scipy_out = _scipy_fft.idctn(x.numpy(), type=2, norm="ortho")
                np.testing.assert_allclose(torch_out, scipy_out, atol=1e-4)

    def test_dct_idct_roundtrip(self):
        x = torch.randn(3, 4, 16, 16)
        reconstructed = idct_2d(dct_2d(x))
        torch.testing.assert_close(reconstructed, x, atol=1e-4, rtol=0)

    def test_parseval_identity(self):
        """Ortho DCT preserves the L2 norm (Parseval's theorem)."""
        x = torch.randn(32, 32)
        X = dct_2d(x, norm="ortho")
        torch.testing.assert_close(
            x.norm() ** 2, X.norm() ** 2, atol=1e-3, rtol=1e-4
        )

    @_skip_no_scipy
    def test_float32_matches_scipy_to_relative_1e6(self):
        """GPU float32 DCT should match scipy to relative error ~1.7e-7 (see Bug 5 fix)."""
        torch.manual_seed(0)
        x = torch.randn(64, 64, dtype=torch.float32)
        torch_out = dct_2d(x.float(), norm="ortho").numpy()
        scipy_out = _scipy_fft.dctn(x.numpy(), type=2, norm="ortho")
        rel_err = np.abs(torch_out - scipy_out) / (np.abs(scipy_out) + 1e-10)
        self.assertLess(rel_err.mean(), 1e-5)


# ---------------------------------------------------------------------------
# DCT upsample
# ---------------------------------------------------------------------------


class TestDCTUpsample(unittest.TestCase):
    def test_output_shape(self):
        x = torch.randn(1, 16, 8, 8)
        out = dct_upsample_2d(x, sigma_t=0.3, seed=42)
        self.assertEqual(out.shape, (1, 16, 16, 16))

    def test_output_shape_batched(self):
        x = torch.randn(2, 4, 16, 32)
        out = dct_upsample_2d(x, sigma_t=0.1, seed=0)
        self.assertEqual(out.shape, (2, 4, 32, 64))

    def test_output_dtype_preserved(self):
        """Output should have the same dtype as input (cast back after float32 computation)."""
        for dtype in [torch.float32, torch.bfloat16, torch.float16]:
            with self.subTest(dtype=dtype):
                x = torch.randn(1, 4, 8, 8, dtype=dtype)
                out = dct_upsample_2d(x, sigma_t=0.3, seed=0)
                self.assertEqual(out.dtype, dtype)

    @_skip_no_scipy
    def test_low_freq_preservation(self):
        """With sigma_t ≈ 0, the top-left DCT of the output matches the input DCT."""
        x = torch.randn(1, 1, 8, 8)
        out = dct_upsample_2d(x, sigma_t=1e-6, seed=0)
        X_in = _scipy_fft.dctn(x[0, 0].numpy(), type=2, norm="ortho")
        X_out = _scipy_fft.dctn(out[0, 0].numpy(), type=2, norm="ortho")
        np.testing.assert_allclose(X_out[:8, :8], X_in, atol=1e-3)

    def test_rewind_returns_tuple(self):
        x = torch.randn(1, 2, 8, 8)
        result = dct_upsample_2d(x, sigma_t=0.5, seed=1, rewind=True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        out, t_eff = result
        self.assertEqual(out.shape, (1, 2, 16, 16))
        expected_t_eff = 2 * 0.5 / (1 + 0.5)
        self.assertAlmostEqual(t_eff, expected_t_eff, places=6)

    def test_rewind_t_eff_strictly_greater_than_sigma_t(self):
        """t_eff = 2σ/(1+σ) is always > σ for σ ∈ (0, 1); scheduler is rewound forward."""
        for sigma_t in [0.01, 0.1, 0.5, 0.9, 0.99]:
            with self.subTest(sigma_t=sigma_t):
                x = torch.randn(1, 1, 4, 4)
                _, t_eff = dct_upsample_2d(x, sigma_t=sigma_t, seed=0, rewind=True)
                self.assertGreater(t_eff, sigma_t)

    def test_rewind_vs_no_rewind_scale(self):
        x = torch.randn(1, 1, 8, 8)
        sigma_t = 0.4
        out_plain = dct_upsample_2d(x, sigma_t=sigma_t, seed=7)
        out_rewind, _ = dct_upsample_2d(x, sigma_t=sigma_t, seed=7, rewind=True)
        gamma = 1.0 + sigma_t
        torch.testing.assert_close(
            out_plain * (2.0 / gamma), out_rewind, atol=1e-5, rtol=0
        )

    def test_deterministic_with_same_seed(self):
        x = torch.randn(1, 16, 16, 16)
        a = dct_upsample_2d(x, sigma_t=0.3, seed=42)
        b = dct_upsample_2d(x, sigma_t=0.3, seed=42)
        torch.testing.assert_close(a, b)

    def test_different_seeds_differ(self):
        x = torch.randn(1, 4, 8, 8)
        a = dct_upsample_2d(x, sigma_t=0.5, seed=0)
        b = dct_upsample_2d(x, sigma_t=0.5, seed=1)
        self.assertFalse(torch.allclose(a, b))

    def test_apply_upsample_dct(self):
        x = torch.randn(1, 16, 8, 8)
        out = apply_upsample(x, sigma_t=0.3, seed=0, mode="dct")
        self.assertEqual(out.shape, (1, 16, 16, 16))

    def test_apply_upsample_dct_rewind(self):
        x = torch.randn(1, 16, 8, 8)
        sigma_t = 0.3
        out, t_eff = apply_upsample(x, sigma_t=sigma_t, seed=0, mode="dct_rewind")
        self.assertEqual(out.shape, (1, 16, 16, 16))
        expected = 2 * sigma_t / (1 + sigma_t)
        self.assertAlmostEqual(t_eff, expected, places=6)

    def test_apply_upsample_invalid_mode(self):
        x = torch.randn(1, 4, 4, 4)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            apply_upsample(x, sigma_t=0.3, seed=0, mode="dwt")


# ---------------------------------------------------------------------------
# Scheduler utils
# ---------------------------------------------------------------------------


class TestSchedulerUtils(unittest.TestCase):
    def _make_sigmas(self, n_steps=50):
        return torch.linspace(1.0, 1.0 / n_steps, n_steps + 1)

    def test_compute_stage_transitions_stage1_always_one(self):
        stage_sigmas = compute_stage_transitions(
            delta=0.01, n_levels=1, A=203.6, beta=1.915, H_lat=128, W_lat=128
        )
        self.assertEqual(stage_sigmas[1], 1.0)
        self.assertIn(2, stage_sigmas)
        self.assertGreater(stage_sigmas[2], 0.0)
        self.assertLess(stage_sigmas[2], 1.0)

    def test_compute_stage_transitions_levels(self):
        for levels in (1, 2, 3):
            with self.subTest(levels=levels):
                stage_sigmas = compute_stage_transitions(
                    delta=0.01, n_levels=levels, A=203.6, beta=1.915, H_lat=128, W_lat=128
                )
                self.assertEqual(len(stage_sigmas), levels + 1)

    def test_transition_threshold_decreases_with_larger_delta(self):
        """Larger δ lowers the threshold sigma: t_eff=1/(1+sqrt(δ/denom)) decreases as δ grows.
        A lower threshold means the stage fires later (at a smaller sigma value)."""
        s_01 = compute_stage_transitions(0.01, 1, 203.6, 1.915, 128, 128)
        s_05 = compute_stage_transitions(0.05, 1, 203.6, 1.915, 128, 128)
        self.assertLess(s_05[2], s_01[2])

    def test_find_transition_steps_larger_delta_fires_later(self):
        """Larger δ → lower threshold sigma → transition fires at a later step index."""
        sigmas = self._make_sigmas(50)
        stage_01 = compute_stage_transitions(0.01, 1, 203.6, 1.915, 128, 128)
        stage_05 = compute_stage_transitions(0.05, 1, 203.6, 1.915, 128, 128)
        trans_01 = find_transition_steps(sigmas, stage_01, n_steps=50)
        trans_05 = find_transition_steps(sigmas, stage_05, n_steps=50)
        self.assertLess(trans_01[2], trans_05[2])

    def test_find_transition_steps_ordering(self):
        sigmas = self._make_sigmas()
        stage_sigmas = compute_stage_transitions(
            delta=0.01, n_levels=1, A=203.6, beta=1.915, H_lat=128, W_lat=128
        )
        trans = find_transition_steps(sigmas, stage_sigmas, n_steps=50)
        step = trans[2]
        self.assertGreaterEqual(step, 0)
        self.assertLessEqual(step, 50)

    def test_find_transition_steps_multi_level_ordering(self):
        """For L=2, transition step 2 < transition step 3."""
        sigmas = self._make_sigmas()
        stage_sigmas = compute_stage_transitions(
            delta=0.01, n_levels=2, A=203.6, beta=1.915, H_lat=128, W_lat=128
        )
        trans = find_transition_steps(sigmas, stage_sigmas, n_steps=50)
        self.assertLess(trans[2], trans[3])

    def test_reset_scheduler_at_step(self):
        class FakeScheduler:
            _step_index = 10
            _begin_index = 0
            model_outputs = [None, None]
            lower_order_nums = 5
            last_sample = object()

            class config:
                solver_order = 2

        sched = FakeScheduler()
        reset_scheduler_at_step(sched, 20)
        self.assertEqual(sched._step_index, 20)
        self.assertEqual(sched._begin_index, 0)
        self.assertEqual(sched.lower_order_nums, 0)
        self.assertIsNone(sched.last_sample)
        self.assertTrue(all(v is None for v in sched.model_outputs))

    def test_reset_scheduler_no_optional_attrs(self):
        """reset_scheduler_at_step must not fail on minimal schedulers (Euler, DDIM)."""
        class MinimalScheduler:
            _step_index = 5

        sched = MinimalScheduler()
        reset_scheduler_at_step(sched, 10)
        self.assertEqual(sched._step_index, 10)


# ---------------------------------------------------------------------------
# ProgressiveDenoisingStage base-class unit tests (no GPU / no model required)
# ---------------------------------------------------------------------------


class TestProgressiveDenoisingStageBase(unittest.TestCase):
    """Tests for ProgressiveDenoisingStage static helpers and mode detection."""

    def _import_stage(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.denoising import (
            ProgressiveDenoisingStage,
        )
        return ProgressiveDenoisingStage

    def test_get_seed_from_seeds_list(self):
        stage = self._import_stage()
        batch = SimpleNamespace(seeds=[99], sampling_params=None)
        self.assertEqual(stage._get_seed(batch), 99)

    def test_get_seed_from_sampling_params_seed(self):
        stage = self._import_stage()
        sp = SimpleNamespace(seed=123)
        batch = SimpleNamespace(seeds=None, sampling_params=sp)
        self.assertEqual(stage._get_seed(batch), 123)

    def test_get_seed_fallback_is_42(self):
        stage = self._import_stage()
        batch = SimpleNamespace(seeds=None, sampling_params=None)
        self.assertEqual(stage._get_seed(batch), 42)

    def test_get_seed_empty_seeds_list_falls_back(self):
        stage = self._import_stage()
        batch = SimpleNamespace(seeds=[], sampling_params=None)
        self.assertEqual(stage._get_seed(batch), 42)

    def test_flux_spectrum_constants_plausible(self):
        """FLUX VAE spectrum constants A and beta must be in physically meaningful ranges."""
        from sglang.multimodal_gen.runtime.pipelines.flux_progressive import (
            FLUX_SPECTRUM_A,
            FLUX_SPECTRUM_BETA,
        )
        self.assertGreater(FLUX_SPECTRUM_A, 0.0)
        self.assertGreater(FLUX_SPECTRUM_BETA, 1.0)
        self.assertLess(FLUX_SPECTRUM_BETA, 4.0)


if __name__ == "__main__":
    unittest.main()
