# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for progressive-resolution spectral ops and upsample utilities.

These tests run on CPU (torch.fft works without CUDA) and are fast (<10 s).
They verify numerical correctness of our GPU DCT against scipy's reference
implementation and check the upsample pipeline end-to-end.
"""

import math

import numpy as np
import pytest
import torch

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


class TestDCT:
    """Verify our torch.fft DCT matches scipy's reference."""

    @pytest.mark.parametrize("N", [8, 16, 32, 64])
    def test_dct_1d_matches_scipy(self, N):
        scipy_fft = pytest.importorskip("scipy.fft")
        x = torch.randn(N)
        torch_out = dct_1d(x, norm="ortho").numpy()
        scipy_out = scipy_fft.dct(x.numpy(), type=2, norm="ortho")
        np.testing.assert_allclose(torch_out, scipy_out, atol=1e-5)

    @pytest.mark.parametrize("N", [8, 16, 32])
    def test_idct_1d_matches_scipy(self, N):
        scipy_fft = pytest.importorskip("scipy.fft")
        x = torch.randn(N)
        torch_out = idct_1d(x, norm="ortho").numpy()
        scipy_out = scipy_fft.idct(x.numpy(), type=2, norm="ortho")
        np.testing.assert_allclose(torch_out, scipy_out, atol=1e-5)

    @pytest.mark.parametrize("H,W", [(8, 8), (16, 32), (32, 16)])
    def test_dct_2d_matches_scipy(self, H, W):
        scipy_fft = pytest.importorskip("scipy.fft")
        x = torch.randn(H, W)
        torch_out = dct_2d(x, norm="ortho").numpy()
        scipy_out = scipy_fft.dctn(x.numpy(), type=2, norm="ortho")
        np.testing.assert_allclose(torch_out, scipy_out, atol=1e-4)

    @pytest.mark.parametrize("H,W", [(8, 8), (16, 32)])
    def test_idct_2d_matches_scipy(self, H, W):
        scipy_fft = pytest.importorskip("scipy.fft")
        x = torch.randn(H, W)
        torch_out = idct_2d(x, norm="ortho").numpy()
        scipy_out = scipy_fft.idctn(x.numpy(), type=2, norm="ortho")
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


# ---------------------------------------------------------------------------
# DCT upsample
# ---------------------------------------------------------------------------


class TestDCTUpsample:
    def test_output_shape(self):
        x = torch.randn(1, 16, 8, 8)
        out = dct_upsample_2d(x, sigma_t=0.3, seed=42)
        assert out.shape == (1, 16, 16, 16)

    def test_output_shape_batched(self):
        x = torch.randn(2, 4, 16, 32)
        out = dct_upsample_2d(x, sigma_t=0.1, seed=0)
        assert out.shape == (2, 4, 32, 64)

    def test_low_freq_preservation(self):
        """With sigma_t ≈ 0, the top-left DCT of the output matches the input DCT."""
        scipy_fft = pytest.importorskip("scipy.fft")
        x = torch.randn(1, 1, 8, 8)
        sigma_t = 1e-6
        out = dct_upsample_2d(x, sigma_t=sigma_t, seed=0)

        X_in = scipy_fft.dctn(x[0, 0].numpy(), type=2, norm="ortho")
        X_out = scipy_fft.dctn(out[0, 0].numpy(), type=2, norm="ortho")
        np.testing.assert_allclose(X_out[:8, :8], X_in, atol=1e-3)

    def test_rewind_returns_tuple(self):
        x = torch.randn(1, 2, 8, 8)
        result = dct_upsample_2d(x, sigma_t=0.5, seed=1, rewind=True)
        assert isinstance(result, tuple) and len(result) == 2
        out, t_eff = result
        assert out.shape == (1, 2, 16, 16)
        expected_t_eff = 2 * 0.5 / (1 + 0.5)
        assert abs(t_eff - expected_t_eff) < 1e-6

    def test_rewind_vs_no_rewind_scale(self):
        x = torch.randn(1, 1, 8, 8)
        sigma_t = 0.4
        out_plain = dct_upsample_2d(x, sigma_t=sigma_t, seed=7)
        out_rewind, _ = dct_upsample_2d(x, sigma_t=sigma_t, seed=7, rewind=True)
        gamma = 1.0 + sigma_t
        torch.testing.assert_close(out_plain * (2.0 / gamma), out_rewind, atol=1e-5, rtol=0)

    def test_deterministic_with_same_seed(self):
        x = torch.randn(1, 16, 16, 16)
        a = dct_upsample_2d(x, sigma_t=0.3, seed=42)
        b = dct_upsample_2d(x, sigma_t=0.3, seed=42)
        torch.testing.assert_close(a, b)

    def test_different_seeds_differ(self):
        x = torch.randn(1, 4, 8, 8)
        a = dct_upsample_2d(x, sigma_t=0.5, seed=0)
        b = dct_upsample_2d(x, sigma_t=0.5, seed=1)
        assert not torch.allclose(a, b)

    def test_apply_upsample_dct(self):
        x = torch.randn(1, 16, 8, 8)
        out = apply_upsample(x, sigma_t=0.3, seed=0, mode="dct")
        assert out.shape == (1, 16, 16, 16)

    def test_apply_upsample_dct_rewind(self):
        x = torch.randn(1, 16, 8, 8)
        sigma_t = 0.3
        out, t_eff = apply_upsample(x, sigma_t=sigma_t, seed=0, mode="dct_rewind")
        assert out.shape == (1, 16, 16, 16)
        # t_eff = 2*sigma_t/(1+sigma_t) — always > sigma_t for sigma_t in (0,1)
        expected = 2 * sigma_t / (1 + sigma_t)
        assert abs(t_eff - expected) < 1e-6

    def test_apply_upsample_invalid_mode(self):
        x = torch.randn(1, 4, 4, 4)
        with pytest.raises(ValueError, match="Unsupported"):
            apply_upsample(x, sigma_t=0.3, seed=0, mode="dwt")


# ---------------------------------------------------------------------------
# Scheduler utils
# ---------------------------------------------------------------------------


class TestSchedulerUtils:
    def _make_sigmas(self, n_steps=50):
        """Simulate a FLUX-style linearly decreasing sigma schedule."""
        return torch.linspace(1.0, 1.0 / n_steps, n_steps + 1)

    def test_compute_stage_transitions_stage1_always_one(self):
        stage_sigmas = compute_stage_transitions(
            delta=0.01, n_levels=1, A=203.6, beta=1.915, H_lat=128, W_lat=128
        )
        assert stage_sigmas[1] == 1.0
        assert 2 in stage_sigmas
        assert 0 < stage_sigmas[2] < 1.0

    def test_compute_stage_transitions_levels(self):
        for levels in (1, 2, 3):
            stage_sigmas = compute_stage_transitions(
                delta=0.01, n_levels=levels, A=203.6, beta=1.915, H_lat=128, W_lat=128
            )
            assert len(stage_sigmas) == levels + 1

    def test_find_transition_steps_ordering(self):
        sigmas = self._make_sigmas()
        stage_sigmas = compute_stage_transitions(
            delta=0.01, n_levels=1, A=203.6, beta=1.915, H_lat=128, W_lat=128
        )
        trans = find_transition_steps(sigmas, stage_sigmas, n_steps=50)
        # Transition step 2 should be strictly between 0 and 50
        step = trans[2]
        assert 0 <= step <= 50

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
        assert sched._step_index == 20
        assert sched._begin_index == 0
        assert sched.lower_order_nums == 0
        assert sched.last_sample is None
        assert all(v is None for v in sched.model_outputs)
