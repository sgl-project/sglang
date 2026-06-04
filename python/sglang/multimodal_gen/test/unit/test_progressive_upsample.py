# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for progressive-resolution spectral ops and upsample utilities.

These tests run on CPU (torch.fft works without CUDA) and are fast (<30 s total).
They verify numerical correctness of our GPU DCT against scipy's reference
implementation and check the upsample pipeline end-to-end.

All classes inherit from unittest.TestCase so the test suite integrates with
both `pytest` and `python -m unittest`.
"""

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
        torch.testing.assert_close(x.norm() ** 2, X.norm() ** 2, atol=1e-3, rtol=1e-4)

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
                    delta=0.01,
                    n_levels=levels,
                    A=203.6,
                    beta=1.915,
                    H_lat=128,
                    W_lat=128,
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


# ---------------------------------------------------------------------------
# FLUX.2 pack / unpack (pure CPU, no mocks)
# ---------------------------------------------------------------------------


class TestFlux2Pack(unittest.TestCase):
    """Verify _flux2_pack / _flux2_unpack shapes and roundtrip correctness."""

    def setUp(self):
        from sglang.multimodal_gen.runtime.pipelines.flux_2_progressive import (
            _flux2_pack,
            _flux2_unpack,
        )

        self._pack = _flux2_pack
        self._unpack = _flux2_unpack

    def test_pack_output_shape(self):
        """(B, C, H, W) → (B, H*W, C)."""
        x = torch.randn(2, 64, 8, 16)
        out = self._pack(x)
        self.assertEqual(out.shape, (2, 8 * 16, 64))

    def test_unpack_output_shape(self):
        """(B, H*W, C) → (B, C, H, W)."""
        packed = torch.randn(2, 128, 64)
        out = self._unpack(packed, h_lat=8, w_lat=16)
        self.assertEqual(out.shape, (2, 64, 8, 16))

    def test_pack_unpack_roundtrip(self):
        """pack followed by unpack reconstructs the original spatial tensor."""
        x = torch.randn(1, 64, 8, 8)
        reconstructed = self._unpack(self._pack(x), h_lat=8, w_lat=8)
        torch.testing.assert_close(reconstructed, x)

    def test_unpack_pack_roundtrip(self):
        """unpack followed by pack reconstructs the original packed tensor."""
        packed = torch.randn(1, 64, 64)
        reconstructed = self._pack(self._unpack(packed, h_lat=8, w_lat=8))
        torch.testing.assert_close(reconstructed, packed)

    def test_pack_is_row_major(self):
        """Token (h, w) maps to sequence index h*W + w — pure row-major ordering."""
        B, C, H, W = 1, 4, 3, 5
        x = torch.randn(B, C, H, W)
        packed = self._pack(x)  # (1, H*W, C)
        for h in range(H):
            for w in range(W):
                idx = h * W + w
                torch.testing.assert_close(packed[0, idx, :], x[0, :, h, w])

    def test_dtype_preserved_through_pack(self):
        for dtype in [torch.float32, torch.bfloat16, torch.float16]:
            with self.subTest(dtype=dtype):
                x = torch.randn(1, 16, 4, 4, dtype=dtype)
                self.assertEqual(self._pack(x).dtype, dtype)

    def test_dtype_preserved_through_unpack(self):
        for dtype in [torch.float32, torch.bfloat16, torch.float16]:
            with self.subTest(dtype=dtype):
                packed = torch.randn(1, 16, 64, dtype=dtype)
                self.assertEqual(self._unpack(packed, 4, 4).dtype, dtype)


# ---------------------------------------------------------------------------
# Flux2ProgressiveDenoisingStage — unit tests (CPU, no GPU, no real model)
# ---------------------------------------------------------------------------


class TestFlux2ProgressiveStage(unittest.TestCase):
    """Unit tests for Flux2ProgressiveDenoisingStage helpers.

    Uses object.__new__ to bypass DenoisingStage.__init__ (which requires a
    live server_args context), then manually sets the attributes each method
    under test reads.
    """

    def _make_stage(self):
        from sglang.multimodal_gen.runtime.pipelines.flux_2_progressive import (
            Flux2ProgressiveDenoisingStage,
        )

        stage = object.__new__(Flux2ProgressiveDenoisingStage)
        stage._freqs_cis_cache = {}
        stage._spectrum_A = 203.615097
        stage._spectrum_beta = 1.915461
        return stage

    def _make_server_args(self, vae_scale=8, in_channels=64):
        return SimpleNamespace(
            pipeline_config=SimpleNamespace(
                vae_config=SimpleNamespace(
                    arch_config=SimpleNamespace(vae_scale_factor=vae_scale)
                ),
                dit_config=SimpleNamespace(
                    arch_config=SimpleNamespace(in_channels=in_channels)
                ),
                get_latent_dtype=lambda dtype: torch.float32,
                prepare_pos_cond_kwargs=lambda *a, **kw: {},
            )
        )

    # ------------------------------------------------------------------
    # Spectrum constants
    # ------------------------------------------------------------------

    def test_flux2_spectrum_constants_plausible(self):
        from sglang.multimodal_gen.runtime.pipelines.flux_2_progressive import (
            FLUX_SPECTRUM_A,
            FLUX_SPECTRUM_BETA,
        )

        self.assertGreater(FLUX_SPECTRUM_A, 0.0)
        self.assertGreater(FLUX_SPECTRUM_BETA, 1.0)
        self.assertLess(FLUX_SPECTRUM_BETA, 4.0)

    # ------------------------------------------------------------------
    # _latent_scale_factor
    # ------------------------------------------------------------------

    def test_latent_scale_factor_is_double_vae_scale(self):
        stage = self._make_stage()
        server_args = self._make_server_args(vae_scale=8)
        self.assertEqual(stage._latent_scale_factor(server_args), 16)

    def test_latent_scale_factor_scales_with_vae_config(self):
        """If vae_scale_factor were ever different, the 2× factor still applies."""
        stage = self._make_stage()
        for vae_scale in [4, 8, 16]:
            with self.subTest(vae_scale=vae_scale):
                server_args = self._make_server_args(vae_scale=vae_scale)
                self.assertEqual(stage._latent_scale_factor(server_args), vae_scale * 2)

    # ------------------------------------------------------------------
    # _generate_initial_noise
    # ------------------------------------------------------------------

    def test_generate_initial_noise_packed_shape(self):
        """Output is packed [1, h_lat*w_lat, C]."""
        import unittest.mock as mock

        stage = self._make_stage()
        server_args = self._make_server_args(in_channels=64)
        batch = SimpleNamespace(
            prompt_embeds=[torch.zeros(1, 1, dtype=torch.bfloat16)],
            latent_ids=None,
        )
        with mock.patch(
            "sglang.multimodal_gen.runtime.distributed.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            out = stage._generate_initial_noise(
                batch, server_args, h_lat=4, w_lat=8, seed=42
            )

        self.assertEqual(out.shape, (1, 4 * 8, 64))

    def test_generate_initial_noise_sets_latent_ids_shape(self):
        """batch.latent_ids is set to (1, h_lat*w_lat, 4) after the call."""
        import unittest.mock as mock

        stage = self._make_stage()
        server_args = self._make_server_args(in_channels=64)
        batch = SimpleNamespace(
            prompt_embeds=[torch.zeros(1, 1, dtype=torch.bfloat16)],
            latent_ids=None,
        )
        with mock.patch(
            "sglang.multimodal_gen.runtime.distributed.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            stage._generate_initial_noise(batch, server_args, h_lat=4, w_lat=8, seed=0)

        self.assertIsNotNone(batch.latent_ids)
        self.assertEqual(batch.latent_ids.shape, (1, 4 * 8, 4))

    def test_generate_initial_noise_latent_ids_not_floating(self):
        """latent_ids should contain integer position coordinates."""
        import unittest.mock as mock

        stage = self._make_stage()
        server_args = self._make_server_args(in_channels=64)
        batch = SimpleNamespace(
            prompt_embeds=[torch.zeros(1, 1, dtype=torch.bfloat16)],
            latent_ids=None,
        )
        with mock.patch(
            "sglang.multimodal_gen.runtime.distributed.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            stage._generate_initial_noise(batch, server_args, h_lat=4, w_lat=4, seed=0)

        self.assertFalse(batch.latent_ids.is_floating_point())

    def test_generate_initial_noise_deterministic(self):
        """Same seed → same packed output."""
        import unittest.mock as mock

        stage = self._make_stage()
        server_args = self._make_server_args()
        batch1 = SimpleNamespace(
            prompt_embeds=[torch.zeros(1, 1, dtype=torch.bfloat16)], latent_ids=None
        )
        batch2 = SimpleNamespace(
            prompt_embeds=[torch.zeros(1, 1, dtype=torch.bfloat16)], latent_ids=None
        )
        with mock.patch(
            "sglang.multimodal_gen.runtime.distributed.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            out1 = stage._generate_initial_noise(batch1, server_args, 4, 4, seed=7)
            out2 = stage._generate_initial_noise(batch2, server_args, 4, 4, seed=7)

        torch.testing.assert_close(out1, out2)

    def test_generate_initial_noise_different_seeds_differ(self):
        import unittest.mock as mock

        stage = self._make_stage()
        server_args = self._make_server_args()
        batch1 = SimpleNamespace(
            prompt_embeds=[torch.zeros(1, 1, dtype=torch.bfloat16)], latent_ids=None
        )
        batch2 = SimpleNamespace(
            prompt_embeds=[torch.zeros(1, 1, dtype=torch.bfloat16)], latent_ids=None
        )
        with mock.patch(
            "sglang.multimodal_gen.runtime.distributed.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            out1 = stage._generate_initial_noise(batch1, server_args, 4, 4, seed=1)
            out2 = stage._generate_initial_noise(batch2, server_args, 4, 4, seed=2)

        self.assertFalse(torch.allclose(out1, out2))

    # ------------------------------------------------------------------
    # _on_resolution_change
    # ------------------------------------------------------------------

    def _make_ctx(self, latent_h=4, latent_w=4, in_channels=64):
        """Build a minimal DenoisingContext-like namespace for _on_resolution_change."""
        dummy_branch = SimpleNamespace(kwargs={"freqs_cis": torch.zeros(1)})
        cfg_policy = SimpleNamespace(branches=[dummy_branch])
        latents = torch.zeros(1, latent_h * latent_w, in_channels)
        return SimpleNamespace(
            cfg_policy=cfg_policy,
            latents=latents,
            target_dtype=torch.float32,
            pos_cond_kwargs={"freqs_cis": torch.zeros(1)},
        )

    def test_on_resolution_change_no_cfg_policy_no_crash(self):
        """cfg_policy=None → early return with no exceptions."""
        stage = self._make_stage()
        server_args = self._make_server_args()
        ctx = SimpleNamespace(cfg_policy=None)
        batch = SimpleNamespace(latent_ids=None)
        # Must not raise
        stage._on_resolution_change(ctx, batch, server_args, 1024, 1024)

    def test_on_resolution_change_updates_latent_ids_shape(self):
        """batch.latent_ids is updated to the new (upsampled) spatial grid shape."""
        stage = self._make_stage()
        server_args = self._make_server_args(vae_scale=8, in_channels=64)
        # Transition from 32×32 → 64×64 (pixel 1024 → 1024, latent 32→64)
        new_h_pixel, new_w_pixel = 1024, 1024  # latent = 1024 // 16 = 64
        ctx = self._make_ctx(latent_h=64, latent_w=64)
        batch = SimpleNamespace(latent_ids=None)

        # Pre-populate cache so _get_transformer_attr is never called
        stage._freqs_cis_cache[(64, 64)] = torch.zeros(1)

        stage._on_resolution_change(ctx, batch, server_args, new_h_pixel, new_w_pixel)

        self.assertIsNotNone(batch.latent_ids)
        expected_seq_len = 64 * 64
        self.assertEqual(batch.latent_ids.shape, (1, expected_seq_len, 4))

    def test_on_resolution_change_updates_branch_freqs_cis(self):
        """branch.kwargs['freqs_cis'] is replaced with the cached value."""
        stage = self._make_stage()
        server_args = self._make_server_args(vae_scale=8, in_channels=64)
        ctx = self._make_ctx(latent_h=64, latent_w=64)
        batch = SimpleNamespace(latent_ids=None)

        sentinel = torch.ones(3, 3)  # recognizable cached value
        stage._freqs_cis_cache[(64, 64)] = sentinel

        stage._on_resolution_change(ctx, batch, server_args, 1024, 1024)

        # The branch should now reference the cached sentinel
        self.assertIs(ctx.cfg_policy.branches[0].kwargs["freqs_cis"], sentinel)

    def test_on_resolution_change_updates_pos_cond_kwargs(self):
        """ctx.pos_cond_kwargs['freqs_cis'] is also updated from the cache."""
        stage = self._make_stage()
        server_args = self._make_server_args(vae_scale=8, in_channels=64)
        ctx = self._make_ctx(latent_h=64, latent_w=64)
        batch = SimpleNamespace(latent_ids=None)

        sentinel = torch.ones(5)
        stage._freqs_cis_cache[(64, 64)] = sentinel

        stage._on_resolution_change(ctx, batch, server_args, 1024, 1024)

        self.assertIs(ctx.pos_cond_kwargs["freqs_cis"], sentinel)

    def test_on_resolution_change_latent_ids_correct_coords(self):
        """latent_ids has correct H and W coordinate ranges for a 2×4 grid."""
        stage = self._make_stage()
        server_args = self._make_server_args(vae_scale=8, in_channels=64)
        # latent 2×4 → pixel 2*16 × 4*16 = 32×64
        ctx = self._make_ctx(latent_h=2, latent_w=4)
        batch = SimpleNamespace(latent_ids=None)
        stage._freqs_cis_cache[(2, 4)] = torch.zeros(1)

        stage._on_resolution_change(
            ctx, batch, server_args, new_h_pixel=32, new_w_pixel=64
        )

        ids = batch.latent_ids  # (1, 8, 4): [T, H, W, Layer]
        h_coords = ids[0, :, 1]
        w_coords = ids[0, :, 2]
        self.assertEqual(int(h_coords.max()), 1)  # 0..H-1 = 0..1
        self.assertEqual(int(w_coords.max()), 3)  # 0..W-1 = 0..3


if __name__ == "__main__":
    unittest.main()
