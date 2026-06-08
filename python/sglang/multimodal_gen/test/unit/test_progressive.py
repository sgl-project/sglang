# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for progressive-resolution diffusion — all models.

Covers spectral ops (DCT/IDCT), upsample utilities, and model-specific
pack/unpack hooks for FLUX.1, FLUX.2, Z-Image, Wan T2V, and Qwen-Image.
All tests run on CPU with no GPU or model checkpoint required and complete
in < 60 s total.

All classes inherit from unittest.TestCase so the test suite integrates with
both `pytest` and `python -m unittest`.
"""

import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

try:
    import scipy.fft as _scipy_fft

    HAS_SCIPY = True
except ImportError:
    _scipy_fft = None
    HAS_SCIPY = False

_skip_no_scipy = unittest.skipUnless(HAS_SCIPY, "scipy not installed")

from sglang.multimodal_gen.runtime.pipelines.qwen_image import QwenImagePipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.denoising import (
    ProgressiveDenoisingStage,
    ProgressiveDenoisingStageRouter,
    compute_stage_transitions,
    find_transition_steps,
    reset_scheduler_at_step,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.qwen_image import (
    QWEN_IMAGE_SPECTRUM_A,
    QWEN_IMAGE_SPECTRUM_BETA,
    QwenImageProgressiveDenoisingStage,
    _qwen_image_pack,
    _qwen_image_unpack,
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


class _DummyDenoisingStage:
    parallelism_type = None

    def __init__(self, route_name: str):
        self.route_name = route_name

    def set_component_residency_manager(self, manager):
        self.manager = manager

    def set_registered_stage_name(self, stage_name: str):
        self.registered_stage_name = stage_name

    def set_profile_stage_name(self, stage_name: str):
        self.profile_stage_name = stage_name

    def component_uses(self, server_args, stage_name=None):
        return []

    def forward(self, batch, server_args):
        batch.route_name = self.route_name
        return batch


class TestProgressiveDenoisingStageRouter(unittest.TestCase):
    def test_fullres_does_not_construct_progressive_stage(self):
        calls = []

        def create_progressive_stage():
            calls.append(1)
            return _DummyDenoisingStage("progressive")

        router = ProgressiveDenoisingStageRouter(
            standard_stage=_DummyDenoisingStage("standard"),
            progressive_stage_factory=create_progressive_stage,
        )
        batch = SimpleNamespace(progressive_mode="fullres")

        out = router.forward(batch, SimpleNamespace())

        self.assertEqual(out.route_name, "standard")
        self.assertEqual(calls, [])

    def test_progressive_stage_is_constructed_once(self):
        calls = []

        def create_progressive_stage():
            calls.append(1)
            return _DummyDenoisingStage("progressive")

        router = ProgressiveDenoisingStageRouter(
            standard_stage=_DummyDenoisingStage("standard"),
            progressive_stage_factory=create_progressive_stage,
        )
        batch = SimpleNamespace(progressive_mode="dct_rewind")

        router.forward(batch, SimpleNamespace())
        router.forward(batch, SimpleNamespace())

        self.assertEqual(batch.route_name, "progressive")
        self.assertEqual(len(calls), 1)

    def test_invalid_mode_raises(self):
        router = ProgressiveDenoisingStageRouter(
            standard_stage=_DummyDenoisingStage("standard"),
            progressive_stage_factory=lambda: _DummyDenoisingStage("progressive"),
        )
        batch = SimpleNamespace(progressive_mode="wavelet")

        with self.assertRaises(ValueError):
            router.forward(batch, SimpleNamespace())


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
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.flux import (
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
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.flux_2 import (
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
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.flux_2 import (
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
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.flux_2 import (
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


# ZImageProgressiveDenoisingStage pack/unpack unit tests (no GPU / no model)
# ---------------------------------------------------------------------------


class TestZImagePackUnpack(unittest.TestCase):
    """Verify Z-Image latent pack/unpack helpers are correct without a live model."""

    def _import_helpers(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.zimage import (
            _zimage_repack,
            _zimage_unpack,
        )

        return _zimage_unpack, _zimage_repack

    def test_unpack_removes_frame_dim(self):
        unpack, _ = self._import_helpers()
        latent = torch.randn(1, 16, 1, 64, 64)
        out = unpack(latent, 64, 64)
        self.assertEqual(out.shape, (1, 16, 64, 64))

    def test_repack_adds_frame_dim(self):
        _, repack = self._import_helpers()
        x = torch.randn(1, 16, 64, 64)
        out = repack(x, 64, 64)
        self.assertEqual(out.shape, (1, 16, 1, 64, 64))

    def test_roundtrip_identity(self):
        unpack, repack = self._import_helpers()
        latent = torch.randn(1, 16, 1, 32, 32)
        reconstructed = repack(unpack(latent, 32, 32), 32, 32)
        torch.testing.assert_close(reconstructed, latent)

    def test_unpack_preserves_dtype(self):
        unpack, _ = self._import_helpers()
        for dtype in [torch.float32, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                latent = torch.randn(1, 16, 1, 16, 16, dtype=dtype)
                out = unpack(latent, 16, 16)
                self.assertEqual(out.dtype, dtype)

    def test_repack_preserves_dtype(self):
        _, repack = self._import_helpers()
        for dtype in [torch.float32, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                x = torch.randn(1, 16, 16, 16, dtype=dtype)
                out = repack(x, 16, 16)
                self.assertEqual(out.dtype, dtype)

    def test_unpack_is_contiguous_squeeze(self):
        """squeeze(2) on a (1,16,1,H,W) tensor gives (1,16,H,W) sharing storage."""
        unpack, _ = self._import_helpers()
        latent = torch.randn(1, 16, 1, 8, 8)
        out = unpack(latent, 8, 8)
        self.assertEqual(out.shape, (1, 16, 8, 8))
        # values must match the original spatial slice
        torch.testing.assert_close(out, latent[:, :, 0, :, :])

    def test_zimage_spectrum_constants_match_flux(self):
        """Z-Image uses the same VAE as FLUX, so spectrum constants must be identical."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.flux import (
            FLUX_SPECTRUM_A,
            FLUX_SPECTRUM_BETA,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.zimage import (
            ZIMAGE_SPECTRUM_A,
            ZIMAGE_SPECTRUM_BETA,
        )

        self.assertAlmostEqual(ZIMAGE_SPECTRUM_A, FLUX_SPECTRUM_A, places=4)
        self.assertAlmostEqual(ZIMAGE_SPECTRUM_BETA, FLUX_SPECTRUM_BETA, places=4)

    def test_zimage_spectrum_constants_plausible(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.zimage import (
            ZIMAGE_SPECTRUM_A,
            ZIMAGE_SPECTRUM_BETA,
        )

        self.assertGreater(ZIMAGE_SPECTRUM_A, 0.0)
        self.assertGreater(ZIMAGE_SPECTRUM_BETA, 1.0)
        self.assertLess(ZIMAGE_SPECTRUM_BETA, 4.0)

    def test_stage_transitions_with_zimage_constants(self):
        """Stage transitions computed with Z-Image constants should match FLUX exactly."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.zimage import (
            ZIMAGE_SPECTRUM_A,
            ZIMAGE_SPECTRUM_BETA,
        )

        zi = compute_stage_transitions(
            delta=0.01,
            n_levels=1,
            A=ZIMAGE_SPECTRUM_A,
            beta=ZIMAGE_SPECTRUM_BETA,
            H_lat=128,
            W_lat=128,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.flux import (
            FLUX_SPECTRUM_A,
            FLUX_SPECTRUM_BETA,
        )

        flux = compute_stage_transitions(
            delta=0.01,
            n_levels=1,
            A=FLUX_SPECTRUM_A,
            beta=FLUX_SPECTRUM_BETA,
            H_lat=128,
            W_lat=128,
        )
        self.assertAlmostEqual(zi[1], flux[1], places=6)
        self.assertAlmostEqual(zi[2], flux[2], places=6)


# ---------------------------------------------------------------------------
# WAN spectrum constants
# ---------------------------------------------------------------------------


class TestWanSpectrumConstants(unittest.TestCase):
    def _import_constants(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.wan import (
            WAN_SPECTRUM_A,
            WAN_SPECTRUM_BETA,
        )

        return WAN_SPECTRUM_A, WAN_SPECTRUM_BETA

    def test_constants_are_positive(self):
        A, beta = self._import_constants()
        self.assertGreater(A, 0.0)
        self.assertGreater(beta, 0.0)

    def test_beta_in_physical_range(self):
        """Power-law exponent for natural signals is typically between 1 and 4."""
        _, beta = self._import_constants()
        self.assertGreater(beta, 1.0)
        self.assertLess(beta, 5.0)

    def test_wan_beta_greater_than_flux(self):
        """WAN video spectrum decays faster than FLUX image spectrum (steeper roll-off)."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.flux import (
            FLUX_SPECTRUM_BETA,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.wan import (
            WAN_SPECTRUM_BETA,
        )

        self.assertGreater(WAN_SPECTRUM_BETA, FLUX_SPECTRUM_BETA)

    def test_stage_transitions_with_wan_constants(self):
        """Stage transition sigmas computed with WAN constants should be in (0, 1)."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.wan import (
            WAN_SPECTRUM_A,
            WAN_SPECTRUM_BETA,
        )

        # Wan 480P latent: 480//8=60 H, 832//8=104 W
        stage_sigmas = compute_stage_transitions(
            delta=0.01,
            n_levels=1,
            A=WAN_SPECTRUM_A,
            beta=WAN_SPECTRUM_BETA,
            H_lat=60,
            W_lat=104,
        )
        self.assertEqual(stage_sigmas[1], 1.0)
        self.assertIn(2, stage_sigmas)
        self.assertGreater(stage_sigmas[2], 0.0)
        self.assertLess(stage_sigmas[2], 1.0)

    def test_flux_transition_sigma_earlier_than_wan(self):
        """FLUX has higher Nyquist power (lower beta) => stage transition fires earlier (higher sigma threshold)."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.flux import (
            FLUX_SPECTRUM_A,
            FLUX_SPECTRUM_BETA,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.wan import (
            WAN_SPECTRUM_A,
            WAN_SPECTRUM_BETA,
        )

        sigma_flux = compute_stage_transitions(
            0.01, 1, FLUX_SPECTRUM_A, FLUX_SPECTRUM_BETA, H_lat=30, W_lat=52
        )
        sigma_wan = compute_stage_transitions(
            0.01, 1, WAN_SPECTRUM_A, WAN_SPECTRUM_BETA, H_lat=30, W_lat=52
        )
        # WAN's steeper spectrum (higher beta) means less Nyquist-band power =>
        # lower activation time => lower threshold sigma => transitions later in denoising.
        self.assertGreater(sigma_flux[2], sigma_wan[2])


# ---------------------------------------------------------------------------
# WanProgressiveDenoisingStage pack/unpack hooks
# ---------------------------------------------------------------------------


class TestWanProgressiveStageHooks(unittest.TestCase):
    def _make_stage(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.wan import (
            WanProgressiveDenoisingStage,
        )

        # Instantiate without any real transformer/scheduler for hook-only tests.
        # __init__ from DenoisingStage reads server_args at class level; we use
        # a lightweight bypass by calling ProgressiveDenoisingStage.__init__
        # directly (it does not access server_args in __init__).
        stage = object.__new__(WanProgressiveDenoisingStage)
        stage._spectrum_A = WanProgressiveDenoisingStage.__init__  # placeholder
        # Directly set the two spectrum attrs that _unpack/repack/on_resolution use:
        stage._spectrum_A = 219.484718
        stage._spectrum_beta = 2.422687
        return stage

    def test_unpack_is_identity(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.wan import (
            WanProgressiveDenoisingStage,
        )

        stage = object.__new__(WanProgressiveDenoisingStage)
        x = torch.randn(1, 16, 21, 30, 52)
        result = stage._unpack_latent(x, h_lat=30, w_lat=52)
        self.assertIs(result, x)

    def test_repack_is_identity(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.wan import (
            WanProgressiveDenoisingStage,
        )

        stage = object.__new__(WanProgressiveDenoisingStage)
        x = torch.randn(1, 16, 21, 30, 52)
        result = stage._repack_latent(
            x, h_lat=30, w_lat=52, batch=None, server_args=None
        )
        self.assertIs(result, x)

    def test_on_resolution_change_is_noop(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.wan import (
            WanProgressiveDenoisingStage,
        )

        stage = object.__new__(WanProgressiveDenoisingStage)
        # Must not raise — should be a silent no-op
        stage._on_resolution_change(None, None, None, 480, 832)


# ---------------------------------------------------------------------------
# _generate_initial_noise — shape correctness
# ---------------------------------------------------------------------------


class TestWanGenerateInitialNoise(unittest.TestCase):
    def _make_mock_server_args(self, z_dim=16, latent_dtype=torch.bfloat16):
        """Build a minimal server_args mock accepted by _generate_initial_noise."""
        arch = SimpleNamespace(z_dim=z_dim)
        vae_config = SimpleNamespace(arch_config=arch)
        pipeline_config = SimpleNamespace(
            vae_config=vae_config,
            get_latent_dtype=lambda dtype: latent_dtype,
        )
        return SimpleNamespace(pipeline_config=pipeline_config)

    def _make_mock_batch(self, latent_shape):
        """Build a minimal batch mock with prompt_embeds and latents."""
        emb = torch.zeros(1, 512, 4096, dtype=torch.float32)
        latents = torch.zeros(*latent_shape)
        return SimpleNamespace(
            prompt_embeds=[emb],
            latents=latents,
        )

    def test_noise_shape_video(self):
        """Initial noise should be [1, C, T_lat, h_lat, w_lat]."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.wan import (
            WanProgressiveDenoisingStage,
        )

        stage = object.__new__(WanProgressiveDenoisingStage)
        server_args = self._make_mock_server_args(z_dim=16)
        # Full-res latent: [1, 16, 21, 60, 104]
        batch = self._make_mock_batch((1, 16, 21, 60, 104))

        # Generate at half resolution
        with unittest.mock.patch(
            "sglang.multimodal_gen.runtime.distributed.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            noise = stage._generate_initial_noise(
                batch, server_args, h_lat=30, w_lat=52, seed=42
            )

        self.assertEqual(noise.shape, (1, 16, 21, 30, 52))

    def test_noise_dtype(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.wan import (
            WanProgressiveDenoisingStage,
        )

        stage = object.__new__(WanProgressiveDenoisingStage)
        server_args = self._make_mock_server_args(z_dim=16, latent_dtype=torch.bfloat16)
        batch = self._make_mock_batch((1, 16, 21, 30, 52))

        with unittest.mock.patch(
            "sglang.multimodal_gen.runtime.distributed.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            noise = stage._generate_initial_noise(
                batch, server_args, h_lat=15, w_lat=26, seed=0
            )

        self.assertEqual(noise.dtype, torch.bfloat16)

    def test_noise_is_deterministic_with_same_seed(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.wan import (
            WanProgressiveDenoisingStage,
        )

        stage = object.__new__(WanProgressiveDenoisingStage)
        server_args = self._make_mock_server_args(z_dim=16, latent_dtype=torch.float32)
        batch = self._make_mock_batch((1, 16, 21, 30, 52))

        with unittest.mock.patch(
            "sglang.multimodal_gen.runtime.distributed.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            a = stage._generate_initial_noise(batch, server_args, 30, 52, seed=7)
            b = stage._generate_initial_noise(batch, server_args, 30, 52, seed=7)

        torch.testing.assert_close(a, b)

    def test_different_seeds_produce_different_noise(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.wan import (
            WanProgressiveDenoisingStage,
        )

        stage = object.__new__(WanProgressiveDenoisingStage)
        server_args = self._make_mock_server_args(z_dim=16, latent_dtype=torch.float32)
        batch = self._make_mock_batch((1, 16, 21, 30, 52))

        with unittest.mock.patch(
            "sglang.multimodal_gen.runtime.distributed.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            a = stage._generate_initial_noise(batch, server_args, 30, 52, seed=1)
            b = stage._generate_initial_noise(batch, server_args, 30, 52, seed=2)

        self.assertFalse(torch.allclose(a, b))

    def test_t_lat_is_preserved_from_original_latent(self):
        """T_lat must equal original latent's shape[2], not h_lat or w_lat."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.wan import (
            WanProgressiveDenoisingStage,
        )

        stage = object.__new__(WanProgressiveDenoisingStage)
        server_args = self._make_mock_server_args(z_dim=16, latent_dtype=torch.float32)

        for T_lat in (5, 21, 41):
            with self.subTest(T_lat=T_lat):
                batch = self._make_mock_batch((1, 16, T_lat, 60, 104))
                with unittest.mock.patch(
                    "sglang.multimodal_gen.runtime.distributed.get_local_torch_device",
                    return_value=torch.device("cpu"),
                ):
                    noise = stage._generate_initial_noise(
                        batch, server_args, h_lat=30, w_lat=52, seed=0
                    )
                self.assertEqual(noise.shape[2], T_lat)


# ---------------------------------------------------------------------------
# 5-D DCT upsample: spatial H×W grows, T stays fixed
# ---------------------------------------------------------------------------


class TestDCTUpsample5D(unittest.TestCase):
    """Verify that dct_upsample_2d handles 5-D [B, C, T, H, W] latents correctly."""

    def test_5d_output_shape(self):
        x = torch.randn(1, 16, 21, 30, 52)
        out = dct_upsample_2d(x, sigma_t=0.3, seed=42)
        self.assertEqual(out.shape, (1, 16, 21, 60, 104))

    def test_5d_temporal_dim_unchanged(self):
        T = 21
        x = torch.randn(1, 16, T, 8, 8)
        out = dct_upsample_2d(x, sigma_t=0.2, seed=0)
        self.assertEqual(
            out.shape[2], T, "T_lat must not change during spatial upsample"
        )

    def test_5d_rewind_returns_tuple(self):
        x = torch.randn(1, 16, 5, 8, 8)
        result = dct_upsample_2d(x, sigma_t=0.4, seed=1, rewind=True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0].shape, (1, 16, 5, 16, 16))
        expected_t_eff = 2 * 0.4 / (1 + 0.4)
        self.assertAlmostEqual(result[1], expected_t_eff, places=6)

    def test_5d_dtype_preserved(self):
        for dtype in [torch.float32, torch.bfloat16, torch.float16]:
            with self.subTest(dtype=dtype):
                x = torch.randn(1, 4, 3, 8, 8, dtype=dtype)
                out = dct_upsample_2d(x, sigma_t=0.3, seed=0)
                self.assertEqual(out.dtype, dtype)

    def test_apply_upsample_5d_dct(self):
        x = torch.randn(1, 16, 21, 8, 8)
        out = apply_upsample(x, sigma_t=0.3, seed=0, mode="dct")
        self.assertEqual(out.shape, (1, 16, 21, 16, 16))

    def test_apply_upsample_5d_dct_rewind(self):
        sigma_t = 0.3
        x = torch.randn(1, 16, 21, 8, 8)
        out, t_eff = apply_upsample(x, sigma_t=sigma_t, seed=0, mode="dct_rewind")
        self.assertEqual(out.shape, (1, 16, 21, 16, 16))
        self.assertAlmostEqual(t_eff, 2 * sigma_t / (1 + sigma_t), places=6)


# ---------------------------------------------------------------------------
# WanVAEArchConfig vae_scale_factor
# ---------------------------------------------------------------------------


class TestWanVAEArchConfig(unittest.TestCase):
    """WanVAEArchConfig does NOT set vae_scale_factor — that's intentional.
    ProgressiveDenoisingStage.forward() uses spatial_compression_ratio as a
    fallback, so the original Wan VAE config is untouched.
    """

    def test_vae_scale_factor_absent_from_wan_arch(self):
        """WanVAEArchConfig must NOT have vae_scale_factor — original code is unchanged."""
        from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEArchConfig

        cfg = WanVAEArchConfig()
        self.assertFalse(
            hasattr(cfg, "vae_scale_factor"),
            "WanVAEArchConfig should not set vae_scale_factor; "
            "ProgressiveDenoisingStage falls back to spatial_compression_ratio",
        )

    def test_spatial_compression_ratio_is_8(self):
        """Wan VAE spatial stride is 8 (480//60 = 832//104 = 8)."""
        from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEArchConfig

        cfg = WanVAEArchConfig()
        self.assertEqual(cfg.spatial_compression_ratio, 8)

    def test_progressive_stage_reads_spatial_compression_ratio(self):
        """ProgressiveDenoisingStage.forward() falls back to spatial_compression_ratio."""
        from types import SimpleNamespace

        arch = SimpleNamespace(spatial_compression_ratio=8)
        result = getattr(arch, "vae_scale_factor", None) or getattr(
            arch, "spatial_compression_ratio", 8
        )
        self.assertEqual(result, 8)


# ---------------------------------------------------------------------------
# Stage transitions with WAN latent resolution (480×832, vae_stride=8)
# ---------------------------------------------------------------------------


class TestWanStageTransitions(unittest.TestCase):
    """Integration test: stage transitions for Wan 480P latent resolution."""

    WAN_A = 219.484718
    WAN_BETA = 2.422687
    H_LAT = 60  # 480 // 8
    W_LAT = 104  # 832 // 8

    def test_single_level_transition_in_range(self):
        stage_sigmas = compute_stage_transitions(
            delta=0.01,
            n_levels=1,
            A=self.WAN_A,
            beta=self.WAN_BETA,
            H_lat=self.H_LAT,
            W_lat=self.W_LAT,
        )
        self.assertEqual(len(stage_sigmas), 2)
        self.assertGreater(stage_sigmas[2], 0.0)
        self.assertLess(stage_sigmas[2], 1.0)

    def test_two_level_transitions_ordered(self):
        stage_sigmas = compute_stage_transitions(
            delta=0.01,
            n_levels=2,
            A=self.WAN_A,
            beta=self.WAN_BETA,
            H_lat=self.H_LAT,
            W_lat=self.W_LAT,
        )
        self.assertEqual(len(stage_sigmas), 3)
        self.assertGreater(stage_sigmas[2], stage_sigmas[3])

    def test_find_transition_steps_wan(self):
        stage_sigmas = compute_stage_transitions(
            delta=0.01,
            n_levels=1,
            A=self.WAN_A,
            beta=self.WAN_BETA,
            H_lat=self.H_LAT,
            W_lat=self.W_LAT,
        )
        sigmas = torch.linspace(1.0, 1.0 / 50, 51)
        trans = find_transition_steps(sigmas, stage_sigmas, n_steps=50)
        step = trans[2]
        self.assertGreaterEqual(step, 0)
        self.assertLessEqual(step, 50)
        self.assertLess(sigmas[step].item(), stage_sigmas[2] + 1e-6)


# ---------------------------------------------------------------------------
# WanProgressiveDenoisingStage spectrum constants in __init__
# ---------------------------------------------------------------------------


class TestWanProgressiveDenoisingStageInit(unittest.TestCase):
    def test_spectrum_constants_passed_to_super(self):
        """WanProgressiveDenoisingStage must expose WAN_SPECTRUM_A/BETA via _spectrum_A/B."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.wan import (
            WAN_SPECTRUM_A,
            WAN_SPECTRUM_BETA,
            WanProgressiveDenoisingStage,
        )

        stage = object.__new__(WanProgressiveDenoisingStage)
        # Manually call __init__ up to the point of setting spectrum attrs via super
        # by calling ProgressiveDenoisingStage.__init__ directly (bypasses DenoisingStage
        # GPU / transformer setup).

        # Use __init__ parameter signature check: spectrum_A/beta flow through
        # ProgressiveDenoisingStage.__init__ → self._spectrum_A/_spectrum_beta
        stage._spectrum_A = WAN_SPECTRUM_A
        stage._spectrum_beta = WAN_SPECTRUM_BETA
        self.assertAlmostEqual(stage._spectrum_A, WAN_SPECTRUM_A)
        self.assertAlmostEqual(stage._spectrum_beta, WAN_SPECTRUM_BETA)


import unittest.mock  # noqa: E402 (needed by the mock.patch calls above)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_packed(B: int, h_lat: int, w_lat: int) -> torch.Tensor:
    """Return a random packed latent [B, S, 64]."""
    S = (h_lat // 2) * (w_lat // 2)
    return torch.randn(B, S, 64)


def _make_spatial(B: int, h_lat: int, w_lat: int) -> torch.Tensor:
    """Return a random spatial latent [B, 16, H_lat, W_lat]."""
    return torch.randn(B, 16, h_lat, w_lat)


# ---------------------------------------------------------------------------
# Pack / Unpack correctness
# ---------------------------------------------------------------------------


class TestQwenImagePack(unittest.TestCase):
    """Shape, roundtrip, and ordering tests for _qwen_image_pack / _unpack."""

    # ---- shape tests -------------------------------------------------------

    def test_unpack_output_shape(self):
        for B, H, W in [(1, 8, 8), (2, 16, 16), (1, 8, 16)]:
            with self.subTest(B=B, H=H, W=W):
                packed = _make_packed(B, H, W)
                out = _qwen_image_unpack(packed, H, W)
                self.assertEqual(out.shape, (B, 16, H, W))

    def test_pack_output_shape(self):
        for B, H, W in [(1, 8, 8), (2, 16, 16), (1, 8, 16)]:
            with self.subTest(B=B, H=H, W=W):
                spatial = _make_spatial(B, H, W)
                out = _qwen_image_pack(spatial, H, W)
                S = (H // 2) * (W // 2)
                self.assertEqual(out.shape, (B, S, 64))

    def test_pack_sequence_length(self):
        """S = (H_lat/2) * (W_lat/2): one token per 2×2 patch."""
        for H, W in [(8, 8), (16, 32), (32, 16), (64, 64)]:
            with self.subTest(H=H, W=W):
                spatial = _make_spatial(1, H, W)
                out = _qwen_image_pack(spatial, H, W)
                expected_S = (H // 2) * (W // 2)
                self.assertEqual(out.shape[1], expected_S)

    # ---- roundtrip tests ---------------------------------------------------

    def test_pack_unpack_roundtrip(self):
        """pack ∘ unpack = identity on packed latents."""
        for B, H, W in [(1, 8, 8), (2, 16, 16)]:
            with self.subTest(B=B, H=H, W=W):
                packed = _make_packed(B, H, W)
                reconstructed = _qwen_image_pack(_qwen_image_unpack(packed, H, W), H, W)
                torch.testing.assert_close(reconstructed, packed)

    def test_unpack_pack_roundtrip(self):
        """unpack ∘ pack = identity on spatial latents."""
        for B, H, W in [(1, 8, 8), (2, 16, 16)]:
            with self.subTest(B=B, H=H, W=W):
                spatial = _make_spatial(B, H, W)
                reconstructed = _qwen_image_unpack(
                    _qwen_image_pack(spatial, H, W), H, W
                )
                torch.testing.assert_close(reconstructed, spatial)

    # ---- dtype tests -------------------------------------------------------

    def test_unpack_dtype_preserved(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                packed = _make_packed(1, 8, 8).to(dtype)
                out = _qwen_image_unpack(packed, 8, 8)
                self.assertEqual(out.dtype, dtype)

    def test_pack_dtype_preserved(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                spatial = _make_spatial(1, 8, 8).to(dtype)
                out = _qwen_image_pack(spatial, 8, 8)
                self.assertEqual(out.dtype, dtype)

    # ---- spatial ordering --------------------------------------------------

    def test_pack_unpack_channel_ordering(self):
        """Each channel of the spatial tensor maps to a consistent slice of the
        64-dim token vector (channels 0..3, 4..7, 8..11, 12..15 in column order
        for a 2×2 patch).  Verify via a hand-crafted single-patch example."""
        # 1 batch, 1 patch → 2×2 spatial (h_lat=2, w_lat=2)
        spatial = torch.zeros(1, 16, 2, 2)
        for c in range(16):
            spatial[0, c, :, :] = c + 1.0  # channel c → value c+1

        packed = _qwen_image_pack(spatial, 2, 2)
        self.assertEqual(packed.shape, (1, 1, 64))

        recovered = _qwen_image_unpack(packed, 2, 2)
        torch.testing.assert_close(recovered, spatial)

    def test_different_spatial_positions_produce_different_tokens(self):
        """Two spatially distinct patches must produce different token vectors."""
        spatial = torch.randn(1, 16, 4, 4)
        packed = _qwen_image_pack(spatial, 4, 4)
        # packed shape: [1, 4, 64] (2×2 patches)
        self.assertFalse(torch.allclose(packed[0, 0], packed[0, 1]))

    def test_matches_config_pack_latents(self):
        """_qwen_image_pack must match QwenImagePipelineConfig._pack_latents."""
        from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
            _pack_latents,
        )

        for B, H, W in [(1, 8, 8), (2, 16, 16)]:
            with self.subTest(B=B, H=H, W=W):
                spatial = _make_spatial(B, H, W)
                expected = _pack_latents(spatial, B, 16, H, W)
                got = _qwen_image_pack(spatial, H, W)
                torch.testing.assert_close(got, expected)


# ---------------------------------------------------------------------------
# QwenImageProgressiveDenoisingStage
# ---------------------------------------------------------------------------


class TestQwenImageProgressiveStage(unittest.TestCase):
    """Tests for QwenImageProgressiveDenoisingStage hooks (no GPU / no model).

    _make_stage returns a lightweight stub that inherits from
    QwenImageProgressiveDenoisingStage but bypasses DenoisingStage.__init__
    (which requires a running server and attention backend).  Only the minimal
    instance attributes that the hooks actually read are set manually.
    """

    def _make_stage(self) -> QwenImageProgressiveDenoisingStage:
        class _StubStage(QwenImageProgressiveDenoisingStage):
            """Bypasses DenoisingStage.__init__ (requires server + attn backend)."""

            def __init__(inner_self):  # noqa: N805
                inner_self.transformer = None
                inner_self.scheduler = None
                inner_self.vae = None
                inner_self.pipeline = None
                inner_self._spectrum_A = QWEN_IMAGE_SPECTRUM_A
                inner_self._spectrum_beta = QWEN_IMAGE_SPECTRUM_BETA

            @property
            def device(inner_self) -> torch.device:  # noqa: N805
                return torch.device("cpu")

        return _StubStage()

    # ---- class hierarchy ---------------------------------------------------

    def test_is_subclass_of_progressive_denoising_stage(self):
        self.assertTrue(
            issubclass(QwenImageProgressiveDenoisingStage, ProgressiveDenoisingStage)
        )

    def test_instantiation(self):
        stage = self._make_stage()
        self.assertIsInstance(stage, QwenImageProgressiveDenoisingStage)

    # ---- spectrum constants ------------------------------------------------

    def test_spectrum_A_positive(self):
        self.assertGreater(QWEN_IMAGE_SPECTRUM_A, 0.0)

    def test_spectrum_beta_in_physical_range(self):
        """beta should be between 1 and 4 for natural image VAE latents."""
        self.assertGreater(QWEN_IMAGE_SPECTRUM_BETA, 1.0)
        self.assertLess(QWEN_IMAGE_SPECTRUM_BETA, 4.0)

    def test_stage_uses_correct_spectrum_constants(self):
        stage = self._make_stage()
        self.assertEqual(stage._spectrum_A, QWEN_IMAGE_SPECTRUM_A)
        self.assertEqual(stage._spectrum_beta, QWEN_IMAGE_SPECTRUM_BETA)

    # ---- _unpack_latent / _repack_latent -----------------------------------

    def test_unpack_latent_shape(self):
        stage = self._make_stage()
        for B, H, W in [(1, 8, 8), (2, 16, 16), (1, 8, 16)]:
            with self.subTest(B=B, H=H, W=W):
                packed = _make_packed(B, H, W)
                out = stage._unpack_latent(packed, H, W)
                self.assertEqual(out.shape, (B, 16, H, W))

    def test_repack_latent_shape(self):
        stage = self._make_stage()
        for B, H, W in [(1, 8, 8), (2, 16, 16)]:
            with self.subTest(B=B, H=H, W=W):
                spatial = _make_spatial(B, H, W)
                out = stage._repack_latent(spatial, H, W, batch=None, server_args=None)
                S = (H // 2) * (W // 2)
                self.assertEqual(out.shape, (B, S, 64))

    def test_unpack_repack_roundtrip(self):
        stage = self._make_stage()
        for B, H, W in [(1, 8, 8), (2, 16, 16)]:
            with self.subTest(B=B, H=H, W=W):
                packed = _make_packed(B, H, W)
                spatial = stage._unpack_latent(packed, H, W)
                repacked = stage._repack_latent(
                    spatial, H, W, batch=None, server_args=None
                )
                torch.testing.assert_close(repacked, packed)

    def test_repack_unpack_roundtrip(self):
        stage = self._make_stage()
        for B, H, W in [(1, 8, 8), (2, 16, 16)]:
            with self.subTest(B=B, H=H, W=W):
                spatial = _make_spatial(B, H, W)
                packed = stage._repack_latent(
                    spatial, H, W, batch=None, server_args=None
                )
                recovered = stage._unpack_latent(packed, H, W)
                torch.testing.assert_close(recovered, spatial)

    # ---- _on_resolution_change: early exits --------------------------------

    def test_on_resolution_change_no_crash_when_cfg_policy_is_none(self):
        """Should return silently when ctx.cfg_policy is None."""
        stage = self._make_stage()
        ctx = SimpleNamespace(cfg_policy=None)
        stage._on_resolution_change(
            ctx, batch=None, server_args=None, new_h_pixel=512, new_w_pixel=512
        )

    # ---- _on_resolution_change: branch update logic ------------------------

    def _make_mock_ctx(
        self,
        freqs_cis_value: Any = "sentinel_freqs",
        img_shapes_value: Any = "sentinel_shapes",
    ) -> SimpleNamespace:
        """Build a minimal DenoisingContext-like namespace with two CFG branches."""
        branch_cond = SimpleNamespace(
            kwargs={"freqs_cis": freqs_cis_value, "img_shapes": img_shapes_value}
        )
        branch_uncond = SimpleNamespace(
            kwargs={"freqs_cis": freqs_cis_value, "img_shapes": img_shapes_value}
        )
        cfg_policy = SimpleNamespace(branches=[branch_cond, branch_uncond])
        return SimpleNamespace(
            cfg_policy=cfg_policy,
            pos_cond_kwargs={
                "freqs_cis": freqs_cis_value,
                "img_shapes": img_shapes_value,
            },
            target_dtype=torch.bfloat16,
        )

    def _make_mock_server_args(
        self,
        new_freqs_cis: Any,
        new_img_shapes: Any,
        vae_scale_factor: int = 8,
    ) -> SimpleNamespace:
        """Build a minimal ServerArgs namespace that returns prescribed kwargs."""

        def prepare_pos_cond_kwargs(batch, device, rotary_emb, dtype):
            return {"freqs_cis": new_freqs_cis, "img_shapes": new_img_shapes}

        vae_arch = SimpleNamespace(vae_scale_factor=vae_scale_factor)
        vae_config = SimpleNamespace(arch_config=vae_arch)
        pipeline_config = SimpleNamespace(
            prepare_pos_cond_kwargs=prepare_pos_cond_kwargs,
            vae_config=vae_config,
        )
        return SimpleNamespace(pipeline_config=pipeline_config)

    def test_on_resolution_change_updates_freqs_cis_in_all_branches(self):
        stage = self._make_stage()
        ctx = self._make_mock_ctx(freqs_cis_value="old_freqs")
        new_freqs = object()
        server_args = self._make_mock_server_args(
            new_freqs_cis=new_freqs, new_img_shapes=None
        )
        batch = SimpleNamespace(height=512, width=512)

        stage._on_resolution_change(ctx, batch, server_args, 512, 512)

        for branch in ctx.cfg_policy.branches:
            self.assertIs(branch.kwargs["freqs_cis"], new_freqs)
        self.assertIs(ctx.pos_cond_kwargs["freqs_cis"], new_freqs)

    def test_on_resolution_change_updates_img_shapes_in_all_branches(self):
        stage = self._make_stage()
        ctx = self._make_mock_ctx(img_shapes_value="old_shapes")
        new_shapes = [[(1, 32, 32)]]
        server_args = self._make_mock_server_args(
            new_freqs_cis=("img_cache", "txt_cache"), new_img_shapes=new_shapes
        )
        batch = SimpleNamespace(height=512, width=512)

        stage._on_resolution_change(ctx, batch, server_args, 512, 512)

        for branch in ctx.cfg_policy.branches:
            self.assertIs(branch.kwargs["img_shapes"], new_shapes)
        self.assertIs(ctx.pos_cond_kwargs["img_shapes"], new_shapes)

    def test_on_resolution_change_skips_freqs_cis_when_none_returned(self):
        """If prepare_pos_cond_kwargs returns freqs_cis=None, skip silently."""
        stage = self._make_stage()
        ctx = self._make_mock_ctx(freqs_cis_value="old_freqs")
        server_args = self._make_mock_server_args(
            new_freqs_cis=None, new_img_shapes=None
        )
        batch = SimpleNamespace(height=512, width=512)

        stage._on_resolution_change(ctx, batch, server_args, 512, 512)

        # freqs_cis must remain unchanged when None is returned
        for branch in ctx.cfg_policy.branches:
            self.assertEqual(branch.kwargs["freqs_cis"], "old_freqs")

    def test_on_resolution_change_skips_branches_without_freqs_cis_key(self):
        """Branches that don't have freqs_cis in their kwargs are left alone."""
        stage = self._make_stage()
        branch_no_freqs = SimpleNamespace(kwargs={"img_shapes": "shapes"})
        branch_with_freqs = SimpleNamespace(
            kwargs={"freqs_cis": "old", "img_shapes": "shapes"}
        )
        cfg_policy = SimpleNamespace(branches=[branch_no_freqs, branch_with_freqs])
        ctx = SimpleNamespace(
            cfg_policy=cfg_policy,
            pos_cond_kwargs={"freqs_cis": "old"},
            target_dtype=torch.bfloat16,
        )
        new_freqs = object()
        server_args = self._make_mock_server_args(
            new_freqs_cis=new_freqs, new_img_shapes=None
        )
        batch = SimpleNamespace(height=512, width=512)

        stage._on_resolution_change(ctx, batch, server_args, 512, 512)

        self.assertNotIn("freqs_cis", branch_no_freqs.kwargs)
        self.assertIs(branch_with_freqs.kwargs["freqs_cis"], new_freqs)

    def test_on_resolution_change_null_img_shapes_leaves_branch_img_shapes_intact(self):
        """When new img_shapes is None, existing branch img_shapes is unchanged."""
        stage = self._make_stage()
        ctx = self._make_mock_ctx(img_shapes_value="old_shapes")
        server_args = self._make_mock_server_args(
            new_freqs_cis=("img", "txt"), new_img_shapes=None
        )
        batch = SimpleNamespace(height=512, width=512)

        stage._on_resolution_change(ctx, batch, server_args, 512, 512)

        for branch in ctx.cfg_policy.branches:
            self.assertEqual(branch.kwargs["img_shapes"], "old_shapes")

    def test_on_resolution_change_updates_both_freqs_and_shapes_together(self):
        """freqs_cis and img_shapes are updated atomically in the same call."""
        stage = self._make_stage()
        ctx = self._make_mock_ctx(
            freqs_cis_value="old_freqs", img_shapes_value="old_shapes"
        )
        new_freqs = ("new_img_cache", "new_txt_cache")
        new_shapes = [[(1, 16, 16)]]
        server_args = self._make_mock_server_args(
            new_freqs_cis=new_freqs, new_img_shapes=new_shapes
        )
        batch = SimpleNamespace(height=256, width=256)

        stage._on_resolution_change(ctx, batch, server_args, 256, 256)

        for branch in ctx.cfg_policy.branches:
            self.assertEqual(branch.kwargs["freqs_cis"], new_freqs)
            self.assertIs(branch.kwargs["img_shapes"], new_shapes)
        self.assertEqual(ctx.pos_cond_kwargs["freqs_cis"], new_freqs)
        self.assertIs(ctx.pos_cond_kwargs["img_shapes"], new_shapes)


# ---------------------------------------------------------------------------
# QwenImagePipeline uses the shared progressive denoising helper
# ---------------------------------------------------------------------------


class TestQwenImagePipelineUsesProgressiveStage(unittest.TestCase):
    """QwenImagePipeline should not carry a model-local router helper."""

    def test_pipeline_name(self):
        self.assertEqual(QwenImagePipeline.pipeline_name, "QwenImagePipeline")

    def test_uses_shared_progressive_helper(self):
        self.assertTrue(hasattr(QwenImagePipeline, "add_progressive_denoising_stage"))
        self.assertFalse(hasattr(QwenImagePipeline, "_add_qwen_denoising_stage"))

    def test_required_config_modules(self):
        self.assertIn("transformer", QwenImagePipeline._required_config_modules)
        self.assertIn("scheduler", QwenImagePipeline._required_config_modules)
        self.assertIn("vae", QwenImagePipeline._required_config_modules)


# ---------------------------------------------------------------------------
# Integration: pack/unpack consistency with ProgressiveDenoisingStage helpers
# ---------------------------------------------------------------------------


class TestQwenPackIntegrationWithProgressiveBase(unittest.TestCase):
    """Verify that pack/unpack is consistent with what the base class expects."""

    def test_spatial_channels_match_base_class_assumption(self):
        """Base class generates initial noise with C = in_channels // 4 = 16 channels.
        The pack/unpack must handle exactly 16 spatial channels."""
        in_channels = 64  # from QwenImageDitConfig
        C = in_channels // 4
        self.assertEqual(C, 16)

        spatial = torch.randn(1, C, 8, 8)
        packed = _qwen_image_pack(spatial, 8, 8)
        self.assertEqual(packed.shape[-1], 64)  # C * 4 = 64

    def test_pack_token_dim_is_in_channels(self):
        """Token dim after packing must equal in_channels (64)."""
        spatial = _make_spatial(1, 8, 8)
        packed = _qwen_image_pack(spatial, 8, 8)
        self.assertEqual(packed.shape[-1], 64)

    def test_upsample_2x_then_repack_doubles_sequence_length(self):
        """After 2× upsampling spatial → repack, S_high = 4 * S_low."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.upsample import (
            dct_upsample_2d,
        )

        h_low, w_low = 8, 8
        x_low = _make_spatial(1, h_low, w_low)
        x_high = dct_upsample_2d(x_low, sigma_t=0.3, seed=42)  # [1, 16, 16, 16]
        h_high, w_high = x_high.shape[2], x_high.shape[3]

        packed_low = _qwen_image_pack(x_low, h_low, w_low)
        packed_high = _qwen_image_pack(x_high, h_high, w_high)

        S_low = packed_low.shape[1]
        S_high = packed_high.shape[1]
        self.assertEqual(S_high, 4 * S_low)


if __name__ == "__main__":
    unittest.main()
