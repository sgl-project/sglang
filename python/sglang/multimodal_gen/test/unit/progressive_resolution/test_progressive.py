# SPDX-License-Identifier: Apache-2.0
"""Focused unit tests for the experimental progressive-resolution path."""

import argparse
import dataclasses
import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.denoising import (
    ProgressiveDenoisingStage,
    ProgressiveDenoisingStageRouter,
    compute_stage_transitions,
    find_transition_steps,
    is_progressive_resolution_mode,
    reset_scheduler_at_step,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.flux import (
    _flux_pack,
    _flux_unpack,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.flux_2 import (
    Flux2ProgressiveDenoisingStage,
    _flux2_pack,
    _flux2_unpack,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.qwen_image import (
    _qwen_image_pack,
    _qwen_image_unpack,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.spectral_ops import (
    dct_2d,
    idct_2d,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.upsample import (
    apply_upsample,
    dct_upsample_2d,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.wan import (
    WanProgressiveDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.zimage import (
    _zimage_repack,
    _zimage_unpack,
)


class _DummyDenoisingStage:
    parallelism_type = None

    def __init__(self, route_name: str):
        self.route_name = route_name
        self.component_manager = None
        self.registered_stage_name = None
        self.profile_stage_name = None

    def set_component_residency_manager(self, manager):
        self.component_manager = manager

    def set_registered_stage_name(self, stage_name: str):
        self.registered_stage_name = stage_name

    def set_profile_stage_name(self, stage_name: str):
        self.profile_stage_name = stage_name

    def component_uses(self, server_args, stage_name=None):
        return []

    def forward(self, batch, server_args):
        batch.route_name = self.route_name
        return batch


class TestProgressiveSamplingParams(unittest.TestCase):
    def _parse_cli_kwargs(self, argv: list[str]) -> dict:
        parser = argparse.ArgumentParser()
        SamplingParams.add_cli_args(parser)
        return SamplingParams.get_cli_args(parser.parse_args(argv))

    def test_defaults_and_valid_modes(self):
        params = SamplingParams()
        self.assertEqual(params.progressive_mode, "fullres")
        self.assertEqual(params.progressive_levels, 1)
        self.assertAlmostEqual(params.progressive_delta, 0.01)

        for mode in ("fullres", "dct", "dct_rewind"):
            with self.subTest(mode=mode):
                self.assertEqual(
                    SamplingParams(progressive_mode=mode).progressive_mode, mode
                )

    def test_validation_rejects_invalid_values(self):
        invalid_cases = [
            {"progressive_mode": "wavelet"},
            {"progressive_levels": 0},
            {"progressive_levels": True},
            {"progressive_delta": 0},
            {"progressive_delta": 1},
        ]
        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SamplingParams(**kwargs)

    def test_fields_stay_in_batch_signature(self):
        fields = {field.name: field for field in dataclasses.fields(SamplingParams)}
        for name in ("progressive_mode", "progressive_levels", "progressive_delta"):
            with self.subTest(field=name):
                self.assertFalse(fields[name].metadata.get("batch_sig_exclude"))

    def test_cli_only_emits_explicit_progressive_args(self):
        self.assertEqual(self._parse_cli_kwargs([]), {})

        kwargs = self._parse_cli_kwargs(
            [
                "--progressive-mode",
                "dct_rewind",
                "--progressive-levels",
                "2",
                "--progressive-delta",
                "0.05",
            ]
        )

        self.assertEqual(kwargs["progressive_mode"], "dct_rewind")
        self.assertEqual(kwargs["progressive_levels"], 2)
        self.assertAlmostEqual(kwargs["progressive_delta"], 0.05)


class TestProgressiveRouter(unittest.TestCase):
    def test_fullres_uses_standard_stage_without_constructing_progressive_stage(self):
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

    def test_progressive_stage_is_lazy_and_reused(self):
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

        with self.assertRaises(ValueError):
            router.forward(
                SimpleNamespace(progressive_mode="wavelet"), SimpleNamespace()
            )

    def test_mode_predicate(self):
        self.assertTrue(is_progressive_resolution_mode("dct"))
        self.assertTrue(is_progressive_resolution_mode("dct_rewind"))
        self.assertFalse(is_progressive_resolution_mode("fullres"))
        self.assertFalse(is_progressive_resolution_mode(None))


class TestStageTransitionHelpers(unittest.TestCase):
    def test_compute_stage_transitions_returns_one_threshold_per_stage(self):
        transitions = compute_stage_transitions(
            delta=0.01,
            n_levels=2,
            A=203.615097,
            beta=1.915461,
            H_lat=128,
            W_lat=128,
        )

        self.assertEqual(set(transitions), {1, 2, 3})
        self.assertEqual(transitions[1], 1.0)
        self.assertTrue(0 < transitions[2] < 1)
        self.assertTrue(0 < transitions[3] < 1)

    def test_find_transition_steps_maps_thresholds_to_scheduler_indices(self):
        scheduler_sigmas = torch.tensor([1.0, 0.8, 0.5, 0.25, 0.1])
        transitions = find_transition_steps(
            scheduler_sigmas,
            {1: 1.0, 2: 0.5, 3: 0.2},
            n_steps=5,
        )

        self.assertEqual(transitions, {2: 2, 3: 4})

    def test_reset_scheduler_clears_solver_state(self):
        scheduler = SimpleNamespace(
            config=SimpleNamespace(solver_order=2),
            model_outputs=[torch.ones(1), torch.ones(1)],
            lower_order_nums=1,
            last_sample=torch.ones(1),
            this_order=1,
            timestep_list=[1, 2],
            _step_index=0,
        )

        reset_scheduler_at_step(scheduler, 3)

        self.assertEqual(scheduler.model_outputs, [None, None])
        self.assertEqual(scheduler.lower_order_nums, 0)
        self.assertIsNone(scheduler.last_sample)
        self.assertEqual(scheduler.this_order, 0)
        self.assertEqual(scheduler.timestep_list, [None, None])
        self.assertEqual(scheduler._step_index, 3)


class TestSpectralUpsample(unittest.TestCase):
    def test_dct_roundtrip(self):
        x = torch.randn(2, 3, 8, 10)

        reconstructed = idct_2d(dct_2d(x))

        torch.testing.assert_close(reconstructed, x, rtol=1e-5, atol=1e-5)

    def test_apply_upsample_shapes_and_rewind_return(self):
        x = torch.randn(2, 3, 4, 5)

        out = apply_upsample(x, sigma_t=0.25, seed=[1, 2], mode="dct")
        rewind_out, t_eff = apply_upsample(
            x, sigma_t=0.25, seed=[1, 2], mode="dct_rewind"
        )

        self.assertEqual(out.shape, (2, 3, 8, 10))
        self.assertEqual(rewind_out.shape, (2, 3, 8, 10))
        self.assertGreater(t_eff, 0.25)
        self.assertEqual(out.dtype, x.dtype)

    def test_seed_list_is_batch_checked_and_deterministic(self):
        x = torch.randn(2, 3, 4, 4)

        out1 = dct_upsample_2d(x, sigma_t=0.1, seed=[7, 8])
        out2 = dct_upsample_2d(x, sigma_t=0.1, seed=[7, 8])

        torch.testing.assert_close(out1, out2)
        with self.assertRaises(ValueError):
            dct_upsample_2d(x, sigma_t=0.1, seed=[7])

    def test_invalid_upsample_mode_raises(self):
        with self.assertRaises(ValueError):
            apply_upsample(torch.zeros(1, 1, 2, 2), 0.1, 0, "wavelet")


class TestProgressiveStageHelpers(unittest.TestCase):
    def test_seed_helpers_support_batch_seed_lists(self):
        stage = object.__new__(ProgressiveDenoisingStage)

        batch = SimpleNamespace(batch_size=2, seeds=[11, 12], sampling_params=None)
        self.assertEqual(stage._get_seed(batch), 11)
        self.assertEqual(stage._get_seeds(batch, seed=0), [11, 12])

        batch = SimpleNamespace(
            prompt_embeds=[torch.zeros(3, 4, 5)],
            seeds=None,
            sampling_params=SimpleNamespace(seed=20),
        )
        self.assertEqual(stage._get_seeds(batch, seed=20), [20, 21, 22])

    def test_seed_helper_rejects_wrong_seed_count(self):
        stage = object.__new__(ProgressiveDenoisingStage)
        batch = SimpleNamespace(batch_size=2, seeds=[1], sampling_params=None)

        with self.assertRaises(ValueError):
            stage._get_seeds(batch, seed=0)

    def test_model_specific_latent_scale_factors(self):
        flux2_stage = object.__new__(Flux2ProgressiveDenoisingStage)
        wan_stage = object.__new__(WanProgressiveDenoisingStage)

        flux2_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                vae_config=SimpleNamespace(
                    arch_config=SimpleNamespace(vae_scale_factor=8)
                )
            )
        )
        wan_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                vae_config=SimpleNamespace(
                    arch_config=SimpleNamespace(spatial_compression_ratio=8)
                )
            )
        )

        self.assertEqual(flux2_stage._latent_scale_factor(flux2_args), 16)
        self.assertEqual(wan_stage._latent_scale_factor(wan_args), 8)


class TestLatentAdapters(unittest.TestCase):
    def test_flux_and_qwen_patchify_roundtrip(self):
        x = torch.arange(1 * 16 * 8 * 12, dtype=torch.float32).reshape(1, 16, 8, 12)

        for name, pack, unpack in (
            ("flux", _flux_pack, _flux_unpack),
            ("qwen_image", _qwen_image_pack, _qwen_image_unpack),
        ):
            with self.subTest(adapter=name):
                packed = pack(x, 8, 12)
                self.assertEqual(packed.shape, (1, (8 // 2) * (12 // 2), 64))
                torch.testing.assert_close(unpack(packed, 8, 12), x)

    def test_flux2_row_major_roundtrip(self):
        x = torch.arange(2 * 4 * 3 * 5, dtype=torch.float32).reshape(2, 4, 3, 5)

        packed = _flux2_pack(x)

        self.assertEqual(packed.shape, (2, 3 * 5, 4))
        torch.testing.assert_close(packed[0, 7], x[0, :, 1, 2])
        torch.testing.assert_close(_flux2_unpack(packed, 3, 5), x)

    def test_zimage_adds_and_removes_frame_dim(self):
        latent = torch.randn(2, 16, 1, 8, 8)

        spatial = _zimage_unpack(latent)

        self.assertEqual(spatial.shape, (2, 16, 8, 8))
        torch.testing.assert_close(_zimage_repack(spatial), latent)

    def test_wan_latent_adapter_is_identity(self):
        stage = object.__new__(WanProgressiveDenoisingStage)
        latent = torch.randn(1, 16, 5, 8, 8)

        self.assertIs(stage._unpack_latent(latent, 8, 8), latent)
        self.assertIs(
            stage._repack_latent(latent, 8, 8, SimpleNamespace(), SimpleNamespace()),
            latent,
        )


if __name__ == "__main__":
    unittest.main()
