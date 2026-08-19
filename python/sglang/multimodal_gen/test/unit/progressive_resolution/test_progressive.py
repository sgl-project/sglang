# SPDX-License-Identifier: Apache-2.0
"""Focused unit tests for the experimental progressive-resolution path."""

import argparse
import dataclasses
import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
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
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.ideogram import (
    Ideogram4ProgressiveDenoisingStage,
    _ideogram4_pack,
    _ideogram4_unpack,
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
        ideogram_stage = object.__new__(Ideogram4ProgressiveDenoisingStage)

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
        ideogram_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(patch_size=2, ae_scale_factor=8)
        )

        self.assertEqual(flux2_stage._latent_scale_factor(flux2_args), 16)
        self.assertEqual(wan_stage._latent_scale_factor(wan_args), 8)
        self.assertEqual(ideogram_stage._latent_scale_factor(ideogram_args), 16)


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


class TestIdeogram4LatentAdapters(unittest.TestCase):
    def test_row_major_roundtrip(self):
        # [B, C, H, W] → pack → unpack → original
        x = torch.arange(2 * 128 * 4 * 6, dtype=torch.float32).reshape(2, 128, 4, 6)

        packed = _ideogram4_pack(x)

        self.assertEqual(packed.shape, (2, 4 * 6, 128))
        torch.testing.assert_close(_ideogram4_unpack(packed, 4, 6), x)

    def test_pack_preserves_row_major_order(self):
        # Each packed token at position [b, row*W + col] should equal x[b, :, row, col]
        x = torch.arange(1 * 4 * 3 * 5, dtype=torch.float32).reshape(1, 4, 3, 5)
        packed = _ideogram4_pack(x)

        # token at spatial position (row=1, col=2) → flat index 1*5+2 = 7
        torch.testing.assert_close(packed[0, 7], x[0, :, 1, 2])

    def test_ideogram_matches_flux2_pack_unpack(self):
        # Ideogram and FLUX.2 share the same row-major token layout
        from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.flux_2 import (
            _flux2_pack,
            _flux2_unpack,
        )

        x = torch.randn(2, 128, 5, 7)
        torch.testing.assert_close(_ideogram4_pack(x), _flux2_pack(x))
        packed = _ideogram4_pack(x)
        torch.testing.assert_close(
            _ideogram4_unpack(packed, 5, 7), _flux2_unpack(packed, 5, 7)
        )


class TestIdeogram4OnResolutionChange(unittest.TestCase):
    """CPU-only test: _on_resolution_change rebuilds batch/ctx tensors correctly."""

    _PATCH = 2
    _AE = 8
    _SCALE = 16  # patch * ae = 16
    _IN_C = 128
    _LLM_DIM = 64  # small stand-in for llm_features_dim

    def _make_ideogram_extra(self, batch_size, grid_h, grid_w, max_text_tokens):
        """Build a minimal batch.extra["ideogram4"] dict matching _prepare_denoising_loop."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ideogram import (
            IMAGE_POSITION_OFFSET,
            LLM_TOKEN_INDICATOR,
            OUTPUT_IMAGE_INDICATOR,
            SEQUENCE_PADDING_INDICATOR,
        )

        num_image_tokens = grid_h * grid_w
        total_seq_len = max_text_tokens + num_image_tokens

        h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
        w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
        image_pos = (
            torch.stack([torch.zeros_like(h_idx), h_idx, w_idx], dim=1)
            + IMAGE_POSITION_OFFSET
        )

        position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
        segment_ids = torch.full(
            (batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long
        )
        indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

        for b in range(batch_size):
            # simulate a single item with no text padding for simplicity
            position_ids[b, :max_text_tokens] = (
                torch.arange(max_text_tokens).unsqueeze(-1).expand(-1, 3)
            )
            position_ids[b, max_text_tokens:] = image_pos
            segment_ids[b] = 1
            indicator[b, :max_text_tokens] = LLM_TOKEN_INDICATOR
            indicator[b, max_text_tokens:] = OUTPUT_IMAGE_INDICATOR

        return {
            "position_ids": position_ids,
            "segment_ids": segment_ids,
            "indicator": indicator,
            "num_image_tokens": num_image_tokens,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "max_text_tokens": max_text_tokens,
        }

    def _make_ctx_extra(self, batch_size, num_image_tokens, max_text_tokens):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ideogram import (
            OUTPUT_IMAGE_INDICATOR,
        )

        neg_llm_features = torch.zeros(batch_size, num_image_tokens, self._LLM_DIM)
        attn_mask = torch.ones(
            batch_size, max_text_tokens + num_image_tokens, dtype=torch.bool
        )
        neg_attn_mask = torch.ones(batch_size, num_image_tokens, dtype=torch.bool)
        return {
            "ideogram4_attn_mask": attn_mask,
            "ideogram4_attn_mask_meta": None,
            "ideogram4_neg_position_ids": torch.zeros(
                batch_size, num_image_tokens, 3, dtype=torch.long
            ),
            "ideogram4_neg_segment_ids": torch.ones(
                batch_size, num_image_tokens, dtype=torch.long
            ),
            "ideogram4_neg_indicator": torch.full(
                (batch_size, num_image_tokens), OUTPUT_IMAGE_INDICATOR, dtype=torch.long
            ),
            "ideogram4_neg_attn_mask": neg_attn_mask,
            "ideogram4_neg_attn_mask_meta": None,
            "ideogram4_neg_llm_features": neg_llm_features,
        }

    def test_resolution_change_doubles_image_tokens(self):
        B = 2
        old_grid_h, old_grid_w = 4, 4
        new_grid_h, new_grid_w = 8, 8
        max_text_tokens = 10

        stage = object.__new__(Ideogram4ProgressiveDenoisingStage)
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                patch_size=self._PATCH, ae_scale_factor=self._AE
            )
        )

        old_num_img = old_grid_h * old_grid_w  # 16
        new_num_img = new_grid_h * new_grid_w  # 64
        new_h_pixel = new_grid_h * self._SCALE
        new_w_pixel = new_grid_w * self._SCALE

        # Build fake ctx and batch
        ctx = SimpleNamespace(
            latents=torch.zeros(B, new_num_img, self._IN_C),
            extra=self._make_ctx_extra(B, old_num_img, max_text_tokens),
            # Non-None so _on_resolution_change proceeds (it early-returns when
            # cfg_policy is None); the value itself is not used further here.
            cfg_policy=CFGPolicy(),
        )
        batch = SimpleNamespace(
            extra={
                "ideogram4": self._make_ideogram_extra(
                    B, old_grid_h, old_grid_w, max_text_tokens
                )
            }
        )

        stage._on_resolution_change(ctx, batch, server_args, new_h_pixel, new_w_pixel)

        data = batch.extra["ideogram4"]
        self.assertEqual(data["num_image_tokens"], new_num_img)
        self.assertEqual(data["grid_h"], new_grid_h)
        self.assertEqual(data["grid_w"], new_grid_w)
        self.assertEqual(
            data["position_ids"].shape, (B, max_text_tokens + new_num_img, 3)
        )
        self.assertEqual(data["segment_ids"].shape, (B, max_text_tokens + new_num_img))
        self.assertEqual(data["indicator"].shape, (B, max_text_tokens + new_num_img))

        # ctx.extra tensors updated to new sizes
        self.assertEqual(
            ctx.extra["ideogram4_attn_mask"].shape,
            (B, max_text_tokens + new_num_img),
        )
        self.assertEqual(
            ctx.extra["ideogram4_neg_position_ids"].shape, (B, new_num_img, 3)
        )
        self.assertEqual(
            ctx.extra["ideogram4_neg_llm_features"].shape,
            (B, new_num_img, self._LLM_DIM),
        )

    def test_text_portion_is_unchanged_after_resolution_change(self):
        B = 1
        old_grid_h, old_grid_w = 4, 4
        new_grid_h, new_grid_w = 8, 8
        max_text_tokens = 6

        stage = object.__new__(Ideogram4ProgressiveDenoisingStage)
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                patch_size=self._PATCH, ae_scale_factor=self._AE
            )
        )

        old_data = self._make_ideogram_extra(B, old_grid_h, old_grid_w, max_text_tokens)
        old_text_position_ids = old_data["position_ids"][:, :max_text_tokens].clone()
        old_text_segment_ids = old_data["segment_ids"][:, :max_text_tokens].clone()
        old_text_indicator = old_data["indicator"][:, :max_text_tokens].clone()

        ctx = SimpleNamespace(
            latents=torch.zeros(B, new_grid_h * new_grid_w, self._IN_C),
            extra=self._make_ctx_extra(B, old_grid_h * old_grid_w, max_text_tokens),
            # Non-None so _on_resolution_change proceeds (it early-returns when
            # cfg_policy is None); the value itself is not used further here.
            cfg_policy=CFGPolicy(),
        )
        batch = SimpleNamespace(extra={"ideogram4": old_data})

        stage._on_resolution_change(
            ctx,
            batch,
            server_args,
            new_grid_h * self._SCALE,
            new_grid_w * self._SCALE,
        )

        data = batch.extra["ideogram4"]
        torch.testing.assert_close(
            data["position_ids"][:, :max_text_tokens], old_text_position_ids
        )
        torch.testing.assert_close(
            data["segment_ids"][:, :max_text_tokens], old_text_segment_ids
        )
        torch.testing.assert_close(
            data["indicator"][:, :max_text_tokens], old_text_indicator
        )

    def test_image_position_ids_use_grid_coordinates(self):
        B = 1
        grid_h, grid_w = 4, 6
        max_text_tokens = 4
        scale = self._SCALE

        stage = object.__new__(Ideogram4ProgressiveDenoisingStage)
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                patch_size=self._PATCH, ae_scale_factor=self._AE
            )
        )

        old_num_img = 2 * 3  # half the new grid
        ctx = SimpleNamespace(
            latents=torch.zeros(B, grid_h * grid_w, self._IN_C),
            extra=self._make_ctx_extra(B, old_num_img, max_text_tokens),
            # Non-None so _on_resolution_change proceeds (it early-returns when
            # cfg_policy is None); the value itself is not used further here.
            cfg_policy=CFGPolicy(),
        )
        batch = SimpleNamespace(
            extra={
                "ideogram4": self._make_ideogram_extra(
                    B, grid_h // 2, grid_w // 2, max_text_tokens
                )
            }
        )

        stage._on_resolution_change(
            ctx, batch, server_args, grid_h * scale, grid_w * scale
        )

        img_pos = batch.extra["ideogram4"]["position_ids"][0, max_text_tokens:]
        self.assertEqual(img_pos.shape, (grid_h * grid_w, 3))
        # All t-coordinates (dim 0) should be IMAGE_POSITION_OFFSET (the t=0 term)
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ideogram import (
            IMAGE_POSITION_OFFSET,
        )

        # t dimension (index 0) should be IMAGE_POSITION_OFFSET + 0
        self.assertTrue((img_pos[:, 0] == IMAGE_POSITION_OFFSET).all())
        # h dimension at row-major index row*grid_w + col should be IMAGE_POSITION_OFFSET + row
        for row in range(grid_h):
            for col in range(grid_w):
                idx = row * grid_w + col
                self.assertEqual(img_pos[idx, 1].item(), IMAGE_POSITION_OFFSET + row)
                self.assertEqual(img_pos[idx, 2].item(), IMAGE_POSITION_OFFSET + col)


if __name__ == "__main__":
    unittest.main()
