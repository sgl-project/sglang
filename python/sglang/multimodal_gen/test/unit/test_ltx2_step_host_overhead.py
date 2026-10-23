import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.multimodal_gen.runtime.models.dits.ltx_2 import (
    LTX2AudioVideoRotaryPosEmbed,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)
from sglang.multimodal_gen.test.unit.host_sync_guard import forbid_host_syncs
from sglang.test.test_utils import CustomTestCase


class _OpRecorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(func)
        return func(*args, **(kwargs or {}))


class TestLTX2StepHostOverhead(CustomTestCase):
    def test_video_coord_constants_uploaded_once(self):
        def rope():
            return LTX2AudioVideoRotaryPosEmbed(
                dim=48, patch_size=1, patch_size_t=1, scale_factors=(8, 32, 32)
            )

        kwargs = dict(
            batch_size=2, num_frames=3, height=4, width=5, device=torch.device("cpu")
        )
        expected = rope().prepare_video_coords(**kwargs, fps=24.0)
        cached = rope()
        cached.prepare_video_coords(**kwargs, fps=24.0)
        # every later call reuses the device constants instead of uploading them
        with patch.object(torch, "tensor", wraps=torch.tensor) as upload:
            for start_frame in (0, 3):
                actual = cached.prepare_video_coords(
                    **kwargs, fps=24.0, start_frame=start_frame
                )
        self.assertEqual(upload.call_count, 0)
        self.assertTrue(
            torch.equal(cached.prepare_video_coords(**kwargs, fps=24.0), expected)
        )
        self.assertTrue(
            torch.equal(
                actual,
                rope().prepare_video_coords(**kwargs, fps=24.0, start_frame=3),
            )
        )

    def test_sde_coefficients_never_read_on_host(self):
        sigma_next = torch.tensor(0.4, dtype=torch.float64)
        # a 0-dim boolean index goes through nonzero, a device sync
        with forbid_host_syncs():
            alpha_ratio, sigma_down, sigma_up = LTX2DenoisingStage._ltx2_get_sde_coeff(
                sigma_next, sigma_up=sigma_next * 0.5
            )
        sigma_up_ref = (sigma_next * 0.5).clamp(max=sigma_next * 0.9999)
        residual = (sigma_next**2 - sigma_up_ref**2).clamp(min=0) ** 0.5
        alpha_ref = (1 - sigma_next) + residual
        self.assertTrue(torch.equal(alpha_ratio, alpha_ref))
        self.assertTrue(torch.equal(sigma_down, residual / alpha_ref))
        self.assertTrue(torch.equal(sigma_up, sigma_up_ref))

    @torch.no_grad()
    def test_bongmath_refinement_casts_once(self):
        stage = object.__new__(LTX2DenoisingStage)
        stage._ltx2_apply_clean_latent_mask = lambda latents, ctx: latents
        generator = torch.Generator().manual_seed(0)
        ctx = SimpleNamespace(
            latents=torch.randn(1, 6, 4, generator=generator),
            audio_latents=torch.randn(1, 3, 4, generator=generator),
            use_native_hq_res2s_sde_noise=False,
        )
        stage._randn_like_with_batch_generators = lambda latents, batch: torch.randn(
            latents.shape, generator=generator
        )

        def midpoint_model_call(video, audio, sigma):
            return video * 0.5, audio * 0.5, None, None

        recorder = _OpRecorder()
        with recorder:
            video, audio = stage._ltx2_stage2_res2s_step(
                ctx=ctx,
                batch=None,
                sigma=torch.tensor(0.6),
                sigma_next=torch.tensor(0.45),
                model_video_velocity=torch.ones(1, 6, 4),
                model_audio_velocity=torch.ones(1, 3, 4),
                model_video_timestep=None,
                model_audio_timestep=None,
                midpoint_model_call=midpoint_model_call,
            )
        self.assertTrue(torch.isfinite(video).all() and torch.isfinite(audio).all())
        # the 100-iteration refinement casts the denoised estimates once, not per
        # iteration (two modalities x 100 casts before)
        casts = sum(op is torch.ops.aten._to_copy.default for op in recorder.ops)
        self.assertLess(casts, 100)


if __name__ == "__main__":
    unittest.main()
