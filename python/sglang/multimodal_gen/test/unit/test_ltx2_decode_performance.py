import os
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding_av import (
    LTX2AVDecodingStage,
)


def _make_stage_without_init() -> LTX2AVDecodingStage:
    """Build the stage without running __init__ (which needs real VAE modules)."""
    stage = LTX2AVDecodingStage.__new__(LTX2AVDecodingStage)
    stage._vae_decode_untiled_failed = False
    stage._vae_decode_untiled_logged = False
    return stage


class _FakeVAE:
    """Minimal VAE: untiled decode raises, tiled decode succeeds."""

    def __init__(self):
        self.tiling_enabled = False
        self.spatial_compression_ratio = 32
        self.temporal_compression_ratio = 8

    def enable_tiling(self):
        self.tiling_enabled = True

    def disable_tiling(self):
        self.tiling_enabled = False

    def decode(self, latents):
        if not self.tiling_enabled:
            raise RuntimeError("untiled decode boom")
        return "tiled-result"


class TestLTX2DecodePostprocess(unittest.TestCase):
    def test_postprocess_matches_legacy_videoprocessor_plus_save_path(self):
        """The on-GPU uint8 output must equal the legacy float postprocess once
        save_outputs applies its own float->uint8 cast (independent reference)."""
        from diffusers.video_processor import VideoProcessor

        torch.manual_seed(0)
        video = torch.empty(1, 3, 2, 4, 5, dtype=torch.float32).uniform_(-1.5, 1.5)

        # Legacy path: VideoProcessor float [0,1] np, then save_outputs uint8 cast.
        legacy_float = VideoProcessor(vae_scale_factor=32).postprocess_video(
            video, output_type="np"
        )
        legacy_uint8 = (np.clip(legacy_float, 0.0, 1.0) * 255.0).astype(np.uint8)

        out = LTX2AVDecodingStage._postprocess_video_to_uint8_np(video)

        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, legacy_uint8.shape)  # [B, T, H, W, C]
        self.assertTrue((out == legacy_uint8).all())

    def test_postprocess_rejects_non_5d(self):
        with self.assertRaises(TypeError):
            LTX2AVDecodingStage._postprocess_video_to_uint8_np(
                torch.zeros(3, 4, 5, dtype=torch.float32)
            )


class TestLTX2UntiledDecodeFallback(unittest.TestCase):
    def test_forced_untiled_failure_falls_back_to_tiled_and_latches(self):
        stage = _make_stage_without_init()
        vae = _FakeVAE()
        stage.vae = vae
        server_args = types.SimpleNamespace(
            pipeline_config=types.SimpleNamespace(vae_tiling=True)
        )
        latents = torch.zeros(1, 128, 2, 4, 4, dtype=torch.float32)

        with patch.dict(
            os.environ, {"SGLANG_DIFFUSION_LTX2_UNTILED_VAE_DECODE": "force"}
        ):
            # sanity: the policy actually resolves to the forced untiled path
            self.assertEqual(stage._vae_decode_untiled_mode(), "force")
            result = stage._decode_video_latents(latents, server_args)

        self.assertEqual(result, "tiled-result")
        self.assertTrue(stage._vae_decode_untiled_failed)  # latched off for the run
        self.assertTrue(vae.tiling_enabled)  # tiled path re-enabled tiling

    def test_off_mode_skips_untiled_and_uses_tiled(self):
        stage = _make_stage_without_init()
        vae = _FakeVAE()
        vae.enable_tiling()  # tiled decode would otherwise raise
        stage.vae = vae
        server_args = types.SimpleNamespace(
            pipeline_config=types.SimpleNamespace(vae_tiling=True)
        )
        latents = torch.zeros(1, 128, 2, 4, 4, dtype=torch.float32)

        with patch.dict(
            os.environ, {"SGLANG_DIFFUSION_LTX2_UNTILED_VAE_DECODE": "off"}
        ):
            self.assertEqual(stage._vae_decode_untiled_mode(), "off")
            result = stage._decode_video_latents(latents, server_args)

        self.assertEqual(result, "tiled-result")
        self.assertFalse(stage._vae_decode_untiled_failed)  # untiled never attempted


if __name__ == "__main__":
    unittest.main()
