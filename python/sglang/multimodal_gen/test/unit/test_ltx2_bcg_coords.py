import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
    _prepare_ltx2_rope_coords_for_bcg,
)
from sglang.test.test_utils import CustomTestCase


class TestLTX2BCGCoords(CustomTestCase):
    def setUp(self):
        self.video_result = torch.ones(1, 4)
        self.audio_result = torch.ones(1, 2)
        self.model = SimpleNamespace(
            rope=SimpleNamespace(
                prepare_video_coords=Mock(return_value=self.video_result)
            ),
            audio_rope=SimpleNamespace(
                prepare_audio_coords=Mock(return_value=self.audio_result)
            ),
        )
        self.video_latents = torch.zeros(2, 8, 16)
        self.audio_latents = torch.zeros(2, 4, 16)

    def _call(self, *, enabled, video_coords=None, audio_coords=None):
        return _prepare_ltx2_rope_coords_for_bcg(
            enabled=enabled,
            current_model=self.model,
            latent_model_input=self.video_latents,
            audio_latent_model_input=self.audio_latents,
            video_coords=video_coords,
            audio_coords=audio_coords,
            num_frames=16,
            height=32,
            width=48,
            audio_num_frames=126,
            fps=24,
        )

    def test_prepares_missing_coords_before_bcg_capture(self):
        video, audio = self._call(enabled=True)

        self.assertIs(video, self.video_result)
        self.assertIs(audio, self.audio_result)
        self.model.rope.prepare_video_coords.assert_called_once_with(
            batch_size=2,
            num_frames=16,
            height=32,
            width=48,
            device=self.video_latents.device,
            fps=24,
        )
        self.model.audio_rope.prepare_audio_coords.assert_called_once_with(
            batch_size=2,
            num_frames=126,
            device=self.audio_latents.device,
        )

    def test_disabled_bcg_keeps_legacy_none_coords(self):
        video, audio = self._call(enabled=False)

        self.assertIsNone(video)
        self.assertIsNone(audio)
        self.model.rope.prepare_video_coords.assert_not_called()
        self.model.audio_rope.prepare_audio_coords.assert_not_called()

    def test_existing_parallel_coords_are_preserved(self):
        existing_video = torch.zeros(3)
        existing_audio = torch.zeros(5)

        video, audio = self._call(
            enabled=True,
            video_coords=existing_video,
            audio_coords=existing_audio,
        )

        self.assertIs(video, existing_video)
        self.assertIs(audio, existing_audio)
        self.model.rope.prepare_video_coords.assert_not_called()
        self.model.audio_rope.prepare_audio_coords.assert_not_called()

    def test_stage_reuses_coords_across_denoising_steps(self):
        stage = object.__new__(LTX2DenoisingStage)
        stage._ltx2_coords_cache = OrderedDict()
        stage._ltx2_coords_cache_max_entries = 4
        pipeline_config = SimpleNamespace(
            prepare_video_rope_coords_for_sp=Mock(return_value=self.video_result),
            prepare_audio_rope_coords_for_sp=Mock(return_value=self.audio_result),
        )
        ctx = SimpleNamespace(
            latent_num_frames_for_model=16,
            latent_height=32,
            latent_width=48,
            use_ltx23_legacy_one_stage=False,
        )
        step = SimpleNamespace(current_model=self.model)
        batch = SimpleNamespace(
            fps=24,
            did_sp_shard_latents=False,
            did_sp_shard_audio_latents=False,
        )
        server_args = SimpleNamespace(
            pipeline_config=pipeline_config,
            enable_breakable_cuda_graph=False,
        )

        first = stage._get_ltx2_rope_coords(
            ctx,
            step,
            batch,
            server_args,
            self.video_latents,
            self.audio_latents,
            audio_num_frames_latent=126,
        )
        second = stage._get_ltx2_rope_coords(
            ctx,
            step,
            batch,
            server_args,
            self.video_latents,
            self.audio_latents,
            audio_num_frames_latent=126,
        )

        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])
        pipeline_config.prepare_video_rope_coords_for_sp.assert_called_once()
        pipeline_config.prepare_audio_rope_coords_for_sp.assert_called_once()


if __name__ == "__main__":
    unittest.main()
