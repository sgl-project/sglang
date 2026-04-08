import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig


class TestLTX2SequenceParallelPadding(unittest.TestCase):
    def test_video_sp_shard_tracks_valid_tokens_for_padded_tail(self):
        cfg = LTX2PipelineConfig()
        batch = SimpleNamespace(
            height=cfg.vae_scale_factor,
            width=cfg.vae_scale_factor * 2,
        )
        latents = torch.randn(1, 6, 4)

        with patch(
            "sglang.multimodal_gen.configs.pipeline_configs.ltx_2.get_sp_world_size",
            return_value=2,
        ), patch(
            "sglang.multimodal_gen.configs.pipeline_configs.ltx_2.get_sp_parallel_rank",
            return_value=1,
        ):
            shard, did_shard = cfg.shard_latents_for_sp(batch, latents)

        self.assertTrue(did_shard)
        self.assertEqual(tuple(shard.shape), (1, 4, 4))
        self.assertEqual(batch.sp_video_latent_num_frames, 2)
        self.assertEqual(batch.sp_video_start_frame, 2)
        self.assertEqual(batch.sp_video_tokens_per_frame, 2)
        self.assertEqual(batch.sp_video_valid_token_count, 2)

    def test_audio_sp_shard_tracks_valid_tokens_for_padded_tail(self):
        cfg = LTX2PipelineConfig()
        batch = SimpleNamespace()
        audio_latents = torch.randn(1, 3, 8)

        with patch(
            "sglang.multimodal_gen.configs.pipeline_configs.ltx_2.get_sp_world_size",
            return_value=2,
        ), patch(
            "sglang.multimodal_gen.configs.pipeline_configs.ltx_2.get_sp_parallel_rank",
            return_value=1,
        ):
            shard, did_shard = cfg.shard_audio_latents_for_sp(batch, audio_latents)

        self.assertTrue(did_shard)
        self.assertEqual(tuple(shard.shape), (1, 2, 8))
        self.assertEqual(batch.sp_audio_latent_num_frames, 2)
        self.assertEqual(batch.sp_audio_start_frame, 2)
        self.assertEqual(batch.sp_audio_valid_token_count, 1)


if __name__ == "__main__":
    unittest.main()
