import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
from sglang.multimodal_gen.runtime.models.dits.ltx_2 import LTX2Attention


class _FakeColumnParallelLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        quant_config=None,
    ) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor):
        return self.linear(x), None


class _FakeRowParallelLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        quant_config=None,
    ) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor):
        return self.linear(x), None


class _CapturingUSPAttention(torch.nn.Module):
    last_attn_mask = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, q, k, v, attn_mask=None, **kwargs):
        _CapturingUSPAttention.last_attn_mask = attn_mask
        return v


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

    def test_ltx2_attention_passes_mask_to_usp_attention(self):
        x = torch.randn(1, 4, 8)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)

        with patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.get_tp_world_size",
            return_value=1,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.ColumnParallelLinear",
            _FakeColumnParallelLinear,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.RowParallelLinear",
            _FakeRowParallelLinear,
        ), patch(
            "sglang.multimodal_gen.runtime.models.dits.ltx_2.USPAttention",
            _CapturingUSPAttention,
        ):
            attn = LTX2Attention(
                query_dim=8,
                heads=2,
                dim_head=4,
                qk_norm=False,
                use_local_attention=False,
            )
            _CapturingUSPAttention.last_attn_mask = None
            _ = attn(x, mask=mask)

        self.assertIs(_CapturingUSPAttention.last_attn_mask, mask)


if __name__ == "__main__":
    unittest.main()
