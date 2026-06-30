import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.zimage import ZImagePipelineConfig
from sglang.multimodal_gen.runtime.models.dits.zimage import (
    ZImageRMSNorm,
    ZImageTransformer2DModel,
)


class TestZImagePipelineConfig(unittest.TestCase):
    def test_rmsnorm_native_formula(self) -> None:
        norm = ZImageRMSNorm(4, eps=1e-5)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([1.0, 0.5, 1.5, 2.0]))
        x = torch.tensor(
            [[1.25, 0.5, -0.75, 3.0], [0.1, 2.3, -4.1, 0.7]],
            dtype=torch.bfloat16,
        )

        output = norm(x)
        expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        expected = expected * norm.weight.to(dtype=x.dtype)

        self.assertEqual(output.dtype, x.dtype)
        self.assertTrue(torch.equal(output, expected))

    def test_explicit_sigmas(self) -> None:
        """Z-Image uses the native explicit flow sigmas schedule."""
        config = ZImagePipelineConfig()

        self.assertEqual(
            config.prepare_sigmas(None, 4).tolist(),
            [1.0, 0.75, 0.5, 0.25],
        )

    def test_autocast_disabled(self) -> None:
        """Official Z-Image runs bf16 weights without an outer autocast context."""
        self.assertFalse(ZImagePipelineConfig().enable_autocast)

    @patch("sglang.multimodal_gen.configs.pipeline_configs.zimage.get_sp_world_size")
    def test_image_rope_patch_tokens(self, mock_get_sp_world_size) -> None:
        mock_get_sp_world_size.return_value = 1

        config = ZImagePipelineConfig()
        config.vae_config.post_init()
        batch = SimpleNamespace(
            prompt_embeds=[torch.ones(113, 2560)],
            negative_prompt_embeds=None,
            height=480,
            width=640,
        )

        def rotary_emb(pos_ids):
            return pos_ids

        _, image_pos_ids = config.prepare_pos_cond_kwargs(
            batch=batch,
            device=torch.device("cpu"),
            rotary_emb=rotary_emb,
            dtype=torch.float32,
        )["freqs_cis"]

        self.assertEqual(image_pos_ids.shape, (1216, 3))
        self.assertEqual(image_pos_ids[0].tolist(), [129, 0, 0])
        self.assertEqual(image_pos_ids[1199].tolist(), [129, 29, 39])
        self.assertEqual(image_pos_ids[-1].tolist(), [0, 0, 0])

    @patch("sglang.multimodal_gen.configs.pipeline_configs.zimage.get_sp_world_size")
    def test_negative_rope_len(self, mock_get_sp_world_size) -> None:
        """Negative CFG branch should build RoPE positions from negative prompt embeds."""
        mock_get_sp_world_size.return_value = 1

        config = ZImagePipelineConfig()
        pos_seq_len = 19
        neg_seq_len = 45
        batch = SimpleNamespace(
            prompt_embeds=[torch.ones(pos_seq_len, 2560)],
            prompt_seq_lens=[[pos_seq_len]],
            negative_prompt_embeds=[torch.ones(neg_seq_len, 2560)],
            negative_prompt_seq_lens=[[neg_seq_len]],
            height=16,
            width=16,
        )

        def rotary_emb(pos_ids):
            return pos_ids

        neg_kwargs = config.prepare_neg_cond_kwargs(
            batch=batch,
            device=torch.device("cpu"),
            rotary_emb=rotary_emb,
            dtype=torch.float32,
        )

        cap_pos_ids, image_pos_ids = neg_kwargs["freqs_cis"]
        neg_cap_padded_len = 64
        self.assertEqual(cap_pos_ids.shape, (neg_cap_padded_len, 3))
        self.assertEqual(image_pos_ids[0].tolist(), [neg_cap_padded_len + 1, 0, 0])

    def test_batched_rope_offsets(self) -> None:
        model = ZImageTransformer2DModel.__new__(ZImageTransformer2DModel)

        def rotary_emb(pos_ids):
            return (
                pos_ids.to(torch.float32),
                (pos_ids + 1000).to(torch.float32),
            )

        model.rotary_emb = rotary_emb

        images = [torch.zeros(16, 1, 60, 80), torch.zeros(16, 1, 60, 80)]
        cap_feats = [torch.zeros(113, 2560), torch.zeros(177, 2560)]

        cap_freqs, image_freqs = model._build_batched_freqs_cis(
            images,
            cap_feats,
            patch_size=2,
            f_patch_size=1,
            image_target_len=1216,
            cap_target_len=192,
        )

        self.assertEqual(cap_freqs[0].shape, (2, 192, 3))
        self.assertEqual(image_freqs[0].shape, (2, 1216, 3))
        self.assertEqual(image_freqs[0][0, 0].tolist(), [129.0, 0.0, 0.0])
        self.assertEqual(image_freqs[0][1, 0].tolist(), [193.0, 0.0, 0.0])
        self.assertEqual(cap_freqs[0][0, 127].tolist(), [128.0, 0.0, 0.0])
        self.assertEqual(cap_freqs[0][0, 128].tolist(), [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
