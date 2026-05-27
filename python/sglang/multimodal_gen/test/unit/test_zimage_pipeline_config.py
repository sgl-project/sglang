import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.zimage import ZImagePipelineConfig


class TestZImagePipelineConfig(unittest.TestCase):
    @patch("sglang.multimodal_gen.configs.pipeline_configs.zimage.get_sp_world_size")
    def test_zimage_negative_prompt_rotary_embeddings_use_negative_prompt_len(
        self, mock_get_sp_world_size
    ) -> None:
        """Negative CFG branch should build RoPE positions from negative prompt embeds."""
        mock_get_sp_world_size.return_value = 1

        config = ZImagePipelineConfig()
        pos_seq_len = 19
        neg_seq_len = 45
        batch = SimpleNamespace(
            prompt_embeds=[torch.ones(pos_seq_len, 2560)],
            negative_prompt_embeds=[torch.ones(neg_seq_len, 2560)],
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


if __name__ == "__main__":
    unittest.main()
