from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs import krea2
from sglang.multimodal_gen.configs.pipeline_configs.krea2 import Krea2PipelineConfig


def test_only_ragged_prompts_carry_a_key_mask():
    """The mask is classified once per request: the DiT only reads it to choose
    the maskless path, and testing it per forward syncs the host every step."""
    config = Krea2PipelineConfig()
    batch = SimpleNamespace(height=256, width=256)
    context = torch.zeros(2, 5, 8)
    image_tokens = (256 // config.get_vae_scale_factor()) ** 2 // (
        config.dit_config.arch_config.patch**2
    )
    with patch.object(krea2, "get_sp_world_size", return_value=1):
        for text_mask in (None, torch.ones(2, 5, dtype=torch.long)):
            kwargs = config._build_pos_and_mask(context, text_mask, batch, "cpu")
            assert kwargs["mask"] is None
            assert kwargs["pos"].shape == (2, 5 + image_tokens, 3)
        ragged = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        kwargs = config._build_pos_and_mask(context, ragged, batch, "cpu")
    expected = torch.cat(
        [ragged.bool(), torch.ones(2, image_tokens, dtype=torch.bool)], dim=1
    )
    assert torch.equal(kwargs["mask"], expected)
