# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn
from transformers import Qwen2_5_VLConfig, Qwen3VLConfig

from sglang.multimodal_gen.runtime.models.encoders.qwen2_5vl import Qwen2_5_VLModel
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import Qwen3VLModel
from sglang.srt.runtime_context import get_parallel


@pytest.fixture(params=[Qwen2_5_VLConfig, Qwen3VLConfig])
def model(request):
    config = request.param(
        image_token_id=2,
        video_token_id=3,
        text_config=dict(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=0,
            num_attention_heads=2,
            num_key_value_heads=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=1,
        ),
        vision_config=dict(
            hidden_size=16,
            intermediate_size=24,
            depth=0,
            num_heads=2,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=16,
            num_position_embeddings=16,
            deepstack_visual_indexes=[],
        ),
    )
    with get_parallel().override(tp_size=1, tp_rank=0, tp_group=None):
        yield (
            Qwen2_5_VLModel(config, enable_image_understanding=True)
            if isinstance(config, Qwen2_5_VLConfig)
            else Qwen3VLModel(config)
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen_vl_placeholder_masks(model, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model.to(device=device, dtype=dtype)
    weight = model.get_input_embeddings().weight
    with torch.no_grad():
        weight.copy_(torch.arange(weight.shape[0], device=device)[:, None])
    ids = torch.tensor([[1, 2, 2, 3], [3, 3, 4, 1]], device=device)
    embeds = model.get_input_embeddings()(ids)
    features = {"image_features": embeds[ids == 2], "video_features": embeds[ids == 3]}
    for input_ids in (ids.cpu(), None):
        masks = model.get_placeholder_mask(input_ids, embeds, **features)
        for token_id, mask in zip((2, 3), masks):
            expected = (ids == token_id).unsqueeze(-1).expand_as(embeds)
            torch.testing.assert_close(mask, expected)
        for kind, message in (("image", "Image"), ("video", "Videos")):
            invalid = dict(features)
            invalid[f"{kind}_features"] = features[f"{kind}_features"][:-1]
            with pytest.raises(ValueError, match=f"{message} features and .* tokens"):
                model.get_placeholder_mask(input_ids, embeds, **invalid)
        empty = model.get_placeholder_mask(input_ids, embeds)
        assert all(
            torch.equal(actual, expected) for actual, expected in zip(empty, masks)
        )
    no_media = model.get_placeholder_mask(ids.new_ones(2, 4), embeds)
    assert not any(mask.any() for mask in no_media)


def test_qwen_vl_embedding_and_decoder_accessors(model):
    keys = set(model.state_dict())
    replacement = nn.Embedding(32, 16)
    model.set_input_embeddings(replacement)
    assert model.get_input_embeddings() is replacement
    assert model.get_decoder().embed_tokens is replacement
    assert set(model.state_dict()) == keys
    assert (
        dict(model.named_parameters())["language_model.embed_tokens.weight"]
        is replacement.weight
    )
    decoder = nn.Module()
    decoder.embed_tokens = replacement
    model.set_decoder(decoder)
    assert model.get_decoder() is decoder
    assert model.get_input_embeddings() is replacement
    assert set(model.state_dict()) == {
        "language_model.embed_tokens.weight",
        *(f"visual.{name}" for name in model.visual.state_dict()),
    }
