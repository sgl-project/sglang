# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.multimodal_gen.configs.models.encoders.bagel_vit import (
    BagelImageEncoderArchConfig,
    BagelImageEncoderConfig,
)
from sglang.multimodal_gen.runtime.models.encoders.bagel_vit import (
    BagelImageEncoder,
    preprocess_bagel_vit_image,
)


@pytest.fixture
def tiny_config() -> BagelImageEncoderConfig:
    return BagelImageEncoderConfig(
        arch_config=BagelImageEncoderArchConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            patch_size=2,
            max_image_size=4,
            min_image_size=2,
            max_num_patches_per_side=2,
            position_embedding_rows=4,
            llm_hidden_size=6,
        )
    )


def test_preprocess_patch_order_and_position_ids(
    tiny_config: BagelImageEncoderConfig,
) -> None:
    pixels = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    image = Image.fromarray(pixels)

    patches, position_ids, grid = preprocess_bagel_vit_image(image, tiny_config)

    assert grid == (1, 2)
    assert patches.shape == (2, 12)
    assert position_ids.tolist() == [0, 1]
    normalized = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
    normalized = normalized * 2 - 1
    expected = normalized.reshape(3, 1, 2, 2, 2)
    expected = torch.einsum("chpwq->hwpqc", expected).reshape(2, 12)
    torch.testing.assert_close(patches, expected)


def test_forward_projects_to_llm_width(tiny_config: BagelImageEncoderConfig) -> None:
    torch.manual_seed(0)
    encoder = BagelImageEncoder(tiny_config).eval()
    patches = torch.randn(4, 12)
    position_ids = torch.tensor([0, 1, 2, 3])

    output = encoder(patches, position_ids)

    assert output.shape == (4, 6)
    assert torch.isfinite(output).all()


def test_weight_loader_is_streaming_strict_and_meta_safe(
    tiny_config: BagelImageEncoderConfig,
) -> None:
    with torch.device("meta"):
        encoder = BagelImageEncoder(tiny_config)
    expected = [
        (name, torch.full(tuple(parameter.shape), index + 1.0))
        for index, (name, parameter) in enumerate(encoder.named_parameters())
    ]

    loaded = encoder.load_weights(
        iter([("language_model.ignored", torch.ones(1)), *expected])
    )
    encoder.to("cpu")

    assert loaded == {name for name, _ in expected}
    assert not any(parameter.is_meta for parameter in encoder.parameters())
    with pytest.raises(ValueError, match="unexpected image encoder weights"):
        encoder.load_weights(
            [("vit_model.vision_model.unknown", torch.ones(1))], strict=True
        )


def test_forward_rejects_invalid_positions(
    tiny_config: BagelImageEncoderConfig,
) -> None:
    encoder = BagelImageEncoder(tiny_config)
    with pytest.raises(ValueError, match="outside the checkpoint table"):
        encoder(torch.zeros(1, 12), torch.tensor([4]))
