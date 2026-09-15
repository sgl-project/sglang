# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.llada_image import (
    LLaDAImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.llada_image.source import (
    LLaDAImageSourceImageConditioningStage,
)


def test_source_image_preprocessing_and_output_replication():
    pixels = torch.arange(3 * 8 * 12, dtype=torch.float32).reshape(1, 3, 8, 12)
    raw = torch.arange(2 * 2 * 4 * 6, dtype=torch.float32).reshape(2, 2, 4, 6)
    semantics = torch.arange(30, dtype=torch.float32).reshape(2, 3, 5)
    processor = SimpleNamespace(preprocess=Mock(return_value=pixels))
    sigvq = Mock(
        return_value=SimpleNamespace(semantic_features=semantics),
        config=SimpleNamespace(patch_size=16),
        parameters=lambda: iter([torch.zeros(1)]),
    )
    vae = SimpleNamespace(
        parameters=lambda: iter([torch.zeros(1)]),
        bn=SimpleNamespace(
            running_mean=torch.arange(8), running_var=torch.full((8,), 3.75)
        ),
        config=SimpleNamespace(batch_norm_eps=0.25),
        encode=Mock(return_value=SimpleNamespace(mode=lambda: raw)),
    )
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args",
        return_value=SimpleNamespace(),
    ):
        stage = LLaDAImageSourceImageConditioningStage(sigvq, vae, processor)
    batch = SimpleNamespace(
        condition_image=None,
        image_embeds=[],
        source_latents=None,
        height=64,
        width=80,
        batch_size=2,
    )
    assert stage.forward(batch, None) is batch
    assert batch.image_embeds == [] and batch.source_latents is None
    processor.preprocess.assert_not_called()
    batch.condition_image = Image.new("RGB", (40, 30))
    stage.forward(batch, None)
    processor.preprocess.assert_called_once_with(
        batch.condition_image, height=64, width=80, resize_mode="crop"
    )
    assert sigvq.call_args.args[0].shape == (2, 3, 32, 48)
    torch.testing.assert_close(vae.encode.call_args.args[0], pixels.repeat(2, 1, 1, 1))
    torch.testing.assert_close(torch.stack(batch.image_embeds), semantics)
    expected = torch.stack(
        [
            raw[:, channel, row::2, column::2]
            for channel in range(2)
            for row in range(2)
            for column in range(2)
        ],
        dim=1,
    )
    expected = (expected - torch.arange(8).reshape(1, 8, 1, 1)) / 2
    torch.testing.assert_close(
        torch.stack(batch.source_latents), expected.unsqueeze(2), rtol=0, atol=0
    )
    batch.condition_image = [batch.condition_image] * 2
    with pytest.raises(ValueError, match="exactly one source image"):
        stage.forward(batch, None)


def test_source_and_decode_preserve_vae_cast_order():
    vae = SimpleNamespace(
        parameters=lambda: iter([torch.zeros(1, dtype=torch.bfloat16)]),
        bn=SimpleNamespace(
            running_mean=torch.tensor([0.01, -0.02, 0.03, -0.04]),
            running_var=torch.full((4,), 0.0129973),
        ),
        config=SimpleNamespace(batch_norm_eps=0.003),
    )
    latents = torch.tensor([0.007, -0.249, 0.126, -0.751]).reshape(1, 4, 1, 1)
    bf16 = latents.bfloat16()
    mean = vae.bn.running_mean.reshape(1, 4, 1, 1).to(bf16)
    std = (
        (vae.bn.running_var + vae.config.batch_norm_eps)
        .sqrt()
        .reshape(1, 4, 1, 1)
        .to(bf16)
    )
    normalized = LLaDAImageSourceImageConditioningStage._normalize_latents(bf16, vae)
    torch.testing.assert_close(normalized, (bf16 - mean) / std, rtol=0, atol=0)
    decoded = LLaDAImagePipelineConfig().preprocess_decoding(latents, vae=vae)
    torch.testing.assert_close(
        decoded, (bf16 * std + mean).reshape(1, 1, 2, 2), rtol=0, atol=0
    )
