import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from PIL import Image

from sglang.srt.disaggregation.encode_server import MMEncoder, _get_mm_grid_dim
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _encoder(model_type="kimi_k3"):
    encoder = MMEncoder.__new__(MMEncoder)
    encoder.model_type = model_type
    encoder.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            vision_config=SimpleNamespace(merge_kernel_size=(2, 2))
        )
    )
    return encoder


def test_kimi_k3_encoder_normalizes_pillow_images_to_media_dicts():
    image = Image.new("RGB", (2, 2))
    encoder = _encoder()

    assert encoder._grid_count_per_leaf(
        [image, {"type": "image", "image": [image, image]}], Modality.IMAGE
    ) == [1, 2]

    normalized = encoder._normalize_kimi_encoder_images(
        [image, {"type": "image", "image": [image, image]}]
    )
    assert len(normalized) == 3
    assert all(item["type"] == "image" for item in normalized)
    assert all(item["image"] is image for item in normalized)


def test_kimi_k3_encoder_passes_media_dicts_to_image_processor():
    image = Image.new("RGB", (2, 2))
    processor_calls = []

    def image_processor(*, images, **kwargs):
        processor_calls.append((images, kwargs))
        return {"pixel_values": torch.ones(1, 3), "grid_thws": [[1, 1, 1]]}

    encoder = _encoder()
    encoder.image_processor = image_processor
    encoder.vision_config = {"image": {"return_tensors": "pt"}}
    encoder._flatten_and_load_images = AsyncMock(return_value=[image])
    encoder.preproc_executor = ThreadPoolExecutor(max_workers=1)
    try:
        output = asyncio.run(encoder._process_image_items([image], None))
    finally:
        encoder.preproc_executor.shutdown()

    assert "pixel_values" in output
    assert len(processor_calls) == 1
    images, kwargs = processor_calls[0]
    assert images[0]["type"] == "image"
    assert images[0]["image"] is image
    assert kwargs == {"return_tensors": "pt"}


def test_kimi_k3_encoder_prefers_grid_thws_and_uses_temporal_pool_length():
    grid_thws = torch.tensor([[3, 8, 12]])
    stale_grid = torch.tensor([[1, 2, 2]])
    mm_inputs = {"grid_thws": grid_thws, "image_grid_thw": stale_grid}

    assert _get_mm_grid_dim(mm_inputs, Modality.IMAGE, "kimi_k3") is grid_thws
    assert _encoder().get_num_tokens(grid_thws[0], Modality.IMAGE) == 24


def test_kimi_k3_encoder_splits_cross_request_batch_into_single_grid_items():
    encoder = _encoder()
    grid_thws = torch.tensor([[1, 2, 2], [2, 2, 4], [1, 4, 2]])
    feature = torch.arange(56, dtype=torch.float32).reshape(28, 2)
    embeddings = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    captured = {}

    def get_feature_fn(items):
        captured["items"] = items
        return embeddings

    output = encoder._encode_missing(
        feature,
        {"pixel_values": feature, "grid_thws": grid_thws},
        indices=[2, 0, 1],
        modality=Modality.IMAGE,
        get_feature_fn=get_feature_fn,
        grid_thw=grid_thws,
        keep_on_gpu=True,
    )

    items = captured["items"]
    assert len(items) == 3
    expected_feature_slices = [feature[20:28], feature[0:4], feature[4:20]]
    expected_grids = [grid_thws[2:3], grid_thws[0:1], grid_thws[1:2]]
    for item, expected_feature, expected_grid in zip(
        items, expected_feature_slices, expected_grids
    ):
        torch.testing.assert_close(item.feature, expected_feature)
        torch.testing.assert_close(item.model_specific_data["grid_thws"], expected_grid)

    assert [embedding.shape[0] for embedding in output] == [2, 1, 2]
    torch.testing.assert_close(torch.cat(output), embeddings)


def test_kimi_k3_encoder_only_wrapper_guards_language_tower_hooks():
    model = SimpleNamespace(language_model=None)

    KimiK3ForConditionalGeneration.post_load_weights(model)
    with pytest.raises(AttributeError, match="lm_head"):
        KimiK3ForConditionalGeneration.lm_head.fget(model)
    with pytest.raises(AttributeError, match="DSPARK"):
        KimiK3ForConditionalGeneration.set_dspark_layers_to_capture(model, [0])


def test_kimi_k3_declares_encoder_only_weight_prefixes():
    assert KimiK3ForConditionalGeneration.encoder_only_safetensors_weight_prefixes == (
        "vision_tower.",
        "mm_projector.",
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
