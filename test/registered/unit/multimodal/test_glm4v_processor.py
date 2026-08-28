import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

from sglang.srt.configs.glm5_next_processing import (
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    smart_resize,
)
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.customized_mm_processor_utils import (
    _CUSTOMIZED_MM_PROCESSOR,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.glm4v import (
    Glm4vImageProcessor,
    _collapse_glm5_next_image_tokens,
)
from sglang.srt.utils.hf_transformers import processor as processor_module


def test_glm5_next_image_processor_uses_dynamic_padded_grid():
    pixels = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)

    output = Glm5NextImageProcessor()(
        images=Image.fromarray(pixels), return_tensors="pt"
    )

    assert output.image_grid_thw.tolist() == [[1, 8, 10]]
    assert output.pixel_values.shape == (80, 1176)
    assert output.pixel_values.isfinite().all()


def test_glm5_next_smart_resize_rejects_impossible_token_budget():
    with pytest.raises(ValueError, match="max_pixels=0 is too small"):
        smart_resize(
            num_frames=2,
            height=64,
            width=64,
            temporal_factor=2,
            factor=28,
            max_pixels=0,
        )


def test_collapse_glm5_next_image_tokens_preserves_image_boundaries():
    image_token_id = 99
    image_start_token_id = 10
    image_end_token_id = 11
    input_ids = [10, 99, 99, 11, 10, 99, 99, 99, 11]

    assert _collapse_glm5_next_image_tokens(input_ids, image_token_id) == [
        image_start_token_id,
        99,
        image_end_token_id,
        image_start_token_id,
        99,
        image_end_token_id,
    ]


def test_collapse_glm5_next_image_tokens_handles_sequence_boundaries():
    assert _collapse_glm5_next_image_tokens([99, 99, 1, 99, 99], 99) == [
        99,
        1,
        99,
    ]
    assert _collapse_glm5_next_image_tokens([], 99) == []


def test_glm5_next_preserves_processor_expanded_image_tokens(monkeypatch):
    image_token_id = 99
    expanded_input_ids = [1, 10, 99, 99, 99, 99, 11, 2]
    collapsed_input_ids = [1, 10, 99, 11, 2]
    image = Image.new("RGB", (28, 28))
    image_item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=torch.ones((4, 3)),
        model_specific_data={"image_grid_thw": torch.tensor([[1, 2, 2]])},
    )
    processor_output = SimpleNamespace(
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        video_grid_thw=None,
        attention_mask=torch.ones((1, len(expanded_input_ids))),
    )

    processor = object.__new__(Glm4vImageProcessor)
    processor.hf_config = SimpleNamespace(model_type="glm5_next")
    processor.IM_TOKEN_ID = image_token_id
    processor.mm_tokens = MultimodalSpecialTokens(
        image_token_id=image_token_id, video_token_id=98
    )
    processor.video_config = {}
    processor._processor = SimpleNamespace(
        video_processor=None,
        _get_num_multimodal_tokens=lambda **_: SimpleNamespace(num_image_tokens=[4]),
    )
    processor._tokenizer = MagicMock()
    processor.preserve_processor_input_ids = False
    processor.precompute_hash_before_cpu_transfer = False
    processor.use_cuda_ipc = False
    processor.load_mm_data = AsyncMock(
        return_value=BaseMultiModalProcessorOutput(
            input_text="unused", input_ids=collapsed_input_ids, images=[image]
        )
    )
    processor._process_and_collect_mm_items = MagicMock(
        return_value=(
            [image_item],
            torch.tensor(expanded_input_ids),
            processor_output,
        )
    )
    monkeypatch.setattr(
        MRotaryEmbedding,
        "get_rope_index_glm4v",
        MagicMock(
            return_value=(
                torch.zeros((3, 1, len(expanded_input_ids))),
                torch.zeros(1),
            )
        ),
    )

    output = asyncio.run(
        processor.process_mm_data_async(
            image_data=["image"],
            input_text=expanded_input_ids,
            request_obj=SimpleNamespace(video_data=None),
        )
    )

    assert processor.load_mm_data.await_args.kwargs["prompt"] == collapsed_input_ids
    assert output.input_ids == expanded_input_ids
    assert len(output.mm_items) == 1
    assert output.mm_items[0].offsets == [(2, 5)]


def test_get_processor_uses_registered_glm5_next_processor(monkeypatch):
    expected_processor = MagicMock()
    expected_processor.tokenizer.chat_template = "test-template"
    monkeypatch.setattr(
        processor_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: type(
            "Config", (), {"model_type": "glm5_next", "language_model_only": False}
        )(),
    )
    monkeypatch.setattr(
        Glm5NextProcessor,
        "from_pretrained",
        lambda *args, **kwargs: expected_processor,
    )

    assert _CUSTOMIZED_MM_PROCESSOR["glm5_next"] is Glm5NextProcessor
    assert processor_module.get_processor("unused-checkpoint") is expected_processor
