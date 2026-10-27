"""LFM3-VL keeps CPU video decoding independent of processor placement."""

from unittest.mock import patch

import pytest

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.lfm3_vl import Lfm3VlImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "processor,preprocessing_device,use_gpu",
    [
        (BaseMultimodalProcessor, None, True),
        (BaseMultimodalProcessor, "cpu", False),
        (Lfm3VlImageProcessor, None, False),
        (Lfm3VlImageProcessor, "cuda", False),
    ],
)
def test_video_decode_device(processor, preprocessing_device, use_gpu, monkeypatch):
    monkeypatch.setattr(processor, "video_preprocessing_device", preprocessing_device)
    with patch(
        "sglang.srt.multimodal.processors.base_processor.load_video"
    ) as load_video:
        result = processor._load_single_item("video.mp4", Modality.VIDEO)
    load_video.assert_called_once_with("video.mp4", use_gpu=use_gpu)
    assert result is load_video.return_value


@pytest.mark.parametrize(
    "name",
    [
        "model.unknown.weight",
        "model.unknown.bias",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
    ],
)
def test_text_loader_rejects_unmapped_weights(name):
    from types import SimpleNamespace

    import torch

    from sglang.srt.models.lfm3 import Lfm3ForCausalLM

    model = SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=False),
        named_parameters=lambda: [],
    )
    with pytest.raises(ValueError, match="Unexpected LFM3 weights"):
        Lfm3ForCausalLM.load_weights(model, [(name, torch.zeros(1))])
