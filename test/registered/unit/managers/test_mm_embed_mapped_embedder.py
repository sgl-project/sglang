"""embed_mm_inputs with a per-modality embedder mapping (Kimi-VL style).

Regression for the extend-memory-profile tag: the tag must follow the modality
whether the embedder comes from ``data_embedding_func_mapping`` or from the
model's ``get_<modality>_feature`` fallback, and must not go stale between
modalities.
"""

from unittest import mock

import pytest
import torch
from torch import nn

from sglang.srt.managers import mm_utils
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.utils import extend_mem_profile
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")

IMAGE_PAD = 100
AUDIO_PAD = 200
HIDDEN = 4


def _item(modality, pad_value, offsets, embedding):
    return MultimodalDataItem(
        modality=modality,
        pad_value=pad_value,
        offsets=offsets,
        precomputed_embeddings=embedding,
    )


def _inputs():
    # 8 tokens: image placeholders at [0, 2], audio placeholders at [4, 5].
    input_ids = torch.tensor([IMAGE_PAD] * 3 + [3] + [AUDIO_PAD] * 2 + [5, 7])
    image = torch.full((3, HIDDEN), 1.0)
    audio = torch.full((2, HIDDEN), 2.0)
    mm_inputs = MultimodalInputs(
        mm_items=[
            _item(Modality.IMAGE, IMAGE_PAD, [(0, 2)], image),
            _item(Modality.AUDIO, AUDIO_PAD, [(4, 5)], audio),
        ]
    )
    return input_ids, image, audio, mm_inputs


class _PhaseSpy:
    def __init__(self):
        self.tags = []

    def __call__(self, tag):
        self.tags.append(tag)
        return extend_mem_profile._NOOP_SCOPE


def _run(input_ids, mm_inputs, **kwargs):
    embedding = nn.Embedding(16, HIDDEN)
    spy = _PhaseSpy()
    with mock.patch.object(extend_mem_profile, "phase", spy):
        input_embeds, _ = mm_utils.embed_mm_inputs(
            mm_inputs_list=[mm_inputs],
            extend_prefix_lens=[0],
            extend_seq_lens=[input_ids.numel()],
            input_ids=input_ids.clone(),
            input_embedding=embedding,
            **kwargs,
        )
    return input_embeds, embedding, spy.tags


def _assert_scattered(input_embeds, embedding, input_ids, image, audio):
    torch.testing.assert_close(input_embeds[0:3], image)
    torch.testing.assert_close(input_embeds[4:6], audio)
    text = embedding(input_ids.clamp(max=15))
    torch.testing.assert_close(input_embeds[3], text[3])
    torch.testing.assert_close(input_embeds[6:8], text[6:8])


def test_mapped_embedders_for_every_modality():
    input_ids, image, audio, mm_inputs = _inputs()
    input_embeds, embedding, tags = _run(
        input_ids,
        mm_inputs,
        multimodal_model=None,
        data_embedding_func_mapping={
            Modality.IMAGE: mock.Mock(name="image_embedder"),
            Modality.AUDIO: mock.Mock(name="audio_embedder"),
        },
    )
    _assert_scattered(input_embeds, embedding, input_ids, image, audio)
    # Audio follows video in Modality.all(); its tag must be its own.
    assert tags == ["mm-embed:image", "mm-embed:audio"]


def test_mixed_mapped_and_model_fallback_embedders():
    input_ids, image, audio, mm_inputs = _inputs()
    model = mock.Mock(spec=["get_image_feature"])
    input_embeds, embedding, tags = _run(
        input_ids,
        mm_inputs,
        multimodal_model=model,
        data_embedding_func_mapping={Modality.AUDIO: mock.Mock(name="audio")},
    )
    _assert_scattered(input_embeds, embedding, input_ids, image, audio)
    assert tags == ["mm-embed:image", "mm-embed:audio"]


def test_missing_embedder_still_asserts_by_modality():
    input_ids, _, _, mm_inputs = _inputs()
    with pytest.raises(AssertionError, match="AUDIO"):
        _run(
            input_ids,
            mm_inputs,
            multimodal_model=mock.Mock(spec=["get_image_feature"]),
            data_embedding_func_mapping=None,
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
