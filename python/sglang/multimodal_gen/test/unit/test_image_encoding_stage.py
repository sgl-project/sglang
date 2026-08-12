# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import BatchFeature

from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageEncodingStage,
)


class _ImageProcessor:
    def __init__(self, include_mm_token_type_ids):
        self.include_mm_token_type_ids = include_mm_token_type_ids

    def __call__(self, images, return_tensors, text=None, padding=None):
        del images, return_tensors, padding
        inputs = BatchFeature(
            data={
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
                "pixel_values": torch.ones((1, 3, 2, 2)),
                "image_grid_thw": torch.tensor([[1, 1, 1]]),
            }
        )
        if self.include_mm_token_type_ids:
            marker = 2 if text == ["negative"] else 1
            inputs["mm_token_type_ids"] = torch.full((1, 3), marker)
        return inputs


class _CapturingTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        output_hidden_states,
        use_cache,
        mm_token_type_ids=None,
    ):
        assert output_hidden_states is True
        assert use_cache is False
        self.calls.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "mm_token_type_ids": mm_token_type_ids,
            }
        )
        hidden_states = torch.zeros((*input_ids.shape, 4))
        return SimpleNamespace(hidden_states=[hidden_states])


class _StrictTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.called = False

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        output_hidden_states,
        use_cache,
    ):
        del attention_mask, pixel_values, image_grid_thw
        assert output_hidden_states is True
        assert use_cache is False
        self.called = True
        hidden_states = torch.zeros((*input_ids.shape, 4))
        return SimpleNamespace(hidden_states=[hidden_states])


def _make_server_args():
    def prepare_image_processor_kwargs(batch, neg=False):
        del batch
        text = "negative" if neg else "positive"
        return {
            "padding": True,
            "per_prompt_images": [[object()]],
            "text": [text],
        }

    pipeline_config = SimpleNamespace(
        image_encoder_extra_args={},
        postprocess_text_funcs=(lambda outputs, _inputs: outputs.hidden_states[-1],),
        prepare_image_processor_kwargs=prepare_image_processor_kwargs,
    )
    return SimpleNamespace(pipeline_config=pipeline_config)


def _make_batch(do_classifier_free_guidance):
    return SimpleNamespace(
        condition_image=object(),
        do_classifier_free_guidance=do_classifier_free_guidance,
        image_embeds=[],
        prompt_embeds=[],
        negative_prompt_embeds=[],
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        prompt_seq_lens=None,
        negative_prompt_seq_lens=None,
    )


@pytest.mark.parametrize("do_classifier_free_guidance", [False, True])
def test_forwards_mm_token_type_ids_to_image_edit_text_encoder(
    do_classifier_free_guidance,
):
    text_encoder = _CapturingTextEncoder()
    stage = ImageEncodingStage(
        image_processor=_ImageProcessor(include_mm_token_type_ids=True),
        text_encoder=text_encoder,
    )

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding.get_local_torch_device",
        return_value=torch.device("cpu"),
    ):
        stage.forward(
            _make_batch(do_classifier_free_guidance),
            _make_server_args(),
        )

    positive_ids = text_encoder.calls[0]["mm_token_type_ids"]
    assert positive_ids.dtype == torch.long
    assert positive_ids.device == torch.device("cpu")
    assert torch.equal(positive_ids, torch.ones((1, 3), dtype=torch.long))
    if do_classifier_free_guidance:
        negative_ids = text_encoder.calls[1]["mm_token_type_ids"]
        assert negative_ids.dtype == torch.long
        assert negative_ids.device == torch.device("cpu")
        assert torch.equal(
            negative_ids,
            torch.full((1, 3), 2, dtype=torch.long),
        )


def test_omits_mm_token_type_ids_when_processor_does_not_return_them():
    text_encoder = _StrictTextEncoder()
    stage = ImageEncodingStage(
        image_processor=_ImageProcessor(include_mm_token_type_ids=False),
        text_encoder=text_encoder,
    )

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding.get_local_torch_device",
        return_value=torch.device("cpu"),
    ):
        stage.forward(_make_batch(False), _make_server_args())

    assert text_encoder.called
