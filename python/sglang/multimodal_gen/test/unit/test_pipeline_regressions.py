"""Regression tests for pipeline and model-helper bugs (CPU only)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline import (
    DiffusersExecutionStage,
    DiffusersPipeline,
)
from sglang.multimodal_gen.runtime.pipelines.ernie_image import ErnieImagePipeline
from sglang.multimodal_gen.runtime.pipelines.sana_video import (
    SanaVideoTextEncodingStage,
)
from sglang.multimodal_gen.runtime.utils.mesh3d_utils import (
    ImageProcessorV2,
    MeshRender,
)


def _sana_text_outputs(valid_tokens: int, sequence_length: int = 300):
    embeddings = torch.arange(sequence_length * 2, dtype=torch.float32).reshape(
        1, sequence_length, 2
    )
    attention_mask = torch.zeros(1, sequence_length, dtype=torch.long)
    hidden_states_mask = torch.zeros(1, sequence_length, dtype=torch.bool)
    attention_mask[:, :valid_tokens] = 1
    hidden_states_mask[:, :valid_tokens] = True
    return [
        [embeddings],
        [attention_mask],
        [torch.tensor([1.0])],
        [hidden_states_mask],
        [[valid_tokens]],
    ]


def test_default_texture_follows_width_height_texture_size():
    """texture_size is (width, height); the blank default texture was built as
    (width, height, 3) instead of the (height, width, 3) of a loaded texture."""
    render = MeshRender.__new__(MeshRender)
    render.tex = None
    render.texture_size = (64, 32)

    assert render.get_texture().shape == (32, 64, 3)


@pytest.mark.parametrize("channels", [3, 4])
def test_image_processor_accepts_ndarray_like_pil(channels):
    """An ndarray input fell through to an UnboundLocalError on `mask`."""
    array = np.zeros((6, 10, channels), dtype=np.uint8)
    array[1:5, 2:8] = 200
    processor = ImageProcessorV2(size=8)

    actual = processor.load_image(array)
    expected = processor.load_image(Image.fromarray(array))

    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)


def test_image_processor_rejects_unsupported_input_type():
    with pytest.raises(TypeError, match="Unsupported image type"):
        ImageProcessorV2(size=8).load_image(torch.zeros(3, 8, 8))


# Distinct value per sample, frame and channel, so swapped axes change the result.
_VALUES = np.fromfunction(lambda b, t, c: 100 * b + 30 * t + 10 * c, (2, 3, 3))
_VIDEOS = torch.from_numpy(_VALUES / 255).float().permute(0, 2, 1)[..., None, None]


@pytest.mark.parametrize(
    "attr, data, expected",
    [
        (
            "frames",
            np.broadcast_to(_VALUES[:, :, None, None] / 255, (2, 3, 2, 4, 3)).copy(),
            _VIDEOS.expand(2, 3, 3, 2, 4),
        ),
        (
            "frames",
            [
                [Image.new("RGB", (4, 2), tuple(map(int, frame))) for frame in video]
                for video in _VALUES
            ],
            _VIDEOS.expand(2, 3, 3, 2, 4),
        ),
        (
            "images",
            [Image.new("RGB", (4, 2), tuple(map(int, video[0]))) for video in _VALUES],
            _VIDEOS[:, :, 0].expand(2, 3, 2, 4),
        ),
    ],
    ids=["np-videos", "pil-videos", "pil-images"],
)
def test_diffusers_output_layout_is_channels_first(attr, data, expected):
    """Video outputs were permuted twice (np), PIL video lists lost all but the
    first sample, and several PIL images were stacked as video frames."""
    stage = DiffusersExecutionStage.__new__(DiffusersExecutionStage)

    output = stage._postprocess_output(
        stage._extract_output(SimpleNamespace(**{attr: data}))
    )

    torch.testing.assert_close(output, expected)


def test_sana_video_negative_prompt_uses_requested_sequence_length():
    """The negative prompt was always encoded at 300 tokens, so a custom
    max_sequence_length gave positive and negative embeddings different lengths."""
    stage = SanaVideoTextEncodingStage.__new__(SanaVideoTextEncodingStage)
    captured = {}
    stage.encode_text = lambda *args, **kwargs: captured.update(kwargs)

    stage._encode_negative_text(
        SimpleNamespace(negative_prompt="Blurry", max_sequence_length=128), None, [0]
    )
    assert captured["max_length"] == 128

    stage._encode_negative_text(
        SimpleNamespace(negative_prompt="Blurry", max_sequence_length=None), None, [0]
    )
    assert captured["max_length"] == 300


@pytest.mark.parametrize(
    "valid_tokens,expected_width",
    [(18, 64), (56, 64), (64, 64), (65, 300), (163, 300)],
)
def test_sana_video_compacts_only_a_fully_masked_default_suffix(
    valid_tokens, expected_width
):
    from sglang.multimodal_gen.runtime.pipelines.sana_video import (
        _compact_sana_video_text_outputs,
    )

    outputs = _sana_text_outputs(valid_tokens)
    compacted = _compact_sana_video_text_outputs(outputs, 300)

    assert compacted[0][0].shape == (1, expected_width, 2)
    assert compacted[1][0].shape == (1, expected_width)
    assert compacted[3][0].shape == (1, expected_width)
    torch.testing.assert_close(compacted[0][0], outputs[0][0][:, :expected_width])
    assert int(compacted[1][0].sum()) == valid_tokens
    assert int(compacted[3][0].sum()) == valid_tokens
    assert compacted[4] == [[valid_tokens]]
    assert compacted[2] is outputs[2]


def test_sana_video_preserves_suffix_validity_and_unsupported_layouts():
    from sglang.multimodal_gen.runtime.pipelines.sana_video import (
        _compact_sana_video_text_outputs,
    )

    for mask_index, value in ((1, 1), (3, True)):
        suffix_valid = _sana_text_outputs(18)
        suffix_valid[mask_index][0][0, 70] = value
        assert _compact_sana_video_text_outputs(suffix_valid, 300) is suffix_valid

    malformed = _sana_text_outputs(18)
    malformed[1][0] = malformed[1][0].unsqueeze(-1)
    assert _compact_sana_video_text_outputs(malformed, 300) is malformed

    non_default = _sana_text_outputs(18, sequence_length=128)
    assert _compact_sana_video_text_outputs(non_default, 128) is non_default


@pytest.mark.parametrize(
    "positive_tokens,negative_tokens",
    [(163, 18), (18, 163)],
)
def test_sana_video_compacts_cfg_branches_independently(
    positive_tokens, negative_tokens
):
    stage = SanaVideoTextEncodingStage.__new__(SanaVideoTextEncodingStage)
    stage.tokenizers = [SimpleNamespace(encode=lambda _: [1, 2, 3])]
    stage.text_encoders = [object()]
    positive_outputs = _sana_text_outputs(positive_tokens)
    negative_outputs = _sana_text_outputs(negative_tokens)
    stage.encode_text = lambda text, *_args, **_kwargs: (
        positive_outputs if isinstance(text, list) else negative_outputs
    )
    batch = SimpleNamespace(
        prompt="A moving scene",
        negative_prompt="Blurry",
        max_sequence_length=None,
        do_classifier_free_guidance=True,
        prompt_embeds=[],
        pooled_embeds=[],
        prompt_attention_mask=None,
        prompt_embeds_mask=None,
        prompt_seq_lens=None,
        negative_prompt_embeds=[],
        neg_pooled_embeds=[],
        negative_attention_mask=None,
        negative_prompt_embeds_mask=None,
        negative_prompt_seq_lens=None,
    )

    stage.forward(batch, None)

    positive_width = 64 if positive_tokens <= 64 else 300
    negative_width = 64 if negative_tokens <= 64 else 300
    assert batch.prompt_embeds[0].shape == (1, positive_width, 2)
    assert batch.negative_prompt_embeds[0].shape == (1, negative_width, 2)
    assert batch.prompt_attention_mask[0].shape == (1, positive_width)
    assert batch.prompt_embeds_mask[0].shape == (1, positive_width)
    assert batch.negative_attention_mask[0].shape == (1, negative_width)
    assert batch.negative_prompt_embeds_mask[0].shape == (1, negative_width)
    assert batch.prompt_seq_lens == [[positive_tokens]]
    assert batch.negative_prompt_seq_lens == [[negative_tokens]]
    assert int(batch.prompt_attention_mask[0].sum()) == positive_tokens
    assert int(batch.negative_attention_mask[0].sum()) == negative_tokens


@pytest.mark.parametrize(
    "pe_entry, expected",
    [([None, None], False), (["transformers", "Mistral3Model"], True)],
)
def test_ernie_absent_pe_component_is_not_detected(monkeypatch, pe_entry, expected):
    """model_index.json records an absent component as [null, null]; that was
    treated as a real prompt-enhancement model and loading then failed."""
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.pipelines.ernie_image.maybe_download_model_index",
        lambda path: {"pe": pe_entry},
    )
    pipeline = ErnieImagePipeline.__new__(ErnieImagePipeline)

    assert (
        pipeline._has_pe_in_model_index(SimpleNamespace(model_path="org/ernie"))
        is expected
    )


@pytest.mark.parametrize("legacy_order", [False, True])
def test_diffusers_pipeline_add_stage_accepts_both_argument_orders(legacy_order):
    pipeline = DiffusersPipeline.__new__(DiffusersPipeline)
    pipeline.modules = {}
    pipeline._disagg_role = RoleType.MONOLITHIC
    pipeline._stages = []
    pipeline._stage_name_mapping = {}
    stage = DiffusersExecutionStage(None)
    if legacy_order:
        pipeline.add_stage("diffusers_execution", stage)
    else:
        pipeline.add_stage(stage, "diffusers_execution")
    assert pipeline._stage_name_mapping == {"diffusers_execution": stage}
    assert pipeline._stages == [stage]
