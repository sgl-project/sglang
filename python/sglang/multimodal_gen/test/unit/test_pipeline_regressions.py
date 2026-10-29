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
    "data, expected",
    [
        (
            np.broadcast_to(_VALUES[:, :, None, None] / 255, (2, 3, 2, 4, 3)).copy(),
            _VIDEOS.expand(2, 3, 3, 2, 4),
        ),
        (
            [
                [Image.new("RGB", (4, 2), tuple(map(int, frame))) for frame in video]
                for video in _VALUES
            ],
            _VIDEOS.expand(2, 3, 3, 2, 4),
        ),
        (
            [Image.new("RGB", (4, 2), tuple(map(int, video[0]))) for video in _VALUES],
            _VIDEOS[:, :, 0].expand(2, 3, 2, 4),
        ),
    ],
    ids=["np-videos", "pil-videos", "pil-images"],
)
def test_diffusers_output_layout_is_channels_first(data, expected):
    """Video outputs were permuted twice (np), PIL video lists lost all but the
    first sample, and several PIL images were stacked as video frames."""
    stage = DiffusersExecutionStage.__new__(DiffusersExecutionStage)

    output = stage._postprocess_output(stage._convert_to_tensor(data))

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
