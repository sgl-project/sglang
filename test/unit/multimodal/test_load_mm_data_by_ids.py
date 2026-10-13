"""A pre-tokenized multimodal prompt locates its images by the image
placeholder id in its ids, never by the regex over its decoded text.

Ordinary tokens that spell the Qwen image regex ("<image>", or the vision
triple written out) used to count as one more image than image_data held and
the request failed with "Mismatch: More 'IMAGE' tokens found than
corresponding data provided". These tests run the loaders with a tokenizer
whose decode() raises, so a decode is a test failure.
"""

import asyncio
import types

import pytest

from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

PAD = 7


class _NoDecodeTokenizer:
    def decode(self, *args, **kwargs):  # pragma: no cover - the assertion
        raise AssertionError("the decoded text must not be scanned for an ids prompt")


def _tokens():
    return MultimodalSpecialTokens(
        image_token="<|vision_start|><|image_pad|><|vision_end|>",
        image_token_id=PAD,
    )


class _Processor(BaseMultimodalProcessor):
    """A BaseMultimodalProcessor shell: only the loader methods run."""

    async def process_mm_data_async(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


def _processor():
    p = _Processor.__new__(_Processor)
    p._tokenizer = _NoDecodeTokenizer()
    p.skip_tokenizer_init = False
    p.support_dynamic_frame_expansion = False
    loaded = {}

    async def fast_load_mm_data(prompt, multimodal_tokens, image_data=None, **kw):
        loaded.update(
            prompt=prompt, image_data=list(image_data), input_ids=kw.get("input_ids")
        )
        return types.SimpleNamespace(**loaded)

    p.fast_load_mm_data = fast_load_mm_data
    return p, loaded


def test_placeholder_runs_count_one_id_per_image_or_an_expanded_run_once():
    f = BaseMultimodalProcessor
    assert f._placeholder_runs([1, PAD, 2, PAD, PAD, PAD, 3], PAD) == 2
    assert f._collapse_placeholder_runs([1, PAD, PAD, 2, PAD], PAD) == [1, PAD, 2, PAD]


@pytest.mark.parametrize("loader", ["load_mm_data", "legacy_load_mm_data"])
def test_an_ids_prompt_with_images_never_decodes(loader):
    p, loaded = _processor()
    ids = [1, PAD, 2, 3, PAD, PAD, 4]  # ordinary tokens 2, 3, 4 may spell "<image>"
    out = asyncio.run(
        getattr(p, loader)(
            prompt=ids, multimodal_tokens=_tokens(), image_data=["a.png", "b.png"]
        )
    )
    assert loaded["prompt"] == "<|vision_start|><|image_pad|><|vision_end|>" * 2
    assert loaded["input_ids"] == [1, PAD, 2, 3, PAD, 4]
    assert loaded["image_data"] == ["a.png", "b.png"]
    assert out.input_ids == [1, PAD, 2, 3, PAD, 4]


def test_a_placeholder_count_that_differs_from_the_images_is_a_client_error():
    p, _ = _processor()
    with pytest.raises(ValueError, match=r"carry 1 image placeholder\(s\) but 2 image"):
        asyncio.run(
            p.legacy_load_mm_data(
                prompt=[1, PAD], multimodal_tokens=_tokens(), image_data=["a", "b"]
            )
        )


def test_text_prompts_video_audio_and_precomputed_features_keep_the_text_path():
    t = _tokens()
    f = BaseMultimodalProcessor._ids_for_media
    assert f("<image> text", None, t, ["a"], None, None) is None
    assert f([1, PAD], None, t, [], None, None) is None
    assert f([1, PAD], None, t, ["a"], ["v.mp4"], None) is None
    assert f([1, PAD], None, t, ["a"], None, ["s.wav"]) is None
    assert f([1, PAD], None, t, [{"precomputed_features": 1}], None, None) is None
    assert f([1, PAD], None, t, ["a"], None, None) == [1, PAD]
    assert f(None, [1, PAD], t, ["a"], None, None) == [1, PAD]
