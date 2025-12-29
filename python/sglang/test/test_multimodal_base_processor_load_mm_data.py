import concurrent.futures
import re
from types import SimpleNamespace

import pytest

from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class _DummyProcessor(BaseMultimodalProcessor):
    """A minimal concrete processor for unit-testing load_mm_data()."""

    def __init__(self):
        # Avoid BaseMultimodalProcessor.__init__ (it creates process pools).
        self._processor = SimpleNamespace(
            tokenizer=SimpleNamespace(decode=lambda x: "")
        )
        self.io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


@pytest.fixture
def dummy_processor():
    p = _DummyProcessor()
    try:
        yield p
    finally:
        p.io_executor.shutdown(wait=True, cancel_futures=True)


def test_load_mm_data_mismatch_more_tokens_than_data_keeps_token(dummy_processor):
    mm_tokens = MultimodalSpecialTokens(
        image_token="<image>", image_token_regex=re.compile(re.escape("<image>"))
    )
    mm_tokens.get_combined_regex()

    out = dummy_processor.load_mm_data(
        prompt="<image><image>",
        multimodal_tokens=mm_tokens,
        image_data=[{"format": "processor_output"}],
    )

    assert out.input_text == "<image><image>"
    assert len(out.images) == 1


def test_load_mm_data_video_audio_token_not_split_into_chars(dummy_processor):
    mm_tokens = MultimodalSpecialTokens(
        video_token="<video>", video_token_regex=re.compile(re.escape("<video>"))
    )
    mm_tokens.get_combined_regex()

    out = dummy_processor.load_mm_data(
        prompt="<video>",
        multimodal_tokens=mm_tokens,
        video_data=[{"format": "processor_output"}],
    )

    assert out.input_text == "<video>"
    assert len(out.videos) == 1


def test_get_combined_regex_with_no_tokens_never_matches(dummy_processor):
    mm_tokens = MultimodalSpecialTokens()
    mm_tokens.get_combined_regex()

    out = dummy_processor.load_mm_data(
        prompt="hello",
        multimodal_tokens=mm_tokens,
        image_data=None,
        video_data=None,
        audio_data=None,
    )

    assert out.input_text == "hello"
    assert out.images == []
    assert out.videos == []
    assert out.audios == []
