"""Unit tests for video decoding in ``BaseMultimodalProcessor._load_single_item``.

Regression test for the positional-argument bug where the video branch called
``load_video(data, frame_count_limit)`` even though ``load_video``'s second
parameter is ``use_gpu: bool = True`` (a leftover from #5888, which renamed
``encode_video(video_path, frame_count_limit)`` without updating call sites).

It was harmless only because ``frame_count_limit`` is computed exclusively for
IMAGE items and is always ``None`` on the video path (falsy -> CPU decode). Had
it ever become a positive int for a video item, video decoding would silently
move to CUDA in the tokenizer process, creating an unexpected CUDA context.

No server, no model loading, no codec backend — pure CPU with the decoder
patched out.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.test.test_utils import CustomTestCase


class _StubProcessor(BaseMultimodalProcessor):
    # gpu_image_decode=False is irrelevant for video but keeps construction
    # consistent with the sibling image-decode tests. The abstract methods are
    # never called: only the _load_single_item classmethod is exercised.
    gpu_image_decode = False

    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


class TestLoadSingleItemVideoDecode(CustomTestCase):
    def _load_video_with_limit(self, frame_count_limit):
        with patch("sglang.srt.utils.common.VideoDecoderWrapper") as mock_decoder:
            _StubProcessor._load_single_item(
                b"payload", Modality.VIDEO, frame_count_limit=frame_count_limit
            )
        mock_decoder.assert_called_once()
        # Positional args of VideoDecoderWrapper(source, device=...): device is
        # passed as a keyword by load_video.
        return mock_decoder.call_args.kwargs["device"]

    def test_video_decode_stays_on_cpu(self):
        # The normal video path: frame_count_limit is None.
        self.assertEqual(self._load_video_with_limit(None), "cpu")

    def test_frame_count_limit_does_not_leak_into_use_gpu(self):
        # The landmine: a positive frame_count_limit must NOT be interpreted as
        # use_gpu=True. Before the fix this returned "cuda".
        self.assertEqual(self._load_video_with_limit(30), "cpu")


if __name__ == "__main__":
    unittest.main()
