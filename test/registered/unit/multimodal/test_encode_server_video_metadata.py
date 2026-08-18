"""Unit tests for the video_metadata kwarg built by ``_flatten_and_load_videos``.

``preprocess_video`` returns ``(video, None)`` for input that was already decoded
upstream, so the metadata list can hold ``None``. Such a list must not reach the
processor: transformers' ``make_batched_metadata`` passes it through unchanged
and the consumer then dereferences ``None``. Omitting the kwarg is better --
transformers synthesizes the fields from the video itself.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import asyncio
import unittest
from concurrent.futures import Future
from unittest.mock import patch

from sglang.srt.disaggregation.encode_server import MMEncoder
from sglang.test.test_utils import CustomTestCase

_META = {
    "fps": 24.0,
    "duration": 1.0,
    "total_num_frames": 24,
    "frames_indices": list(range(24)),
}


class _StubEncoder:
    """Carries only what _flatten_and_load_videos touches -- no model, no GPU."""

    model_type = "qwen2_5_vl"
    vision_config = {"video": {}}

    def submit_data_loading_tasks(self, items, modalities):
        futures = []
        for item in items:
            future: Future = Future()
            future.set_result(item)
            futures.append(future)
        return futures, None

    # Exercise the real implementation, bound to the stub.
    _flatten_and_load_videos = MMEncoder._flatten_and_load_videos


def _run(videos, metadata):
    """Run _flatten_and_load_videos with preprocess_video yielding `metadata`."""
    calls = iter(metadata)

    async def _fake_preprocess_video(video, video_config=None):
        return video, next(calls)

    async def _go():
        with patch(
            "sglang.srt.disaggregation.encode_server.preprocess_video",
            _fake_preprocess_video,
        ):
            return await _StubEncoder()._flatten_and_load_videos(videos)

    return asyncio.run(_go())


class TestFlattenAndLoadVideosMetadata(CustomTestCase):
    def test_all_none_metadata_is_not_forwarded(self):
        # The regression: [None] is truthy, so an emptiness-only guard forwards it.
        _, kwargs = _run(["a", "b"], [None, None])
        self.assertNotIn("video_metadata", kwargs)
        self.assertIs(kwargs["do_sample_frames"], False)

    def test_real_metadata_is_forwarded(self):
        _, kwargs = _run(["a", "b"], [_META, _META])
        self.assertEqual(kwargs["video_metadata"], [_META, _META])

    def test_mixed_metadata_is_not_forwarded(self):
        # The list is positionally parallel to `videos`, so a None entry cannot
        # be dropped without misaligning the rest -- omit the kwarg instead.
        _, kwargs = _run(["a", "b"], [_META, None])
        self.assertNotIn("video_metadata", kwargs)

    def test_videos_are_returned_in_order(self):
        videos, _ = _run(["a", "b", "c"], [None, None, None])
        self.assertEqual(videos, ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
