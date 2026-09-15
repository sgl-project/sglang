"""Unit test for ``NVILAMultimodalProcessor`` video-frame handling.

Regression test for the bug where ``process_mm_data_async`` called ``.asnumpy()`` on
video frames that ``VideoDecoderWrapper`` already returns as numpy arrays (NHWC uint8),
raising ``AttributeError: 'numpy.ndarray' object has no attribute 'asnumpy'`` on the
first frame of every decoded (file/URL/bytes) NVILA / NVILA-Lite / JetVLM video request.

The loop materializes only genuine frame sequences (the ``VideoDecoderWrapper`` and
caller-supplied lists/tuples), converting legacy decord frames and keeping numpy frames as
is. Everything else is passed through untouched -- a whole-clip ndarray/tensor, a
preprocessed dict, or a ``None`` placeholder from the preprocessed fast path -- since
iterating those would slice an array into rows, reduce a dict to its keys, or raise on
``None``.

No server, no model loading — pure CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import asyncio
import types
import unittest

import numpy as np
import torch

from sglang.srt.multimodal.processors.nvila import NVILAMultimodalProcessor
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.test_utils import CustomTestCase


class _NumpyFrameVideo(VideoDecoderWrapper):
    """Stand-in for a real ``VideoDecoderWrapper`` (the processor's positive ``isinstance``
    guard matches that exact type): a sequence whose frames are already numpy ndarrays,
    like the real wrapper's ``__getitem__``. Bypasses the decoding base ``__init__``."""

    def __init__(self, num_frames: int = 3):
        self._frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(num_frames)]

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        return self._frames[idx]  # raises IndexError past the end -> iteration stops

    def close(self):
        # The base __init__ is bypassed, so the fake holds none of the real wrapper's
        # temp-file state. Override cleanup to a no-op so the inherited __del__ -> close()
        # cannot raise an unraisable AttributeError on GC, independent of the base internals.
        pass


class _DecordFrame:
    """Stand-in for a legacy decord frame: not numpy, but exposes ``.asnumpy()``."""

    def asnumpy(self):
        return np.ones((2, 2, 3), dtype=np.uint8)


def _run_processor(videos):
    """Drive the real ``process_mm_data_async`` with ``base_output.videos = videos`` and
    everything else stubbed, returning the post-conversion ``videos`` list."""
    proc = NVILAMultimodalProcessor.__new__(NVILAMultimodalProcessor)
    proc.mm_tokens = types.SimpleNamespace(image_token_id=1, video_token_id=2)
    base_output = types.SimpleNamespace(videos=videos)
    captured = {}

    async def fake_load_mm_data(*args, **kwargs):
        return base_output

    def fake_process_and_combine(bo, *args, **kwargs):
        captured["videos"] = bo.videos
        return ["mm_item"], torch.tensor([1, 2, 3]), None

    proc.load_mm_data = fake_load_mm_data
    proc.process_and_combine_mm_data = fake_process_and_combine
    request_obj = types.SimpleNamespace(image_data=None, video_data=["fake"])
    asyncio.run(
        proc.process_mm_data_async(
            image_data=None,
            audio_data=None,
            input_text="<video>",
            request_obj=request_obj,
        )
    )
    return captured["videos"]


class TestNVILAVideoFrameHandling(CustomTestCase):
    def test_numpy_frames_pass_through_without_asnumpy(self):
        # The original crash: VideoDecoderWrapper frames are already numpy, so the old
        # `[x.asnumpy() for x in video]` raised AttributeError. They must survive as-is,
        # by identity -- materialized into a list but each frame left uncopied.
        wrapper = _NumpyFrameVideo(4)
        frames = _run_processor([wrapper])[0]
        self.assertIsInstance(frames, list)
        self.assertEqual(len(frames), 4)
        self.assertTrue(all(isinstance(f, np.ndarray) for f in frames))
        self.assertTrue(all(frames[k] is wrapper._frames[k] for k in range(4)))

    def test_decord_frames_are_converted_to_numpy(self):
        # Legacy / caller-supplied decord frames still expose `.asnumpy()` and must be
        # converted (behavior preserved from before the VideoDecoderWrapper refactor);
        # a plain `list(video)` fix would wrongly leave them as decord objects.
        frames = _run_processor([[_DecordFrame(), _DecordFrame()]])[0]
        self.assertEqual(len(frames), 2)
        self.assertTrue(all(isinstance(f, np.ndarray) for f in frames))

    def test_preprocessed_dict_is_passed_through_unchanged(self):
        # The loader passes preprocessed video dicts through untouched; the processor
        # must NOT turn them into a list of keys (which `list(video)` would do). A real
        # preprocessed dict carries a `format` marker (base_processor
        # ._get_preprocessed_input_format).
        preprocessed = {"format": "processor_output", "pixel_values_videos": "tensor"}
        out = _run_processor([preprocessed])[0]
        self.assertIs(out, preprocessed)

    def test_bare_ndarray_video_is_passed_through_untouched(self):
        # `load_video` returns ndarray/tensor videos verbatim. A 3-D H*W*C single frame
        # must NOT be iterated (that slices it into 2-D rows); a 4-D T*H*W*C clip must
        # survive intact rather than be rebuilt into a Python list of frames.
        single = np.zeros((2, 2, 3), dtype=np.uint8)
        self.assertIs(_run_processor([single])[0], single)
        clip = np.zeros((4, 2, 2, 3), dtype=np.uint8)
        self.assertIs(_run_processor([clip])[0], clip)

    def test_bare_torch_tensor_video_is_passed_through_untouched(self):
        clip = torch.zeros((4, 2, 2, 3), dtype=torch.uint8)
        self.assertIs(_run_processor([clip])[0], clip)

    def test_multiple_raw_videos_are_handled_independently(self):
        # Several videos in one request, each loaded as a different raw type, are
        # normalized per-index without cross-talk. (A preprocessed dict cannot be mixed
        # with raw videos in the same modality, so this covers only the raw types.)
        decord_list = [_DecordFrame()]
        decord_tuple = (_DecordFrame(), _DecordFrame())
        wrapper_vid = _NumpyFrameVideo(2)
        ndarray_vid = np.zeros((2, 2, 3), dtype=np.uint8)
        out = _run_processor([decord_list, decord_tuple, wrapper_vid, ndarray_vid])
        self.assertEqual(len(out[0]), 1)  # list of decord frames -> converted
        self.assertEqual(len(out[1]), 2)  # tuple of decord frames -> converted + listed
        self.assertTrue(all(isinstance(f, np.ndarray) for f in out[0] + out[1]))
        self.assertEqual(len(out[2]), 2)  # wrapper -> frames materialized
        self.assertIs(out[3], ndarray_vid)  # ndarray -> passed through untouched

    def test_none_placeholder_is_passed_through_untouched(self):
        # The preprocessed fast path can leave a `None` in `videos` (another modality
        # preprocessed while `video_data=[None]`). The loop must NOT iterate it -- the
        # stale code raised `TypeError: 'NoneType' object is not iterable` -- it passes
        # through for the downstream to handle.
        self.assertIsNone(_run_processor([None])[0])


if __name__ == "__main__":
    unittest.main()
