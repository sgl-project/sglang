"""CPU tests for model-specific Qwen3-VL video preprocessing."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import asyncio
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers.image_utils import SizeDict
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
)

from sglang.srt.multimodal.processors import qwen3_vl, qwen_vl
from sglang.test.test_utils import CustomTestCase


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def permute(self, *dims):
        self.shape = tuple(self.shape[dim] for dim in dims)
        return self

    def pin_memory(self):
        return self

    def new_empty(self, shape):
        return _FakeTensor(shape)

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise TypeError(f"unsupported fake tensor index: {item!r}")
        start = item.start or 0
        stop = item.stop if item.stop is not None else self.shape[0]
        return _FakeTensor((stop - start, *self.shape[1:]))

    def copy_(self, other):
        if self.shape != other.shape:
            raise ValueError(f"shape mismatch: {self.shape} != {other.shape}")
        return self


class _FakeVideoDecoder:
    def __init__(self, *, total_frames, fps, height=64, width=80):
        self.total_frames = total_frames
        self.avg_fps = fps
        self.height = height
        self.width = width
        self.requested_indices = None
        self.requested_chunks = []

    def __len__(self):
        return self.total_frames

    @property
    def frame_shape(self):
        return self.height, self.width, 3

    def get_frames_as_tensor(self, indices):
        self.requested_indices = indices
        self.requested_chunks.append(indices)
        return _FakeTensor((len(indices), self.height, self.width, 3))


class _FakeVideoProcessor:
    def __init__(self):
        self.size = SizeDict(shortest_edge=1, longest_edge=10**9)
        self.patch_size = 16
        self.merge_size = 2
        self.temporal_patch_size = 2
        self.fps = 2.0
        self.min_frames = 4
        self.max_frames = 768
        self.resample = None
        self.resize_calls = []

    def resize(self, video, *, size, resample):
        self.resize_calls.append((video.shape, size, resample))
        return _FakeTensor((video.shape[0], video.shape[1], size.height, size.width))


class _TensorVideoDecoder:
    def __init__(self, frames, fps):
        self.frames = frames
        self.avg_fps = fps
        self.requested_chunks = []

    def __len__(self):
        return self.frames.shape[0]

    @property
    def frame_shape(self):
        return tuple(self.frames.shape[1:])

    def get_frames_as_tensor(self, indices):
        self.requested_chunks.append(indices)
        return self.frames[indices]


class TestQwen3VLVideoPreprocess(CustomTestCase):
    def test_qwen3_resizes_source_frames_in_bounded_chunks(self):
        decoder = _FakeVideoDecoder(total_frames=10, fps=2.0)
        video_processor = _FakeVideoProcessor()
        frame_bytes = decoder.height * decoder.width * 3
        with patch.object(qwen3_vl, "VideoDecoderWrapper", _FakeVideoDecoder):
            video, metadata = asyncio.run(
                qwen3_vl.preprocess_video(
                    decoder,
                    video_processor=video_processor,
                    video_config={"nframes": 4},
                    max_decode_chunk_bytes=2 * frame_bytes,
                )
            )

        self.assertEqual(video.shape, (4, 3, 64, 64))
        self.assertEqual(decoder.requested_chunks, [[0, 3], [6, 9]])
        self.assertEqual(
            [call[0] for call in video_processor.resize_calls],
            [(2, 3, 64, 80), (2, 3, 64, 80)],
        )
        np.testing.assert_array_equal(metadata["frames_indices"], [0, 3, 6, 9])

    def test_qwen3_max_frames_still_spans_full_video(self):
        decoder = _FakeVideoDecoder(total_frames=3000, fps=30.0)
        video_processor = _FakeVideoProcessor()
        with patch.object(qwen3_vl, "VideoDecoderWrapper", _FakeVideoDecoder):
            video, _ = asyncio.run(
                qwen3_vl.preprocess_video(
                    decoder,
                    video_processor=video_processor,
                    video_config={"fps": 2, "max_frames": 10},
                )
            )

        requested_indices = [
            index for chunk in decoder.requested_chunks for index in chunk
        ]
        self.assertEqual(video.shape, (10, 3, 64, 64))
        self.assertEqual(requested_indices[0], 0)
        self.assertEqual(requested_indices[-1], 2999)
        self.assertTrue(np.all(np.diff(np.asarray(requested_indices)) > 0))

    def test_qwen3_uses_loaded_processor_sampling_defaults(self):
        decoder = _FakeVideoDecoder(total_frames=3000, fps=30.0)
        video_processor = _FakeVideoProcessor()
        video_processor.fps = 1.0
        video_processor.max_frames = 13
        video_processor.temporal_patch_size = 4
        with patch.object(qwen3_vl, "VideoDecoderWrapper", _FakeVideoDecoder):
            video, _ = asyncio.run(
                qwen3_vl.preprocess_video(
                    decoder,
                    video_processor=video_processor,
                )
            )

        requested_indices = [
            index for chunk in decoder.requested_chunks for index in chunk
        ]
        self.assertEqual(video.shape[0], 12)
        self.assertEqual(requested_indices[0], 0)
        self.assertEqual(requested_indices[-1], 2999)

    def test_qwen3_rejects_a_source_frame_larger_than_the_chunk_limit(self):
        decoder = _FakeVideoDecoder(total_frames=10, fps=2.0)
        video_processor = _FakeVideoProcessor()
        frame_bytes = decoder.height * decoder.width * 3
        with (
            patch.object(qwen3_vl, "VideoDecoderWrapper", _FakeVideoDecoder),
            self.assertRaisesRegex(ValueError, "single decoded video frame"),
        ):
            asyncio.run(
                qwen3_vl.preprocess_video(
                    decoder,
                    video_processor=video_processor,
                    video_config={"nframes": 4},
                    max_decode_chunk_bytes=frame_bytes - 1,
                )
            )

        self.assertEqual(decoder.requested_chunks, [])

    def test_chunked_resize_matches_the_hf_processor(self):
        frames = torch.arange(4 * 64 * 96 * 3, dtype=torch.int64)
        frames = frames.remainder(256).to(torch.uint8).reshape(4, 64, 96, 3)
        decoder = _TensorVideoDecoder(frames, fps=2.0)
        native_size = {
            "shortest_edge": 4 * 32 * 32,
            "longest_edge": 8 * 32 * 32,
        }
        video_processor = Qwen3VLVideoProcessor(
            patch_size=16,
            merge_size=2,
            temporal_patch_size=2,
            do_sample_frames=False,
        )
        source_video = frames.permute(0, 3, 1, 2)
        expected = video_processor(
            videos=[source_video],
            size=native_size,
            do_sample_frames=False,
            return_tensors="pt",
        )
        frame_bytes = decoder.frame_shape[0] * decoder.frame_shape[1] * 3

        with patch.object(qwen3_vl, "VideoDecoderWrapper", _TensorVideoDecoder):
            resized_video, _ = asyncio.run(
                qwen3_vl.preprocess_video(
                    decoder,
                    video_processor=video_processor,
                    video_config={"nframes": 4, "size": native_size},
                    max_decode_chunk_bytes=2 * frame_bytes,
                )
            )

        actual = video_processor(
            videos=[resized_video],
            do_sample_frames=False,
            do_resize=False,
            return_tensors="pt",
        )
        torch.testing.assert_close(
            actual["pixel_values_videos"],
            expected["pixel_values_videos"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(actual["video_grid_thw"], expected["video_grid_thw"])

    def test_qwen3_disables_the_downstream_video_resize(self):
        processor = qwen3_vl.Qwen3VLImageProcessor.__new__(
            qwen3_vl.Qwen3VLImageProcessor
        )
        processor.video_config = {
            "do_normalize": False,
            "resample": 3,
            "size": {"shortest_edge": 4096, "longest_edge": 8192},
        }

        self.assertEqual(
            processor._processor_video_config([{"fps": 30.0}]),
            {"do_normalize": False, "do_resize": False},
        )

    def test_qwen3_rejects_legacy_qwen2_size_fields(self):
        for key in (
            "min_pixels",
            "max_pixels",
            "total_pixels",
            "resized_height",
            "resized_width",
        ):
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "legacy Qwen2 sizing fields"),
            ):
                qwen3_vl.validate_qwen3_video_config({key: 1})

    def test_qwen3_rejects_processor_geometry_overrides(self):
        for key in ("patch_size", "merge_size", "temporal_patch_size"):
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "loaded from the Hugging Face"),
            ):
                qwen3_vl.validate_qwen3_video_config({key: 1})

        with self.assertRaisesRegex(ValueError, "do_resize.*managed by SGLang"):
            qwen3_vl.validate_qwen3_video_config({"do_resize": False})

    def test_qwen3_validates_native_size(self):
        qwen3_vl.validate_qwen3_video_config(
            {"size": {"shortest_edge": 4096, "longest_edge": 8192}}
        )

        invalid_sizes = (
            {"height": 64, "width": 96},
            {"shortest_edge": 8192, "longest_edge": 4096},
            {"shortest_edge": 0, "longest_edge": 8192},
        )
        for size in invalid_sizes:
            with self.subTest(size=size), self.assertRaises((TypeError, ValueError)):
                qwen3_vl.validate_qwen3_video_config({"size": size})

    def test_qwen3_rejects_legacy_size_during_processor_initialization(self):
        def fake_base_init(processor, *_args, **_kwargs):
            processor.video_config = {"max_pixels": 8192}

        with (
            patch.object(qwen3_vl.QwenVLImageProcessor, "__init__", fake_base_init),
            self.assertRaisesRegex(ValueError, "legacy Qwen2 sizing fields"),
        ):
            qwen3_vl.Qwen3VLImageProcessor(None, None, None)

    def test_qwen2_keeps_legacy_factor_28_resize(self):
        decoder = _FakeVideoDecoder(total_frames=10, fps=2.0, height=56, width=84)
        resized = _FakeTensor((4, 3, 56, 84))
        with (
            patch.object(qwen_vl, "VideoDecoderWrapper", _FakeVideoDecoder),
            patch.object(
                qwen_vl.torchvision.transforms.functional,
                "resize",
                return_value=resized,
            ) as resize,
        ):
            video, _ = asyncio.run(
                qwen_vl.preprocess_video(decoder, video_config={"nframes": 4})
            )

        resize.assert_called_once()
        self.assertEqual(resize.call_args.args[1], [280, 392])
        self.assertIs(video, resized)


if __name__ == "__main__":
    unittest.main(verbosity=2)
