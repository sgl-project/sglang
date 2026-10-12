"""Qwen-VL video preprocessing must bound raw frames without changing pixels."""

import asyncio
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torchvision
from torchvision.transforms import InterpolationMode

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.multimodal.processors import qwen_vl
from sglang.srt.utils import video_decoder

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _DecodedFrames:
    def __init__(self, data):
        self.data = data

    def asnumpy(self):
        return self.data.numpy()


class _RecordingDecoder:
    def __init__(self, height, width, total_frames=20):
        self.frame = (
            torch.arange(height * width * 3).reshape(height, width, 3).to(torch.uint8)
        )
        self.total_frames = total_frames
        self.metadata = SimpleNamespace(height=height, width=width, average_fps=5.0)
        self.batches = []
        self.batch_references = []
        self.max_live_batches = 0

    def __len__(self):
        return self.total_frames

    def __getitem__(self, index):
        return _DecodedFrames(self.frame + index)

    def get_avg_fps(self):
        return self.metadata.average_fps

    def get_frames_at(self, indices):
        self.batches.append(indices)
        self.max_live_batches = max(
            self.max_live_batches,
            1 + sum(ref() is not None for ref in self.batch_references),
        )
        data = torch.stack([self.frame + index for index in indices])
        self.batch_references.append(weakref.ref(data))
        return _DecodedFrames(data)

    get_batch = get_frames_at


class TestQwenVLVideoPreprocess(CustomTestCase):
    @staticmethod
    def _run(decoder, config, backend="torchcodec", image_factor=28):
        vr = video_decoder.VideoDecoderWrapper.__new__(
            video_decoder.VideoDecoderWrapper
        )
        vr._decoder = decoder
        vr._num_decode_threads = 1
        with (
            patch.object(video_decoder, "_BACKEND", backend),
            patch.object(video_decoder, "_is_cpu_engine", return_value=True),
            patch.object(qwen_vl, "is_cpu", return_value=True),
        ):
            return asyncio.run(
                qwen_vl.preprocess_video(
                    vr, image_factor=image_factor, video_config=config
                )
            )

    def _assert_pixels_and_metadata(self, decoder, video, metadata, target):
        indices = metadata["frames_indices"]
        golden = torchvision.transforms.functional.resize(
            torch.stack([decoder.frame + int(index) for index in indices]).permute(
                0, 3, 1, 2
            ),
            target,
            interpolation=InterpolationMode.BILINEAR,
        )
        self.assertTrue(torch.equal(video, golden))
        self.assertEqual(video.dtype, golden.dtype)
        self.assertEqual(video.device, golden.device)
        self.assertEqual(metadata["fps"], 5.0)
        self.assertEqual(metadata["duration"], len(decoder) / 5.0)
        self.assertEqual(metadata["total_num_frames"], len(decoder))
        self.assertEqual(metadata["video_backend"], "torchvision")
        self.assertEqual(
            [index for batch in decoder.batches for index in batch], indices.tolist()
        )

    def test_downscale_bounds_decoded_pixels_and_preserves_output(self):
        for backend in ("torchcodec", "decord"):
            with self.subTest(backend=backend):
                decoder = _RecordingDecoder(112, 168)
                video, metadata = self._run(
                    decoder,
                    {"nframes": 12, "min_pixels": 28**2, "max_pixels": 6 * 28**2},
                    backend,
                )
                self._assert_pixels_and_metadata(decoder, video, metadata, [56, 84])
                self.assertLessEqual(max(map(len, decoder.batches)), 3)
                self.assertEqual(decoder.max_live_batches, 1)

    def test_ordinary_video_keeps_single_decode_batch(self):
        decoder = _RecordingDecoder(60, 86)
        video, metadata = self._run(
            decoder, {"nframes": 12, "min_pixels": 28**2, "max_pixels": 8 * 28**2}
        )
        self._assert_pixels_and_metadata(decoder, video, metadata, [56, 84])
        self.assertEqual(len(decoder.batches), 1)

    def test_no_resize_and_upscale_keep_single_decode_batch(self):
        for source, target, min_pixels in (
            ((56, 84), [56, 84], 28**2),
            ((28, 28), [56, 56], 4 * 28**2),
        ):
            with self.subTest(source=source):
                decoder = _RecordingDecoder(*source)
                video, metadata = self._run(
                    decoder,
                    {"nframes": 12, "min_pixels": min_pixels, "max_pixels": 6 * 28**2},
                )
                self._assert_pixels_and_metadata(decoder, video, metadata, target)
                self.assertEqual(len(decoder.batches), 1)

    def test_total_pixel_budget_applies_before_decode(self):
        decoder = _RecordingDecoder(112, 168)
        video, metadata = self._run(
            decoder, {"nframes": 12, "min_pixels": 28**2, "total_pixels": 28224}
        )
        self._assert_pixels_and_metadata(decoder, video, metadata, [56, 84])
        self.assertLessEqual(max(map(len, decoder.batches)), 3)

    def test_explicit_resize_and_sampling_are_preserved(self):
        decoder = _RecordingDecoder(112, 168)
        video, metadata = self._run(
            decoder,
            {
                "fps": 1,
                "min_frames": 6,
                "max_frames": 8,
                "min_pixels": 14**2,
                "max_pixels": 3 * 14**2,
                "resized_height": 84,
                "resized_width": 126,
            },
            image_factor=14,
        )
        self._assert_pixels_and_metadata(decoder, video, metadata, [84, 126])
        np.testing.assert_array_equal(
            metadata["frames_indices"], np.linspace(0, 19, 6, dtype=np.int64)
        )
        self.assertLessEqual(max(map(len, decoder.batches)), 3)

    def test_one_source_frame_can_exceed_entire_output_budget(self):
        decoder = _RecordingDecoder(224, 336)
        video, metadata = self._run(
            decoder, {"nframes": 4, "min_pixels": 28**2, "max_pixels": 6 * 28**2}
        )
        self._assert_pixels_and_metadata(decoder, video, metadata, [56, 84])
        self.assertEqual(list(map(len, decoder.batches)), [1, 1, 1, 1])

    def test_preprocessed_tensor_passes_through(self):
        video = torch.zeros(4, 3, 56, 84, dtype=torch.uint8)
        result, metadata = asyncio.run(qwen_vl.preprocess_video(video))
        self.assertIs(result, video)
        self.assertIsNone(metadata)

    def test_invalid_resize_is_rejected_before_decoding(self):
        decoder = _RecordingDecoder(112, 168)
        with self.assertRaisesRegex(ValueError, "absolute aspect ratio"):
            self._run(
                decoder,
                {"nframes": 12, "resized_height": 1, "resized_width": 400},
            )
        self.assertEqual(decoder.batches, [])


if __name__ == "__main__":
    unittest.main()
