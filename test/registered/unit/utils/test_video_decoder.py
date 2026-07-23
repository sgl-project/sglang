"""CPU unit tests for the unified video decoder wrapper."""

import os
import sys
import threading
import types
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np

from sglang.srt.utils import video_decoder
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_decord_module(video_reader):
    module = types.ModuleType("decord")
    module.VideoReader = video_reader
    module.cpu = MagicMock(return_value="cpu:0")
    return module


def make_wrapper(decoder, *, source="video.mp4", num_decode_threads=1):
    wrapper = video_decoder.VideoDecoderWrapper.__new__(
        video_decoder.VideoDecoderWrapper
    )
    wrapper._source = source
    wrapper._num_decode_threads = num_decode_threads
    wrapper._source_bytes = source if isinstance(source, bytes) else None
    wrapper._source_path = source if isinstance(source, str) else None
    wrapper._tmp_path = None
    wrapper._decoder = decoder
    wrapper._tc_kwargs = {"dimension_order": "NHWC"}
    return wrapper


class TestVideoDecoderWrapper(CustomTestCase):
    def test_cuda_backend_success_is_cached(self):
        decoders = types.ModuleType("torchcodec.decoders")
        decoders.set_cuda_backend = MagicMock()

        with (
            patch.object(video_decoder, "_cuda_backend_enabled", None),
            patch.dict(sys.modules, {"torchcodec.decoders": decoders}),
        ):
            self.assertTrue(video_decoder._try_cuda_backend())
            self.assertTrue(video_decoder._try_cuda_backend())

        decoders.set_cuda_backend.assert_called_once_with("beta")

    def test_cuda_backend_failure_is_cached(self):
        decoders = types.ModuleType("torchcodec.decoders")
        decoders.set_cuda_backend = MagicMock(side_effect=RuntimeError("unsupported"))

        with (
            patch.object(video_decoder, "_cuda_backend_enabled", None),
            patch.dict(sys.modules, {"torchcodec.decoders": decoders}),
        ):
            self.assertFalse(video_decoder._try_cuda_backend())
            self.assertFalse(video_decoder._try_cuda_backend())

        decoders.set_cuda_backend.assert_called_once_with("beta")

    def test_torchcodec_initialization_and_numpy_accessors(self):
        frame_array = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        batch_array = np.stack([frame_array, frame_array])
        frame = MagicMock()
        frame.numpy.return_value = frame_array
        batch = SimpleNamespace(data=MagicMock())
        batch.data.numpy.return_value = batch_array

        decoder = MagicMock()
        decoder.__len__.return_value = 7
        decoder.__getitem__.return_value = frame
        decoder.metadata.average_fps = 29.97
        decoder.get_frames_at.return_value = batch
        decoder_factory = MagicMock(return_value=decoder)

        with (
            patch.object(video_decoder, "_BACKEND", "torchcodec"),
            patch.object(video_decoder, "VideoDecoder", decoder_factory, create=True),
        ):
            wrapper = video_decoder.VideoDecoderWrapper("video.mp4")
            self.assertEqual(len(wrapper), 7)
            np.testing.assert_array_equal(wrapper[3], frame_array)
            self.assertEqual(wrapper.avg_fps, 29.97)
            np.testing.assert_array_equal(wrapper.get_frames_at([1, 4]), batch_array)

        decoder_factory.assert_called_once_with("video.mp4", dimension_order="NHWC")
        decoder.__getitem__.assert_called_once_with(3)
        decoder.get_frames_at.assert_called_once_with([1, 4])

    def test_torchcodec_cuda_failure_falls_back_to_cpu(self):
        cpu_decoder = MagicMock()
        decoder_factory = MagicMock(
            side_effect=[RuntimeError("CUDA decode failed"), cpu_decoder]
        )

        with (
            patch.object(video_decoder, "_BACKEND", "torchcodec"),
            patch.object(video_decoder, "_try_cuda_backend", return_value=True),
            patch.object(video_decoder, "VideoDecoder", decoder_factory, create=True),
            self.assertLogs(video_decoder.logger, level="WARNING") as logs,
        ):
            wrapper = video_decoder.VideoDecoderWrapper("video.mp4", device="cuda")

        self.assertIs(wrapper._decoder, cpu_decoder)
        self.assertEqual(wrapper._tc_kwargs, {"dimension_order": "NHWC"})
        self.assertIn("falling back to CPU", logs.output[0])
        self.assertEqual(
            decoder_factory.call_args_list,
            [
                call("video.mp4", dimension_order="NHWC", device="cuda"),
                call("video.mp4", dimension_order="NHWC"),
            ],
        )

    def test_torchcodec_cpu_failure_is_not_suppressed(self):
        decoder_factory = MagicMock(side_effect=RuntimeError("corrupt input"))

        with (
            patch.object(video_decoder, "_BACKEND", "torchcodec"),
            patch.object(video_decoder, "VideoDecoder", decoder_factory, create=True),
            self.assertRaisesRegex(RuntimeError, "corrupt input"),
        ):
            video_decoder.VideoDecoderWrapper("video.mp4")

    def test_decord_path_accessors_and_source_bytes(self):
        payload = b"video-file-contents"
        frame_array = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        batch_array = np.stack([frame_array, frame_array])
        frame = MagicMock()
        frame.asnumpy.return_value = frame_array
        batch = MagicMock()
        batch.asnumpy.return_value = batch_array

        decoder = MagicMock()
        decoder.__len__.return_value = 5
        decoder.__getitem__.side_effect = [frame, frame_array]
        decoder.get_avg_fps.return_value = 24.0
        decoder.get_batch.return_value = batch
        video_reader = MagicMock(return_value=decoder)
        decord = make_decord_module(video_reader)

        with TemporaryDirectory() as directory:
            source_path = os.path.join(directory, "clip.mp4")
            with open(source_path, "wb") as source_file:
                source_file.write(payload)

            with (
                patch.object(video_decoder, "_BACKEND", "decord"),
                patch.dict(sys.modules, {"decord": decord}),
            ):
                wrapper = video_decoder.VideoDecoderWrapper(source_path)
                self.assertEqual(len(wrapper), 5)
                np.testing.assert_array_equal(wrapper[0], frame_array)
                np.testing.assert_array_equal(wrapper[1], frame_array)
                self.assertEqual(wrapper.avg_fps, 24.0)
                np.testing.assert_array_equal(
                    wrapper.get_frames_at([0, 3]), batch_array
                )
                self.assertEqual(wrapper.source_bytes, payload)
                wrapper.close()

            self.assertTrue(os.path.exists(source_path))

        decord.cpu.assert_called_once_with(0)
        video_reader.assert_called_once_with(source_path, ctx="cpu:0")

    def test_decord_bytes_create_and_cleanup_temporary_file(self):
        payload = b"encoded-video-bytes"
        observed_paths = []

        def video_reader(path, *, ctx):
            observed_paths.append(path)
            self.assertEqual(ctx, "cpu:0")
            with open(path, "rb") as temporary_file:
                self.assertEqual(temporary_file.read(), payload)
            return MagicMock()

        decord = make_decord_module(video_reader)
        with (
            patch.object(video_decoder, "_BACKEND", "decord"),
            patch.dict(sys.modules, {"decord": decord}),
        ):
            with video_decoder.VideoDecoderWrapper(payload) as wrapper:
                temporary_path = wrapper._tmp_path
                self.assertIs(wrapper.__enter__(), wrapper)
                self.assertEqual(wrapper.source_bytes, payload)
                self.assertTrue(os.path.exists(temporary_path))
                self.assertTrue(temporary_path.endswith(".mp4"))

            self.assertFalse(os.path.exists(temporary_path))
            self.assertIsNone(wrapper._tmp_path)
            wrapper.close()

        self.assertEqual(observed_paths, [temporary_path])

    def test_get_frames_as_tensor_uses_backend_conversion(self):
        torchcodec_data = MagicMock()
        torchcodec_data.device.type = "cpu"
        torchcodec_pinned = object()
        torchcodec_data.pin_memory.return_value = torchcodec_pinned
        torchcodec_decoder = MagicMock()
        torchcodec_decoder.get_frames_at.return_value = SimpleNamespace(
            data=torchcodec_data
        )
        torchcodec_wrapper = make_wrapper(torchcodec_decoder)

        with patch.object(video_decoder, "_BACKEND", "torchcodec"):
            self.assertIs(
                torchcodec_wrapper.get_frames_as_tensor([1, 2]), torchcodec_pinned
            )
        torchcodec_decoder.get_frames_at.assert_called_once_with([1, 2])
        torchcodec_data.pin_memory.assert_called_once_with()

        batch_array = np.zeros((2, 2, 2, 3), dtype=np.uint8)
        decord_batch = MagicMock()
        decord_batch.asnumpy.return_value = batch_array
        decord_decoder = MagicMock()
        decord_decoder.get_batch.return_value = decord_batch
        decord_wrapper = make_wrapper(decord_decoder)
        converted = MagicMock()
        converted.device.type = "cpu"
        decord_pinned = object()
        converted.pin_memory.return_value = decord_pinned
        fake_torch = types.ModuleType("torch")
        fake_torch.from_numpy = MagicMock(return_value=converted)

        with (
            patch.object(video_decoder, "_BACKEND", "decord"),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            self.assertIs(decord_wrapper.get_frames_as_tensor([0, 3]), decord_pinned)

        decord_decoder.get_batch.assert_called_once_with([0, 3])
        fake_torch.from_numpy.assert_called_once_with(batch_array)
        converted.pin_memory.assert_called_once_with()

    def test_parallel_thread_count_is_bounded(self):
        fake_torch = types.ModuleType("torch")
        cases = [
            (0, 64, list(range(20)), 16),
            (0, None, list(range(20)), 8),
            (8, 64, [0, 1, 2], 3),
        ]

        with (
            patch.object(video_decoder, "_BACKEND", "torchcodec"),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            for configured_threads, cpu_count, indices, expected_threads in cases:
                with self.subTest(
                    configured_threads=configured_threads,
                    cpu_count=cpu_count,
                    expected_threads=expected_threads,
                ):
                    wrapper = make_wrapper(
                        MagicMock(), num_decode_threads=configured_threads
                    )
                    sentinel = object()
                    with (
                        patch.object(os, "cpu_count", return_value=cpu_count),
                        patch.object(
                            wrapper, "_parallel_decode", return_value=sentinel
                        ) as parallel_decode,
                    ):
                        self.assertIs(wrapper.get_frames_as_tensor(indices), sentinel)
                    parallel_decode.assert_called_once_with(indices, expected_threads)

    def test_parallel_decode_preserves_chunk_order(self):
        second_chunk_started = threading.Event()
        decoder_calls = []

        class FakeVideoDecoder:
            def __init__(self, source, **kwargs):
                decoder_calls.append((source, kwargs))

            def get_frames_at(self, chunk):
                if chunk == [0, 1]:
                    second_chunk_started.wait(timeout=1)
                else:
                    second_chunk_started.set()
                return SimpleNamespace(data=tuple(chunk))

        concatenated = MagicMock()
        pinned = object()
        concatenated.pin_memory.return_value = pinned
        fake_torch = types.ModuleType("torch")
        fake_torch.cat = MagicMock(return_value=concatenated)
        wrapper = make_wrapper(MagicMock(), source=b"video", num_decode_threads=2)

        with (
            patch.object(video_decoder, "VideoDecoder", FakeVideoDecoder, create=True),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            self.assertIs(wrapper._parallel_decode([0, 1, 2, 3], 2), pinned)

        fake_torch.cat.assert_called_once_with([(0, 1), (2, 3)], dim=0)
        concatenated.pin_memory.assert_called_once_with()
        self.assertEqual(
            decoder_calls,
            [
                (b"video", {"dimension_order": "NHWC"}),
                (b"video", {"dimension_order": "NHWC"}),
            ],
        )


if __name__ == "__main__":
    unittest.main()
