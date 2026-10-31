# SPDX-License-Identifier: Apache-2.0
"""A warm output and a different resolution must keep separate NVENC sessions."""

import importlib.util
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from sglang.test.test_utils import CustomTestCase


class TestNvencAvailability(CustomTestCase):
    def test_h264_encoder_capability_controls_nvenc(self):
        """A GPU without an H.264 encoder must fall back even when CUDA is available."""
        from sglang.multimodal_gen.runtime.entrypoints import (
            nvenc_video_encoder as nvenc,
        )

        native = types.ModuleType("PyNvVideoCodec")
        native.GetEncoderCaps = mock.Mock(return_value={"num_encoder_engines": 0})
        nvenc._device_has_nvenc.cache_clear()
        try:
            with (
                mock.patch.dict(sys.modules, {"PyNvVideoCodec": native}),
                mock.patch.object(torch.cuda, "is_available", return_value=True),
            ):
                self.assertFalse(nvenc.supports_nvenc(torch.device("cuda:0")))
                native.GetEncoderCaps.assert_called_once_with(0, "h264")
                native.GetEncoderCaps.return_value = {"num_encoder_engines": 1}
                nvenc._device_has_nvenc.cache_clear()
                self.assertTrue(nvenc.supports_nvenc(torch.device("cuda:0")))
                native.GetEncoderCaps.side_effect = RuntimeError()
                nvenc._device_has_nvenc.cache_clear()
                self.assertFalse(nvenc.supports_nvenc(torch.device("cuda:0")))
        finally:
            nvenc._device_has_nvenc.cache_clear()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_h100_h200_fall_back_to_cpu(self):
        """H100/H200 expose CUDA but have no NVENC encoder engines."""
        from sglang.multimodal_gen.runtime.entrypoints import (
            nvenc_video_encoder as nvenc,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        if not any(
            gpu in torch.cuda.get_device_name(device) for gpu in ("H100", "H200")
        ):
            self.skipTest("requires H100 or H200")
        self.assertFalse(nvenc.supports_nvenc(device))
        self.assertIsNone(
            nvenc.NvencVideoEncoder.open(
                "/tmp/nvenc-unavailable.mp4",
                device=device,
                height=480,
                width=832,
                fps=24,
                num_frames=4,
            )
        )


@unittest.skipUnless(
    torch.cuda.is_available() and importlib.util.find_spec("PyNvVideoCodec"),
    "requires CUDA and PyNvVideoCodec",
)
class TestNvencVideoEncoder(CustomTestCase):
    def setUp(self):
        super().setUp()
        from sglang.multimodal_gen.runtime.entrypoints.nvenc_video_encoder import (
            supports_nvenc,
        )

        if not supports_nvenc(torch.device("cuda", torch.cuda.current_device())):
            self.skipTest("requires H.264 NVENC")

    def test_warmup_cache_hits_by_resolution(self):
        import av

        from sglang.multimodal_gen.runtime.entrypoints import (
            nvenc_video_encoder as nvenc,
        )

        with nvenc._cache_lock:
            nvenc._idle_sessions.clear()
        device = torch.device("cuda", torch.cuda.current_device())
        cases = (
            (832, 480, False),
            (640, 360, False),
            (832, 480, True),
            (640, 360, True),
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                for index, (width, height, expected_hit) in enumerate(cases):
                    path = Path(directory, f"video_{index}.mp4")
                    frames = torch.full(
                        (3, 4, height, width),
                        index / 5,
                        dtype=torch.float32,
                        device=device,
                    )
                    encoder = nvenc.NvencVideoEncoder.open(
                        str(path),
                        device=device,
                        height=height,
                        width=width,
                        fps=24,
                        num_frames=4,
                    )
                    self.assertIsNotNone(encoder)
                    self.assertEqual(encoder.cache_hit, expected_hit)
                    try:
                        encoder.write(frames)
                        encoder.close()
                    except BaseException:
                        encoder.abort()
                        raise
                    with av.open(str(path)) as container:
                        stream = container.streams.video[0]
                        decoded = list(container.decode(video=0))
                        self.assertEqual((stream.width, stream.height), (width, height))
                        self.assertEqual(len(decoded), 4)
                        self.assertEqual(decoded[0].pts, 0)
                        self.assertTrue(decoded[0].key_frame)
                        self.assertAlmostEqual(
                            float(stream.duration * stream.time_base), 4 / 24
                        )
        finally:
            with nvenc._cache_lock:
                nvenc._idle_sessions.clear()

    def test_streamed_write_returns_while_nvenc_worker_is_busy(self):
        """The decode callback can return before the NVENC worker finishes."""
        import av

        from sglang.multimodal_gen.runtime.entrypoints.nvenc_video_encoder import (
            NvencVideoEncoder,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "overlapped.mp4")
            encoder = NvencVideoEncoder.open(
                str(path),
                device=device,
                height=64,
                width=64,
                fps=24,
                num_frames=4,
                max_queued_frames=4,
            )
            self.assertIsNotNone(encoder)
            started = threading.Event()
            release = threading.Event()
            original = encoder._write_frames

            def held_write(frames, ready):
                started.set()
                if not release.wait(5):
                    raise TimeoutError("NVENC worker was not released")
                return original(frames, ready)

            encoder._write_frames = held_write
            errors = []

            def submit_frames():
                try:
                    frames = torch.full((3, 4, 64, 64), 0.5, device=device)
                    encoder.write(frames)
                except BaseException as exc:
                    errors.append(exc)

            producer = threading.Thread(target=submit_frames)
            producer.start()
            try:
                self.assertTrue(started.wait(5))
                producer.join(timeout=0.25)
                self.assertFalse(producer.is_alive())
            finally:
                release.set()
                producer.join(timeout=5)
            self.assertEqual(errors, [])
            encoder.close()
            with av.open(str(path)) as container:
                frames = list(container.decode(video=0))
                self.assertEqual(len(frames), 4)
                self.assertAlmostEqual(
                    frames[0].to_ndarray(format="rgb24").mean(), 128, delta=12
                )

    def test_file_only_output_uses_nvenc_for_cpu_frames_and_audio(self):
        import av

        from sglang.multimodal_gen.configs.sample.sampling_params import DataType
        from sglang.multimodal_gen.runtime.entrypoints import utils
        from sglang.srt.environ import envs

        rgb = np.full((4, 360, 640, 3), 128, dtype=np.uint8)
        cpu_frames = torch.from_numpy(rgb.astype(np.float32) / 255).permute(3, 0, 1, 2)
        samples = (
            cpu_frames,
            rgb.astype(np.float32) / 255,
            rgb,
            cpu_frames.to("cuda"),
        )
        audio = torch.sin(torch.arange(3200) / 10).float()
        with tempfile.TemporaryDirectory() as directory:
            for index, sample in enumerate(samples):
                with self.subTest(sample_type=type(sample), index=index):
                    path = Path(directory, f"video_{index}.mp4")
                    with (
                        envs.SGLANG_USE_NVENC.override(True),
                        mock.patch.object(
                            utils,
                            "_save_video_ffmpeg",
                            side_effect=AssertionError("unexpected CPU encode"),
                        ),
                    ):
                        utils.save_outputs(
                            [sample],
                            DataType.VIDEO,
                            24,
                            True,
                            lambda _idx: str(path),
                            audio=audio if index == 0 else None,
                            audio_sample_rate=16000,
                        )
                    with av.open(str(path)) as container:
                        decoded = list(container.decode(video=0))
                        self.assertEqual(len(decoded), 4)
                        mean_rgb = decoded[0].to_ndarray(format="rgb24").mean()
                        self.assertAlmostEqual(mean_rgb, 128, delta=12)
                        self.assertEqual(len(container.streams.audio), index == 0)


if __name__ == "__main__":
    unittest.main()
