"""Regression tests for explicit ``input_audio`` media containers."""

import asyncio
import base64
import concurrent.futures
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import soundfile

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.audio_from_video import (
    decode_audio_container,
    extract_audio_from_video_bytes,
    is_audio_container,
)
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils.common import load_audio
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _tone(*, sample_rate: int, channels: int = 1) -> np.ndarray:
    samples = np.arange(sample_rate // 10)
    mono = 0.2 * np.sin(2 * np.pi * 440 * samples / sample_rate)
    return np.repeat(mono[np.newaxis, :], channels, axis=0).astype(np.float32)


def _encode_audio_container(
    *,
    codec: str,
    container_format: str,
    sample_rate: int,
    channels: int = 1,
) -> bytes:
    output = io.BytesIO()
    layout = "mono" if channels == 1 else "stereo"
    sample_format = "s16" if codec == "libopencore_amrnb" else "fltp"
    samples = _tone(sample_rate=sample_rate, channels=channels)
    if sample_format == "s16":
        samples = (samples * np.iinfo(np.int16).max).astype(np.int16)

    with av.open(output, mode="w", format=container_format) as container:
        stream = container.add_stream(codec, rate=sample_rate)
        stream.layout = layout
        frame = av.AudioFrame.from_ndarray(
            samples,
            format=sample_format,
            layout=layout,
        )
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return output.getvalue()


def _encode_video_only_mp4() -> bytes:
    output = io.BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=1)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(
            np.zeros((16, 16, 3), dtype=np.uint8),
            format="rgb24",
        )
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return output.getvalue()


def _encode_wav() -> bytes:
    output = io.BytesIO()
    soundfile.write(output, _tone(sample_rate=16000)[0], 16000, format="WAV")
    return output.getvalue()


class _StubProcessor(BaseMultimodalProcessor):
    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ):
        raise NotImplementedError


class TestAudioContainerDetection(CustomTestCase):
    def test_supported_container_signatures(self):
        """Guards the external container signatures used for decoder routing."""
        supported_headers = (
            b"\x00\x00\x00\x18ftypisom",
            b"\x00\x00\x00\x18ftypM4A ",
            b"RIFF\x00\x00\x00\x00AVI ",
            b"#!AMR\n",
            b"#!AMR-WB\n",
        )
        for header in supported_headers:
            with self.subTest(header=header):
                self.assertTrue(is_audio_container(header))

    def test_non_container_audio_is_not_routed(self):
        """Keeps WAV and unknown short inputs on their existing decoder path."""
        self.assertFalse(is_audio_container(b"RIFF\x00\x00\x00\x00WAVE"))
        self.assertFalse(is_audio_container(b"random"))


class TestAudioContainerDecode(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.mp4 = _encode_audio_container(
            codec="aac",
            container_format="mp4",
            sample_rate=16000,
            channels=2,
        )
        cls.amr = _encode_audio_container(
            codec="libopencore_amrnb",
            container_format="amr",
            sample_rate=8000,
        )

    def test_mp4_aac_decodes_and_resamples_to_mono(self):
        """Reproduces the AAC-in-MP4 input that libsndfile rejected."""
        waveform = decode_audio_container(
            self.mp4,
            target_sr=8000,
            mono=True,
        )
        self.assertEqual(waveform.ndim, 1)
        self.assertEqual(waveform.dtype, np.float32)
        self.assertTrue(waveform.flags.c_contiguous)
        self.assertGreater(len(waveform), 700)

    def test_mp4_bypasses_torchcodec_backend(self):
        """Recognized containers must use PyAV regardless of video backend."""
        with patch("sglang.srt.utils.common._BACKEND", "torchcodec"):
            waveform = load_audio(self.mp4, sr=16000)
        self.assertGreater(len(waveform), 0)

    def test_wav_keeps_existing_decoder_path(self):
        """Ordinary WAV must not be redirected to the container decoder."""
        with (
            patch("sglang.srt.utils.common._BACKEND", "decord"),
            patch(
                "sglang.srt.multimodal.audio_from_video.decode_audio_container"
            ) as decoder,
        ):
            waveform = load_audio(_encode_wav(), sr=16000)
        decoder.assert_not_called()
        self.assertEqual(waveform.shape, (1600,))

    def test_amr_decodes_despite_wav_data_url_mime(self):
        """Reproduces AMR bytes mislabeled as WAV by production clients."""
        data_url = "data:audio/wav;base64," + base64.b64encode(self.amr).decode()
        waveform = load_audio(data_url, sr=16000)
        self.assertEqual(waveform.ndim, 1)
        self.assertGreater(len(waveform), 1500)

    def test_recognized_path_is_passed_directly_to_pyav(self):
        """Prevents path inputs from regressing to a full in-memory read."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.m4a"
            path.write_bytes(self.mp4)
            for source in (str(path), path.as_uri()):
                with (
                    self.subTest(source=source),
                    patch(
                        "sglang.srt.multimodal.audio_from_video.decode_audio_container",
                        wraps=decode_audio_container,
                    ) as decoder,
                ):
                    waveform = load_audio(source, sr=16000)
                    self.assertGreater(len(waveform), 0)
                    self.assertEqual(decoder.call_args.args[0], str(path))

    def test_invalid_container_remains_a_value_error(self):
        """Malformed recognized media must remain a client input error."""
        with self.assertRaisesRegex(ValueError, "Invalid input_audio"):
            load_audio(b"\x00\x00\x00\x18ftypisom-invalid", sr=16000)

    def test_container_without_audio_is_a_value_error(self):
        """A valid video-only MP4 must not be accepted as empty audio."""
        with self.assertRaisesRegex(ValueError, "Invalid input_audio"):
            decode_audio_container(
                _encode_video_only_mp4(),
                target_sr=16000,
                mono=True,
            )

    def test_invalid_container_is_classified_as_bad_input(self):
        """Invalid input_audio must not become an internal server error."""
        with self.assertRaisesRegex(ValueError, "Invalid input_audio"):
            _StubProcessor._load_single_item(
                b"\x00\x00\x00\x18ftypisom-invalid",
                Modality.AUDIO,
                audio_sample_rate=16000,
            )

    def test_fast_loader_preserves_bad_input_error(self):
        """Fast multimodal loading must preserve client input errors."""
        processor = _StubProcessor.__new__(_StubProcessor)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            processor.io_executor = executor
            with self.assertRaisesRegex(ValueError, "Invalid input_audio"):
                asyncio.run(
                    processor.fast_load_mm_data(
                        prompt="",
                        multimodal_tokens=None,
                        audio_data=[b"\x00\x00\x00\x18ftypisom-invalid"],
                        audio_sample_rate=16000,
                    )
                )

    def test_lenient_video_wrapper_returns_none(self):
        """Silent or corrupt video remains optional for video-specific callers."""
        self.assertIsNone(extract_audio_from_video_bytes(b"invalid"))


if __name__ == "__main__":
    unittest.main()
