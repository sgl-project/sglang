"""Invariant tests for application-level audio decode limits."""

import gc
import io
import math
import os
import sys
import types
import unittest
import warnings
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import sglang.srt.utils.common as common_utils
from sglang.srt.environ import envs
from sglang.srt.utils.common import load_audio
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "base-a-test-cpu")

SAMPLE_RATE = 16_000
DURATION_ENV = "SGLANG_MAX_AUDIO_DECODE_DURATION_S"
BYTE_ENV = "SGLANG_MAX_AUDIO_DECODE_BYTES"


def _audio_bytes(duration_s: float, *, sample_rate: int = SAMPLE_RATE) -> bytes:
    import soundfile as sf

    samples = np.zeros(round(sample_rate * duration_s), dtype=np.float32)
    output = io.BytesIO()
    sf.write(output, samples, sample_rate, format="FLAC")
    return output.getvalue()


def _patch_torchcodec(decoder_cls):
    torchcodec = types.ModuleType("torchcodec")
    decoders = types.ModuleType("torchcodec.decoders")
    decoders.AudioDecoder = decoder_cls
    torchcodec.decoders = decoders
    return patch.dict(
        sys.modules,
        {"torchcodec": torchcodec, "torchcodec.decoders": decoders},
    )


class TestLoadAudioDecodeLimits(unittest.TestCase):
    def setUp(self):
        self.audio = _audio_bytes(2.0)

    def test_real_compressed_audio_honors_composed_limits(self):
        with patch.object(common_utils, "_BACKEND", "soundfile"):
            decoded = load_audio(
                self.audio,
                sr=SAMPLE_RATE,
                max_duration_s=2,
                max_decode_bytes=256_000,
            )
            self.assertEqual(decoded.shape, (2 * SAMPLE_RATE,))

            with self.assertRaisesRegex(ValueError, "decode duration"):
                load_audio(
                    self.audio,
                    sr=SAMPLE_RATE,
                    max_duration_s=1,
                    max_decode_bytes=0,
                )
            with self.assertRaisesRegex(ValueError, "decoded size"):
                load_audio(
                    self.audio,
                    sr=SAMPLE_RATE,
                    max_duration_s=0,
                    max_decode_bytes=255_999,
                )

            decoded = load_audio(
                self.audio,
                sr=SAMPLE_RATE,
                max_duration_s=0,
                max_decode_bytes=0,
            )
            self.assertEqual(decoded.shape, (2 * SAMPLE_RATE,))

    def test_soundfile_exact_boundary_requires_eof(self):
        import soundfile as sf

        class FakeSoundFile:
            samplerate = SAMPLE_RATE
            channels = 1
            frames = 4
            returned_frames = 4
            requested_frames = None

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def read(self, *, frames, dtype):
                self.requested_frames = frames
                return np.zeros(self.returned_frames, dtype=dtype)

        audio_file = FakeSoundFile()
        with patch.object(sf, "SoundFile", return_value=audio_file):
            audio, _ = common_utils._decode_audio_with_soundfile(
                b"compressed-audio",
                dtype="float32",
                max_duration_s=0,
                max_decode_bytes=16,
            )
        self.assertEqual(audio.shape, (4,))
        self.assertEqual(audio_file.requested_frames, 5)

        audio_file = FakeSoundFile()
        audio_file.returned_frames = 5
        with (
            patch.object(sf, "SoundFile", return_value=audio_file),
            self.assertRaisesRegex(ValueError, "decoded size"),
        ):
            common_utils._decode_audio_with_soundfile(
                b"compressed-audio",
                dtype="float32",
                max_duration_s=0,
                max_decode_bytes=16,
            )

    def test_soundfile_rejects_oversized_header_before_read(self):
        import soundfile as sf

        class FakeSoundFile:
            samplerate = 192_000
            channels = 32
            frames = 192_000
            read_called = False

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def read(self, **_kwargs):
                self.read_called = True
                raise AssertionError("oversized PCM must be rejected before read")

        audio_file = FakeSoundFile()
        with (
            patch.object(sf, "SoundFile", return_value=audio_file),
            self.assertRaisesRegex(ValueError, "decoded size"),
        ):
            common_utils._decode_audio_with_soundfile(
                b"compressed-audio",
                max_duration_s=2,
                max_decode_bytes=1_048_576,
            )
        self.assertFalse(audio_file.read_called)

    def test_torchcodec_exact_boundary_requires_eof(self):
        class FakeAudioDecoder:
            returned_frames = 4
            requested_range = None

            def __init__(self, _source, **_kwargs):
                self.metadata = SimpleNamespace(
                    duration_seconds=math.nextafter(4 / SAMPLE_RATE, math.inf),
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                )

            def get_samples_played_in_range(self, start_seconds, *, stop_seconds):
                FakeAudioDecoder.requested_range = (start_seconds, stop_seconds)
                return SimpleNamespace(
                    data=torch.zeros((1, self.returned_frames), dtype=torch.float32),
                    sample_rate=SAMPLE_RATE,
                )

        with _patch_torchcodec(FakeAudioDecoder):
            samples = common_utils._decode_audio_with_torchcodec(
                b"compressed-audio",
                max_duration_s=0,
                max_decode_bytes=16,
            )
        self.assertEqual(samples.data.shape, (1, 4))
        self.assertEqual(
            FakeAudioDecoder.requested_range,
            (0.0, 5 / SAMPLE_RATE),
        )

        FakeAudioDecoder.returned_frames = 5
        with (
            _patch_torchcodec(FakeAudioDecoder),
            self.assertRaisesRegex(ValueError, "decoded size"),
        ):
            common_utils._decode_audio_with_torchcodec(
                b"compressed-audio",
                max_duration_s=0,
                max_decode_bytes=16,
            )

    def test_torchcodec_uses_metadata_and_bounded_range(self):
        class FakeAudioDecoder:
            range_called = False

            def __init__(self, _source, **_kwargs):
                self.metadata = SimpleNamespace(
                    duration_seconds=1.0,
                    sample_rate=192_000,
                    num_channels=32,
                )

            def get_all_samples(self):
                raise AssertionError("range-limited decode must not read all samples")

            def get_samples_played_in_range(self, *_args, **_kwargs):
                FakeAudioDecoder.range_called = True
                raise AssertionError("oversized metadata must reject before decode")

        with (
            _patch_torchcodec(FakeAudioDecoder),
            self.assertRaisesRegex(ValueError, "decoded size"),
        ):
            common_utils._decode_audio_with_torchcodec(
                b"compressed-audio",
                max_duration_s=2,
                max_decode_bytes=1_048_576,
            )
        self.assertFalse(FakeAudioDecoder.range_called)

    def test_torchcodec_unknown_duration_still_limits_logical_output_bytes(self):
        class FakeAudioDecoder:
            requested_range = None

            def __init__(self, _source, **_kwargs):
                self.metadata = SimpleNamespace(
                    duration_seconds=None,
                    sample_rate=192_000,
                    num_channels=8,
                )

            def get_all_samples(self):
                raise AssertionError("range-limited decode must not read all samples")

            def get_samples_played_in_range(self, start_seconds, *, stop_seconds):
                FakeAudioDecoder.requested_range = (start_seconds, stop_seconds)
                return SimpleNamespace(
                    data=torch.zeros((8, 31), dtype=torch.float32),
                    sample_rate=192_000,
                )

        with _patch_torchcodec(FakeAudioDecoder):
            samples = common_utils._decode_audio_with_torchcodec(
                b"compressed-audio",
                max_duration_s=0,
                max_decode_bytes=1024,
            )
        self.assertEqual(FakeAudioDecoder.requested_range, (0.0, 33 / 192_000))
        self.assertEqual(samples.data.numel() * samples.data.element_size(), 992)

    def test_torchcodec_limit_error_does_not_fallback(self):
        with (
            patch.object(common_utils, "_BACKEND", "torchcodec"),
            patch.object(
                common_utils,
                "_decode_audio_with_torchcodec",
                side_effect=common_utils._AudioDecodeLimitError("decode limit"),
            ),
            patch.object(common_utils, "_decode_audio_with_soundfile") as fallback,
            self.assertRaisesRegex(ValueError, "decode limit"),
        ):
            load_audio(b"compressed-audio")
        fallback.assert_not_called()

    def test_resample_expansion_is_rejected_before_allocation(self):
        with (
            patch.object(common_utils, "_BACKEND", "soundfile"),
            patch.object(
                common_utils,
                "_check_audio_resample_size",
                wraps=common_utils._check_audio_resample_size,
            ) as size_check,
            self.assertRaisesRegex(ValueError, "decoded size"),
        ):
            load_audio(
                _audio_bytes(1.0, sample_rate=8_000),
                sr=24_000,
                max_duration_s=0,
                max_decode_bytes=80_000,
            )
        size_check.assert_called_once_with(8_000, 1, 4, 8_000, 24_000, 80_000)

    def test_invalid_limits_and_sample_rates_fail_closed(self):
        for duration in (float("nan"), float("inf"), "invalid"):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                load_audio(self.audio, max_duration_s=duration)
        for byte_cap in (float("nan"), float("inf"), 1.5, "invalid"):
            with self.subTest(byte_cap=byte_cap), self.assertRaises(ValueError):
                load_audio(self.audio, max_decode_bytes=byte_cap)
        for sample_rate in (0, -1):
            with self.subTest(sample_rate=sample_rate), self.assertRaises(ValueError):
                load_audio(self.audio, sr=sample_rate)

    def test_environment_defaults_and_invalid_duration_fallback(self):
        with (
            patch.object(common_utils, "_BACKEND", "soundfile"),
            patch.dict(os.environ, {DURATION_ENV: "1", BYTE_ENV: "0"}),
            self.assertRaisesRegex(ValueError, "decode duration"),
        ):
            load_audio(self.audio, sr=SAMPLE_RATE)

        with (
            patch.object(common_utils, "_BACKEND", "soundfile"),
            patch.dict(os.environ, {DURATION_ENV: "0", BYTE_ENV: "255999"}),
            self.assertRaisesRegex(ValueError, "decoded size"),
        ):
            load_audio(self.audio, sr=SAMPLE_RATE)

        for value in ("invalid", "nan", "inf"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {DURATION_ENV: value}),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always")
                self.assertEqual(envs.SGLANG_MAX_AUDIO_DECODE_DURATION_S.get(), 600.0)
                self.assertRegex(str(caught[-1].message), "using default")


class TestChunkedTranscriptionMemory(unittest.TestCase):
    def test_iterator_uses_bounded_loader(self):
        from sglang.srt.entrypoints.openai import streaming_asr

        with (
            patch.object(
                streaming_asr,
                "_decode_audio_with_soundfile",
                side_effect=ValueError("decode limit"),
            ) as decoder,
            self.assertRaisesRegex(ValueError, "decode limit"),
        ):
            next(streaming_asr.iter_audio_chunks(b"compressed-audio", 2.0))
        decoder.assert_called_once_with(b"compressed-audio", dtype="float32")

    def test_iterator_releases_prior_prefix_and_preserves_cumulative_output(self):
        import soundfile as sf

        from sglang.srt.entrypoints.openai import streaming_asr

        class Payload:
            pass

        class FakeBuffer:
            def getvalue(self):
                return Payload()

            def close(self):
                pass

        with (
            patch.object(
                streaming_asr,
                "_decode_audio_with_soundfile",
                return_value=(np.zeros(4, dtype=np.float32), 2),
            ),
            patch.object(streaming_asr.io, "BytesIO", FakeBuffer),
            patch.object(streaming_asr.sf, "write"),
        ):
            chunks = streaming_asr.iter_audio_chunks(b"compressed-audio", 1.0)
            first, first_is_last = next(chunks)
            first_ref = weakref.ref(first)
            del first
            gc.collect()
            self.assertIsNone(first_ref())
            second, second_is_last = next(chunks)

        self.assertFalse(first_is_last)
        self.assertTrue(second_is_last)
        self.assertIsInstance(second, Payload)

        chunks = list(streaming_asr.iter_audio_chunks(_audio_bytes(4.0), 2.0))
        lengths = [len(sf.read(io.BytesIO(chunk))[0]) for chunk, _ in chunks]
        self.assertEqual(lengths, [2 * SAMPLE_RATE, 4 * SAMPLE_RATE])
        self.assertEqual([is_last for _, is_last in chunks], [False, True])


if __name__ == "__main__":
    unittest.main()
