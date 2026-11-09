"""Unit tests for lazy cumulative WAV chunking in streaming_asr.split_audio_chunks."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import io
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from sglang.srt.entrypoints.openai.streaming_asr import split_audio_chunks
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SR = 16000


def _make_wav_bytes(samples: np.ndarray, sample_rate: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


class TestStreamingASRSplitAudioChunks(CustomTestCase):
    def test_cumulative_wav_prefixes_and_indexing(self):
        t = np.arange(int(2.5 * SR), dtype=np.float32) / SR
        waveform = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        wav_bytes = _make_wav_bytes(waveform, SR)

        chunks = split_audio_chunks(wav_bytes, chunk_size_sec=1.0)
        self.assertEqual(len(chunks), 3)

        expected_lengths = [SR, 2 * SR, int(2.5 * SR)]
        for idx, expected_len in enumerate(expected_lengths):
            decoded, decoded_sr = sf.read(io.BytesIO(chunks[idx]), dtype="float32")
            self.assertEqual(decoded_sr, SR)
            self.assertEqual(len(decoded), expected_len)

        # Negative indexing and slicing behave like a standard Sequence[bytes].
        self.assertEqual(chunks[-1], chunks[2])
        self.assertEqual(chunks[1:3], [chunks[1], chunks[2]])
        self.assertEqual(list(chunks), [chunks[0], chunks[1], chunks[2]])

    def test_lazy_encoding_on_demand(self):
        waveform = np.zeros(4 * SR, dtype=np.float32)
        wav_bytes = _make_wav_bytes(waveform, SR)

        with patch.object(sf, "write", wraps=sf.write) as mock_write:
            chunks = split_audio_chunks(wav_bytes, chunk_size_sec=1.0)
            self.assertEqual(len(chunks), 4)
            # No WAV encoding happens until a chunk is accessed.
            self.assertEqual(mock_write.call_count, 0)

            # Simulating early client disconnect after chunk 0 only encodes 1 chunk.
            for i, _chunk in enumerate(chunks):
                if i == 0:
                    break
            self.assertEqual(mock_write.call_count, 1)

    def test_stereo_downmixed_and_subsample_chunk_size(self):
        stereo = np.ones((100, 2), dtype=np.float32)
        wav_bytes = _make_wav_bytes(stereo, SR)
        # chunk_size_sec smaller than 1 / SR should clamp step to 1 sample instead of ValueError.
        chunks = split_audio_chunks(wav_bytes, chunk_size_sec=1e-5)
        self.assertEqual(len(chunks), 100)
        first_decoded, _ = sf.read(io.BytesIO(chunks[0]), dtype="float32")
        self.assertEqual(first_decoded.shape, (1,))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            split_audio_chunks(b"", chunk_size_sec=1.0)
        with self.assertRaises(ValueError):
            split_audio_chunks(b"not-empty", chunk_size_sec=0.0)
        with self.assertRaises(ValueError):
            split_audio_chunks(b"not-a-valid-wav", chunk_size_sec=1.0)


if __name__ == "__main__":
    unittest.main()
