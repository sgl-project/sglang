"""Unit tests for energy-aware audio chunking (long-audio transcription).

Whisper's encoder window is 30 s; longer audio must be split before
prompting. The splitter must cut at low-energy points (pauses) inside the
tail search window of each stride — never blindly at the stride boundary,
which could land mid-word — and the chunks must be contiguous,
non-overlapping, and reproduce the full waveform when concatenated.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import io
import unittest

import numpy as np
import soundfile as sf

from sglang.srt.entrypoints.openai.audio_chunking import (
    find_split_point,
    split_audio_energy_aware,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SR = 16000


def _tone_with_silences(duration_s: float, silences: list, sr: int = SR) -> np.ndarray:
    """A 440 Hz tone with zeroed-out gaps at the given (start_s, end_s) spans."""
    t = np.arange(int(duration_s * sr)) / sr
    wav = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    for start_s, end_s in silences:
        wav[int(start_s * sr) : int(end_s * sr)] = 0.0
    return wav


def _wav_bytes(wav: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return buf.getvalue()


def _decode(chunk: bytes) -> np.ndarray:
    data, sr = sf.read(io.BytesIO(chunk), dtype="float32")
    assert sr == SR
    return data


class TestFindSplitPoint(CustomTestCase):
    def test_picks_quietest_window(self):
        # Loud tone with one silent 200 ms gap; the split point must land
        # inside the gap.
        wav = _tone_with_silences(4.0, [(2.0, 2.2)])
        idx = find_split_point(wav, int(1.0 * SR), int(3.0 * SR))
        self.assertGreaterEqual(idx, int(2.0 * SR))
        self.assertLess(idx, int(2.2 * SR))

    def test_uniform_energy_returns_search_start(self):
        wav = _tone_with_silences(4.0, [])
        idx = find_split_point(wav, int(1.0 * SR), int(3.0 * SR))
        # All windows are equally loud; the first one wins.
        self.assertEqual(idx, int(1.0 * SR))


class TestSplitAudioEnergyAware(CustomTestCase):
    def test_short_audio_single_chunk(self):
        wav = _tone_with_silences(10.0, [])
        chunks, offsets = split_audio_energy_aware(_wav_bytes(wav), max_clip_s=30.0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(offsets, [0.0])
        self.assertEqual(len(_decode(chunks[0])), len(wav))

    def test_long_audio_splits_at_silence(self):
        # 70 s tone with silence gaps planted inside each stride's search
        # window (the last 1 s before the 30 s boundary): the first cut must
        # land in [29.2, 29.5], which puts the second search window at
        # ~[58.2, 59.2] where the second gap [58.5, 58.8] lives.
        gaps = [(29.2, 29.5), (58.5, 58.8)]
        wav = _tone_with_silences(70.0, gaps)
        chunks, offsets = split_audio_energy_aware(_wav_bytes(wav), max_clip_s=30.0)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(offsets[0], 0.0)
        # Each cut lands inside its silence gap, not at the blind 30 s mark.
        for offset, (gap_start, gap_end) in zip(offsets[1:], gaps):
            self.assertGreaterEqual(offset, gap_start)
            self.assertLess(offset, gap_end)

        decoded = [_decode(c) for c in chunks]
        # No chunk exceeds the model's window.
        for d in decoded:
            self.assertLessEqual(len(d), 30 * SR)
        # Chunks are contiguous and non-overlapping: offsets line up with
        # cumulative chunk lengths and the total sample count is preserved.
        cumulative = 0
        for d, offset in zip(decoded, offsets):
            self.assertEqual(cumulative, int(round(offset * SR)))
            cumulative += len(d)
        self.assertEqual(cumulative, len(wav))
        # Concatenation reproduces the waveform (modulo PCM16 quantization).
        stitched = np.concatenate(decoded)
        np.testing.assert_allclose(stitched, wav, atol=2.0 / 32768)

    def test_no_silence_still_makes_progress(self):
        # A constant-energy tone has no preferred split point; the splitter
        # must still terminate with bounded chunks covering all samples.
        wav = _tone_with_silences(65.0, [])
        chunks, offsets = split_audio_energy_aware(_wav_bytes(wav), max_clip_s=30.0)
        self.assertGreaterEqual(len(chunks), 3)
        decoded = [_decode(c) for c in chunks]
        for d in decoded:
            self.assertLessEqual(len(d), 30 * SR)
        self.assertEqual(sum(len(d) for d in decoded), len(wav))
        self.assertEqual(len(offsets), len(chunks))


if __name__ == "__main__":
    unittest.main()
