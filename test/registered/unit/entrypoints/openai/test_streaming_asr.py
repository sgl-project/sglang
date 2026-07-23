"""Unit tests for srt/entrypoints/openai/streaming_asr.py.

Covers the sliding audio window (compute_window_drop / start_new_window)
that bounds per-chunk inference cost, and delta continuity across a
window roll.
"""

import io
import unittest

import numpy as np
import soundfile as sf

from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
    compute_window_drop,
    decode_audio_mono,
    encode_wav,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_state(**overrides):
    cfg = {
        "chunk_size_sec": 2.0,
        "unfixed_chunk_num": 2,
        "unfixed_token_num": 5,
        "window_chunk_num": 8,
    }
    cfg.update(overrides)
    return StreamingASRState(**cfg)


class TestComputeWindowDrop(CustomTestCase):
    # One chunk = 100 units in these tests.
    C = 100

    def test_disabled_window_never_drops(self):
        state = make_state(window_chunk_num=0)
        self.assertEqual(
            compute_window_drop(
                buffered=100 * self.C,
                inferred=99 * self.C,
                chunk_size=self.C,
                state=state,
            ),
            0,
        )

    def test_buffer_within_window_never_drops(self):
        state = make_state()
        self.assertEqual(
            compute_window_drop(
                buffered=8 * self.C, inferred=7 * self.C, chunk_size=self.C, state=state
            ),
            0,
        )

    def test_normal_cadence_rolls_to_keep_zone(self):
        # 9 chunks buffered, 8 inferred: drop 6, keeping the unfixed zone
        # (2 chunks) plus the newest not-yet-inferred chunk.
        state = make_state()
        self.assertEqual(
            compute_window_drop(
                buffered=9 * self.C, inferred=8 * self.C, chunk_size=self.C, state=state
            ),
            6 * self.C,
        )

    def test_never_drops_uninferred_audio(self):
        # One giant append before any inference: nothing is safe to drop,
        # regardless of how far the buffer exceeds the window.
        state = make_state()
        self.assertEqual(
            compute_window_drop(
                buffered=30 * self.C, inferred=0, chunk_size=self.C, state=state
            ),
            0,
        )

    def test_partially_inferred_drop_capped_at_safe_prefix(self):
        # 20 chunks buffered but only 5 inferred: safe = 5 - 2 = 3 chunks,
        # even though the keep rule alone would allow dropping 17.
        state = make_state()
        self.assertEqual(
            compute_window_drop(
                buffered=20 * self.C,
                inferred=5 * self.C,
                chunk_size=self.C,
                state=state,
            ),
            3 * self.C,
        )


class TestWindowRollState(CustomTestCase):
    def test_start_new_window_resets_frame_keeps_prefix(self):
        state = make_state(unfixed_token_num=2)
        state.update("a b c d")
        self.assertEqual(state.emitted_text, "a b")
        state.start_new_window()
        self.assertEqual(state.confirmed_text, "")
        self.assertEqual(state.full_transcript, "")
        # The prompt prefix survives the roll.
        self.assertEqual(state.emitted_text, "a b")

    def test_delta_continuity_across_roll(self):
        """No duplicated or lost words when the audio window rolls."""
        state = make_state(unfixed_token_num=2)
        deltas = [state.update("a b c d")]
        deltas.append(state.update("a b c d e f"))
        # Window rolls: the model will now see prefix "a b c d" plus only
        # recent audio, and return the continuation.
        state.start_new_window()
        deltas.append(state.update("e f g h"))
        state.full_transcript = "e f g h i"
        deltas.append(state.finalize())
        self.assertEqual(deltas, ["a b", "c d", "e f", "g h i"])
        self.assertEqual(state.emitted_text, "a b c d e f g h i")


class TestAudioHelpers(CustomTestCase):
    def test_wav_round_trip(self):
        sr = 16000
        tone = np.sin(np.linspace(0, 2 * np.pi * 440, sr)).astype(np.float32) * 0.5
        wav = encode_wav(tone, sr)
        decoded, decoded_sr = decode_audio_mono(wav)
        self.assertEqual(decoded_sr, sr)
        self.assertEqual(len(decoded), len(tone))
        np.testing.assert_allclose(decoded, tone, atol=1e-4)

    def test_decode_stereo_downmixes_to_mono(self):
        sr = 16000
        stereo = np.zeros((sr, 2), dtype=np.float32)
        stereo[:, 0] = 0.5
        buf = io.BytesIO()
        sf.write(buf, stereo, sr, format="WAV")
        decoded, _ = decode_audio_mono(buf.getvalue())
        self.assertEqual(decoded.ndim, 1)
        np.testing.assert_allclose(decoded, np.full(sr, 0.25), atol=1e-4)

    def test_decode_empty_raises(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            decode_audio_mono(b"")


if __name__ == "__main__":
    unittest.main()
