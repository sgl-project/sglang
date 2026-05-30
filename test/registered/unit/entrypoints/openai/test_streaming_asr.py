"""Unit tests for realtime ASR slicing-path helpers.

Edge cases for ``dedupe_overlap`` (normalization rules, CJK fallback, the
suffix-only-history invariant the perf optimization depends on), the
bit-equality invariant for ``_pcm_to_float_samples``, and ``_slice_pcm_from``
validation. Trivial happy-path assertions that restated Python primitives were
dropped. The slicing trigger logic and its interaction with
``StreamingASRState`` are exercised by the manual GPU suite, not by CI.
"""

import io
import unittest

import numpy as np
import soundfile as sf

from sglang.srt.entrypoints.openai.realtime.session import (
    _pcm_to_float_samples,
    _slice_pcm_from,
)
from sglang.srt.entrypoints.openai.streaming_asr import dedupe_overlap
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDedupeOverlap(CustomTestCase):
    """Edge cases for the dedupe heuristic.

    Drops trivial happy-path assertions; keeps cases that lock
    normalization rules, CJK fallback paths, and the suffix-only-history
    invariant that the perf optimization relies on.
    """

    def test_full_candidate_overlaps_returns_empty(self):
        # Whole-candidate match must emit empty so StreamingASRState doesn't
        # double-record the previous chunk's content.
        self.assertEqual(dedupe_overlap("hello world", "hello world"), "")

    def test_empty_committed_returns_candidate_unchanged(self):
        self.assertEqual(dedupe_overlap("", "anything goes"), "anything goes")

    def test_empty_candidate_returns_empty(self):
        self.assertEqual(dedupe_overlap("anything", ""), "")

    def test_em_dash_normalized_during_match(self):
        # Trailing em dash and case differences are stripped during match.
        # Regression test for the dedupe rule documented in _DEDUPE_NORM_STRIP.
        self.assertEqual(
            dedupe_overlap("stew for dinner—", "Dinner: turnips"), "turnips"
        )

    def test_cjk_char_level_fallback(self):
        # No whitespace → word-level returns unchanged → CJK fallback engages.
        self.assertEqual(dedupe_overlap("你好世界", "世界今天很好"), "今天很好")

    def test_cjk_overlap_with_punctuation(self):
        # CJK punctuation in committed_text must not block the char-level
        # match on the ideographs that follow.
        self.assertEqual(dedupe_overlap("你好,世界", "世界今天很好"), "今天很好")

    def test_long_committed_history_uses_suffix_overlap(self):
        # Locks the suffix-only invariant the tail-only optimization
        # depends on: a massive committed prefix unrelated to the candidate
        # must not change the match outcome.
        committed = " ".join(["old"] * 1000 + ["a", "b", "c"])
        self.assertEqual(dedupe_overlap(committed, "b c d"), "d")


class TestPcmToFloatSamples(CustomTestCase):
    """The bit-equality invariant the PR's perf claim depends on, plus the
    one corruption boundary worth catching loudly."""

    def test_matches_soundfile_round_trip(self):
        # The PCM→WAV→sf.read path was the legacy converter; this PR's
        # direct conversion must remain bit-equal to it.
        rng = np.random.default_rng(42)
        ints = rng.integers(-32768, 32768, size=4096, dtype=np.int16)
        pcm = ints.tobytes()

        direct = _pcm_to_float_samples(pcm)

        buf = io.BytesIO()
        sf.write(buf, ints, 16000, format="WAV")
        buf.seek(0)
        round_trip, _ = sf.read(buf)

        np.testing.assert_array_equal(direct, round_trip)

    def test_odd_length_pcm_raises(self):
        # int16 frames are 2 bytes; an odd-length buffer means upstream
        # corruption. Keep the np.frombuffer ValueError loud — silent
        # rounding would mask the bug.
        with self.assertRaises(ValueError):
            _pcm_to_float_samples(b"\x00")


class TestSlicePcmFrom(CustomTestCase):
    """Only the validation behavior — the trivial slice cases were
    Python-built-in tests, not ours."""

    def test_negative_start_raises(self):
        with self.assertRaises(ValueError):
            _slice_pcm_from(b"abcdef", -1)

    def test_past_end_raises(self):
        with self.assertRaises(ValueError):
            _slice_pcm_from(b"abcdef", 7)


if __name__ == "__main__":
    unittest.main()
