"""Unit tests for entrypoints/openai/streaming_asr.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=6, suite="base-b-test-cpu")

import io
import unittest
import wave

import soundfile as sf

from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
    _is_cjk,
    needs_space,
    normalize_whitespace,
    split_audio_chunks,
)
from sglang.test.test_utils import CustomTestCase


def _make_wav_bytes(duration_sec=1.0, sample_rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * int(duration_sec * sample_rate))
    return buf.getvalue()


class TestStreamingASRState(CustomTestCase):
    def _state(self, unfixed_chunk_num=2, unfixed_token_num=2):
        return StreamingASRState(
            chunk_size_sec=1.0,
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
        )

    def test_get_prefix_empty_before_threshold(self):
        state = self._state(unfixed_chunk_num=3)
        state.emitted_text = "hello"
        self.assertEqual(state.get_prefix_text(), "")
        state.chunk_index = 2
        self.assertEqual(state.get_prefix_text(), "")

    def test_get_prefix_returns_emitted_after_threshold(self):
        state = self._state(unfixed_chunk_num=2)
        state.emitted_text = "hello world"
        state.chunk_index = 2
        self.assertEqual(state.get_prefix_text(), "hello world")

    def test_get_prefix_returns_empty_when_no_emitted(self):
        state = self._state(unfixed_chunk_num=2)
        state.chunk_index = 5
        state.emitted_text = ""
        self.assertEqual(state.get_prefix_text(), "")

    def test_record_emit_first_delta(self):
        state = self._state()
        result = state._record_emit("hello")
        self.assertEqual(result, "hello")
        self.assertEqual(state.emitted_text, "hello")

    def test_record_emit_appends_with_space(self):
        state = self._state()
        state._record_emit("hello")
        state._record_emit("world")
        self.assertEqual(state.emitted_text, "hello world")

    def test_record_emit_skips_empty(self):
        state = self._state()
        result = state._record_emit("")
        self.assertEqual(result, "")
        self.assertEqual(state.emitted_text, "")

    def test_update_normal_growth(self):
        state = self._state(unfixed_token_num=2)
        delta = state.update("hello world from asr today")
        self.assertEqual(state.confirmed_text, "hello world from")
        self.assertEqual(delta, "hello world from")

    def test_update_second_chunk_emits_only_new_words(self):
        state = self._state(unfixed_token_num=2)
        state.update("hello world from asr today")
        delta = state.update("hello world from asr right now")
        self.assertEqual(delta, "asr")

    def test_update_short_transcript_confirmed_becomes_empty(self):
        state = self._state(unfixed_token_num=3)
        state.update("hello world")
        self.assertEqual(state.confirmed_text, "")

    def test_update_model_revises_earlier_text(self):
        state = self._state(unfixed_token_num=2)
        state.update("hello world foo bar baz")
        delta = state.update("hello earth foo bar baz")
        self.assertEqual(state.confirmed_text, "hello earth foo")
        self.assertNotIn("world", delta)

    def test_update_increments_chunk_index(self):
        state = self._state(unfixed_token_num=2)
        self.assertEqual(state.chunk_index, 0)
        state.update("one two three four")
        self.assertEqual(state.chunk_index, 1)
        state.update("one two three five")
        self.assertEqual(state.chunk_index, 2)

    def test_finalize_exact_match(self):
        state = self._state(unfixed_token_num=2)
        state.update("hello world from asr now")
        state.full_transcript = "hello world from asr now"
        delta = state.finalize()
        self.assertEqual(delta, "asr now")

    def test_finalize_with_revision(self):
        state = self._state(unfixed_token_num=2)
        state.update("hello world from asr now")
        state.full_transcript = "hello earth from asr now"
        delta = state.finalize()
        self.assertNotIn("world", delta)

    def test_finalize_no_common_prefix(self):
        state = self._state(unfixed_token_num=2)
        state.update("hello world foo bar")
        state.full_transcript = "completely different text here now"
        delta = state.finalize()
        self.assertEqual(delta, "completely different text here now")


class TestSplitAudioChunks(CustomTestCase):
    def test_empty_audio_raises(self):
        with self.assertRaises(ValueError):
            split_audio_chunks(b"", 1.0)

    def test_nonpositive_chunk_size_raises(self):
        wav = _make_wav_bytes(duration_sec=1.0)
        with self.assertRaises(ValueError):
            split_audio_chunks(wav, 0)
        with self.assertRaises(ValueError):
            split_audio_chunks(wav, -1)

    def test_single_chunk_when_chunk_size_equals_duration(self):
        wav = _make_wav_bytes(duration_sec=1.0, sample_rate=8000)
        chunks = split_audio_chunks(wav, 1.0)
        self.assertEqual(len(chunks), 1)
        self.assertIsInstance(chunks[0], bytes)

    def test_multiple_chunks_when_duration_exceeds_chunk_size(self):
        wav = _make_wav_bytes(duration_sec=1.0, sample_rate=8000)
        chunks = split_audio_chunks(wav, 0.3)
        self.assertEqual(len(chunks), 4)

    def test_chunks_are_growing(self):
        wav = _make_wav_bytes(duration_sec=1.0, sample_rate=8000)
        chunks = split_audio_chunks(wav, 0.4)
        self.assertEqual(len(chunks), 3)
        prev_size = 0
        for c in chunks:
            self.assertGreater(len(c), prev_size)
            prev_size = len(c)

    def test_chunks_are_valid_wav(self):
        wav = _make_wav_bytes(duration_sec=1.0, sample_rate=8000)
        chunks = split_audio_chunks(wav, 0.4)
        for c in chunks:
            data, _sr = sf.read(io.BytesIO(c), dtype="float32")
            self.assertGreater(len(data), 0)

    def test_invalid_audio_raises(self):
        with self.assertRaises(ValueError):
            split_audio_chunks(b"not audio data", 1.0)


class TestNormalizeWhitespace(CustomTestCase):
    def test_no_punctuation_no_change(self):
        self.assertEqual(normalize_whitespace("hello world"), "hello world")

    def test_removes_space_before_comma(self):
        self.assertEqual(normalize_whitespace("hello , world"), "hello, world")

    def test_removes_space_before_period(self):
        self.assertEqual(normalize_whitespace("end ."), "end.")

    def test_removes_space_before_question_mark(self):
        self.assertEqual(normalize_whitespace("what ?"), "what?")

    def test_removes_space_before_cjk_punctuation(self):
        self.assertEqual(normalize_whitespace("你好 ， 世界"), "你好， 世界")
        self.assertEqual(normalize_whitespace("真的 ？"), "真的？")

    def test_multiple_punctuation_in_one_string(self):
        self.assertEqual(
            normalize_whitespace("x , y . z !"),
            "x, y. z!",
        )


class TestIsCJK(CustomTestCase):
    def test_cjk_ideograph(self):
        self.assertTrue(_is_cjk("世"))

    def test_hiragana(self):
        self.assertTrue(_is_cjk("あ"))

    def test_katakana(self):
        self.assertTrue(_is_cjk("ア"))

    def test_cjk_punctuation(self):
        self.assertTrue(_is_cjk("、"))
        self.assertTrue(_is_cjk("。"))

    def test_fullwidth_ascii(self):
        self.assertTrue(_is_cjk("Ａ"))

    def test_latin_letters_are_not_cjk(self):
        self.assertFalse(_is_cjk("a"))
        self.assertFalse(_is_cjk("Z"))

    def test_ascii_digit_is_not_cjk(self):
        self.assertFalse(_is_cjk("5"))

    def test_ascii_space_is_not_cjk(self):
        self.assertFalse(_is_cjk(" "))

    def test_korean_hangul_is_not_cjk(self):
        self.assertFalse(_is_cjk("가"))

    def test_devanagari_is_not_cjk(self):
        self.assertFalse(_is_cjk("अ"))


class TestNeedsSpace(CustomTestCase):
    def test_empty_strings_return_false(self):
        self.assertFalse(needs_space("", ""))
        self.assertFalse(needs_space("hello", ""))
        self.assertFalse(needs_space("", "world"))

    def test_already_has_space_returns_false(self):
        self.assertFalse(needs_space("hello ", "world"))
        self.assertFalse(needs_space("hello", " world"))

    def test_all_no_space_before_chars(self):
        from sglang.srt.entrypoints.openai.streaming_asr import _NO_SPACE_BEFORE
        for ch in _NO_SPACE_BEFORE:
            self.assertFalse(needs_space("hello", ch), f"char={ch!r}")

    def test_all_no_space_after_chars(self):
        from sglang.srt.entrypoints.openai.streaming_asr import _NO_SPACE_AFTER
        for ch in _NO_SPACE_AFTER:
            self.assertFalse(needs_space(ch, "world"), f"char={ch!r}")

    def test_adjacent_cjk_no_space(self):
        self.assertFalse(needs_space("你好", "世界"))

    def test_cjk_to_latin_needs_space(self):
        self.assertTrue(needs_space("你好", "hello"))

    def test_latin_to_cjk_needs_space(self):
        self.assertTrue(needs_space("hello", "世界"))

    def test_normal_words_need_space(self):
        self.assertTrue(needs_space("hello", "world"))


if __name__ == "__main__":
    unittest.main()
