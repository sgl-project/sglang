"""Unit tests for the Whisper transcription adapter.

Focused on ``WhisperAdapter.parse_fused_output`` — a pure static method that
parses the fused auto-detect output (``<|lang|><|transcribe|><|notimestamps|>
text``) and must fail strictly rather than silently defaulting to English on
malformed input.
"""

import unittest

from sglang.srt.entrypoints.openai.transcription_adapters.whisper import WhisperAdapter
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestWhisperParseFusedOutput(unittest.TestCase):
    """parse_fused_output: (language_code | None, transcription)."""

    def test_happy_english(self):
        lang, text = WhisperAdapter.parse_fused_output(
            "<|en|><|transcribe|><|notimestamps|> Hello world"
        )
        self.assertEqual(lang, "en")
        self.assertEqual(text, "Hello world")

    def test_happy_non_english(self):
        lang, text = WhisperAdapter.parse_fused_output(
            "<|zh|><|transcribe|><|notimestamps|>你好世界"
        )
        self.assertEqual(lang, "zh")
        self.assertEqual(text, "你好世界")

    def test_missing_language_prefix_returns_none(self):
        lang, text = WhisperAdapter.parse_fused_output("raw untagged output")
        self.assertIsNone(lang)
        self.assertEqual(text, "raw untagged output")

    def test_missing_sentinel_returns_none(self):
        # Repro from the PR review: truncation/FSM abort leaves the language
        # tag but no <|notimestamps|>. Must not leak the special-token prefix.
        lang, text = WhisperAdapter.parse_fused_output("<|zh|> Hi")
        self.assertIsNone(lang)

    def test_truncated_after_transcribe_returns_none(self):
        lang, _ = WhisperAdapter.parse_fused_output("<|en|><|transcribe|>")
        self.assertIsNone(lang)

    def test_unsupported_language_code_returns_none(self):
        # The FSM regex only allows codes from ISO639_1_SUPPORTED_LANGS.
        # If a code outside the allow-list ever reaches this parser (e.g.
        # FSM bypassed in a future refactor), it must be rejected, not
        # silently accepted as a language.
        lang, _ = WhisperAdapter.parse_fused_output(
            "<|xx|><|transcribe|><|notimestamps|>hi"
        )
        self.assertIsNone(lang)

    def test_empty_transcription_preserved(self):
        lang, text = WhisperAdapter.parse_fused_output(
            "<|en|><|transcribe|><|notimestamps|>"
        )
        self.assertEqual(lang, "en")
        self.assertEqual(text, "")


class TestWhisperFusedPrefixEnd(unittest.TestCase):
    """fused_prefix_end: streaming boundary offset (-1 when sentinel absent)."""

    def test_returns_offset_after_sentinel_and_leading_whitespace(self):
        text = "<|en|><|transcribe|><|notimestamps|> Hello"
        end = WhisperAdapter.fused_prefix_end(text)
        self.assertEqual(text[end:], "Hello")

    def test_returns_offset_when_no_leading_whitespace(self):
        text = "<|zh|><|transcribe|><|notimestamps|>你好"
        end = WhisperAdapter.fused_prefix_end(text)
        self.assertEqual(text[end:], "你好")

    def test_returns_minus_one_when_sentinel_missing(self):
        # Partial prefix still arriving on the wire.
        self.assertEqual(WhisperAdapter.fused_prefix_end("<|en|><|transcribe|>"), -1)
        self.assertEqual(WhisperAdapter.fused_prefix_end(""), -1)
        self.assertEqual(WhisperAdapter.fused_prefix_end("<|zh|> Hi"), -1)

    def test_defers_when_only_whitespace_follows_sentinel(self):
        # Sentinel arrived at chunk boundary but transcription text hasn't —
        # anchor now would leak the leading space on the next delta.
        self.assertEqual(
            WhisperAdapter.fused_prefix_end("<|en|><|transcribe|><|notimestamps|>"),
            -1,
        )
        self.assertEqual(
            WhisperAdapter.fused_prefix_end("<|en|><|transcribe|><|notimestamps|>  "),
            -1,
        )


if __name__ == "__main__":
    unittest.main()
