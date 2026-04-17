"""Unit tests for the Whisper transcription adapter.

Focused on ``WhisperAdapter.parse_fused_output`` — a pure static method that
parses the fused auto-detect output (``<|lang|><|transcribe|><|notimestamps|>
text``) and must fail strictly rather than silently defaulting to English on
malformed input.
"""

import unittest

from sglang.srt.entrypoints.openai.protocol import TranscriptionRequest
from sglang.srt.entrypoints.openai.transcription_adapters.whisper import (
    WHISPER_AUTODETECT_REGEX,
    WHISPER_AUTODETECT_TS_REGEX,
    WhisperAdapter,
)
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

    def test_trailing_endoftext_stripped(self):
        # skip_special_tokens=False means <|endoftext|> can survive into the
        # raw text; make sure it doesn't leak into the user-visible result.
        lang, text = WhisperAdapter.parse_fused_output(
            "<|en|><|transcribe|><|notimestamps|> Hello world<|endoftext|>"
        )
        self.assertEqual(lang, "en")
        self.assertEqual(text, "Hello world")

    def test_embedded_timestamp_tokens_stripped_from_text(self):
        # Timestamps variant: segment-boundary tokens must not appear in the
        # plain-text field (verbose_json builds segments from output_ids via
        # _parse_segments, a separate path).
        lang, text = WhisperAdapter.parse_fused_output(
            "<|en|><|transcribe|><|0.00|> Hello<|5.00|> world<|10.00|><|endoftext|>"
        )
        self.assertEqual(lang, "en")
        self.assertEqual(text, "Hello world")


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


class TestWhisperFusedTimestampsVariant(unittest.TestCase):
    """Timestamps-aware fused regex: <|lang|><|transcribe|><|0.00|> text <|X.XX|>..."""

    def test_parse_fused_output_with_ts_sentinel(self):
        # The first <|0.00|> is the forced prefix sentinel. Subsequent
        # <|5.00|>, <|10.00|> etc are segment-boundary tokens — meaningful
        # for verbose_json's _parse_segments (which decodes output_ids
        # separately) but unwanted noise in the plain text field, so they
        # get scrubbed here along with any trailing <|endoftext|>.
        text = "<|en|><|transcribe|><|0.00|> Hello<|5.00|> World<|10.00|>"
        lang, out = WhisperAdapter.parse_fused_output(text)
        self.assertEqual(lang, "en")
        self.assertEqual(out, "Hello World")

    def test_fused_prefix_end_with_ts_sentinel(self):
        text = "<|zh|><|transcribe|><|0.00|> 你好<|5.00|>"
        end = WhisperAdapter.fused_prefix_end(text)
        self.assertEqual(text[end:], "你好<|5.00|>")

    def test_strip_special_tokens_removes_timestamps_and_eos(self):
        self.assertEqual(
            WhisperAdapter.strip_special_tokens("Hello<|5.00|> world<|endoftext|>"),
            "Hello world",
        )
        self.assertEqual(
            WhisperAdapter.strip_special_tokens("plain text"), "plain text"
        )
        self.assertEqual(WhisperAdapter.strip_special_tokens(""), "")

    def test_fused_prefix_end_defers_on_partial_ts_prefix(self):
        self.assertEqual(WhisperAdapter.fused_prefix_end("<|en|><|transcribe|>"), -1)
        self.assertEqual(
            WhisperAdapter.fused_prefix_end("<|en|><|transcribe|><|0.00|>"), -1
        )
        self.assertEqual(
            WhisperAdapter.fused_prefix_end("<|en|><|transcribe|><|0.00|>  "), -1
        )


class TestWhisperBuildFusedAutodetectParams(unittest.TestCase):
    """build_fused_autodetect_params picks the right regex + propagates ts param."""

    def _request(self, **kwargs) -> TranscriptionRequest:
        base = dict(model="whisper", temperature=0.0)
        base.update(kwargs)
        return TranscriptionRequest(**base)

    def test_no_timestamps_uses_notimestamps_regex(self):
        params = WhisperAdapter().build_fused_autodetect_params(self._request())
        self.assertEqual(params["regex"], WHISPER_AUTODETECT_REGEX)
        self.assertNotIn("timestamp_granularities", params)

    def test_timestamps_uses_ts_regex_and_propagates_granularities(self):
        req = self._request(timestamp_granularities=["segment"])
        params = WhisperAdapter().build_fused_autodetect_params(req)
        self.assertEqual(params["regex"], WHISPER_AUTODETECT_TS_REGEX)
        self.assertEqual(params["timestamp_granularities"], ["segment"])

    def test_empty_timestamps_list_uses_notimestamps_regex(self):
        # Empty list is falsy — treat as "no timestamps requested".
        req = self._request(timestamp_granularities=[])
        params = WhisperAdapter().build_fused_autodetect_params(req)
        self.assertEqual(params["regex"], WHISPER_AUTODETECT_REGEX)
        self.assertNotIn("timestamp_granularities", params)


if __name__ == "__main__":
    unittest.main()
