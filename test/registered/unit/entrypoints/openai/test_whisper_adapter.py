"""Unit tests for the Whisper transcription adapter.

Focused on ``WhisperAdapter.parse_fused_output`` — a pure static method
that parses the fused auto-detect output into ``(language, user_visible_text)``.
``visible=None`` means "forced prefix not yet locatable; streaming callers
should keep buffering, non-streaming callers should fall back to a
best-effort scrub".
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
    """parse_fused_output: (language, visible) where visible=None means defer."""

    def test_happy_english(self):
        lang, visible = WhisperAdapter.parse_fused_output(
            "<|en|><|transcribe|><|notimestamps|> Hello world"
        )
        self.assertEqual((lang, visible), ("en", "Hello world"))

    def test_happy_non_english(self):
        lang, visible = WhisperAdapter.parse_fused_output(
            "<|zh|><|transcribe|><|notimestamps|>你好世界"
        )
        self.assertEqual((lang, visible), ("zh", "你好世界"))

    def test_missing_language_prefix_defers(self):
        # Partial prefix or raw untagged text — streaming callers should
        # keep buffering; non-streaming callers fall back to best-effort.
        self.assertEqual(
            WhisperAdapter.parse_fused_output("raw untagged output"), (None, None)
        )
        self.assertEqual(WhisperAdapter.parse_fused_output(""), (None, None))

    def test_missing_sentinel_defers(self):
        # Reviewer's repro: <|zh|> Hi — language tag in but no sentinel.
        self.assertEqual(WhisperAdapter.parse_fused_output("<|zh|> Hi"), (None, None))

    def test_truncated_after_transcribe_defers(self):
        self.assertEqual(
            WhisperAdapter.parse_fused_output("<|en|><|transcribe|>"), (None, None)
        )

    def test_unsupported_language_code_defers(self):
        # FSM regex only allows ISO639_1_SUPPORTED_LANGS. A bypassed-FSM
        # <|xx|> must not leak through as a valid detection.
        self.assertEqual(
            WhisperAdapter.parse_fused_output("<|xx|><|transcribe|><|notimestamps|>hi"),
            (None, None),
        )

    def test_sentinel_in_but_whitespace_only_returns_empty_visible(self):
        # Prefix arrived at a chunk boundary before the first word. The
        # .strip() collapses to "" so streaming callers see no delta yet;
        # the language is still reported as soon as the sentinel lands.
        self.assertEqual(
            WhisperAdapter.parse_fused_output("<|en|><|transcribe|><|notimestamps|>"),
            ("en", ""),
        )
        self.assertEqual(
            WhisperAdapter.parse_fused_output("<|en|><|transcribe|><|notimestamps|>  "),
            ("en", ""),
        )

    def test_trailing_endoftext_scrubbed(self):
        lang, visible = WhisperAdapter.parse_fused_output(
            "<|en|><|transcribe|><|notimestamps|> Hello world<|endoftext|>"
        )
        self.assertEqual((lang, visible), ("en", "Hello world"))

    def test_embedded_timestamp_tokens_scrubbed(self):
        # Timestamps variant: segment-boundary tokens must not appear in
        # the plain-text field. verbose_json segment timing comes from
        # _parse_segments over output_ids on a separate path.
        lang, visible = WhisperAdapter.parse_fused_output(
            "<|en|><|transcribe|><|0.00|> Hello<|5.00|> world<|10.00|><|endoftext|>"
        )
        self.assertEqual((lang, visible), ("en", "Hello world"))

    def test_visible_grows_monotonically_across_snapshots(self):
        # Streaming property: cumulative text produces cumulative visible.
        snapshots = [
            "<|en|><|transcribe|>",
            "<|en|><|transcribe|><|notimestamps|>",
            "<|en|><|transcribe|><|notimestamps|> Hello",
            "<|en|><|transcribe|><|notimestamps|> Hello world",
            "<|en|><|transcribe|><|notimestamps|> Hello world<|endoftext|>",
        ]
        visibles = [WhisperAdapter.parse_fused_output(s)[1] for s in snapshots]
        # (None, "", "Hello", "Hello world", "Hello world")
        self.assertEqual(visibles, [None, "", "Hello", "Hello world", "Hello world"])
        # Every non-None entry is a prefix of the next non-None entry.
        real = [v for v in visibles if v is not None]
        for a, b in zip(real, real[1:]):
            self.assertTrue(b.startswith(a), f"monotonicity broken: {a!r} -> {b!r}")


class TestWhisperStripSpecialTokens(unittest.TestCase):
    """Fallback scrub used when parse_fused_output defers."""

    def test_strips_all_whisper_specials(self):
        self.assertEqual(
            WhisperAdapter.strip_special_tokens(
                "<|en|><|transcribe|><|0.00|>hi<|5.00|>world<|endoftext|>"
            ),
            "hiworld",
        )

    def test_identity_on_plain_text(self):
        self.assertEqual(
            WhisperAdapter.strip_special_tokens("plain text"), "plain text"
        )
        self.assertEqual(WhisperAdapter.strip_special_tokens(""), "")


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
