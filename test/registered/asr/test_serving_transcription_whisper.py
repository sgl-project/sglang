"""
Test the OpenAI-compatible /v1/audio/transcriptions endpoint with Whisper.

Covers non-streaming and normal (token-level) streaming transcription,
plus Whisper-specific fused auto-detect (language=None) regression tests.

Usage:
    python3 test_serving_transcription_whisper.py -v
"""

import unittest

from sglang.test.asr_utils import ASRTestBase, AudioTestCase
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-test-1-gpu-small")


class TestServingTranscriptionWhisper(ASRTestBase):
    """Test Whisper transcription via /v1/audio/transcriptions endpoint."""

    model = "openai/whisper-large-v3"
    served_model_name = "whisper"
    streaming_exact_match = True
    audio_cases = [
        AudioTestCase(
            url="https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/Trump_WEF_2018_10s.mp3",
            keywords=["privilege", "leader", "science", "art"],
            min_keyword_matches=2,
            filename="audio.mp3",
            language="en",
        ),
    ]

    # ---- Whisper fused auto-detect (language=None) regression tests ----

    def test_auto_detect_language_verbose_json(self):
        """language omitted + verbose_json returns detected language + clean text."""
        case = self.audio_cases[0]
        result = self._transcribe_json(
            case, language=None, response_format="verbose_json"
        )
        self.assertEqual(result.get("language"), "en")
        text = result.get("text", "")
        self.assertTrue(len(text) > 0, "Transcription should not be empty")
        self.assertNotIn("<|", text, f"Special token leaked into text: {text!r}")
        self._assert_keywords(text, case, "[auto-detect] ")

    def test_auto_detect_matches_explicit_english(self):
        """Auto-detected (language=None) text should match explicit language=en."""
        case = self.audio_cases[0]
        auto = self._transcribe_json(case, language=None).get("text", "")
        explicit = self._transcribe_json(case, language="en").get("text", "")
        self.assertEqual(
            auto.strip(),
            explicit.strip(),
            "Auto-detect should produce the same transcription as language=en "
            "on an English clip.",
        )
        self.assertNotIn("<|", auto)

    def test_auto_detect_with_segment_timestamps(self):
        """language=None + timestamp_granularities uses the timestamps fused regex."""
        case = self.audio_cases[0]
        result = self._transcribe_json(
            case,
            language=None,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
        self.assertEqual(result.get("language"), "en")
        segments = result.get("segments") or []
        self.assertGreater(len(segments), 0, "Expected at least one segment")
        for seg in segments:
            self.assertIn("start", seg)
            self.assertIn("end", seg)
            self.assertIn("text", seg)
            self.assertGreaterEqual(seg["end"], seg["start"])
            self.assertNotIn(
                "<|", seg["text"], f"Special token leaked into segment: {seg!r}"
            )

    def test_auto_detect_streaming(self):
        """language=None + stream=True: deltas scrubbed, concat matches non-streaming."""
        case = self.audio_cases[0]
        events, _ = self._transcribe_stream(case, language=None)
        deltas = self._iter_deltas(events)
        self.assertTrue(len(deltas) > 0, "Expected at least one streamed delta")
        for d in deltas:
            self.assertNotIn(
                "<|", d, f"Special token leaked into streaming delta: {d!r}"
            )
        streamed = "".join(deltas).strip()
        reference = self._transcribe_json(case, language=None).get("text", "").strip()
        self.assertEqual(
            streamed,
            reference,
            "Streamed auto-detect text should match the non-streaming result.",
        )


if __name__ == "__main__":
    unittest.main()
