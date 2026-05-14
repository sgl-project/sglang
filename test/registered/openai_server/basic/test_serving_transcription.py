"""
Test the OpenAI-compatible /v1/audio/transcriptions endpoint with Whisper.

Usage:
    python3 test_serving_transcription.py -v
"""

import io
import json
import unittest
from typing import List, Optional

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=60, stage="stage-b", runner_config="1-gpu-small")

WHISPER_MODEL = "openai/whisper-large-v3"
AUDIO_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/Trump_WEF_2018_10s.mp3"


def download_audio_bytes(url=AUDIO_URL):
    """Download audio file and return raw bytes."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


class TestServingTranscription(CustomTestCase):
    """Test Whisper transcription via /v1/audio/transcriptions endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.model = WHISPER_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--served-model-name",
                "whisper",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _transcribe(
        self,
        language: Optional[str] = "en",
        response_format: Optional[str] = None,
        timestamp_granularities: Optional[List[str]] = None,
    ):
        """Send a non-streaming transcription request and return the JSON response.

        Passing ``language=None`` omits the field entirely, which exercises
        the fused auto-detect path.
        """
        audio_bytes = download_audio_bytes()
        data = {"model": "whisper"}
        if language is not None:
            data["language"] = language
        if response_format is not None:
            data["response_format"] = response_format
        if timestamp_granularities is not None:
            # Form-encoded list fields repeat the key
            data["timestamp_granularities[]"] = timestamp_granularities
        response = requests.post(
            self.base_url + "/v1/audio/transcriptions",
            files={"file": ("audio.mp3", io.BytesIO(audio_bytes), "audio/mpeg")},
            data=data,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _transcribe_stream(self, language: Optional[str] = None) -> List[str]:
        """Send a streaming transcription request and return the delta strings."""
        audio_bytes = download_audio_bytes()
        data = {"model": "whisper", "stream": "true"}
        if language is not None:
            data["language"] = language
        with requests.post(
            self.base_url + "/v1/audio/transcriptions",
            files={"file": ("audio.mp3", io.BytesIO(audio_bytes), "audio/mpeg")},
            data=data,
            stream=True,
            timeout=120,
        ) as response:
            self.assertEqual(response.status_code, 200, response.text)
            deltas: List[str] = []
            for raw in response.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                for choice in obj.get("choices", []):
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        deltas.append(content)
            return deltas

    def test_basic_transcription(self):
        """Test that transcription returns a valid non-empty response."""
        result = self._transcribe()
        self.assertIn("text", result)
        self.assertTrue(len(result["text"]) > 0, "Transcription should not be empty")

    def test_transcription_content_quality(self):
        """Test that transcription captures key content from the audio."""
        result = self._transcribe()
        text = result["text"].lower()
        keywords = ["privilege", "leader", "science", "art"]
        matches = [kw for kw in keywords if kw in text]
        self.assertGreaterEqual(
            len(matches),
            2,
            f"Expected at least 2 of {keywords} in transcription, "
            f"found {matches}. Full text: {text}",
        )

    def test_multiple_sequential_requests(self):
        """Test that sequential requests produce consistent results."""
        results = []
        for _ in range(3):
            result = self._transcribe()
            self.assertIn("text", result)
            self.assertTrue(len(result["text"]) > 0)
            results.append(result["text"])

        for i in range(1, len(results)):
            self.assertEqual(
                results[0],
                results[i],
                f"Transcription {i + 1} differs from first transcription",
            )

    # -- fused auto-detect (language=None) ---------------------------------
    # The clip is English, so the fused path must both produce a valid
    # transcription AND expose "en" as the detected language. None of the
    # deltas / text fields should leak Whisper special tokens.

    def test_auto_detect_language_verbose_json(self):
        """language omitted + verbose_json returns detected language + clean text."""
        result = self._transcribe(language=None, response_format="verbose_json")
        self.assertEqual(result.get("language"), "en")
        text = result.get("text", "")
        self.assertTrue(len(text) > 0, "Transcription should not be empty")
        self.assertNotIn("<|", text, f"Special token leaked into text: {text!r}")
        # Sanity-check content against the same keywords the English test uses.
        keywords = ["privilege", "leader", "science", "art"]
        matches = [kw for kw in keywords if kw in text.lower()]
        self.assertGreaterEqual(
            len(matches),
            2,
            f"Expected at least 2 of {keywords} in auto-detected transcription, "
            f"found {matches}. Full text: {text!r}",
        )

    def test_auto_detect_matches_explicit_english(self):
        """Auto-detected (language=None) text should match explicit language=en."""
        auto = self._transcribe(language=None).get("text", "")
        explicit = self._transcribe(language="en").get("text", "")
        self.assertEqual(
            auto.strip(),
            explicit.strip(),
            "Auto-detect should produce the same transcription as language=en "
            "on an English clip.",
        )
        self.assertNotIn("<|", auto)

    def test_auto_detect_with_segment_timestamps(self):
        """language=None + timestamp_granularities uses the timestamps fused regex."""
        result = self._transcribe(
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
        """language=None + stream=True: deltas scrubbed, concat matches non-streaming.

        Verified against a real server: sglang's streaming path for Whisper
        produces clean deltas (complete words, no BPE fragmentation), so the
        fused path only needs to hide the forced prefix — which this PR
        does. Asserts both the prefix-leak guard and text equivalence.
        """
        deltas = self._transcribe_stream(language=None)
        self.assertTrue(len(deltas) > 0, "Expected at least one streamed delta")
        for d in deltas:
            self.assertNotIn(
                "<|", d, f"Special token leaked into streaming delta: {d!r}"
            )
        streamed = "".join(deltas).strip()
        reference = self._transcribe(language=None).get("text", "").strip()
        self.assertEqual(
            streamed,
            reference,
            "Streamed auto-detect text should match the non-streaming result.",
        )


if __name__ == "__main__":
    unittest.main()
