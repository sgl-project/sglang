"""
Test the OpenAI-compatible /v1/audio/transcriptions endpoint with Whisper.

Usage:
    python3 test_serving_transcription.py -v
"""

import io
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=50, suite="stage-b-test-1-gpu-small")

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

    def _transcribe(self, language="en"):
        """Send a transcription request and return the JSON response."""
        audio_bytes = download_audio_bytes()
        response = requests.post(
            self.base_url + "/v1/audio/transcriptions",
            files={"file": ("audio.mp3", io.BytesIO(audio_bytes), "audio/mpeg")},
            data={
                "model": "whisper",
                "language": language,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

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


if __name__ == "__main__":
    unittest.main()
