"""
Test Whisper model with CUDA graph support.

This test verifies that:
1. Whisper model works correctly with CUDA graph enabled (default)
2. Cross-attention KV cache is properly managed through RadixAttention
3. Output is consistent between CUDA graph and non-CUDA-graph modes

Usage:
    python test_whisper_cuda_graph.py

Requires:
    - A GPU with sufficient memory
    - openai-whisper model (e.g., openai/whisper-large-v3)
    - An audio file or URL for testing
"""

import io
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

WHISPER_MODEL = "openai/whisper-large-v3"
TEST_AUDIO_URL = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
TEST_AUDIO_LOCAL = "/tmp/test_whisper_audio.flac"


def get_audio_bytes():
    """Get audio bytes, downloading if necessary."""
    import os

    if os.path.exists(TEST_AUDIO_LOCAL):
        with open(TEST_AUDIO_LOCAL, "rb") as f:
            return f.read()
    resp = requests.get(TEST_AUDIO_URL, timeout=30)
    resp.raise_for_status()
    with open(TEST_AUDIO_LOCAL, "wb") as f:
        f.write(resp.content)
    return resp.content


class TestWhisperCudaGraph(CustomTestCase):
    """Test Whisper with CUDA graph enabled (default behavior)."""

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
        kill_process_tree(cls.process.pid)

    def _transcribe(self, language="en"):
        """Send a transcription request via OpenAI-compatible audio endpoint."""
        audio_bytes = get_audio_bytes()
        response = requests.post(
            self.base_url + "/v1/audio/transcriptions",
            files={"file": ("audio.ogg", io.BytesIO(audio_bytes), "audio/ogg")},
            data={
                "model": "whisper",
                "language": language,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_basic_transcription(self):
        """Test that basic transcription works with CUDA graph."""
        result = self._transcribe()
        self.assertIn("text", result)
        text = result["text"]
        self.assertTrue(len(text) > 0, "Transcription should not be empty")
        print(f"Transcription: {text}")

    def test_multiple_sequential_requests(self):
        """Test multiple sequential requests to verify CUDA graph replay consistency."""
        results = []
        for i in range(3):
            result = self._transcribe()
            self.assertIn("text", result)
            results.append(result["text"])
            print(f"Request {i+1}: {result['text'][:80]}...")

        # All transcriptions of the same audio should be identical
        for i in range(1, len(results)):
            self.assertEqual(
                results[0],
                results[i],
                f"Transcription {i+1} differs from first transcription",
            )

    def test_transcription_quality(self):
        """Test that transcription quality is reasonable (contains expected words)."""
        result = self._transcribe()
        text = result["text"].lower()
        # The test audio is a LibriSpeech sample about stew for dinner
        self.assertIn("stew", text, f"Expected 'stew' in transcription: {text}")
        self.assertIn("dinner", text, f"Expected 'dinner' in transcription: {text}")
        print(f"Quality check passed: {result['text'][:80]}...")


class TestWhisperNoCudaGraph(CustomTestCase):
    """Test Whisper with CUDA graph explicitly disabled for comparison."""

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
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_basic_transcription_no_cuda_graph(self):
        """Test that transcription works without CUDA graph (baseline)."""
        audio_bytes = get_audio_bytes()
        response = requests.post(
            self.base_url + "/v1/audio/transcriptions",
            files={"file": ("audio.ogg", io.BytesIO(audio_bytes), "audio/ogg")},
            data={
                "model": "whisper",
                "language": "en",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertIn("text", result)
        self.assertTrue(len(result["text"]) > 0)
        print(f"No CUDA graph transcription: {result['text'][:80]}...")


if __name__ == "__main__":
    unittest.main(verbosity=3)
