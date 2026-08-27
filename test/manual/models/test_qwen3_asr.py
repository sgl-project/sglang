"""
Test Qwen3-ASR model support in SGLang.

Tests /v1/audio/transcriptions endpoint (OpenAI-compatible).

Usage:
    python test/manual/models/test_qwen3_asr.py
"""

import io
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MODEL = "Qwen/Qwen3-ASR-0.6B"
# MODEL = "Qwen/Qwen3-ASR-1.7B"
TEST_AUDIO_EN_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
)
TEST_AUDIO_ZH_URL = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav"
)
TEST_AUDIO_EN_LOCAL = "/tmp/test_qwen3_asr_en.wav"
TEST_AUDIO_ZH_LOCAL = "/tmp/test_qwen3_asr_zh.wav"


def download_audio(url, local_path):
    """Download audio file if not already cached."""
    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            return f.read()
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(resp.content)
    return resp.content


class TestQwen3ASRTranscription(CustomTestCase):
    """Test Qwen3-ASR via /v1/audio/transcriptions endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--served-model-name",
                "qwen3-asr",
                "--trust-remote-code",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _transcribe(self, audio_url, local_path, language=None):
        """Send a transcription request."""
        audio_bytes = download_audio(audio_url, local_path)
        data = {"model": "qwen3-asr"}
        if language:
            data["language"] = language
        response = requests.post(
            self.base_url + "/v1/audio/transcriptions",
            files={"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")},
            data=data,
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_english_transcription(self):
        """Test English audio transcription."""
        result = self._transcribe(TEST_AUDIO_EN_URL, TEST_AUDIO_EN_LOCAL)
        self.assertIn("text", result)
        text = result["text"]
        self.assertTrue(len(text) > 0, "Transcription should not be empty")
        print(f"[EN Transcription] {text}")

    def test_chinese_transcription(self):
        """Test Chinese audio transcription."""
        result = self._transcribe(TEST_AUDIO_ZH_URL, TEST_AUDIO_ZH_LOCAL)
        self.assertIn("text", result)
        text = result["text"]
        self.assertTrue(len(text) > 0, "Transcription should not be empty")
        print(f"[ZH Transcription] {text}")

    def test_multiple_requests_consistency(self):
        """Test that repeated requests produce consistent output."""
        results = []
        for _ in range(3):
            result = self._transcribe(TEST_AUDIO_EN_URL, TEST_AUDIO_EN_LOCAL)
            results.append(result["text"])

        for i in range(1, len(results)):
            self.assertEqual(
                results[0],
                results[i],
                f"Request {i+1} differs from first request",
            )
        print(f"[Consistency] All 3 requests match: {results[0][:80]}...")


if __name__ == "__main__":
    unittest.main(verbosity=3)
