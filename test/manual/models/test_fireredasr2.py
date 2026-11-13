"""Run with: python test/manual/models/test_fireredasr2.py."""

import io
import unittest

import numpy as np
import requests
import soundfile as sf

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

MODEL = "allendou/FireRedASR2-LLM-vllm"
AUDIO_URL = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"


class TestFireRedASR2(CustomTestCase):
    process = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--tp", "1", "--dtype", "bfloat16"],
        )
        response = requests.get(AUDIO_URL, timeout=60)
        response.raise_for_status()
        cls.audio, cls.sample_rate = sf.read(io.BytesIO(response.content))

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)

    def _transcribe(self, audio, response_format="json"):
        data = io.BytesIO()
        sf.write(data, audio, self.sample_rate, format="WAV")
        response = requests.post(
            f"{self.base_url}/v1/audio/transcriptions",
            files={"file": ("speech.wav", data.getvalue(), "audio/wav")},
            data={"model": MODEL, "temperature": 0, "response_format": response_format},
            timeout=180,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_short_transcription(self):
        result = self._transcribe(self.audio)
        self.assertIn("i have a dream", result["text"].lower())

    def test_long_audio_chunking(self):
        repeats = int(31 * self.sample_rate / len(self.audio)) + 1
        result = self._transcribe(np.tile(self.audio, repeats), "verbose_json")
        self.assertIn("i have a dream", result["text"].lower())
        self.assertGreater(result["duration"], 30)
        self.assertEqual(result["segments"], [])


if __name__ == "__main__":
    unittest.main()
