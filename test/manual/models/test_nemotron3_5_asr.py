"""
Test NVIDIA Nemotron 3.5 ASR (nvidia/nemotron-3.5-asr-streaming-0.6b).

Exercises /v1/audio/transcriptions: transcripts against transformers' greedy
RNN-T output, batched decoding, the language prompt and long-audio chunking.
Requires librosa (the checkpoint's feature extractor uses it).

Usage:
    python test/manual/models/test_nemotron3_5_asr.py
"""

import concurrent.futures
import io
import re
import unittest

import numpy as np
import requests
import soundfile as sf

from sglang.srt.utils import load_audio
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
AUDIO_URLS = {
    "libri": "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
    "mlk": "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
    "en": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
    "zh": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
}

# transformers Nemotron3_5AsrForRNNT.generate, fp32, language="auto"
# (transformers 5.19.0, 2026-10-27).
EXPECTED_TRANSCRIPTS = {
    "libri": (
        "He hoped there would be stew for dinner, turnips and carrots and bruised"
        " potatoes and fat mutton pieces to be ladled out in thick peppered flour"
        " fat and sauce."
    ),
    "mlk": (
        "I have a dream that one day this nation will rise up and live up the"
        " true meaning of its creed."
    ),
    "en": (
        "mm oh yeah yeah he wasn't even that big when I started listening to him"
        " but and his solo music didn't do overly well but he did very well when"
        " he started writing for other people"
    ),
    "zh": "甚 至 出 现 交 易 几 乎 停 滞 的 情 况。",
}

# fp16 (the server's default for this fp32 checkpoint) moves an occasional
# comma or word relative to the fp32 reference.
MAX_WER = 0.05


def _words(text: str) -> list:
    return re.sub(r"[^\w\s]+", " ", text.lower()).split()


def _wer(hypothesis: str, reference: str) -> float:
    hyp, ref = _words(hypothesis), _words(reference)
    dp = list(range(len(ref) + 1))
    for i in range(1, len(hyp) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(ref) + 1):
            cur = dp[j]
            dp[j] = prev if hyp[i - 1] == ref[j - 1] else 1 + min(prev, dp[j - 1], cur)
            prev = cur
    return dp[len(ref)] / max(len(ref), 1)


class TestNemotron3_5Asr(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.audio = {
            name: requests.get(url, timeout=60).content
            for name, url in AUDIO_URLS.items()
        }
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            MODEL, cls.base_url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def _transcribe(self, audio: bytes, **form) -> requests.Response:
        return requests.post(
            self.base_url + "/v1/audio/transcriptions",
            files={"file": ("audio", audio)},
            data={"model": MODEL, **form},
            timeout=120,
        )

    def _text(self, audio: bytes, **form) -> str:
        response = self._transcribe(audio, **form)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["text"]

    def test_transcripts_match_transformers(self):
        for name, expected in EXPECTED_TRANSCRIPTS.items():
            with self.subTest(audio=name):
                text = self._text(self.audio[name])
                if name == "zh":
                    self.assertEqual(text.strip(), expected)
                else:
                    self.assertLessEqual(_wer(text, expected), MAX_WER, text)

    def test_batched_requests_match_sequential(self):
        # Each request's RNN-T state lives in its pool slot; batching must not
        # mix them up. fp16 rounding may differ with the batch, so not exact.
        names = list(EXPECTED_TRANSCRIPTS)
        sequential = {name: self._text(self.audio[name]) for name in names}
        with concurrent.futures.ThreadPoolExecutor(16) as pool:
            batched = list(
                pool.map(lambda n: (n, self._text(self.audio[n])), names * 6)
            )
        for name, text in batched:
            self.assertLessEqual(_wer(text, sequential[name]), MAX_WER, name)

    def test_language_prompt(self):
        text = self._text(self.audio["mlk"], language="en")
        self.assertLessEqual(_wer(text, EXPECTED_TRANSCRIPTS["mlk"]), MAX_WER, text)
        response = self._transcribe(self.audio["mlk"], language="xx")
        self.assertEqual(response.status_code, 400, response.text)

    def test_long_audio_is_chunked_with_word_boundaries(self):
        # Over the adapter's 30 s clip limit, so the server splits and stitches.
        names = ["libri", "mlk", "en"]
        silence = np.zeros(8000, dtype=np.float32)
        pieces = []
        for name in names:
            pieces += [load_audio(self.audio[name]), silence]
        buffer = io.BytesIO()
        sf.write(buffer, np.concatenate(pieces), 16000, format="WAV")
        text = self._text(buffer.getvalue())
        reference = " ".join(EXPECTED_TRANSCRIPTS[name] for name in names)
        self.assertLessEqual(_wer(text, reference), 0.1, text)


if __name__ == "__main__":
    unittest.main(verbosity=3)
