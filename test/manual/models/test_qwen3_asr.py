"""
Test Qwen3-ASR model support in SGLang.

Tests /v1/audio/transcriptions (HTTP) and
/v1/audio/transcriptions/stream (WebSocket live audio input).

Usage:
    python test/manual/models/test_qwen3_asr.py
"""

import asyncio
import io
import json
import os
import re
import unittest

import numpy as np
import requests
import soundfile as sf

try:
    import websockets

    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

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
TEST_AUDIO_MLK_URL = (
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
)
TEST_AUDIO_LIBRI_URL = (
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
)
TEST_AUDIO_SPANISH_URL = (
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/4.flac"
)
TEST_AUDIO_HINDI_URL = (
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/hindi.ogg"
)
TEST_AUDIO_MP3_URL = (
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/i-know-kung-fu.mp3"
)
TEST_AUDIO_EN_LOCAL = "/tmp/test_qwen3_asr_en.wav"
TEST_AUDIO_ZH_LOCAL = "/tmp/test_qwen3_asr_zh.wav"
TEST_AUDIO_MLK_LOCAL = "/tmp/test_qwen3_asr_mlk.flac"
TEST_AUDIO_LIBRI_LOCAL = "/tmp/test_qwen3_asr_libri.flac"
TEST_AUDIO_SPANISH_LOCAL = "/tmp/test_qwen3_asr_spanish.flac"
TEST_AUDIO_HINDI_LOCAL = "/tmp/test_qwen3_asr_hindi.ogg"
TEST_AUDIO_MP3_LOCAL = "/tmp/test_qwen3_asr_kungfu.mp3"

# Captured from Qwen3-ASR-0.6B non-streaming inference (2026-04-14).
# Refresh if model weights or sampling params change.
EXPECTED_TRANSCRIPTS = {
    "en": (
        "Oh yeah, yeah. He wasn't even that big when I started listening to him."
        " But and his solo music didn't do overly well, but he did very well"
        " when he started writing for other people."
    ),
    "zh": "甚至出现交易几乎停滞的情况。",
    "mlk": (
        "I have a dream that one day this nation will rise up and live out"
        " the true meaning of its creed."
    ),
    "libri": (
        "He hoped there would be stew for dinner—turnips and carrots and"
        " bruised potatoes and fat mutton pieces—to be ladled out in thick"
        " peppered flour-fatted sauce."
    ),
    "spanish": (
        "y en las ramas medio sumergidas revoloteaban algunos pájaros"
        " de químico y legendario plumaje"
    ),
    "hindi": "मिर्ची में कितने विभिन्न प्रजातियाँ हैं",
    "mp3": "I know kung fu.",
}


def _normalize_for_wer(text: str) -> list:
    """Lowercase, strip punctuation, split on whitespace.

    Used by ``_wer`` so that chunked-streaming artifacts that differ from
    one-shot only in punctuation / casing (``—`` vs ``:``, trailing period,
    leading capitalization) don't count as errors.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s\u0900-\u097f\u4e00-\u9fff]+", " ", text)
    return text.split()


def _wer(hypothesis: str, reference: str) -> float:
    """Word error rate via Levenshtein distance on normalized tokens.

    Returns ``edit_distance(hyp_words, ref_words) / len(ref_words)``. For
    CJK text where ``str.split()`` degenerates, we fall back to character
    distance so the metric still means something.
    """
    hyp = _normalize_for_wer(hypothesis)
    ref = _normalize_for_wer(reference)
    if len(ref) <= 1 and not any(" " in w for w in ref):
        # CJK fallback: compare at char level
        hyp = list(hypothesis.replace(" ", ""))
        ref = list(reference.replace(" ", ""))
    if not ref:
        return 0.0 if not hyp else float("inf")
    # Standard Levenshtein DP
    n, m = len(hyp), len(ref)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            if hyp[i - 1] == ref[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j - 1], dp[j])
            prev = cur
    return dp[m] / len(ref)


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


def _pcm16_from_audio_bytes(audio_bytes):
    """Decode audio bytes, resample to 16kHz mono, return (pcm_bytes, sample_rate)."""
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    target_sr = 16000
    if sr != target_sr:
        num_samples = int(len(data) / sr * target_sr)
        indices = np.linspace(0, len(data) - 1, num_samples)
        data = np.interp(indices, np.arange(len(data)), data)
        sr = target_sr
    pcm = (data * 32767).astype(np.int16).tobytes()
    return pcm, sr


async def _stream_websocket_async(
    websocket_url, pcm_bytes, sample_rate, language=None, realtime=False
):
    """Stream PCM over WebSocket; return {text, deltas, session_id, duration_sec}.

    If realtime=True, sleeps between chunks to simulate live audio pacing.
    """
    chunk_duration = 0.5
    chunk_bytes = int(chunk_duration * sample_rate * 2)  # int16 = 2 bytes

    async with websockets.connect(websocket_url) as websocket:
        start_msg = {"type": "session.start"}
        if language:
            start_msg["language"] = language
        await websocket.send(json.dumps(start_msg))
        ack = json.loads(await websocket.recv())
        assert ack.get("type") == "session.started", f"unexpected ack: {ack}"
        session_id = ack["session_id"]

        deltas = []
        final_msg = {}

        async def receive_loop():
            async for raw in websocket:
                resp = json.loads(raw)
                if resp["type"] == "transcript.delta":
                    deltas.append(resp["delta"])
                elif resp["type"] == "transcript.final":
                    final_msg.update(resp)
                    return
                elif resp["type"] == "error":
                    raise RuntimeError(
                        f"websocket error [{resp.get('code', '?')}]: {resp.get('message', '')}"
                    )

        receiver = asyncio.create_task(receive_loop())

        for offset in range(0, len(pcm_bytes), chunk_bytes):
            chunk = pcm_bytes[offset : offset + chunk_bytes]
            await websocket.send(chunk)
            if realtime:
                await asyncio.sleep(chunk_duration)

        await websocket.send(json.dumps({"type": "session.end"}))
        await receiver

    assert final_msg, "no transcript.final received"
    return {
        "text": final_msg.get("text", ""),
        "deltas": deltas,
        "session_id": session_id,
        "duration_sec": final_msg.get("duration_sec", 0.0),
    }


class TestQwen3ASRTranscription(CustomTestCase):
    """Test Qwen3-ASR via HTTP /v1/audio/transcriptions and WebSocket /v1/audio/transcriptions/stream."""

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

    # ------------------------------------------------------------------
    # HTTP path
    # ------------------------------------------------------------------

    def _transcribe(self, audio_url, local_path, language=None):
        """Send an HTTP transcription request."""
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

    def test_mlk_transcription(self):
        """13s MLK speech (FLAC 22050 Hz) — HTTP non-stream ground truth."""
        result = self._transcribe(TEST_AUDIO_MLK_URL, TEST_AUDIO_MLK_LOCAL)
        self.assertIn("text", result)
        text = result["text"]
        self.assertTrue(len(text) > 0, "Transcription should not be empty")
        print(f"[MLK Transcription] {text}")

    def test_librispeech_transcription(self):
        """10s LibriSpeech-style FLAC 16 kHz — HTTP non-stream ground truth."""
        result = self._transcribe(TEST_AUDIO_LIBRI_URL, TEST_AUDIO_LIBRI_LOCAL)
        self.assertIn("text", result)
        text = result["text"]
        self.assertTrue(len(text) > 0, "Transcription should not be empty")
        print(f"[LibriSpeech Transcription] {text}")

    def test_spanish_transcription(self):
        """Spanish FLAC 48 kHz PCM_24 — HTTP non-stream ground truth."""
        result = self._transcribe(
            TEST_AUDIO_SPANISH_URL, TEST_AUDIO_SPANISH_LOCAL, language="es"
        )
        self.assertIn("text", result)
        text = result["text"]
        self.assertTrue(len(text) > 0, "Transcription should not be empty")
        print(f"[Spanish Transcription] {text}")

    def test_hindi_transcription(self):
        """Hindi OGG/Opus 16 kHz — HTTP non-stream ground truth."""
        result = self._transcribe(
            TEST_AUDIO_HINDI_URL, TEST_AUDIO_HINDI_LOCAL, language="hi"
        )
        self.assertIn("text", result)
        text = result["text"]
        self.assertTrue(len(text) > 0, "Transcription should not be empty")
        print(f"[Hindi Transcription] {text}")

    def test_mp3_stereo_transcription(self):
        """MP3 stereo 44.1 kHz — HTTP non-stream ground truth."""
        result = self._transcribe(TEST_AUDIO_MP3_URL, TEST_AUDIO_MP3_LOCAL)
        self.assertIn("text", result)
        text = result["text"]
        self.assertTrue(len(text) > 0, "Transcription should not be empty")
        print(f"[MP3 Transcription] {text}")

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

    # ------------------------------------------------------------------
    # WebSocket path
    # ------------------------------------------------------------------

    def _websocket_url(self):
        return (
            self.base_url.replace("http://", "ws://").replace("https://", "wss://")
            + "/v1/audio/transcriptions/stream"
        )

    def _stream_websocket(self, audio_url, local_path, language=None, realtime=False):
        audio_bytes = download_audio(audio_url, local_path)
        pcm, sr = _pcm16_from_audio_bytes(audio_bytes)
        return asyncio.run(
            _stream_websocket_async(
                self._websocket_url(), pcm, sr, language=language, realtime=realtime
            )
        )

    def _assert_close_to_ref(
        self, hypothesis: str, ref_key: str, max_wer: float = 0.15
    ):
        """Assert a streamed transcript stays within ``max_wer`` of the reference.

        Chunked streaming inherits a few artifacts from #22089 that one-shot
        does not — "Uh huh." short-context hallucination on long English
        audio, mid-sentence punctuation drift (``—`` → ``:``), and trailing
        periods. We accept up to 15% WER (normalized, case/punct stripped)
        so these don't break CI, while still catching real regressions
        like dropped words or double-emitted phrases.
        """
        reference = EXPECTED_TRANSCRIPTS[ref_key]
        wer = _wer(hypothesis, reference)
        self.assertLessEqual(
            wer,
            max_wer,
            f"WER {wer:.3f} > {max_wer} for {ref_key!r}\n"
            f"  hyp: {hypothesis!r}\n  ref: {reference!r}",
        )

    @unittest.skipUnless(HAS_WEBSOCKETS, "websockets package not installed")
    def test_english_websocket_streaming(self):
        """Test English audio transcription over WebSocket (fast mode)."""
        result = self._stream_websocket(TEST_AUDIO_EN_URL, TEST_AUDIO_EN_LOCAL)
        self._assert_close_to_ref(result["text"], "en")
        self.assertGreater(len(result["deltas"]), 0)
        print(
            f"[EN WS] final={result['text']} "
            f"({len(result['deltas'])} deltas, {result['duration_sec']}s)"
        )

    @unittest.skipUnless(HAS_WEBSOCKETS, "websockets package not installed")
    def test_chinese_websocket_streaming(self):
        """Test Chinese audio transcription over WebSocket with session.start.language."""
        result = self._stream_websocket(
            TEST_AUDIO_ZH_URL, TEST_AUDIO_ZH_LOCAL, language="zh"
        )
        self._assert_close_to_ref(result["text"], "zh")
        print(
            f"[ZH WS] final={result['text']} "
            f"({len(result['deltas'])} deltas, {result['duration_sec']}s)"
        )

    @unittest.skipUnless(HAS_WEBSOCKETS, "websockets package not installed")
    def test_websocket_streaming_realtime(self):
        """Exercise real-time pacing: sleep between chunks to simulate live audio."""
        result = self._stream_websocket(
            TEST_AUDIO_EN_URL, TEST_AUDIO_EN_LOCAL, realtime=True
        )
        self._assert_close_to_ref(result["text"], "en")
        self.assertGreaterEqual(
            len(result["deltas"]), 2, "realtime mode should yield multiple deltas"
        )
        print(
            f"[Realtime WS] final={result['text']} "
            f"({len(result['deltas'])} deltas, {result['duration_sec']}s)"
        )

    @unittest.skipUnless(HAS_WEBSOCKETS, "websockets package not installed")
    def test_mlk_speech_websocket_streaming(self):
        """13s English MLK speech, FLAC @ 22050 Hz (exercises client-side resampling)."""
        result = self._stream_websocket(TEST_AUDIO_MLK_URL, TEST_AUDIO_MLK_LOCAL)
        self._assert_close_to_ref(result["text"], "mlk")
        print(
            f"[MLK WS] final={result['text']} "
            f"({len(result['deltas'])} deltas, {result['duration_sec']}s)"
        )

    @unittest.skipUnless(HAS_WEBSOCKETS, "websockets package not installed")
    def test_librispeech_dummy_websocket_streaming(self):
        """10s LibriSpeech-style FLAC @ 16 kHz (no resampling needed)."""
        result = self._stream_websocket(TEST_AUDIO_LIBRI_URL, TEST_AUDIO_LIBRI_LOCAL)
        self._assert_close_to_ref(result["text"], "libri")
        print(
            f"[LibriSpeech WS] final={result['text']} "
            f"({len(result['deltas'])} deltas, {result['duration_sec']}s)"
        )

    @unittest.skipUnless(HAS_WEBSOCKETS, "websockets package not installed")
    def test_websocket_concurrent_sessions(self):
        """3 parallel WebSocket sessions; verify state isolation and independent finals."""
        audio_bytes = download_audio(TEST_AUDIO_EN_URL, TEST_AUDIO_EN_LOCAL)
        pcm, sr = _pcm16_from_audio_bytes(audio_bytes)

        async def run_n_concurrent(n):
            return await asyncio.gather(
                *[
                    _stream_websocket_async(self._websocket_url(), pcm, sr)
                    for _ in range(n)
                ]
            )

        results = asyncio.run(run_n_concurrent(3))

        session_ids = {r["session_id"] for r in results}
        self.assertEqual(len(session_ids), 3, "each session must have a unique id")
        for i, r in enumerate(results):
            self.assertTrue(
                len(r["text"]) > 0, f"session {i} should produce non-empty transcript"
            )
        # All sessions ran the same audio, so finals should match.
        finals = [r["text"] for r in results]
        self.assertEqual(
            len(set(finals)),
            1,
            f"3 concurrent sessions on identical audio should yield identical finals, got {finals}",
        )
        print(
            f"[Concurrent x3 WS] all finals match: {finals[0]} "
            f"(session_ids={sorted(session_ids)})"
        )

    @unittest.skipUnless(HAS_WEBSOCKETS, "websockets package not installed")
    def test_spanish_websocket_streaming(self):
        """6.6s Spanish audio, FLAC @ 48 kHz PCM_24 (resampling + high-bit-depth)."""
        result = self._stream_websocket(
            TEST_AUDIO_SPANISH_URL, TEST_AUDIO_SPANISH_LOCAL, language="es"
        )
        self._assert_close_to_ref(result["text"], "spanish")
        print(
            f"[Spanish WS] final={result['text']} "
            f"({len(result['deltas'])} deltas, {result['duration_sec']}s)"
        )

    @unittest.skipUnless(HAS_WEBSOCKETS, "websockets package not installed")
    def test_hindi_websocket_streaming(self):
        """4s Hindi audio, OGG/Opus @ 16 kHz (multilingual + non-WAV/FLAC container)."""
        result = self._stream_websocket(
            TEST_AUDIO_HINDI_URL, TEST_AUDIO_HINDI_LOCAL, language="hi"
        )
        self._assert_close_to_ref(result["text"], "hindi")
        print(
            f"[Hindi WS] final={result['text']} "
            f"({len(result['deltas'])} deltas, {result['duration_sec']}s)"
        )

    @unittest.skipUnless(HAS_WEBSOCKETS, "websockets package not installed")
    def test_mp3_stereo_websocket_streaming(self):
        """4s English MP3 stereo @ 44.1 kHz (mp3 decode + stereo->mono + resample)."""
        result = self._stream_websocket(TEST_AUDIO_MP3_URL, TEST_AUDIO_MP3_LOCAL)
        self._assert_close_to_ref(result["text"], "mp3")
        print(
            f"[MP3 stereo WS] final={result['text']} "
            f"({len(result['deltas'])} deltas, {result['duration_sec']}s)"
        )

    @unittest.skipUnless(HAS_WEBSOCKETS, "websockets package not installed")
    def test_websocket_short_clip(self):
        """Sub-chunk clip: less than chunk_size_sec of real speech.

        Qwen3-ASR hallucinates badly on <2s inputs (short-context artifact),
        so we use ~3s of the MP3 kungfu clip where speech actually starts.
        This path still exercises the "session.end before any inference
        trigger" branch because 3s < the session's accumulated-audio
        threshold path when PCM is streamed in 0.5s frames.
        """
        audio_bytes = download_audio(TEST_AUDIO_MP3_URL, TEST_AUDIO_MP3_LOCAL)
        full_pcm, sr = _pcm16_from_audio_bytes(audio_bytes)
        short_pcm = full_pcm[: sr * 2 * 3]  # first 3 seconds
        result = asyncio.run(
            _stream_websocket_async(self._websocket_url(), short_pcm, sr)
        )
        self._assert_close_to_ref(result["text"], "mp3")
        print(
            f"[Short clip WS] final={result['text']} "
            f"({len(result['deltas'])} deltas, {result['duration_sec']}s)"
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
