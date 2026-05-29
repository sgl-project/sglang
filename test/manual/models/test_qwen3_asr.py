"""
Test Qwen3-ASR model support in SGLang.

Tests /v1/audio/transcriptions (HTTP) and /v1/realtime (OpenAI Realtime
transcription WebSocket).

Usage:
    python test/manual/models/test_qwen3_asr.py
"""

import asyncio
import base64
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
    text = text.lower()
    text = re.sub(r"[^\w\s\u0900-\u097f\u4e00-\u9fff]+", " ", text)
    return text.split()


def _wer(hypothesis: str, reference: str) -> float:
    hyp = _normalize_for_wer(hypothesis)
    ref = _normalize_for_wer(reference)
    if len(ref) <= 1 and not any(" " in w for w in ref):
        # CJK fallback: str.split() degenerates, compare at char level.
        hyp = list(hypothesis.replace(" ", ""))
        ref = list(reference.replace(" ", ""))
    if not ref:
        return 0.0 if not hyp else float("inf")
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


def _pcm16_from_audio_bytes(audio_bytes, target_sr=16000):
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if len(data.shape) > 1:
        data = data.mean(axis=1)
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
    chunk_duration = 0.5
    chunk_bytes = int(chunk_duration * sample_rate * 2)
    duration_sec = round(len(pcm_bytes) / (sample_rate * 2), 2)

    async with websockets.connect(websocket_url) as websocket:
        created = json.loads(await websocket.recv())
        assert (
            created.get("type") == "session.created"
        ), f"expected session.created, got {created!r}"
        session_id = created["session"]["id"]

        transcription_cfg = {"model": "qwen3-asr"}
        if language:
            transcription_cfg["language"] = language
        await websocket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "transcription",
                        "audio": {
                            "input": {
                                "format": {"type": "audio/pcm", "rate": sample_rate},
                                "transcription": transcription_cfg,
                                "noise_reduction": None,
                                "turn_detection": None,
                            }
                        },
                    },
                }
            )
        )

        while True:
            evt = json.loads(await websocket.recv())
            if evt.get("type") == "session.updated":
                break
            if evt.get("type") == "error":
                raise RuntimeError(f"websocket error during update: {evt!r}")

        deltas = []
        completed_msg = {}

        async def receive_loop():
            async for raw in websocket:
                resp = json.loads(raw)
                t = resp.get("type")
                if t == "conversation.item.input_audio_transcription.delta":
                    deltas.append(resp["delta"])
                elif t == "conversation.item.input_audio_transcription.completed":
                    assert (
                        "usage" in resp
                    ), f"transcription.completed missing required usage field: {resp!r}"
                    assert resp["usage"].get("type") == "duration", resp["usage"]
                    completed_msg.update(resp)
                    return
                elif t in (
                    "input_audio_buffer.committed",
                    "conversation.item.created",
                ):
                    continue
                elif t == "error":
                    err = resp.get("error", {})
                    raise RuntimeError(
                        f"websocket error [{err.get('code', '?')}]: "
                        f"{err.get('message', '')}"
                    )
                elif t == "conversation.item.input_audio_transcription.failed":
                    raise RuntimeError(f"transcription failed: {resp!r}")

        receiver = asyncio.create_task(receive_loop())

        for offset in range(0, len(pcm_bytes), chunk_bytes):
            chunk = pcm_bytes[offset : offset + chunk_bytes]
            await websocket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            )
            if realtime:
                await asyncio.sleep(chunk_duration)

        await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        try:
            await asyncio.wait_for(receiver, timeout=60)
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"timed out waiting for transcription.completed; "
                f"got {len(deltas)} deltas, last={deltas[-1] if deltas else None!r}"
            ) from e

    assert completed_msg, "no transcription.completed received"
    return {
        "text": completed_msg.get("transcript", ""),
        "deltas": deltas,
        "session_id": session_id,
        "duration_sec": duration_sec,
    }


class TestQwen3ASRTranscription(CustomTestCase):
    """Test Qwen3-ASR via HTTP /v1/audio/transcriptions and OpenAI Realtime WebSocket /v1/realtime."""

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

    def _websocket_url(self):
        return (
            self.base_url.replace("http://", "ws://").replace("https://", "wss://")
            + "/v1/realtime"
        )

    def _stream_websocket(
        self,
        audio_url,
        local_path,
        language=None,
        realtime=False,
        target_sr=16000,
    ):
        audio_bytes = download_audio(audio_url, local_path)
        pcm, sr = _pcm16_from_audio_bytes(audio_bytes, target_sr=target_sr)
        return asyncio.run(
            _stream_websocket_async(
                self._websocket_url(), pcm, sr, language=language, realtime=realtime
            )
        )

    def _assert_close_to_ref(
        self, hypothesis: str, ref_key: str, max_wer: float = 0.15
    ):
        # 15% tolerates chunked-streaming artifacts without hiding regressions.
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
        result = self._stream_websocket(TEST_AUDIO_EN_URL, TEST_AUDIO_EN_LOCAL)
        self._assert_close_to_ref(result["text"], "en")
        self.assertGreater(len(result["deltas"]), 0)
        print(
            f"[EN WS] final={result['text']} "
            f"({len(result['deltas'])} deltas, {result['duration_sec']}s)"
        )

    @unittest.skipUnless(HAS_WEBSOCKETS, "websockets package not installed")
    def test_chinese_websocket_streaming(self):
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
        # Pace appends at wall-clock so multiple deltas land before commit.
        result = self._stream_websocket(
            TEST_AUDIO_EN_URL, TEST_AUDIO_EN_LOCAL, realtime=True
        )
        self._assert_close_to_ref(result["text"], "en")
        self.assertGreaterEqual(len(result["deltas"]), 2, result["deltas"])
        print(
            f"[Realtime WS] final={result['text']} "
            f"({len(result['deltas'])} deltas, {result['duration_sec']}s)"
        )

    @unittest.skipUnless(HAS_WEBSOCKETS, "websockets package not installed")
    def test_mlk_speech_websocket_streaming(self):
        # FLAC 22050 Hz — exercises client-side resample to 16 kHz.
        result = self._stream_websocket(TEST_AUDIO_MLK_URL, TEST_AUDIO_MLK_LOCAL)
        self._assert_close_to_ref(result["text"], "mlk")
        print(
            f"[MLK WS] final={result['text']} "
            f"({len(result['deltas'])} deltas, {result['duration_sec']}s)"
        )

    @unittest.skipUnless(HAS_WEBSOCKETS, "websockets package not installed")
    def test_websocket_concurrent_sessions(self):
        # Verify state isolation: 3 concurrent sessions on identical audio
        # must yield identical finals + 3 distinct session ids.
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
        self.assertEqual(len(session_ids), 3)
        for r in results:
            self.assertTrue(len(r["text"]) > 0)
        finals = [r["text"] for r in results]
        self.assertEqual(len(set(finals)), 1, f"finals diverged: {finals}")
        print(
            f"[Concurrent x3 WS] all finals match: {finals[0]} "
            f"(session_ids={sorted(session_ids)})"
        )

    @unittest.skipUnless(HAS_WEBSOCKETS, "websockets package not installed")
    def test_spanish_websocket_streaming(self):
        # FLAC 48 kHz PCM_24 — keep native rate so server-side resample runs.
        result = self._stream_websocket(
            TEST_AUDIO_SPANISH_URL,
            TEST_AUDIO_SPANISH_LOCAL,
            language="es",
            target_sr=48000,
        )
        self._assert_close_to_ref(result["text"], "spanish")
        print(
            f"[Spanish WS] final={result['text']} "
            f"({len(result['deltas'])} deltas, {result['duration_sec']}s)"
        )

    @unittest.skipUnless(HAS_WEBSOCKETS, "websockets package not installed")
    def test_websocket_short_clip(self):
        # 3s clip exercises the mid-chunk tail flush at commit.
        audio_bytes = download_audio(TEST_AUDIO_MP3_URL, TEST_AUDIO_MP3_LOCAL)
        full_pcm, sr = _pcm16_from_audio_bytes(audio_bytes)
        short_pcm = full_pcm[: sr * 2 * 3]
        result = asyncio.run(
            _stream_websocket_async(self._websocket_url(), short_pcm, sr)
        )
        self._assert_close_to_ref(result["text"], "mp3")
        print(
            f"[Short clip WS] final={result['text']} "
            f"({len(result['deltas'])} deltas, {result['duration_sec']}s)"
        )

    @unittest.skipUnless(HAS_WEBSOCKETS, "websockets package not installed")
    def test_websocket_chunk_boundary_flush(self):
        # Exact 4s = 2 × chunk_size_sec to hit the exact-boundary tail-flush
        # path at commit. EN clip (not mp3) because mp3 starts with silence.
        audio_bytes = download_audio(TEST_AUDIO_EN_URL, TEST_AUDIO_EN_LOCAL)
        full_pcm, sr = _pcm16_from_audio_bytes(audio_bytes)
        boundary_bytes = int(4.0 * sr * 2)  # 2 × chunk_size_sec at 16 kHz int16 mono
        boundary_pcm = full_pcm[:boundary_bytes]
        assert len(boundary_pcm) == boundary_bytes, "audio shorter than 4s"
        result = asyncio.run(
            _stream_websocket_async(self._websocket_url(), boundary_pcm, sr)
        )
        self.assertTrue(len(result["text"]) > 0, result)
        print(
            f"[Chunk boundary WS] final={result['text']} "
            f"({len(result['deltas'])} deltas, {result['duration_sec']}s)"
        )

    @unittest.skipUnless(HAS_WEBSOCKETS, "websockets package not installed")
    def test_websocket_rejects_unsupported_sample_rate(self):
        async def run():
            async with websockets.connect(self._websocket_url()) as ws:
                created = json.loads(await ws.recv())
                self.assertEqual(created.get("type"), "session.created", created)
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {
                                "type": "transcription",
                                "audio": {
                                    "input": {
                                        "format": {
                                            "type": "audio/pcm",
                                            "rate": 22050,
                                        },
                                        "transcription": {"model": "qwen3-asr"},
                                        "noise_reduction": None,
                                        "turn_detection": None,
                                    }
                                },
                            },
                        }
                    )
                )
                evt = json.loads(await ws.recv())
                self.assertEqual(evt.get("type"), "error", evt)
                err = evt.get("error", {})
                self.assertEqual(err.get("code"), "invalid_value", err)
                self.assertEqual(
                    err.get("param"), "session.audio.input.format.rate", err
                )

        asyncio.run(run())
        print("[Unsupported rate WS] 22050 rejected with invalid_value")

    @unittest.skipUnless(HAS_WEBSOCKETS, "websockets package not installed")
    def test_websocket_rejects_non_dict_transcription(self):
        # Use a valid nested format so Pydantic surfaces the transcription
        # error rather than the format error first.
        async def run():
            async with websockets.connect(self._websocket_url()) as ws:
                created = json.loads(await ws.recv())
                self.assertEqual(created.get("type"), "session.created", created)
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {
                                "type": "transcription",
                                "audio": {
                                    "input": {
                                        "format": {
                                            "type": "audio/pcm",
                                            "rate": 16000,
                                        },
                                        "transcription": "qwen3-asr",
                                        "noise_reduction": None,
                                        "turn_detection": None,
                                    }
                                },
                            },
                        }
                    )
                )
                evt = json.loads(await ws.recv())
                self.assertEqual(evt.get("type"), "error", evt)
                err = evt.get("error", {})
                self.assertEqual(err.get("code"), "invalid_value", err)
                self.assertEqual(
                    err.get("param"), "session.audio.input.transcription", err
                )

        asyncio.run(run())
        print("[Non-dict transcription WS] string rejected with invalid_value")

    @unittest.skipUnless(HAS_WEBSOCKETS, "websockets package not installed")
    def test_websocket_two_commits_propagates_previous_item_id(self):
        # Two commits in one session must (a) emit `previous_item_id: null` on
        # the first committed event, (b) emit `previous_item_id` equal to the
        # first item's id on the second committed event, (c) produce two
        # distinct item_ids, (d) reset per-item state between commits so the
        # second transcript reflects only the second audio (no leak).
        audio_zh = download_audio(TEST_AUDIO_ZH_URL, TEST_AUDIO_ZH_LOCAL)
        pcm_zh, sr = _pcm16_from_audio_bytes(audio_zh)
        audio_kungfu = download_audio(TEST_AUDIO_MP3_URL, TEST_AUDIO_MP3_LOCAL)
        pcm_kungfu, _ = _pcm16_from_audio_bytes(audio_kungfu)

        async def run_one_cycle(ws, pcm, sample_rate):
            """Send `pcm` as 0.5s base64 appends, commit, drain until completed."""
            chunk_bytes = int(0.5 * sample_rate * 2)
            for offset in range(0, len(pcm), chunk_bytes):
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(
                                pcm[offset : offset + chunk_bytes]
                            ).decode("ascii"),
                        }
                    )
                )
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            committed = None
            while True:
                evt = json.loads(await ws.recv())
                t = evt.get("type")
                if t == "input_audio_buffer.committed":
                    committed = evt
                elif t == "conversation.item.input_audio_transcription.completed":
                    return committed, evt["transcript"]
                elif t in (
                    "error",
                    "conversation.item.input_audio_transcription.failed",
                ):
                    raise RuntimeError(f"unexpected event: {evt!r}")

        async def run():
            async with websockets.connect(self._websocket_url()) as ws:
                created = json.loads(await ws.recv())
                self.assertEqual(created.get("type"), "session.created", created)
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {
                                "type": "transcription",
                                "audio": {
                                    "input": {
                                        "format": {"type": "audio/pcm", "rate": sr},
                                        "transcription": {"model": "qwen3-asr"},
                                        "noise_reduction": None,
                                        "turn_detection": None,
                                    }
                                },
                            },
                        }
                    )
                )
                while True:
                    evt = json.loads(await ws.recv())
                    if evt.get("type") == "session.updated":
                        break
                    if evt.get("type") == "error":
                        raise RuntimeError(f"session.update failed: {evt!r}")

                committed_1, transcript_1 = await run_one_cycle(ws, pcm_zh, sr)
                self.assertIsNone(
                    committed_1["previous_item_id"],
                    f"first commit's previous_item_id must be JSON null, got {committed_1!r}",
                )
                first_item_id = committed_1["item_id"]
                self.assertTrue(len(transcript_1) > 0, transcript_1)

                committed_2, transcript_2 = await run_one_cycle(ws, pcm_kungfu, sr)
                self.assertEqual(
                    committed_2["previous_item_id"],
                    first_item_id,
                    f"second commit's previous_item_id must equal first item_id; "
                    f"got prev={committed_2['previous_item_id']!r} "
                    f"vs first={first_item_id!r}",
                )
                self.assertNotEqual(
                    committed_2["item_id"],
                    first_item_id,
                    "item_ids must be distinct across commits",
                )
                # State reset: second transcript must reflect only the second
                # audio, not leak from the first.
                wer = _wer(transcript_2, EXPECTED_TRANSCRIPTS["mp3"])
                self.assertLess(
                    wer,
                    0.15,
                    f"second transcript leaked first audio's content; "
                    f"got {transcript_2!r} (WER {wer:.3f} vs canonical {EXPECTED_TRANSCRIPTS['mp3']!r})",
                )

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main(verbosity=3)
