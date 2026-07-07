"""Unit tests for srt/entrypoints/openai/segment_streaming.py.

Drives SegmentStreamingASR against a mocked TokenizerManager — verifies
the streaming-session lifecycle (lazy open, per-turn request shape,
idempotent close) without a server or GPU.
"""

import asyncio
import unittest

from sglang.srt.entrypoints.openai.segment_streaming import SegmentStreamingASR
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeAdapter:
    prompt_template = "<|user|><audio><|assistant|>"

    def postprocess_text(self, text):
        return text.strip()


class FakeTokenizerManager:
    """Records calls; returns scripted per-turn texts."""

    def __init__(self, turn_texts, open_result="sess-1"):
        self.turn_texts = list(turn_texts)
        self.open_result = open_result
        self.open_calls = []
        self.close_calls = []
        self.generate_calls = []

    async def open_session(self, obj, request=None):
        self.open_calls.append(obj)
        return self.open_result

    async def close_session(self, obj, request=None):
        self.close_calls.append(obj)

    async def generate_request(self, obj, request=None):
        self.generate_calls.append(obj)
        yield {"text": self.turn_texts.pop(0)}


def make_asr(tm):
    return SegmentStreamingASR(
        tokenizer_manager=tm,
        adapter=FakeAdapter(),
        sampling_params={"temperature": 0.0},
    )


class TestSegmentStreamingASR(CustomTestCase):
    def test_session_opened_lazily_and_once(self):
        tm = FakeTokenizerManager(["hello", "world"])
        asr = make_asr(tm)
        self.assertEqual(len(tm.open_calls), 0)
        asyncio.run(self._two_segments(asr))
        self.assertEqual(len(tm.open_calls), 1)
        self.assertTrue(tm.open_calls[0].streaming)
        self.assertEqual(asr.transcript, "hello world")

    async def _two_segments(self, asr):
        await asr.transcribe_segment(b"wav-1")
        await asr.transcribe_segment(b"wav-2")

    def test_turn_request_shape(self):
        tm = FakeTokenizerManager(["hi"])
        asr = make_asr(tm)
        asyncio.run(asr.transcribe_segment(b"wav-bytes"))
        (req,) = tm.generate_calls
        self.assertEqual(req.text, FakeAdapter.prompt_template)
        self.assertEqual(req.audio_data, b"wav-bytes")
        self.assertEqual(req.session_params, {"id": "sess-1", "rid": None})
        self.assertEqual(req.modalities, ["audio"])
        self.assertFalse(req.stream)

    def test_open_failure_raises(self):
        tm = FakeTokenizerManager(["x"], open_result=None)
        asr = make_asr(tm)
        with self.assertRaisesRegex(RuntimeError, "enable-streaming-session"):
            asyncio.run(asr.transcribe_segment(b"wav"))

    def test_close_is_idempotent_and_releases_session(self):
        tm = FakeTokenizerManager(["a"])
        asr = make_asr(tm)

        async def run():
            await asr.transcribe_segment(b"wav")
            await asr.close()
            await asr.close()

        asyncio.run(run())
        self.assertEqual(len(tm.close_calls), 1)
        self.assertEqual(tm.close_calls[0].session_id, "sess-1")
        self.assertIsNone(asr.session_id)

    def test_close_before_any_segment_is_noop(self):
        tm = FakeTokenizerManager([])
        asr = make_asr(tm)
        asyncio.run(asr.close())
        self.assertEqual(len(tm.open_calls), 0)
        self.assertEqual(len(tm.close_calls), 0)

    def test_whitespace_normalized_and_accumulated(self):
        tm = FakeTokenizerManager(["hello ,", " world ."])
        asr = make_asr(tm)

        async def run():
            first = await asr.transcribe_segment(b"w1")
            second = await asr.transcribe_segment(b"w2")
            return first, second

        first, second = asyncio.run(run())
        self.assertEqual(first, "hello,")
        self.assertEqual(second, "world.")
        self.assertEqual(asr.transcript, "hello, world.")


if __name__ == "__main__":
    unittest.main()
