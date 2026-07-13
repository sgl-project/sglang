"""Unit tests for srt/entrypoints/openai/segment_streaming.py.

Segment streaming transcribes each fixed-duration segment as an
independent, standalone request (no engine session, no carried context),
then concatenates the per-segment texts. These tests verify the per-turn
request shape and transcript accumulation against a mocked
TokenizerManager — no server or GPU.
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
    """Records generate_request calls; returns scripted per-segment texts."""

    def __init__(self, turn_texts):
        self.turn_texts = list(turn_texts)
        self.generate_calls = []

    async def generate_request(self, obj, request=None):
        self.generate_calls.append(obj)
        yield {"text": self.turn_texts.pop(0)}


def make_asr(tm, routing_key=None):
    return SegmentStreamingASR(
        tokenizer_manager=tm,
        adapter=FakeAdapter(),
        sampling_params={"temperature": 0.0},
        routing_key=routing_key,
    )


class TestSegmentStreamingASR(CustomTestCase):
    def test_each_segment_is_an_independent_request(self):
        tm = FakeTokenizerManager(["hello", "world"])
        asr = make_asr(tm)

        async def run():
            await asr.transcribe_segment(b"wav-1")
            await asr.transcribe_segment(b"wav-2")

        asyncio.run(run())
        # One standalone request per segment, no session bookkeeping.
        self.assertEqual(len(tm.generate_calls), 2)
        self.assertEqual(asr.transcript, "hello world")

    def test_request_shape_carries_only_this_segment(self):
        tm = FakeTokenizerManager(["hi"])
        asr = make_asr(tm)
        asyncio.run(asr.transcribe_segment(b"wav-bytes"))
        (req,) = tm.generate_calls
        self.assertEqual(req.text, FakeAdapter.prompt_template)
        self.assertEqual(req.audio_data, b"wav-bytes")
        self.assertEqual(req.modalities, ["audio"])
        self.assertFalse(req.stream)
        # No session context is threaded between segments.
        self.assertIsNone(req.session_params)

    def test_routing_key_propagated(self):
        tm = FakeTokenizerManager(["hi"])
        asr = make_asr(tm, routing_key="rk-1")
        asyncio.run(asr.transcribe_segment(b"wav"))
        (req,) = tm.generate_calls
        self.assertEqual(req.routing_key, "rk-1")

    def test_empty_response_returns_empty_and_keeps_transcript(self):
        tm = FakeTokenizerManager([""])
        asr = make_asr(tm)
        out = asyncio.run(asr.transcribe_segment(b"wav"))
        self.assertEqual(out, "")
        self.assertEqual(asr.transcript, "")

    def test_close_is_a_noop(self):
        tm = FakeTokenizerManager([])
        asr = make_asr(tm)
        # No segment sent, nothing to release; must not raise or call the TM.
        asyncio.run(asr.close())
        self.assertEqual(len(tm.generate_calls), 0)

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
