"""Unit tests for OpenAIServingTranscription's streaming fused-autodetect path.

Exercises the new stream-aware fused-prefix stripping added in the Whisper
auto-detect review response: buffer deltas until the
``<|notimestamps|>`` sentinel lands, re-anchor ``stream_buffer`` to the
boundary, and never leak the forced prefix (or its trailing whitespace) to
the client.

The tests mock ``TokenizerManager.generate_request`` to yield cumulative
``text`` chunks synthesizing both the happy path and the FSM-abort case.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import json
import unittest
from typing import List
from unittest.mock import Mock

from sglang.srt.entrypoints.openai.protocol import TranscriptionRequest
from sglang.srt.entrypoints.openai.serving_transcription import (
    OpenAIServingTranscription,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="stage-a-test-cpu")


def _chunk(text: str, finish: str = None) -> dict:
    """Shape of what TokenizerManager.generate_request yields per step."""
    return {
        "text": text,
        "meta_info": {
            "finish_reason": {"type": finish} if finish else None,
        },
    }


class _MockTokenizerManager:
    """Minimal mock satisfying OpenAIServingTranscription.__init__ and stream loop."""

    def __init__(self, stream_chunks: List[dict]):
        self.model_config = Mock()
        self.model_config.hf_config = Mock()
        self.model_config.hf_config.architectures = ["WhisperForConditionalGeneration"]
        # Not a real ServerArgs, so base class sets allowed_custom_labels=None.
        self.server_args = Mock()
        self.tokenizer = Mock()
        self._stream_chunks = stream_chunks

    def generate_request(self, adapted_request, raw_request):
        chunks = self._stream_chunks

        async def gen():
            for c in chunks:
                yield c

        return gen()

    def create_abort_task(self, adapted_request):
        return None


def _deltas_from_sse(sse_lines: List[str]) -> List[str]:
    """Extract ``choices[0].delta.content`` strings from a list of SSE frames."""
    out = []
    for line in sse_lines:
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :].strip()
        if payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for choice in obj.get("choices", []):
            content = (choice.get("delta") or {}).get("content")
            if content:
                out.append(content)
    return out


class TestStreamingFusedAutodetect(unittest.TestCase):
    """_generate_transcription_stream with _fused_autodetect=True."""

    def _run_stream(self, chunks: List[dict], fused: bool = True):
        tm = _MockTokenizerManager(chunks)
        serving = OpenAIServingTranscription(tm)

        request = TranscriptionRequest(model="whisper", stream=True)
        if fused:
            request._fused_autodetect = True
        adapted = GenerateReqInput(text="", modalities=["audio"])
        raw_request = Mock()

        async def drive():
            frames = []
            async for frame in serving._generate_transcription_stream(
                adapted, request, raw_request
            ):
                frames.append(frame)
            return frames

        loop = get_or_create_event_loop()
        frames = loop.run_until_complete(drive())
        return request, frames

    def test_prefix_stripped_and_language_extracted(self):
        chunks = [
            _chunk("<|en|>"),
            _chunk("<|en|><|transcribe|>"),
            _chunk("<|en|><|transcribe|><|notimestamps|>"),
            _chunk("<|en|><|transcribe|><|notimestamps|> Hello"),
            _chunk("<|en|><|transcribe|><|notimestamps|> Hello world", finish="stop"),
        ]
        request, frames = self._run_stream(chunks)
        deltas = _deltas_from_sse(frames)
        self.assertEqual(deltas, ["Hello", " world"])
        self.assertEqual(request.language, "en")
        # No delta ever starts with the forced prefix or leading whitespace.
        self.assertFalse(any("<|" in d for d in deltas))
        self.assertFalse(deltas[0].startswith(" "))

    def test_non_english_language_extracted(self):
        chunks = [
            _chunk("<|zh|><|transcribe|><|notimestamps|>你好"),
            _chunk("<|zh|><|transcribe|><|notimestamps|>你好世界", finish="stop"),
        ]
        request, frames = self._run_stream(chunks)
        self.assertEqual(request.language, "zh")
        self.assertEqual(_deltas_from_sse(frames), ["你好", "世界"])

    def test_fsm_abort_before_sentinel_flushes_tail(self):
        # Sentinel never arrives; stream terminates on finish_reason. The
        # handler logs a warning and falls through. After the per-delta
        # special-token scrub, a tail that contains only forced-prefix
        # specials collapses to an empty string — no content delta is
        # emitted, but the warning still surfaces and language stays unset.
        chunks = [
            _chunk("<|en|>"),
            _chunk("<|en|><|transcribe|>", finish="length"),
        ]
        request, frames = self._run_stream(chunks)
        self.assertEqual(_deltas_from_sse(frames), [])
        self.assertIsNone(request.language)

    def test_non_fused_stream_passes_through(self):
        # When _fused_autodetect is False, no buffering or anchoring happens.
        chunks = [
            _chunk("Hello"),
            _chunk("Hello world", finish="stop"),
        ]
        request, frames = self._run_stream(chunks, fused=False)
        self.assertEqual(_deltas_from_sse(frames), ["Hello", " world"])

    def test_streaming_ts_variant_sentinel_at_chunk_boundary(self):
        # The <|0.00|> sentinel can land in its own chunk ahead of any
        # transcription text, and the trailing-space arrives later. The
        # handler must buffer silently until a non-whitespace char shows
        # up (so the first delta doesn't leak a leading space) and then
        # scrub subsequent embedded timestamp tokens.
        chunks = [
            _chunk("<|en|>"),
            _chunk("<|en|><|transcribe|>"),
            _chunk("<|en|><|transcribe|><|0.00|>"),  # sentinel alone
            _chunk("<|en|><|transcribe|><|0.00|> "),  # + whitespace only
            _chunk("<|en|><|transcribe|><|0.00|> Hello"),  # first word
            _chunk("<|en|><|transcribe|><|0.00|> Hello<|5.00|> World"),
            _chunk(
                "<|en|><|transcribe|><|0.00|> Hello<|5.00|> World<|endoftext|>",
                finish="stop",
            ),
        ]
        tm = _MockTokenizerManager(chunks)
        serving = OpenAIServingTranscription(tm)
        request = TranscriptionRequest(
            model="whisper", stream=True, timestamp_granularities=["segment"]
        )
        request._fused_autodetect = True

        async def drive():
            frames = []
            async for f in serving._generate_transcription_stream(
                GenerateReqInput(text="", modalities=["audio"]), request, Mock()
            ):
                frames.append(f)
            return frames

        frames = get_or_create_event_loop().run_until_complete(drive())
        deltas = _deltas_from_sse(frames)
        self.assertEqual(request.language, "en")
        self.assertFalse(any("<|" in d for d in deltas))
        # No delta starts with a leading space (the one Whisper emits
        # between <|0.00|> and "Hello" was consumed by the defer-on-
        # whitespace path).
        self.assertFalse(deltas[0].startswith(" "))
        self.assertEqual("".join(deltas), "Hello World")

    def test_streaming_timestamps_variant_scrubs_embedded_segment_tokens(self):
        # Streaming + timestamp_granularities + language=None uses the fused
        # timestamps variant (<|0.00|> sentinel). Segment-boundary tokens
        # <|5.00|>, <|10.00|> land mid-stream; each delta must have them
        # scrubbed before reaching the client. Auto-detection still works
        # — the SSE stream carries clean text, and callers who want
        # segment timing can use response_format=verbose_json which builds
        # segments from output_ids on a separate path.
        chunks = [
            _chunk("<|en|><|transcribe|><|0.00|> Hello"),
            _chunk("<|en|><|transcribe|><|0.00|> Hello<|5.00|> World"),
            _chunk(
                "<|en|><|transcribe|><|0.00|> Hello<|5.00|> World<|10.00|><|endoftext|>",
                finish="stop",
            ),
        ]
        # Mark request as having requested timestamps; the stream handler
        # itself doesn't branch on that, but this documents the scenario.
        tm = _MockTokenizerManager(chunks)
        serving = OpenAIServingTranscription(tm)
        request = TranscriptionRequest(
            model="whisper", stream=True, timestamp_granularities=["segment"]
        )
        request._fused_autodetect = True

        async def drive():
            frames = []
            async for f in serving._generate_transcription_stream(
                GenerateReqInput(text="", modalities=["audio"]), request, Mock()
            ):
                frames.append(f)
            return frames

        frames = get_or_create_event_loop().run_until_complete(drive())
        deltas = _deltas_from_sse(frames)
        self.assertEqual(request.language, "en")
        self.assertFalse(any("<|" in d for d in deltas))
        self.assertEqual("".join(deltas), "Hello World")

    def test_trailing_endoftext_scrubbed_from_last_delta(self):
        # skip_special_tokens=False means the detokenizer may emit
        # <|endoftext|> at the tail. The fused streaming path must scrub it
        # per-delta so clients never see special tokens in SSE chunks.
        chunks = [
            _chunk("<|en|><|transcribe|><|notimestamps|> Hello"),
            _chunk(
                "<|en|><|transcribe|><|notimestamps|> Hello world<|endoftext|>",
                finish="stop",
            ),
        ]
        _, frames = self._run_stream(chunks)
        deltas = _deltas_from_sse(frames)
        self.assertEqual(deltas, ["Hello", " world"])
        self.assertFalse(any("<|" in d for d in deltas))


if __name__ == "__main__":
    unittest.main()
