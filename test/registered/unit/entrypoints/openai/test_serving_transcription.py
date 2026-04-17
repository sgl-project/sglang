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
        # handler should log a warning and emit whatever raw text it has
        # rather than swallow it silently.
        chunks = [
            _chunk("<|en|>"),
            _chunk("<|en|><|transcribe|>", finish="length"),
        ]
        request, frames = self._run_stream(chunks)
        deltas = _deltas_from_sse(frames)
        self.assertEqual(deltas, ["<|en|><|transcribe|>"])
        # Language was never confidently extracted — do not overwrite.
        self.assertIsNone(request.language)

    def test_non_fused_stream_passes_through(self):
        # When _fused_autodetect is False, no buffering or anchoring happens.
        chunks = [
            _chunk("Hello"),
            _chunk("Hello world", finish="stop"),
        ]
        request, frames = self._run_stream(chunks, fused=False)
        self.assertEqual(_deltas_from_sse(frames), ["Hello", " world"])


if __name__ == "__main__":
    unittest.main()
