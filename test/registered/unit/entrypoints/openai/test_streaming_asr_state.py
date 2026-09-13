from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import json
from typing import List
from unittest.mock import AsyncMock, Mock, patch

from sglang.srt.entrypoints.openai.protocol import TranscriptionRequest
from sglang.srt.entrypoints.openai.serving_transcription import (
    OpenAIServingTranscription,
)
from sglang.srt.entrypoints.openai.streaming_asr import StreamingASRState
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _ScriptedQwen3ASRTokenizerManager:
    def __init__(self, transcripts: List[str]):
        self.model_config = Mock()
        self.model_config.hf_config = Mock()
        self.model_config.hf_config.architectures = ["Qwen3ASRForConditionalGeneration"]
        self.server_args = Mock(asr_max_concurrent_sessions=32)
        self._transcripts = iter(transcripts)

    def generate_request(self, adapted_request, raw_request):
        transcript = next(self._transcripts)

        async def gen():
            yield {"text": f"language English<asr_text>{transcript}"}

        return gen()


def _deltas_from_sse(frames: List[str]) -> List[str]:
    deltas = []
    for frame in frames:
        if not frame.startswith("data: "):
            continue
        payload = frame[len("data: ") :].strip()
        if payload == "[DONE]":
            continue
        response = json.loads(payload)
        for choice in response.get("choices", []):
            content = (choice.get("delta") or {}).get("content")
            if content:
                deltas.append(content)
    return deltas


class TestStreamingASRState(CustomTestCase):
    @staticmethod
    def _state() -> StreamingASRState:
        return StreamingASRState(
            chunk_size_sec=2.0,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
        )

    def test_word_extension_is_treated_as_revision(self):
        state = self._state()
        self.assertEqual(
            state.update("the cat one two three four five"),
            "the cat",
        )

        delta = state.update("the caterpillar one two three four five")

        self.assertEqual(delta, "caterpillar")
        self.assertEqual(state.emitted_text, "the cat caterpillar")

    def test_punctuation_suffix_does_not_add_space_to_prompt(self):
        state = self._state()
        self.assertEqual(
            state.update("hello world one two three four five"),
            "hello world",
        )

        delta = state.update("hello world, one two three four five")

        self.assertEqual(delta, ",")
        self.assertEqual(state.emitted_text, "hello world,")

    def test_normal_append_keeps_word_separator(self):
        state = self._state()
        self.assertEqual(
            state.update("hello one two three four five"),
            "hello",
        )

        delta = state.update("hello world one two three four five")

        self.assertEqual(delta, "world")
        self.assertEqual(state.emitted_text, "hello world")

    def test_temporary_prefix_shrink_does_not_repeat_words(self):
        state = self._state()
        self.assertEqual(
            state.update("alpha beta gamma one two three four five"),
            "alpha beta gamma",
        )
        self.assertEqual(
            state.update("alpha beta one two three four five"),
            "",
        )

        delta = state.update("alpha beta gamma delta one two three four five")

        self.assertEqual(delta, "delta")
        self.assertEqual(state.emitted_text, "alpha beta gamma delta")


class TestChunkedStreamingASRSSE(CustomTestCase):
    def _stream_deltas(self, transcripts: List[str]) -> List[str]:
        tokenizer_manager = _ScriptedQwen3ASRTokenizerManager(transcripts)
        serving = OpenAIServingTranscription(tokenizer_manager)
        request = TranscriptionRequest(
            model="Qwen/Qwen3-ASR-0.6B",
            stream=True,
            audio_data=b"mock audio",
        )
        adapted_request = GenerateReqInput(
            text="",
            sampling_params={},
            modalities=["audio"],
        )
        raw_request = Mock(headers={})
        raw_request.is_disconnected = AsyncMock(return_value=False)

        async def drive_stream():
            frames = []
            async for frame in serving._generate_chunked_asr_stream(
                adapted_request, request, raw_request
            ):
                frames.append(frame)
            return frames

        chunks = [f"chunk-{i}".encode() for i in range(len(transcripts))]
        with patch(
            "sglang.srt.entrypoints.openai.serving_transcription.split_audio_chunks",
            return_value=chunks,
        ):
            frames = get_or_create_event_loop().run_until_complete(drive_stream())
        return _deltas_from_sse(frames)

    def test_punctuation_revision_reaches_client_without_extra_space(self):
        deltas = self._stream_deltas(
            [
                "hello world one two three four five",
                "hello world, one two three four five",
                "hello world, one two three four five",
            ]
        )

        self.assertEqual(
            deltas,
            ["hello", " world", ",", " one", " two", " three", " four", " five"],
        )
        self.assertEqual("".join(deltas), "hello world, one two three four five")

    def test_word_extension_reaches_client_as_a_complete_word(self):
        deltas = self._stream_deltas(
            [
                "the cat one two three four five",
                "the caterpillar one two three four five",
                "the caterpillar one two three four five",
            ]
        )

        self.assertEqual(deltas[2], " caterpillar")
        self.assertNotIn("erpillar", [delta.strip() for delta in deltas])


if __name__ == "__main__":
    import unittest

    unittest.main()
