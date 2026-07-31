"""Focused CPU tests for realtime ASR transcript and request behavior."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np

from sglang.srt.entrypoints.openai.realtime.asr_processor import (
    RealtimeASRProcessor,
)
from sglang.srt.entrypoints.openai.realtime.session import RealtimeConnection
from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
    process_asr_chunk,
)
from sglang.srt.multimodal.audio_encoder_windowing import AudioEncoderWindowConfig
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_AUDIO = np.zeros(1600, dtype=np.float32)
_ENCODER_WINDOW_GEOMETRY = AudioEncoderWindowConfig(
    min_input_samples=4, window_samples=16, window_tokens=8
)
_ENCODER_WINDOW_POLICY = {
    "decoder_prefix_max_tokens": 3,
    "decoder_prefix_holdback_words": 1,
    "max_audio_context_windows": 2,
}


def _adapter(
    *,
    min_audio_sec=0.0,
    encoder_window_config=None,
    unfixed_token_num=1,
):
    return SimpleNamespace(
        prompt_template="PROMPT:",
        model_sample_rate=1,
        postprocess_text=lambda text: text,
        realtime_encoder_window_config=(
            {"min_audio_sec": min_audio_sec} | (encoder_window_config or {})
        ),
        chunked_streaming_config={
            "chunk_size_sec": 2.0,
            "unfixed_chunk_num": 2,
            "unfixed_token_num": unfixed_token_num,
        },
    )


class _MockTokenizerManager:
    def __init__(self, transcripts=None, *, supports_encoder_windows=False, fail=False):
        self._transcripts = (
            list(transcripts) if isinstance(transcripts, list) else [transcripts]
        )
        self._fail = fail
        self.requests = []
        self.processor_kwargs = []
        self.tokenizer = Mock(
            encode=lambda text, add_special_tokens=False: text.split(),
            decode=lambda token_ids, **kwargs: " ".join(token_ids),
        )
        if supports_encoder_windows:
            self.mm_processor = Mock(
                resolve_audio_encoder_window_config=lambda *args: (
                    _ENCODER_WINDOW_GEOMETRY
                )
            )

    def generate_request(self, adapted_request, raw_request=None, **kwargs):
        self.requests.append(adapted_request)
        self.processor_kwargs.append(kwargs.get("internal_mm_processor_kwargs"))
        transcript = self._transcripts.pop(0)

        async def gen():
            if self._fail:
                raise ValueError("synthetic failure")
            if transcript is not None:
                yield {"text": transcript}

        return gen()


def _run(coro):
    return get_or_create_event_loop().run_until_complete(coro)


def _server_args(max_buffer_seconds=60):
    return SimpleNamespace(asr_max_buffer_seconds=max_buffer_seconds)


def _state(**kwargs):
    config = dict(chunk_size_sec=1.0, unfixed_chunk_num=2, unfixed_token_num=2)
    config.update(kwargs)
    return StreamingASRState(**config)


def _chunk(state, transcript, *, is_last=False):
    tokenizer_manager = _MockTokenizerManager(transcript)
    delta = _run(
        process_asr_chunk(
            tokenizer_manager=tokenizer_manager,
            adapter=_adapter(),
            state=state,
            audio_data=_AUDIO,
            sampling_params={},
            is_last=is_last,
        )
    )
    return tokenizer_manager, delta


def _encoder_window_processor(transcripts, *, min_audio_sec=0.0):
    tokenizer_manager = _MockTokenizerManager(
        transcripts, supports_encoder_windows=True
    )
    processor = RealtimeASRProcessor(
        tokenizer_manager,
        _adapter(
            min_audio_sec=min_audio_sec,
            encoder_window_config=_ENCODER_WINDOW_POLICY,
        ),
        _server_args(120),
    )
    return tokenizer_manager, processor, processor.create_state()


class TestStreamingASR(CustomTestCase):
    def test_cumulative_prompt_and_word_reconciliation(self):
        state = _state(emitted_text="hello", decode_count=5)
        tokenizer_manager, _ = _chunk(state, "hello world foo")
        self.assertEqual(tokenizer_manager.requests[0].text, "PROMPT:hello")

        state = _state(
            unfixed_chunk_num=0,
            unfixed_token_num=1,
            confirmed_text="hello world",
        )
        self.assertEqual(_chunk(state, "hello worldly test tail")[1], "worldly test")

        state = _state(confirmed_text="hello world", emitted_text="hello world")
        self.assertEqual(_chunk(state, "Hello world again", is_last=True)[1], "again")

    def test_cumulative_cjk_reconciliation(self):
        state = _state(unfixed_token_num=1)
        self.assertEqual(_chunk(state, "你好世界")[1], "你好世")
        _chunk(state, "你好世界好")
        self.assertEqual(state.emitted_text, "你好世界")

        repeated = "你好世界" * 10
        self.assertEqual(
            state.reconcile_cumulative_transcript(repeated, is_last=False), ""
        )
        state.latest_text = repeated
        state.flush_cumulative_transcript()
        self.assertEqual(state.emitted_text, repeated)

        state = _state(emitted_text="你好，世界。你好")
        state.latest_text = "你好。世界。你好世界"
        self.assertEqual(state.flush_cumulative_transcript(), "世界")

    def test_encoder_window_path_emits_only_the_agreed_suffix(self):
        tokenizer_manager, processor, state = _encoder_window_processor(
            ["three four five", "four five six", "five six seven"]
        )
        state.transcript.emitted_text = "one two three"
        sampling_params = {"max_new_tokens": 256}

        state.audio.append_pcm(b"\x01\x00\x02\x00")
        self.assertEqual(
            _run(
                processor.process(state, is_last=False, sampling_params=sampling_params)
            ),
            "",
        )
        self.assertEqual(tokenizer_manager.requests[0].text, "PROMPT:one two three")
        self.assertEqual(
            tokenizer_manager.requests[0].sampling_params["max_new_tokens"], 32
        )
        self.assertEqual(
            tokenizer_manager.processor_kwargs[0]["audio_encoder_window_config"],
            _ENCODER_WINDOW_GEOMETRY,
        )

        state.audio.append_pcm(b"\x03\x00\x04\x00")
        self.assertEqual(
            _run(
                processor.process(state, is_last=False, sampling_params=sampling_params)
            ),
            "four",
        )

        state.audio.append_pcm(b"\x05\x00\x06\x00")
        self.assertEqual(
            _run(
                processor.process(state, is_last=True, sampling_params=sampling_params)
            ),
            "five six seven",
        )
        self.assertEqual(
            state.transcript.emitted_text, "one two three four five six seven"
        )
        self.assertEqual(
            tokenizer_manager.requests[-1].sampling_params["max_new_tokens"], 256
        )

    def test_encoder_window_mode_starts_only_after_the_audio_gate(self):
        tokenizer_manager, processor, state = _encoder_window_processor(
            "alpha beta", min_audio_sec=60.0
        )
        state.audio.append_pcm(bytes(120))
        _run(processor.process(state, is_last=False, sampling_params={}))
        self.assertIsNone(
            tokenizer_manager.processor_kwargs[0]["audio_encoder_window_config"]
        )

        tokenizer_manager, processor, state = _encoder_window_processor(
            "four five", min_audio_sec=60.0
        )
        state.transcript.emitted_text = "one two three"
        state.audio.append_pcm(bytes(122))
        _run(processor.process(state, is_last=False, sampling_params={}))
        self.assertEqual(
            tokenizer_manager.processor_kwargs[0]["audio_encoder_window_config"],
            _ENCODER_WINDOW_GEOMETRY,
        )

    def test_encoder_window_mode_compacts_on_encoder_boundaries(self):
        tokenizer_manager, processor, state = _encoder_window_processor(
            ["three four five", "four five six"], min_audio_sec=60.0
        )
        state.transcript.emitted_text = "one two three"

        state.audio.append_pcm(bytes(122))
        _run(processor.process(state, is_last=False, sampling_params={}))
        state.audio.append_pcm(bytes(78))
        _run(processor.process(state, is_last=False, sampling_params={}))

        self.assertEqual(tokenizer_manager.requests[0].audio_data.size, 61)
        self.assertEqual(tokenizer_manager.requests[1].audio_data.size, 52)
        self.assertEqual(state.audio.base_offset_bytes, 96)
        self.assertEqual(
            state.audio.base_offset_bytes + len(state.audio.data),
            state.audio.received_bytes,
        )

    def test_encoder_window_cjk_retries_cumulatively(self):
        tokenizer_manager, processor, state = _encoder_window_processor(
            ["你好世界", "你好世界"]
        )
        state.audio.append_pcm(b"\x01\x00\x02\x00")

        delta = _run(processor.process(state, is_last=False, sampling_params={}))

        self.assertEqual(delta, "你好世")
        self.assertTrue(state.encoder_window_disabled)
        self.assertFalse(state.encoder_window_active)
        self.assertEqual(len(tokenizer_manager.requests), 2)
        self.assertEqual(
            tokenizer_manager.processor_kwargs[0]["audio_encoder_window_config"],
            _ENCODER_WINDOW_GEOMETRY,
        )
        self.assertIsNone(
            tokenizer_manager.processor_kwargs[1]["audio_encoder_window_config"]
        )

    def test_unconfigured_adapter_stays_cumulative(self):
        tokenizer_manager = _MockTokenizerManager("hello world")
        processor = RealtimeASRProcessor(tokenizer_manager, _adapter(), _server_args())
        state = processor.create_state()
        state.audio.append_pcm(b"\x01\x00\x02\x00")

        _run(processor.process(state, is_last=False, sampling_params={}))

        self.assertEqual(tokenizer_manager.requests[0].text, "PROMPT:")
        self.assertIsNone(
            tokenizer_manager.processor_kwargs[0]["audio_encoder_window_config"]
        )

    def test_failed_inference_keeps_audio_and_reset_clears_offsets(self):
        tokenizer_manager = _MockTokenizerManager(fail=True)
        websocket = Mock(send_text=AsyncMock(), close=AsyncMock())
        connection = RealtimeConnection(
            websocket, tokenizer_manager, _adapter(), _server_args()
        )
        connection.config.sampling_params = {}
        connection.asr_state.audio.append_pcm(bytes(range(12)))
        connection.asr_state.audio.last_attempted_offset_bytes = 8
        connection.asr_state.audio.last_processed_offset_bytes = 8

        with self.assertLogs(level="WARNING"):
            self.assertFalse(_run(connection._run_inference(is_last=False)))

        self.assertEqual(bytes(connection.asr_state.audio.data), bytes(range(12)))
        self.assertEqual(connection.asr_state.audio.last_processed_offset_bytes, 8)
        websocket.close.assert_awaited_with(code=1011)

        connection._reset_inference_state()
        self.assertEqual(connection.asr_state.audio.data, bytearray())
        self.assertEqual(connection.asr_state.audio.received_bytes, 0)
        self.assertEqual(connection.asr_state.audio.last_attempted_offset_bytes, 0)
        self.assertEqual(connection.asr_state.audio.last_processed_offset_bytes, 0)


if __name__ == "__main__":
    unittest.main()
