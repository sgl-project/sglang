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
from sglang.srt.entrypoints.openai.realtime.protocol import SessionUpdateEvent
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
        build_sampling_params=lambda request: {"language": request.language},
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
    def __init__(
        self,
        transcripts=None,
        *,
        supports_encoder_windows=False,
        fail=False,
        finish_reasons=None,
    ):
        self._transcripts = (
            list(transcripts) if isinstance(transcripts, list) else [transcripts]
        )
        self._finish_reasons = (
            list(finish_reasons)
            if finish_reasons is not None
            else ["stop"] * len(self._transcripts)
        )
        self._fail = fail
        self.served_model_name = "qwen3-asr"
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
                yield {
                    "text": transcript,
                    "meta_info": {
                        "finish_reason": {"type": self._finish_reasons.pop(0)}
                    },
                }

        return gen()


def _run(coro):
    return get_or_create_event_loop().run_until_complete(coro)


def _server_args(max_buffer_seconds=60, long_audio_strategy="encoder_window"):
    return SimpleNamespace(
        asr_max_buffer_seconds=max_buffer_seconds,
        asr_long_audio_strategy=long_audio_strategy,
        tp_size=1,
        dp_size=1,
        pp_size=1,
        nnodes=1,
        language_only=False,
        disaggregation_mode="null",
    )


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


def _realtime_connection(transcripts, finish_reasons, *, encoder_windows=True):
    tokenizer_manager = _MockTokenizerManager(
        transcripts,
        supports_encoder_windows=encoder_windows,
        finish_reasons=finish_reasons,
    )
    connection = RealtimeConnection(
        Mock(send_text=AsyncMock(), close=AsyncMock()),
        tokenizer_manager,
        _adapter(
            encoder_window_config=(_ENCODER_WINDOW_POLICY if encoder_windows else None)
        ),
        _server_args(
            120,
            long_audio_strategy=("encoder_window" if encoder_windows else "cumulative"),
        ),
    )
    connection.config.configured = True
    connection.config.sampling_params = {"max_new_tokens": 256}
    connection._send = AsyncMock()
    if encoder_windows:
        connection.asr_state.transcript.emitted_text = "one two three"
    connection.asr_state.audio.append_pcm(b"\x01\x00\x02\x00")
    return tokenizer_manager, connection


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

        echo_state = _state(chunk_size_sec=2.0)
        emitted_words = [f"w{i}" for i in range(50)]
        echo_state.emitted_text = " ".join(emitted_words)
        decoder_prefix = " ".join(emitted_words[-35:])
        self.assertEqual(
            echo_state.trim_decoder_prefix_echo(decoder_prefix, decoder_prefix), ""
        )
        self.assertEqual(
            echo_state.trim_decoder_prefix_echo("yeah yeah", "yeah yeah"),
            "yeah yeah",
        )
        self.assertEqual(
            echo_state.trim_decoder_prefix_echo(
                "one two three four five",
                "one two three four",
                trim_short_prefix=True,
            ),
            "five",
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

        thai = "ฉันมีความฝันว่าวันหนึ่งประเทศนี้จะลุกขึ้น"
        tokenizer_manager, processor, state = _encoder_window_processor([thai, thai])
        state.audio.append_pcm(b"\x01\x00\x02\x00")

        _run(processor.process(state, is_last=False, sampling_params={}))

        self.assertTrue(state.encoder_window_disabled)
        self.assertFalse(state.encoder_window_active)
        self.assertEqual(len(tokenizer_manager.requests), 2)

    def test_length_limited_decode_replays_with_full_budget(self):
        tokenizer_manager, connection = _realtime_connection(
            ["three four", "three four five"],
            ["length", "stop"],
        )

        self.assertTrue(_run(connection._run_inference(is_last=False)))
        self.assertFalse(connection.asr_state.has_new_audio)
        _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))

        self.assertEqual(len(tokenizer_manager.requests), 2)
        self.assertEqual(
            tokenizer_manager.requests[-1].sampling_params["max_new_tokens"], 256
        )

        tokenizer_manager, connection = _realtime_connection("three four", ["stop"])

        self.assertTrue(_run(connection._run_inference(is_last=False)))
        _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))

        self.assertEqual(len(tokenizer_manager.requests), 1)

        tokenizer_manager, connection = _realtime_connection(
            ["one two three", "", "three four five", "four five six"],
            ["length", "stop", "stop", "stop"],
        )
        connection.asr_state.audio.append_pcm(bytes(96))

        self.assertTrue(_run(connection._run_inference(is_last=False)))
        self.assertTrue(connection.asr_state.final_replay_required)
        connection.asr_state.audio.append_pcm(bytes(4))
        self.assertTrue(_run(connection._run_inference(is_last=False)))
        self.assertTrue(connection.asr_state.final_replay_required)
        self.assertEqual(connection.asr_state.audio.base_offset_bytes, 0)
        self.assertEqual(
            tokenizer_manager.requests[-1].sampling_params["max_new_tokens"], 256
        )

        connection.asr_state.audio.append_pcm(bytes(4))
        self.assertTrue(_run(connection._run_inference(is_last=False)))
        self.assertFalse(connection.asr_state.final_replay_required)
        self.assertEqual(connection.asr_state.audio.base_offset_bytes, 32)
        self.assertEqual(
            tokenizer_manager.requests[-1].sampling_params["max_new_tokens"], 256
        )

        connection.asr_state.audio.append_pcm(bytes(4))
        self.assertTrue(_run(connection._run_inference(is_last=False)))
        self.assertEqual(tokenizer_manager.requests[-1].audio_data.size, 40)
        self.assertEqual(
            tokenizer_manager.requests[-1].sampling_params["max_new_tokens"], 36
        )

        _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))
        self.assertEqual(len(tokenizer_manager.requests), 4)

        tokenizer_manager, connection = _realtime_connection(
            ["three four", "three four five"],
            ["length", "length"],
        )
        self.assertTrue(_run(connection._run_inference(is_last=False)))
        _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))
        self.assertEqual(len(tokenizer_manager.requests), 2)
        sent_event_types = [
            call.args[0].type for call in connection._send.call_args_list
        ]
        self.assertIn(
            "conversation.item.input_audio_transcription.failed", sent_event_types
        )
        self.assertNotIn(
            "conversation.item.input_audio_transcription.completed", sent_event_types
        )

        tokenizer_manager, connection = _realtime_connection(
            ["hello world", "hello world tail"],
            ["length", "stop"],
            encoder_windows=False,
        )
        self.assertTrue(_run(connection._run_inference(is_last=False)))
        self.assertTrue(connection.asr_state.final_replay_required)
        _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))
        self.assertEqual(len(tokenizer_manager.requests), 2)
        sent_event_types = [
            call.args[0].type for call in connection._send.call_args_list
        ]
        self.assertIn(
            "conversation.item.input_audio_transcription.completed", sent_event_types
        )

        tokenizer_manager, connection = _realtime_connection(
            ["hello world", "hello world tail"],
            ["length", "length"],
            encoder_windows=False,
        )
        self.assertTrue(_run(connection._run_inference(is_last=False)))
        _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))
        self.assertEqual(len(tokenizer_manager.requests), 2)
        sent_event_types = [
            call.args[0].type for call in connection._send.call_args_list
        ]
        self.assertIn(
            "conversation.item.input_audio_transcription.failed", sent_event_types
        )
        self.assertNotIn(
            "conversation.item.input_audio_transcription.completed", sent_event_types
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

        tokenizer_manager = _MockTokenizerManager(
            "hello world", supports_encoder_windows=True
        )
        processor = RealtimeASRProcessor(
            tokenizer_manager,
            _adapter(encoder_window_config=_ENCODER_WINDOW_POLICY),
            _server_args(120, long_audio_strategy="cumulative"),
        )
        state = processor.create_state()
        state.audio.append_pcm(b"\x01\x00\x02\x00")
        _run(processor.process(state, is_last=False, sampling_params={}))
        self.assertIsNone(
            tokenizer_manager.processor_kwargs[0]["audio_encoder_window_config"]
        )

        tokenizer_manager = _MockTokenizerManager(
            "hello world", supports_encoder_windows=True
        )
        server_args = _server_args(120)
        server_args.dp_size = 2
        with self.assertLogs(level="WARNING"):
            processor = RealtimeASRProcessor(
                tokenizer_manager,
                _adapter(encoder_window_config=_ENCODER_WINDOW_POLICY),
                server_args,
            )
        state = processor.create_state()
        state.audio.append_pcm(b"\x01\x00\x02\x00")
        _run(processor.process(state, is_last=False, sampling_params={}))
        self.assertIsNone(
            tokenizer_manager.processor_kwargs[0]["audio_encoder_window_config"]
        )

    def test_session_update_preserves_omitted_nested_fields(self):
        connection = RealtimeConnection(
            Mock(send_text=AsyncMock(), close=AsyncMock()),
            _MockTokenizerManager("unused"),
            _adapter(),
            _server_args(),
        )
        connection._send = AsyncMock()

        def update(audio_input):
            event = SessionUpdateEvent.model_validate(
                {
                    "type": "session.update",
                    "session": {
                        "type": "transcription",
                        "audio": {"input": audio_input},
                    },
                }
            )
            _run(connection._on_session_update(event))

        update(
            {
                "format": {"type": "audio/pcm", "rate": 16000},
                "transcription": {"model": "qwen3-asr", "language": "en"},
            }
        )
        update({"transcription": {"language": "es"}})

        self.assertEqual(connection.config.input_sample_rate, 16000)
        self.assertEqual(connection.config.client_model, "qwen3-asr")
        self.assertEqual(connection.config.language, "es")
        self.assertEqual(connection.config.sampling_params, {"language": "es"})

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

        voiced_pcm = (1000).to_bytes(2, "little", signed=True) * 2
        _, processor, state = _encoder_window_processor(["", ""])
        state.audio.append_pcm(voiced_pcm)
        self.assertEqual(
            _run(processor.process(state, is_last=False, sampling_params={})), ""
        )
        self.assertEqual(state.audio.last_processed_offset_bytes, 0)

        state.audio.append_pcm(voiced_pcm * 8)
        self.assertEqual(
            _run(processor.process(state, is_last=False, sampling_params={})), ""
        )
        self.assertEqual(
            state.audio.last_processed_offset_bytes, state.audio.received_bytes
        )
        self.assertTrue(state.encoder_window_active)

        _, processor, state = _encoder_window_processor("")
        state.audio.append_pcm(voiced_pcm)
        state.encoder_window_active = True
        self.assertEqual(
            _run(processor.process(state, is_last=True, sampling_params={})), ""
        )
        self.assertEqual(
            state.audio.last_processed_offset_bytes, state.audio.received_bytes
        )


if __name__ == "__main__":
    unittest.main()
