"""Focused CPU tests for realtime ASR transcript and request behavior."""

import base64

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np

from sglang.srt.entrypoints.openai.realtime.asr_processor import (
    RealtimeASRProcessor,
)
from sglang.srt.entrypoints.openai.realtime.decoder_suffix import (
    DecoderSuffixState,
)
from sglang.srt.entrypoints.openai.realtime.protocol import SessionUpdateEvent
from sglang.srt.entrypoints.openai.realtime.session import RealtimeConnection
from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
    hash_audio_content,
    process_asr_chunk,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    RealtimeEncoderWindowPolicy,
)
from sglang.srt.multimodal.audio_encoder_windowing import AudioEncoderWindowConfig
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_AUDIO = np.zeros(1600, dtype=np.float32)
_ENCODER_WINDOW_GEOMETRY = AudioEncoderWindowConfig(
    min_input_samples=4,
    window_samples=16,
    window_frames=8,
    window_tokens=8,
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
    chunk_size_sec=2.0,
    unfixed_token_num=1,
    supported_languages=("en",),
):
    window_config = None
    if encoder_window_config is not None:
        window_config = RealtimeEncoderWindowPolicy(
            min_audio_sec=min_audio_sec,
            max_audio_context_windows=encoder_window_config[
                "max_audio_context_windows"
            ],
            decoder_prefix_max_tokens=encoder_window_config.get(
                "decoder_prefix_max_tokens", 192
            ),
            decoder_prefix_holdback_words=encoder_window_config.get(
                "decoder_prefix_holdback_words", 1
            ),
            supported_languages=supported_languages,
        )
    return SimpleNamespace(
        prompt_template="PROMPT:",
        model_sample_rate=1,
        postprocess_text=lambda text: text,
        build_sampling_params=lambda request: {"language": request.language},
        realtime_encoder_window_policy=window_config,
        chunked_streaming_config={
            "chunk_size_sec": chunk_size_sec,
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
        config = kwargs.get("audio_encoder_window_config")
        self.processor_kwargs.append(
            {"audio_encoder_window_config": config} if config is not None else None
        )
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


def _process(processor, state, *, is_last=False, language="en", sampling_params=None):
    return _run(
        processor.process(
            state,
            is_last=is_last,
            language=language,
            sampling_params=sampling_params or {},
        )
    )


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


class TestRealtimeASRProcessorConfig(CustomTestCase):
    def test_rejects_invalid_encoder_window_policy(self):
        invalid_values = (
            {"min_audio_sec": float("inf")},
            {"max_audio_context_windows": 0},
            {"supported_languages": ()},
        )
        defaults = {
            "min_audio_sec": 60.0,
            "max_audio_context_windows": 8,
            "supported_languages": ("en",),
        }
        for override in invalid_values:
            with self.subTest(override=override), self.assertRaises(ValueError):
                RealtimeEncoderWindowPolicy(**(defaults | override))

    def test_rejects_invalid_chunk_size(self):
        tokenizer_manager = _MockTokenizerManager("text")
        for chunk_size_sec in (0.0, -1.0, float("inf"), 1e-20):
            with self.subTest(chunk_size_sec=chunk_size_sec):
                with self.assertRaises(ValueError):
                    RealtimeASRProcessor(
                        tokenizer_manager,
                        _adapter(chunk_size_sec=chunk_size_sec),
                        _server_args(),
                    )


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
    connection.config.language = "en"
    connection.config.sampling_params = {"max_new_tokens": 256}
    connection._send = AsyncMock()
    if encoder_windows:
        connection.asr_state.transcript.emitted_text = "one two three"
    connection.asr_state.audio.append_pcm(b"\x01\x00\x02\x00")
    return tokenizer_manager, connection


class TestStreamingASR(CustomTestCase):
    def test_cumulative_prompt_and_word_reconciliation(self):
        state = _state(emitted_text="hello", chunk_index=5)
        tokenizer_manager, _ = _chunk(state, "hello world foo")
        self.assertEqual(tokenizer_manager.requests[0].text, "PROMPT:hello")
        self.assertEqual(
            tokenizer_manager.requests[0].mm_hashes,
            [hash_audio_content(_AUDIO)],
        )

        state = _state(
            unfixed_chunk_num=0,
            unfixed_token_num=1,
            confirmed_text="hello world",
        )
        self.assertEqual(_chunk(state, "hello world test tail")[1], "test")

    def test_encoder_window_path_emits_only_the_agreed_suffix(self):
        tokenizer_manager, processor, state = _encoder_window_processor(
            ["four five", "four five six", "five six seven"]
        )
        state.transcript.emitted_text = "one two three"
        sampling_params = {"max_new_tokens": 256}

        state.audio.append_pcm(b"\x01\x00\x02\x00")
        self.assertEqual(
            _process(processor, state, sampling_params=sampling_params),
            "",
        )
        self.assertEqual(tokenizer_manager.requests[0].text, "PROMPT:one two three")
        self.assertEqual(
            tokenizer_manager.requests[0].sampling_params["max_new_tokens"], 256
        )
        state.audio.append_pcm(b"\x03\x00\x04\x00")
        self.assertEqual(
            _process(processor, state, sampling_params=sampling_params),
            "four",
        )

        state.audio.append_pcm(b"\x05\x00\x06\x00")
        self.assertEqual(
            _process(processor, state, is_last=True, sampling_params=sampling_params),
            "five six seven",
        )
        self.assertEqual(
            state.decoder_suffix.emitted_text,
            "one two three four five six seven",
        )
        self.assertEqual(
            tokenizer_manager.requests[-1].sampling_params["max_new_tokens"], 256
        )

        emitted_words = [f"w{i}" for i in range(50)]
        echo_state = DecoderSuffixState(emitted_text=" ".join(emitted_words))
        decoder_prefix = " ".join(emitted_words[-35:])
        self.assertEqual(
            echo_state.trim_prefix_echo(decoder_prefix, decoder_prefix), ("", True)
        )
        self.assertEqual(
            echo_state.trim_prefix_echo("yeah yeah", "yeah yeah"),
            ("yeah yeah", False),
        )
        self.assertEqual(
            echo_state.trim_prefix_echo(
                "one two three four five",
                "one two three four",
                trim_short_prefix=True,
            ),
            ("five", True),
        )

    def test_encoder_window_handoff_echo_is_not_reemitted(self):
        tokenizer_manager, processor, state = _encoder_window_processor("C D E")
        state.transcript.emitted_text = "A B"
        state.transcript.confirmed_text = "A B"
        state.transcript.full_transcript = "A B C D"
        state.audio.append_pcm(b"\x01\x00\x02\x00")

        self.assertEqual(
            _process(processor, state),
            "",
        )
        self.assertEqual(tokenizer_manager.requests[0].text, "PROMPT:B C D")
        self.assertEqual(state.decoder_suffix.pending_suffix, "C D E")
        self.assertEqual(processor.flush_pending_transcript(state), "C D E")

    def test_encoder_window_preserves_a_real_repetition(self):
        tokenizer_manager, processor, state = _encoder_window_processor("B C D C D E")
        state.transcript.emitted_text = "A B"
        state.transcript.confirmed_text = "A B"
        state.transcript.full_transcript = "A B C D"
        state.audio.append_pcm(b"\x01\x00\x02\x00")

        self.assertEqual(
            _process(processor, state),
            "",
        )
        self.assertEqual(tokenizer_manager.requests[0].text, "PROMPT:B C D")
        self.assertEqual(state.decoder_suffix.pending_suffix, "C D C D E")
        self.assertEqual(processor.flush_pending_transcript(state), "C D C D E")

        final_update = DecoderSuffixState(emitted_text="I agree").reconcile(
            "I agree with that", is_last=True, holdback_words=1
        )
        self.assertEqual(final_update.delta, "I agree with that")

    def test_encoder_window_mode_starts_only_after_the_audio_gate(self):
        tokenizer_manager, processor, state = _encoder_window_processor(
            "alpha beta", min_audio_sec=60.0
        )
        state.audio.append_pcm(bytes(120))
        _process(processor, state)
        self.assertIsNone(tokenizer_manager.processor_kwargs[0])

        tokenizer_manager, processor, state = _encoder_window_processor(
            "four five", min_audio_sec=60.0
        )
        state.transcript.emitted_text = "one two three"
        state.audio.append_pcm(bytes(122))
        state.audio.last_attempted_offset_bytes = 120
        state.audio.last_processed_offset_bytes = 120
        _process(processor, state)
        self.assertEqual(
            tokenizer_manager.processor_kwargs[0]["audio_encoder_window_config"],
            _ENCODER_WINDOW_GEOMETRY,
        )

    def test_encoder_window_mode_requires_an_explicit_supported_language(self):
        for language in (None, "auto", "zh"):
            with self.subTest(language=language):
                tokenizer_manager, processor, state = _encoder_window_processor(
                    "alpha beta"
                )
                state.audio.append_pcm(b"\x01\x00\x02\x00")
                _process(processor, state, language=language)
                self.assertIsNone(tokenizer_manager.processor_kwargs[0])
                self.assertFalse(state.encoder_window_active)

        tokenizer_manager, processor, state = _encoder_window_processor("alpha beta")
        state.audio.append_pcm(b"\x01\x00\x02\x00")
        _process(processor, state, language="en-US")
        self.assertEqual(
            tokenizer_manager.processor_kwargs[0]["audio_encoder_window_config"],
            _ENCODER_WINDOW_GEOMETRY,
        )

    def test_unsupported_language_keeps_the_original_item_limit(self):
        _, processor, state = _encoder_window_processor(
            "alpha beta", min_audio_sec=60.0
        )

        self.assertEqual(processor.max_item_bytes(state, None), 120)
        self.assertEqual(processor.max_item_bytes(state, "en"), 240)

        state.decoder_suffix = DecoderSuffixState(emitted_text="alpha")
        state.audio.append_pcm(b"\x01\x00")
        with self.assertRaisesRegex(RuntimeError, "language cannot change"):
            _process(processor, state, language="zh")

    def test_large_append_is_drained_on_chunk_boundaries(self):
        tokenizer_manager = _MockTokenizerManager(
            ["one"] * 45,
            supports_encoder_windows=True,
        )
        connection = RealtimeConnection(
            Mock(send_text=AsyncMock(), close=AsyncMock()),
            tokenizer_manager,
            _adapter(
                min_audio_sec=60.0,
                encoder_window_config=_ENCODER_WINDOW_POLICY,
            ),
            _server_args(120),
        )
        connection.config.configured = True
        connection.config.input_sample_rate = 1
        connection.config.language = "en"
        connection.config.sampling_params = {"max_new_tokens": 256}
        connection._send = AsyncMock()

        pcm = bytes(90 * 2)
        closed = _run(
            connection._on_input_audio_buffer_append(
                SimpleNamespace(audio=base64.b64encode(pcm).decode())
            )
        )

        self.assertFalse(closed)
        self.assertEqual(len(tokenizer_manager.requests), 45)
        self.assertEqual(
            [request.audio_data.size for request in tokenizer_manager.requests[:3]],
            [2, 4, 6],
        )
        self.assertLessEqual(
            max(request.audio_data.size for request in tokenizer_manager.requests),
            64,
        )
        self.assertEqual(
            connection.asr_state.audio.last_attempted_offset_bytes,
            len(pcm),
        )

    def test_encoder_window_mode_compacts_on_encoder_boundaries(self):
        tokenizer_manager, processor, state = _encoder_window_processor(
            ["three four five", "four five six"], min_audio_sec=60.0
        )
        state.transcript.emitted_text = "one two three"

        state.audio.append_pcm(bytes(122))
        state.audio.last_attempted_offset_bytes = 120
        state.audio.last_processed_offset_bytes = 120
        _process(processor, state)
        state.audio.append_pcm(bytes(4))
        _process(processor, state)

        self.assertEqual(tokenizer_manager.requests[0].audio_data.size, 61)
        self.assertEqual(tokenizer_manager.requests[1].audio_data.size, 47)
        self.assertEqual(state.audio.base_offset_bytes, 32)
        self.assertEqual(
            state.audio.base_offset_bytes + len(state.audio.data),
            state.audio.received_bytes,
        )

    def test_encoder_window_cjk_retries_cumulatively(self):
        tokenizer_manager, processor, state = _encoder_window_processor(
            ["你好世界", "你好世界"]
        )
        state.audio.append_pcm(b"\x01\x00\x02\x00")

        delta = _process(processor, state)

        self.assertEqual(delta, "")
        self.assertTrue(state.encoder_window_disabled)
        self.assertFalse(state.encoder_window_active)
        self.assertEqual(len(tokenizer_manager.requests), 2)
        self.assertEqual(
            tokenizer_manager.processor_kwargs[0]["audio_encoder_window_config"],
            _ENCODER_WINDOW_GEOMETRY,
        )
        self.assertIsNone(tokenizer_manager.processor_kwargs[1])

        thai = "ฉันมีความฝันว่าวันหนึ่งประเทศนี้จะลุกขึ้น"
        tokenizer_manager, processor, state = _encoder_window_processor([thai, thai])
        state.audio.append_pcm(b"\x01\x00\x02\x00")

        _process(processor, state)

        self.assertTrue(state.encoder_window_disabled)
        self.assertFalse(state.encoder_window_active)
        self.assertEqual(len(tokenizer_manager.requests), 2)

    def test_length_limited_decode_fails_closed(self):
        for encoder_windows in (True, False):
            _, connection = _realtime_connection(
                "three four",
                ["length"],
                encoder_windows=encoder_windows,
            )
            with self.assertLogs(level="ERROR"):
                self.assertFalse(_run(connection._run_inference(is_last=False)))
            connection.websocket.close.assert_awaited_with(code=1011)

        _, connection = _realtime_connection("three four", ["length"])
        _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))
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

        _process(processor, state, language=None)

        self.assertEqual(tokenizer_manager.requests[0].text, "PROMPT:")
        request = tokenizer_manager.requests[0]
        self.assertEqual(
            request.mm_hashes,
            [hash_audio_content(request.audio_data)],
        )
        self.assertIsNone(tokenizer_manager.processor_kwargs[0])

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
        _process(processor, state, language=None)
        self.assertIsNone(tokenizer_manager.processor_kwargs[0])

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
        _process(processor, state, language=None)
        self.assertIsNone(tokenizer_manager.processor_kwargs[0])

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
        connection.asr_state.decoder_suffix = DecoderSuffixState(emitted_text="hello")

        update({"transcription": {}})
        self.assertEqual(connection.config.input_sample_rate, 16000)
        self.assertEqual(connection.config.client_model, "qwen3-asr")
        self.assertEqual(connection.config.language, "en")

        connection.asr_state.decoder_suffix = None
        update({"transcription": {"language": "es"}})
        self.assertEqual(connection.config.input_sample_rate, 16000)
        self.assertEqual(connection.config.client_model, "qwen3-asr")
        self.assertEqual(connection.config.language, "es")
        self.assertEqual(connection.config.sampling_params, {"language": "es"})

        update({"transcription": None})
        self.assertIsNone(connection.config.client_model)
        self.assertIsNone(connection.config.language)

        update({"format": None})
        self.assertEqual(connection.config.input_sample_rate, 24000)

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

        pcm = (1000).to_bytes(2, "little", signed=True) * 2
        _, processor, state = _encoder_window_processor(["", ""])
        state.audio.append_pcm(pcm)
        self.assertEqual(
            _process(processor, state),
            "",
        )
        self.assertEqual(state.audio.last_processed_offset_bytes, 0)

        state.audio.append_pcm(pcm)
        self.assertEqual(
            _process(processor, state),
            "",
        )
        self.assertEqual(
            state.audio.last_processed_offset_bytes, state.audio.received_bytes
        )
        self.assertTrue(state.encoder_window_active)

        _, processor, state = _encoder_window_processor("")
        state.audio.append_pcm(pcm)
        state.decoder_suffix = DecoderSuffixState(emitted_text="")
        self.assertEqual(
            _process(processor, state, is_last=True),
            "",
        )
        self.assertEqual(
            state.audio.last_processed_offset_bytes, state.audio.received_bytes
        )


if __name__ == "__main__":
    unittest.main()
