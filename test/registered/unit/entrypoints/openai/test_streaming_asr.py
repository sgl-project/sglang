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
    generate_asr_transcript,
    hash_audio_content,
    process_asr_chunk,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    RealtimeEncoderWindowPolicy,
)
from sglang.srt.entrypoints.openai.transcription_adapters.qwen3_asr import (
    Qwen3ASRAdapter,
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
        supports_chunked_streaming=True,
        postprocess_text=lambda text: text,
        postprocess_streaming_text=lambda text: text,
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
                finish_reason = self._finish_reasons.pop(0)
                if not isinstance(finish_reason, dict):
                    finish_reason = {"type": finish_reason}
                yield {
                    "text": transcript,
                    "meta_info": {
                        "finish_reason": finish_reason,
                    },
                }

        return gen()


class _StreamingTokenizerManager(_MockTokenizerManager):
    def __init__(self, chunks, *, incremental=True, supports_encoder_windows=False):
        super().__init__("unused", supports_encoder_windows=supports_encoder_windows)
        self._chunks = chunks
        self.server_args = SimpleNamespace(incremental_streaming_output=incremental)

    def generate_request(self, adapted_request, raw_request=None, **kwargs):
        self.requests.append(adapted_request)
        config = kwargs.get("audio_encoder_window_config")
        self.processor_kwargs.append(
            {"audio_encoder_window_config": config} if config is not None else None
        )

        async def gen():
            for index, text in enumerate(self._chunks):
                finish_reason = "stop" if index == len(self._chunks) - 1 else None
                yield {
                    "text": text,
                    "meta_info": {
                        "finish_reason": (
                            {"type": finish_reason} if finish_reason else None
                        )
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


def _gated_append_connection(transcripts):
    """Connection with the 60s windowing gate whose audio arrives through the
    real append handler (1 Hz input, so seconds == samples)."""
    tokenizer_manager = _MockTokenizerManager(
        transcripts, supports_encoder_windows=True
    )
    websocket = Mock(send_text=AsyncMock(), close=AsyncMock())
    connection = RealtimeConnection(
        websocket,
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
    return tokenizer_manager, websocket, connection


def _append_seconds(connection, seconds):
    return _run(
        connection._on_input_audio_buffer_append(
            SimpleNamespace(audio=base64.b64encode(bytes(seconds * 2)).decode())
        )
    )


class TestStreamingASR(CustomTestCase):
    def test_decoder_stream_rejects_abort_before_publishing_terminal_text(self):
        async def collect(text):
            updates.append(text)

        for status_code in (None, 400, 500, 503):
            with self.subTest(status_code=status_code):
                updates = []
                finish_reason = {
                    "type": "abort",
                    "message": "synthetic backend abort",
                    "status_code": status_code,
                }
                tokenizer_manager = _MockTokenizerManager(
                    "partial transcript",
                    finish_reasons=[finish_reason],
                )

                with self.assertRaisesRegex(RuntimeError, "synthetic backend abort"):
                    _run(
                        generate_asr_transcript(
                            tokenizer_manager=tokenizer_manager,
                            adapter=_adapter(),
                            audio_data=_AUDIO,
                            sampling_params={},
                            prompt="PROMPT:",
                            on_update=collect,
                        )
                    )

                self.assertEqual(updates, [])

    def test_decoder_stream_hides_qwen_prefix_and_reconstructs_text(self):
        tokenizer_manager = _StreamingTokenizerManager(
            ["language en<asr", "_text>Hello", " world"]
        )
        updates = []

        async def collect(text):
            updates.append(text)

        result = _run(
            generate_asr_transcript(
                tokenizer_manager=tokenizer_manager,
                adapter=Qwen3ASRAdapter(),
                audio_data=_AUDIO,
                sampling_params={},
                prompt="PROMPT:",
                on_update=collect,
            )
        )

        self.assertTrue(tokenizer_manager.requests[0].stream)
        self.assertEqual(updates, ["Hello", "Hello world"])
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(result.finish_reason, "stop")

    def test_decoder_stream_appends_deltas_without_duplicates(self):
        tokenizer_manager = _StreamingTokenizerManager(["hello", " world", " again"])
        connection = RealtimeConnection(
            Mock(send_text=AsyncMock(), close=AsyncMock()),
            tokenizer_manager,
            _adapter(),
            _server_args(long_audio_strategy="cumulative"),
        )
        connection.config.configured = True
        connection.config.language = "en"
        connection.config.sampling_params = {"max_new_tokens": 256}
        connection._send = AsyncMock()
        connection.asr_state.audio.append_pcm(b"\x01\x00\x02\x00")

        self.assertTrue(_run(connection._run_inference(is_last=False)))

        deltas = [
            call.args[0]
            for call in connection._send.call_args_list
            if call.args[0].type == "conversation.item.input_audio_transcription.delta"
        ]
        self.assertEqual([event.delta for event in deltas], ["hello", " world"])

        _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))
        completed = [
            call.args[0]
            for call in connection._send.call_args_list
            if call.args[0].type
            == "conversation.item.input_audio_transcription.completed"
        ]
        self.assertEqual(completed[-1].transcript, "hello world again")

    def test_encoder_window_stream_holds_prefix_replay_before_appending(self):
        tokenizer_manager = _StreamingTokenizerManager(
            ["one", " two", " three", " four", " five"],
            supports_encoder_windows=True,
        )
        processor = RealtimeASRProcessor(
            tokenizer_manager,
            _adapter(encoder_window_config=_ENCODER_WINDOW_POLICY),
            _server_args(120),
        )
        state = processor.create_state()
        state.decoder_suffix = DecoderSuffixState(
            emitted_text="one two", pending_suffix="three four"
        )
        state.audio.append_pcm(b"\x01\x00\x02\x00")
        deltas = []

        async def collect(text):
            deltas.append(text)

        remaining = _run(
            processor.process(
                state,
                is_last=False,
                language="en",
                sampling_params={},
                on_transcript_delta=collect,
            )
        )

        self.assertTrue(tokenizer_manager.requests[0].stream)
        self.assertEqual(deltas, ["three"])
        self.assertEqual(remaining, "")
        self.assertEqual(state.decoder_suffix.latest_text, "one two three four five")
        self.assertEqual(
            tokenizer_manager.processor_kwargs[0]["audio_encoder_window_config"],
            _ENCODER_WINDOW_GEOMETRY,
        )

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

        # The final decode supersedes provisional text from an earlier window.
        final_update = DecoderSuffixState(
            emitted_text="one", pending_suffix="two"
        ).reconcile("three", is_last=True, holdback_words=1)
        self.assertEqual(final_update.delta, "three")

    def test_encoder_window_final_echo_leading_into_pending_is_trimmed(self):
        tokenizer_manager, processor, state = _encoder_window_processor(
            "one two three four"
        )
        state.decoder_suffix = DecoderSuffixState(
            emitted_text="one two", pending_suffix="three"
        )
        state.audio.append_pcm(b"\x01\x00\x02\x00")

        self.assertEqual(
            _process(processor, state, is_last=True),
            "three four",
        )
        self.assertEqual(state.decoder_suffix.emitted_text, "one two three four")

        # Anchor-reject branch: the decode matches the prefix but its remainder
        # does not continue pending, so it is genuine speech and must be kept.
        suffix_state = DecoderSuffixState(
            emitted_text="I agree", pending_suffix="I agree with that"
        )
        self.assertEqual(
            suffix_state.trim_pending_prefix_echo("I agree with that", "I agree"),
            ("I agree with that", False),
        )

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

        self.assertEqual(processor.max_item_bytes(None), 120)
        self.assertEqual(processor.max_item_bytes("en"), 240)

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

    def test_large_append_stops_after_dynamic_cumulative_fallback(self):
        """Bug regression: after a mid-drain fallback to cumulative decoding,
        the state-blind cap let the drain loop keep issuing full cumulative
        re-decodes at 64s, 66s, ... up to the raised windowed cap — the
        quadratic growth this feature exists to bound, reachable by declaring
        a supported language and sending no-word-boundary audio. The fallback
        retry still completes; the next growth decode must not start."""
        tokenizer_manager, websocket, connection = _gated_append_connection(
            [""] * 30 + ["你好世界", "one"]
        )

        closed = _append_seconds(connection, 90)

        request_durations = [
            request.audio_data.size for request in tokenizer_manager.requests
        ]
        self.assertTrue(closed)
        self.assertEqual(request_durations[-2:], [62, 62])
        self.assertNotIn(64, request_durations)
        self.assertTrue(connection.asr_state.encoder_window_disabled)
        self.assertEqual(
            connection.asr_processor.max_item_bytes("en", encoder_window_disabled=True),
            120,
        )
        websocket.close.assert_awaited_with(code=1009)
        error_payloads = [call.args[0] for call in websocket.send_text.call_args_list]
        self.assertTrue(
            any("fell back to cumulative" in payload for payload in error_payloads)
        )

    def test_post_fallback_append_is_rejected_at_the_cumulative_cap(self):
        """Bug regression: the receipt-time cap ignored the runtime fallback,
        so a paced client kept the raised windowed cap after the item went
        cumulative and every extra append re-decoded the whole item."""
        tokenizer_manager, websocket, connection = _gated_append_connection(
            [""] * 30 + ["你好世界", "one"]
        )

        self.assertFalse(_append_seconds(connection, 62))
        websocket.close.assert_not_awaited()
        self.assertTrue(connection.asr_state.encoder_window_disabled)
        self.assertEqual(len(tokenizer_manager.requests), 32)

        self.assertTrue(_append_seconds(connection, 2))
        websocket.close.assert_awaited_with(code=1009)
        self.assertEqual(len(tokenizer_manager.requests), 32)

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
        with self.assertRaisesRegex(ValueError, "PCM16 sample-aligned"):
            state.audio.discard_before(state.audio.base_offset_bytes + 1)

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

    def test_first_window_cjk_with_handoff_retries_cumulatively(self):
        """Bug regression: the no-boundary check ran on the joined text, so a
        whitespace-bearing cumulative handoff ("C D") masked a no-boundary
        model continuation and the one-time cumulative retry was skipped,
        committing the one-way windowed transition on unsupported text."""
        tokenizer_manager, processor, state = _encoder_window_processor(
            ["你好世界", "A B C D 你好世界"]
        )
        state.transcript.emitted_text = "A B"
        state.transcript.confirmed_text = "A B"
        state.transcript.full_transcript = "A B C D"
        state.audio.append_pcm(b"\x01\x00\x02\x00")

        _process(processor, state)

        self.assertTrue(state.encoder_window_disabled)
        self.assertFalse(state.encoder_window_active)
        self.assertIsNone(state.decoder_suffix)
        self.assertEqual(len(tokenizer_manager.requests), 2)
        self.assertIsNotNone(tokenizer_manager.processor_kwargs[0])
        self.assertIsNone(tokenizer_manager.processor_kwargs[1])
        # The rejected windowed attempt must not have compacted any PCM.
        self.assertEqual(state.audio.base_offset_bytes, 0)

    def test_first_window_handoff_replay_with_cjk_retries_cumulatively(self):
        """Bug regression: the no-boundary flag was computed before the
        handoff echo was trimmed, so a decode that replayed the whitespace
        handoff ahead of a CJK continuation ("C D 你好世界") was classified
        as whitespace-bearing and the one-time cumulative retry was skipped.
        Spelled without the space, the handoff also outlived word matching and
        was joined onto itself as "C D C D你好世界"."""
        for separator in (" ", ""):
            with self.subTest(separator=separator):
                tokenizer_manager, processor, state = _encoder_window_processor(
                    [f"C D{separator}你好世界", f"A B C D{separator}你好世界"]
                )
                state.transcript.emitted_text = "A B"
                state.transcript.confirmed_text = "A B"
                state.transcript.full_transcript = "A B C D"
                state.audio.append_pcm(b"\x01\x00\x02\x00")

                _process(processor, state)

                self.assertTrue(state.encoder_window_disabled)
                self.assertIsNone(state.decoder_suffix)
                self.assertEqual(len(tokenizer_manager.requests), 2)
                self.assertIsNone(tokenizer_manager.processor_kwargs[1])
                self.assertEqual(state.audio.base_offset_bytes, 0)

    def test_fused_replay_match_keeps_word_normalization(self):
        """Bug regression: the junction matcher compared raw characters, so a
        replay that only recased or repunctuated its context — which whole-word
        matching deliberately tolerates — went undetected and its echo reached
        the transcript. A Latin continuation still must not match: only a
        no-boundary script writes itself against the replay without a space."""
        suffix_state = DecoderSuffixState(emitted_text="")
        for prefix, decoded, expected in (
            ("a b", "A B你好世界", ("你好世界", True)),
            ("Hello, world!", "hello world你好世界", ("你好世界", True)),
            ("A B", " A B你好世界", ("你好世界", True)),
            ("A B", "A Bfoo", ("A Bfoo", False)),
        ):
            with self.subTest(prefix=prefix, decoded=decoded):
                self.assertEqual(
                    suffix_state.trim_prefix_echo(
                        decoded, prefix, trim_short_prefix=True
                    ),
                    expected,
                )

    def test_handoff_only_replay_defers_like_an_empty_continuation(self):
        """Bug regression: a decode that replayed only the handoff ("C D")
        was non-empty when the deferral check ran, but became empty after the
        handoff trim — the one-shot empty deferral was bypassed and the
        audio was accepted without any new continuation."""
        tokenizer_manager, processor, state = _encoder_window_processor(["C D"])
        state.transcript.emitted_text = "A B"
        state.transcript.confirmed_text = "A B"
        state.transcript.full_transcript = "A B C D"
        state.audio.append_pcm(b"\x01\x00\x02\x00")

        self.assertEqual(_process(processor, state), "")

        self.assertIsNone(state.decoder_suffix)
        self.assertEqual(state.audio.last_attempted_offset_bytes, 4)
        self.assertEqual(state.audio.last_processed_offset_bytes, 0)

    def test_steady_state_without_pending_trims_a_short_prefix_replay(self):
        """Bug regression: an accepted empty continuation reaches steady state
        with no pending text, and the steady-state trim refused short prefixes
        because only pending could anchor them. The replay stayed in the text,
        and the next agreeing decode re-emitted the delivered "one two"."""
        _, processor, state = _encoder_window_processor(
            ["", "", "one two three four", "one two three four five"]
        )
        state.transcript.emitted_text = "one two"
        state.transcript.confirmed_text = "one two"
        state.transcript.full_transcript = "one two"

        deltas = []
        for _ in range(4):
            state.audio.append_pcm(b"\x01\x00\x02\x00")
            deltas.append(_process(processor, state))

        self.assertEqual(deltas, ["", "", "", "three"])
        self.assertEqual(state.decoder_suffix.emitted_text, "one two three")

    def test_steady_state_short_prefix_echo_is_not_reemitted(self):
        """Bug regression: with a sub-threshold decoder prefix the standard
        echo trim refuses, and before the pending-anchored trim also ran on
        non-final steps, a replayed prefix entered pending_suffix and the next
        agreeing decode re-emitted already-delivered words."""
        tokenizer_manager, processor, state = _encoder_window_processor(
            ["one two three four", "one two three four five"]
        )
        state.decoder_suffix = DecoderSuffixState(
            emitted_text="one two", pending_suffix="three"
        )
        state.audio.append_pcm(b"\x01\x00\x02\x00")

        self.assertEqual(_process(processor, state), "")

        state.audio.append_pcm(b"\x01\x00\x02\x00")
        self.assertEqual(_process(processor, state), "three")
        self.assertEqual(state.decoder_suffix.emitted_text, "one two three")

    def test_active_window_cjk_continuation_anchors_by_characters(self):
        """Bug regression: word-based anchoring kept an echoed short prefix
        before a CJK continuation, so a later agreeing decode re-emitted the
        delivered "A" ("A B A" on the wire) and a final decode republished
        "A B". Both junctions failed: spaced, and spaceless as CJK is spelled.
        """
        for separator in (" ", ""):
            with self.subTest(separator=separator):
                decodes = [
                    f"A B{separator}你好世界",
                    f"A B{separator}你好世界啊",
                    f"A B{separator}你好世界啊",
                ]
                _, processor, state = _encoder_window_processor(decodes)
                state.decoder_suffix = DecoderSuffixState(
                    emitted_text="A B", pending_suffix="你好"
                )

                state.audio.append_pcm(b"\x01\x00\x02\x00")
                self.assertEqual(_process(processor, state), "")
                state.audio.append_pcm(b"\x01\x00\x02\x00")
                self.assertEqual(_process(processor, state), "")
                self.assertEqual(_process(processor, state, is_last=True), "你好世界啊")
                self.assertEqual(state.decoder_suffix.emitted_text, "A B 你好世界啊")

    def test_active_window_unanchorable_no_boundary_echo_fails_closed(self):
        """Bug regression companion: when no-boundary text after the echoed
        prefix does not continue pending, no anchor can separate echo from
        real repetition and keeping it re-emits history. The one-way
        transition forbids a cumulative retry, so the step fails closed."""
        for separator in (" ", ""):
            with self.subTest(separator=separator):
                _, processor, state = _encoder_window_processor(
                    [f"A B{separator}你好世界"]
                )
                state.decoder_suffix = DecoderSuffixState(
                    emitted_text="A B", pending_suffix="hello"
                )
                state.audio.append_pcm(b"\x01\x00\x02\x00")

                with self.assertRaisesRegex(RuntimeError, "ambiguous decoder-prefix"):
                    _process(processor, state)

    def test_no_boundary_reconcile_holds_back_characters(self):
        """Bug regression: the no-boundary branch subtracted the word-count
        holdback from a character-level prefix length, so only one character
        was held back and unstable boundary characters were emitted early."""
        update = DecoderSuffixState(
            emitted_text="", pending_suffix="你好世界"
        ).reconcile("你好世界啊", is_last=False, holdback_words=1)
        self.assertEqual(update.delta, "")
        self.assertEqual(update.pending_suffix, "你好世界啊")

        update = DecoderSuffixState(
            emitted_text="", pending_suffix="你好世界你好世界"
        ).reconcile("你好世界你好世界啊", is_last=False, holdback_words=1)
        self.assertEqual(update.delta, "你好世界")

    def test_cumulative_empty_response_keeps_audio_for_retry(self):
        """Bug regression: a transient empty backend response must neither
        close the WebSocket nor mark the audio processed — otherwise a commit
        right after it sees no new audio and completes without retrying."""
        _, connection = _realtime_connection(
            [None, "one two three four"], ["stop", "stop"], encoder_windows=False
        )

        self.assertTrue(_run(connection._run_inference(is_last=False)))
        connection.websocket.close.assert_not_awaited()
        self.assertEqual(connection.asr_state.audio.last_attempted_offset_bytes, 4)
        self.assertEqual(connection.asr_state.audio.last_processed_offset_bytes, 0)
        self.assertTrue(connection.asr_state.has_new_audio)

        # The retained audio is recovered by the next (final) decode.
        self.assertTrue(_run(connection._run_inference(is_last=True)))

    def test_cumulative_final_empty_response_fails_item(self):
        """A final decode that returns nothing must fail the item instead of
        publishing a completed transcript missing the last audio."""
        _, connection = _realtime_connection([None], ["stop"], encoder_windows=False)

        self.assertFalse(_run(connection._run_inference(is_last=True)))
        connection.websocket.close.assert_not_awaited()

    def test_cumulative_truncated_stream_forces_final_decode(self):
        """A truncated intermediate decode may stream, but commit must force
        one final decode even though the audio is fully processed — the gate
        lives in the commit handler, so this drives the real commit path. A
        still-truncated final fails the item instead of completing it."""
        tokenizer_manager, connection = _realtime_connection(
            ["one two", "one two"], ["length", "length"], encoder_windows=False
        )

        self.assertTrue(_run(connection._run_inference(is_last=False)))
        self.assertTrue(connection.asr_state.cumulative_truncated)
        self.assertFalse(connection.asr_state.has_new_audio)

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

    def test_cumulative_truncated_commit_recovers_and_completes(self):
        """Commit-lifecycle counterpart: the forced final decode publishes
        completed only from the clean final result."""
        tokenizer_manager, connection = _realtime_connection(
            ["one two", "one two three"], ["length", "stop"], encoder_windows=False
        )

        self.assertTrue(_run(connection._run_inference(is_last=False)))
        _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))

        self.assertEqual(len(tokenizer_manager.requests), 2)
        completed = [
            call.args[0]
            for call in connection._send.call_args_list
            if call.args[0].type
            == "conversation.item.input_audio_transcription.completed"
        ]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].transcript, "one two three")

    def test_windowed_decode_clears_stale_cumulative_truncated_flag(self):
        """Bug regression: a truncated cumulative decode before the windowed
        transition left cumulative_truncated set forever, so commit forced a
        spurious windowed final decode that cannot revisit the cumulative
        prefix anyway."""
        tokenizer_manager, connection = _realtime_connection(
            ["one two three four five"], ["stop"], encoder_windows=True
        )
        connection.asr_state.cumulative_truncated = True

        self.assertTrue(_run(connection._run_inference(is_last=False)))
        self.assertFalse(connection.asr_state.cumulative_truncated)

        _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))

        self.assertEqual(len(tokenizer_manager.requests), 1)
        sent_event_types = [
            call.args[0].type for call in connection._send.call_args_list
        ]
        self.assertIn(
            "conversation.item.input_audio_transcription.completed", sent_event_types
        )

    def test_encoder_window_length_limited_decode_fails_closed(self):
        _, connection = _realtime_connection(
            "three four",
            ["length"],
            encoder_windows=True,
        )
        with self.assertLogs(level="ERROR"):
            self.assertFalse(_run(connection._run_inference(is_last=False)))
        connection.websocket.close.assert_awaited_with(code=1011)

        _, connection = _realtime_connection("three four", ["length"])
        connection.asr_state.decoder_suffix = DecoderSuffixState(
            emitted_text="one two three"
        )
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

    def test_streaming_abort_does_not_commit_audio_or_transcript(self):
        for encoder_windows in (False, True):
            with self.subTest(encoder_windows=encoder_windows):
                _, connection = _realtime_connection(
                    "partial transcript",
                    [
                        {
                            "type": "abort",
                            "message": "synthetic backend abort",
                            "status_code": 500,
                        }
                    ],
                    encoder_windows=encoder_windows,
                )

                with self.assertLogs(level="ERROR"):
                    self.assertFalse(_run(connection._run_inference(is_last=False)))

                self.assertEqual(
                    connection.asr_state.audio.last_processed_offset_bytes, 0
                )
                self.assertIsNone(connection.asr_state.decoder_suffix)
                self.assertEqual(connection.asr_state.transcript.full_transcript, "")
                sent_event_types = [
                    call.args[0].type for call in connection._send.call_args_list
                ]
                self.assertNotIn(
                    "conversation.item.input_audio_transcription.delta",
                    sent_event_types,
                )
                self.assertNotIn(
                    "conversation.item.input_audio_transcription.completed",
                    sent_event_types,
                )
                connection.websocket.close.assert_awaited_with(code=1011)

    def test_cumulative_length_keeps_existing_reconciliation_behavior(self):
        _, connection = _realtime_connection(
            "three four",
            ["length"],
            encoder_windows=False,
        )

        self.assertTrue(_run(connection._run_inference(is_last=False)))
        connection.websocket.close.assert_not_awaited()

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
                _adapter(
                    min_audio_sec=60.0,
                    encoder_window_config=_ENCODER_WINDOW_POLICY,
                ),
                server_args,
            )
        state = processor.create_state()
        self.assertEqual(processor.max_item_bytes("en"), 120)
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

        connection._send_error = AsyncMock()
        update({"transcription": {"language": "es"}})
        self.assertEqual(connection.config.language, "en")
        self.assertEqual(connection._send_error.await_args.args[0], "invalid_state")

        connection.asr_state.decoder_suffix = None
        connection._send_error.reset_mock()
        update({"transcription": {"language": "es"}})
        connection._send_error.assert_not_awaited()
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
