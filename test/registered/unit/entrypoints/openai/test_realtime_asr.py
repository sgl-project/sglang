"""Focused CPU coverage for realtime ASR request and transcript contracts."""

import base64
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that pull in sgl_kernel

import numpy as np

from sglang.srt.entrypoints.openai.realtime.asr_processor import (
    RealtimeASRProcessor,
)
from sglang.srt.entrypoints.openai.realtime.decoder_suffix import (
    DecoderSuffixState,
)
from sglang.srt.entrypoints.openai.realtime.session import RealtimeConnection
from sglang.srt.entrypoints.openai.streaming_asr import (
    generate_asr_transcript,
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
_WINDOW_GEOMETRY = AudioEncoderWindowConfig(
    min_input_samples=4,
    window_samples=16,
    feature_batch_frames=9,
    window_tokens=8,
)
_WINDOW_POLICY = {
    "decoder_prefix_max_tokens": 3,
    "decoder_prefix_holdback_words": 1,
    "max_audio_context_windows": 2,
}


def _adapter(
    *,
    min_audio_sec=0.0,
    encoder_windows=False,
    chunk_size_sec=2.0,
    unfixed_token_num=1,
):
    policy = None
    if encoder_windows:
        policy = RealtimeEncoderWindowPolicy(
            min_audio_sec=min_audio_sec,
            max_audio_context_windows=_WINDOW_POLICY["max_audio_context_windows"],
            decoder_prefix_max_tokens=_WINDOW_POLICY["decoder_prefix_max_tokens"],
            decoder_prefix_holdback_words=_WINDOW_POLICY[
                "decoder_prefix_holdback_words"
            ],
            supported_languages=("en",),
        )
    return SimpleNamespace(
        prompt_template="PROMPT:",
        model_sample_rate=1,
        supports_chunked_streaming=True,
        postprocess_text=lambda text: text,
        postprocess_streaming_text=lambda text: text,
        build_sampling_params=lambda request: {"language": request.language},
        realtime_encoder_window_policy=policy,
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
        encoder_windows=False,
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
        self.server_args = SimpleNamespace(incremental_streaming_output=False)
        self.requests = []
        self.processor_kwargs = []
        self.tokenizer = Mock(
            encode=lambda text, add_special_tokens=False: text.split(),
            decode=lambda token_ids, **kwargs: " ".join(token_ids),
        )
        if encoder_windows:
            self.mm_processor = Mock(
                resolve_audio_encoder_window_config=lambda *args: _WINDOW_GEOMETRY
            )

    def generate_request(self, adapted_request, raw_request=None, **kwargs):
        self.requests.append(adapted_request)
        config = kwargs.get("audio_encoder_window_config")
        self.processor_kwargs.append(
            {"audio_encoder_window_config": config} if config is not None else None
        )
        transcript = self._transcripts.pop(0)

        async def generate():
            if self._fail:
                raise ValueError("synthetic failure")
            if transcript is not None:
                finish_reason = self._finish_reasons.pop(0)
                if not isinstance(finish_reason, dict):
                    finish_reason = {"type": finish_reason}
                yield {
                    "text": transcript,
                    "meta_info": {"finish_reason": finish_reason},
                }

        return generate()


class _StreamingTokenizerManager(_MockTokenizerManager):
    def __init__(self, chunks, *, incremental=True, encoder_windows=False):
        super().__init__("unused", encoder_windows=encoder_windows)
        self._chunks = chunks
        self.server_args.incremental_streaming_output = incremental

    def generate_request(self, adapted_request, raw_request=None, **kwargs):
        self.requests.append(adapted_request)
        config = kwargs.get("audio_encoder_window_config")
        self.processor_kwargs.append(
            {"audio_encoder_window_config": config} if config is not None else None
        )

        async def generate():
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

        return generate()


def _run(coro):
    return get_or_create_event_loop().run_until_complete(coro)


def _server_args(max_buffer_seconds=60, *, encoder_windows=True):
    return SimpleNamespace(
        asr_max_buffer_seconds=max_buffer_seconds,
        enable_asr_encoder_window=encoder_windows,
    )


def _processor(transcripts, *, min_audio_sec=0.0, encoder_windows=True):
    manager = _MockTokenizerManager(
        transcripts,
        encoder_windows=encoder_windows,
    )
    processor = RealtimeASRProcessor(
        manager,
        _adapter(min_audio_sec=min_audio_sec, encoder_windows=encoder_windows),
        _server_args(120, encoder_windows=encoder_windows),
    )
    return manager, processor, processor.create_state()


def _process(processor, state, *, is_last=False, language="en", on_delta=None):
    return _run(
        processor.process(
            state,
            is_last=is_last,
            language=language,
            sampling_params={},
            on_transcript_delta=on_delta,
        )
    )


def _connection(transcripts, finish_reasons, *, encoder_windows=True):
    manager = _MockTokenizerManager(
        transcripts,
        encoder_windows=encoder_windows,
        finish_reasons=finish_reasons,
    )
    connection = RealtimeConnection(
        Mock(send_text=AsyncMock(), close=AsyncMock()),
        manager,
        _adapter(encoder_windows=encoder_windows),
        _server_args(120, encoder_windows=encoder_windows),
    )
    connection.config.configured = True
    connection.config.language = "en"
    connection.config.sampling_params = {"max_new_tokens": 256}
    connection._send = AsyncMock()
    if encoder_windows:
        connection.asr_state.transcript.emitted_text = "one two three"
    connection.asr_state.audio.append_pcm(b"\x01\x00\x02\x00")
    return manager, connection


def _gated_connection(transcripts):
    manager = _MockTokenizerManager(transcripts, encoder_windows=True)
    websocket = Mock(send_text=AsyncMock(), close=AsyncMock())
    connection = RealtimeConnection(
        websocket,
        manager,
        _adapter(min_audio_sec=60.0, encoder_windows=True),
        _server_args(120),
    )
    connection.config.configured = True
    connection.config.input_sample_rate = 1
    connection.config.language = "en"
    connection.config.sampling_params = {"max_new_tokens": 256}
    connection._send = AsyncMock()
    return manager, websocket, connection


def _append_seconds(connection, seconds):
    return _run(
        connection._on_input_audio_buffer_append(
            SimpleNamespace(audio=base64.b64encode(bytes(seconds * 2)).decode())
        )
    )


def _sent_events(connection, suffix):
    return [
        call.args[0]
        for call in connection._send.call_args_list
        if call.args[0].type.endswith(suffix)
    ]


class TestRealtimeASRConfig(CustomTestCase):
    def test_rejects_invalid_configuration(self):
        defaults = {
            "min_audio_sec": 60.0,
            "max_audio_context_windows": 8,
            "supported_languages": ("en",),
        }
        for override in (
            {"min_audio_sec": float("inf")},
            {"max_audio_context_windows": 0},
            {"supported_languages": ()},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                RealtimeEncoderWindowPolicy(**(defaults | override))

        manager = _MockTokenizerManager("text")
        for chunk_size in (0.0, -1.0, float("inf"), 1e-20, 0.5):
            with self.subTest(chunk_size=chunk_size), self.assertRaises(ValueError):
                RealtimeASRProcessor(
                    manager,
                    _adapter(chunk_size_sec=chunk_size),
                    _server_args(),
                )


class TestRealtimeASR(CustomTestCase):
    def test_backend_streaming_reconstructs_snapshots_and_qwen_output(self):
        async def run_case(manager, adapter):
            updates = []

            async def collect(text):
                updates.append(text)

            result = await generate_asr_transcript(
                tokenizer_manager=manager,
                adapter=adapter,
                audio_data=_AUDIO,
                sampling_params={},
                prompt="PROMPT:",
                on_update=collect,
            )
            return updates, result

        updates, result = _run(
            run_case(
                _StreamingTokenizerManager(
                    ["hello", "hello world", "hello world again"], incremental=False
                ),
                _adapter(),
            )
        )
        self.assertEqual(updates, ["hello", "hello world", "hello world again"])
        self.assertEqual(result.text, "hello world again")

        updates, result = _run(
            run_case(
                _StreamingTokenizerManager(
                    ["language en<asr", "_text>Hello", " world"]
                ),
                Qwen3ASRAdapter(),
            )
        )
        self.assertEqual(updates, ["Hello", "Hello world"])
        self.assertEqual(result.text, "Hello world")

    def test_windowed_stream_keeps_published_text_append_only(self):
        async def run_chunks(chunks, pending, *, incremental=True):
            manager = _StreamingTokenizerManager(
                chunks,
                incremental=incremental,
                encoder_windows=True,
            )
            processor = RealtimeASRProcessor(
                manager,
                _adapter(encoder_windows=True),
                _server_args(120),
            )
            state = processor.create_state()
            state.decoder_suffix = DecoderSuffixState(
                emitted_text="", pending_suffix=pending
            )
            state.audio.append_pcm(b"\x01\x00\x02\x00")
            updates = []

            async def collect(text):
                updates.append(text)

            result = await processor.process(
                state,
                is_last=False,
                language="en",
                sampling_params={},
                on_transcript_delta=collect,
            )
            return updates, result, state

        with self.assertRaisesRegex(RuntimeError, "append-only streamed prefix"):
            _run(
                run_chunks(
                    ["one two three ", "one four"],
                    "one two",
                    incremental=False,
                )
            )

        updates, result, state = _run(
            run_chunks(["one", " two", " twenty", "-five"], "one two twenty")
        )
        self.assertEqual(updates, ["one"])
        self.assertEqual(result, "")
        self.assertEqual(state.decoder_suffix.latest_text, "one two twenty-five")

    def test_wire_deltas_join_to_completed_transcript(self):
        manager = _StreamingTokenizerManager(["hello", " world", " again"])
        connection = RealtimeConnection(
            Mock(send_text=AsyncMock(), close=AsyncMock()),
            manager,
            _adapter(),
            _server_args(encoder_windows=False),
        )
        connection.config.configured = True
        connection.config.language = "en"
        connection.config.sampling_params = {"max_new_tokens": 256}
        connection._send = AsyncMock()
        connection.asr_state.audio.append_pcm(b"\x01\x00\x02\x00")

        self.assertTrue(_run(connection._run_inference(is_last=False)))
        _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))

        deltas = [event.delta for event in _sent_events(connection, ".delta")]
        completed = _sent_events(connection, ".completed")[-1].transcript
        self.assertEqual("".join(deltas), completed)

    def test_windowed_handoff_emits_only_agreed_suffix(self):
        manager, processor, state = _processor(
            ["four five", "four five six", "five six seven"]
        )
        state.transcript.emitted_text = "one two three"

        deltas = []
        for pcm, is_last in (
            (b"\x01\x00\x02\x00", False),
            (b"\x03\x00\x04\x00", False),
            (b"\x05\x00\x06\x00", True),
        ):
            state.audio.append_pcm(pcm)
            deltas.append(_process(processor, state, is_last=is_last))

        self.assertEqual(deltas, ["", "four", "five six seven"])
        self.assertEqual(
            state.decoder_suffix.emitted_text,
            "one two three four five six seven",
        )
        self.assertEqual(manager.requests[0].text, "PROMPT:one two three")
        self.assertEqual(
            manager.processor_kwargs[0]["audio_encoder_window_config"],
            _WINDOW_GEOMETRY,
        )

    def test_handoff_echo_is_trimmed_without_dropping_real_repetition(self):
        for decoded, expected in (
            ("C D E", "C D E"),
            ("B C D C D E", "C D C D E"),
        ):
            with self.subTest(decoded=decoded):
                _, processor, state = _processor(decoded)
                state.transcript.emitted_text = "A B"
                state.transcript.confirmed_text = "A B"
                state.transcript.full_transcript = "A B C D"
                state.audio.append_pcm(b"\x01\x00\x02\x00")
                self.assertEqual(_process(processor, state), "")
                self.assertEqual(processor.flush_pending_transcript(state), expected)

        update = DecoderSuffixState(emitted_text="I agree").reconcile(
            "I agree with that", is_last=True, holdback_words=1
        )
        self.assertEqual(update.delta, "I agree with that")

        update = DecoderSuffixState("hello", "provisional").reconcile(
            "", is_last=True, holdback_words=1
        )
        self.assertEqual((update.delta, update.pending_suffix), ("", ""))

    def test_window_activation_requires_gate_language_and_flag(self):
        manager, processor, state = _processor("alpha beta", min_audio_sec=60.0)
        state.audio.append_pcm(bytes(120))
        _process(processor, state)
        self.assertIsNone(manager.processor_kwargs[0])

        manager, processor, state = _processor("four five", min_audio_sec=60.0)
        state.transcript.emitted_text = "one two three"
        state.audio.append_pcm(bytes(122))
        state.audio.last_attempted_offset_bytes = 120
        state.audio.last_processed_offset_bytes = 120
        _process(processor, state)
        self.assertIsNotNone(manager.processor_kwargs[0])

        for language in (None, "auto", "zh"):
            manager, processor, state = _processor("alpha beta", min_audio_sec=60.0)
            state.audio.append_pcm(b"\x01\x00\x02\x00")
            _process(processor, state, language=language)
            self.assertIsNone(manager.processor_kwargs[0])
            self.assertEqual(processor.max_item_bytes(language), 120)

        manager, processor, state = _processor("alpha beta", encoder_windows=False)
        state.audio.append_pcm(b"\x01\x00\x02\x00")
        _process(processor, state, language="en")
        self.assertIsNone(manager.processor_kwargs[0])

    def test_large_append_is_drained_with_bounded_context(self):
        manager = _MockTokenizerManager(["one"] * 45, encoder_windows=True)
        connection = RealtimeConnection(
            Mock(send_text=AsyncMock(), close=AsyncMock()),
            manager,
            _adapter(min_audio_sec=60.0, encoder_windows=True),
            _server_args(120),
        )
        connection.config.configured = True
        connection.config.input_sample_rate = 1
        connection.config.language = "en"
        connection.config.sampling_params = {"max_new_tokens": 256}
        connection._send = AsyncMock()

        self.assertFalse(_append_seconds(connection, 90))
        sizes = [request.audio_data.size for request in manager.requests]
        self.assertEqual(len(sizes), 45)
        self.assertEqual(sizes[:3], [2, 4, 6])
        self.assertLessEqual(max(sizes[30:]), 64)
        self.assertEqual(
            connection.asr_state.audio.last_attempted_offset_bytes,
            180,
        )

    def test_no_boundary_handoff_retries_cumulative_once(self):
        manager, processor, state = _processor(["你好世界", "你好世界"])
        state.audio.append_pcm(b"\x01\x00\x02\x00")

        self.assertEqual(_process(processor, state), "")
        self.assertTrue(state.encoder_window_disabled)
        self.assertFalse(state.encoder_window_active)
        self.assertEqual(len(manager.requests), 2)
        self.assertIsNotNone(manager.processor_kwargs[0])
        self.assertIsNone(manager.processor_kwargs[1])

    def test_cumulative_fallback_stops_before_unbounded_redecode(self):
        manager, websocket, connection = _gated_connection(
            [""] * 30 + ["你好世界", "one"]
        )

        self.assertTrue(_append_seconds(connection, 90))
        sizes = [request.audio_data.size for request in manager.requests]
        self.assertEqual(sizes[-2:], [62, 62])
        self.assertNotIn(64, sizes)
        self.assertTrue(connection.asr_state.encoder_window_disabled)
        websocket.close.assert_awaited_with(code=1009)

    def test_empty_response_retries_audio_but_final_empty_fails(self):
        _, connection = _connection(
            [None, "one two three"], ["stop", "stop"], encoder_windows=False
        )
        self.assertTrue(_run(connection._run_inference(is_last=False)))
        self.assertEqual(connection.asr_state.audio.last_processed_offset_bytes, 0)
        self.assertTrue(_run(connection._run_inference(is_last=True)))

        _, connection = _connection([None], ["stop"], encoder_windows=False)
        self.assertFalse(_run(connection._run_inference(is_last=True)))
        connection.websocket.close.assert_not_awaited()

    def test_truncated_cumulative_commit_uses_clean_final_result(self):
        for final_reason, completed in (("length", False), ("stop", True)):
            with self.subTest(final_reason=final_reason):
                manager, connection = _connection(
                    ["one two", "one two three"],
                    ["length", final_reason],
                    encoder_windows=False,
                )
                self.assertTrue(_run(connection._run_inference(is_last=False)))
                _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))
                self.assertEqual(len(manager.requests), 2)
                self.assertEqual(
                    bool(_sent_events(connection, ".completed")), completed
                )
                self.assertEqual(
                    bool(_sent_events(connection, ".failed")), not completed
                )

        for final_text, completed in (("one two three", True), ("", False)):
            with self.subTest(recovery_after_empty=final_text):
                manager, connection = _connection(
                    ["one two", "", final_text],
                    ["length", "stop", "stop"],
                    encoder_windows=False,
                )
                self.assertTrue(_run(connection._run_inference(is_last=False)))
                connection.asr_state.audio.append_pcm(b"\x03\x00\x04\x00")
                self.assertTrue(_run(connection._run_inference(is_last=False)))
                self.assertEqual(
                    connection.asr_state.audio.last_processed_offset_bytes, 0
                )

                _run(connection._on_input_audio_buffer_commit(SimpleNamespace()))

                self.assertEqual(len(manager.requests), 3)
                self.assertEqual(
                    bool(_sent_events(connection, ".completed")), completed
                )
                self.assertEqual(
                    bool(_sent_events(connection, ".failed")), not completed
                )

    def test_windowed_length_and_abort_fail_without_committing_state(self):
        _, connection = _connection("three four", ["length"])
        with self.assertLogs(level="ERROR"):
            self.assertFalse(_run(connection._run_inference(is_last=False)))
        connection.websocket.close.assert_awaited_with(code=1011)

        _, connection = _connection(
            "partial",
            [{"type": "abort", "message": "abort", "status_code": 500}],
        )
        with self.assertLogs(level="ERROR"):
            self.assertFalse(_run(connection._run_inference(is_last=False)))
        self.assertEqual(connection.asr_state.audio.last_processed_offset_bytes, 0)
        self.assertEqual(connection.asr_state.transcript.full_transcript, "")
        self.assertFalse(_sent_events(connection, ".delta"))
        connection.websocket.close.assert_awaited_with(code=1011)

    def test_failed_inference_preserves_audio_and_reset_clears_offsets(self):
        manager = _MockTokenizerManager(fail=True)
        websocket = Mock(send_text=AsyncMock(), close=AsyncMock())
        connection = RealtimeConnection(websocket, manager, _adapter(), _server_args())
        connection.config.sampling_params = {}
        connection.asr_state.audio.append_pcm(bytes(range(12)))
        connection.asr_state.audio.last_attempted_offset_bytes = 8
        connection.asr_state.audio.last_processed_offset_bytes = 8

        with self.assertLogs(level="WARNING"):
            self.assertFalse(_run(connection._run_inference(is_last=False)))
        self.assertEqual(bytes(connection.asr_state.audio.data), bytes(range(12)))
        self.assertEqual(connection.asr_state.audio.last_processed_offset_bytes, 8)

        connection._reset_inference_state()
        self.assertEqual(connection.asr_state.audio.data, bytearray())
        self.assertEqual(connection.asr_state.audio.received_bytes, 0)
        self.assertEqual(connection.asr_state.audio.last_attempted_offset_bytes, 0)
        self.assertEqual(connection.asr_state.audio.last_processed_offset_bytes, 0)


if __name__ == "__main__":
    unittest.main()
