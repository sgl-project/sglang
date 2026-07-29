"""Unit tests for the realtime ASR request and transcript paths.

Covers the two inference modes ``RealtimeASRProcessor`` can pick — cumulative
(re-send the whole buffer) and windowed (encoder-aligned windows plus a decoder
prefix) — along with ``StreamingASRState``'s reconciliation rules for word- and
character-delimited transcripts.

``TokenizerManager.generate_request`` is mocked to yield synthetic ``text``
chunks so each case can pin one decode sequence.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np

from sglang.srt.entrypoints.openai.realtime.asr_processor import (
    RealtimeASRProcessor,
    decoder_suffix_token_budget,
)
from sglang.srt.entrypoints.openai.realtime.session import RealtimeConnection
from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
    _dedupe_by_word,
    is_cjk_no_whitespace,
    process_asr_chunk,
)
from sglang.srt.multimodal.audio_encoder_windowing import AudioEncoderWindowConfig
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_AUDIO = np.zeros(1600, dtype=np.float32)
_WINDOW_CONFIG = AudioEncoderWindowConfig(
    min_input_samples=4, window_samples=16, window_tokens=8
)
_WINDOWED_ADAPTER_CONFIG = {
    "decoder_prefix_max_tokens": 3,
    "max_audio_context_windows": 2,
}


def _adapter(
    *,
    model_sample_rate=1,
    min_audio_sec=0.0,
    windowed_config=None,
    unfixed_token_num=1,
):
    """Stand-in adapter; every knob the cases vary is a keyword here."""
    return SimpleNamespace(
        prompt_template="PROMPT:",
        model_sample_rate=model_sample_rate,
        postprocess_text=lambda text: text,
        realtime_long_audio_config=(
            {"min_audio_sec": min_audio_sec} | (windowed_config or {})
        ),
        chunked_streaming_config={
            "chunk_size_sec": 2.0,
            "unfixed_chunk_num": 2,
            "unfixed_token_num": unfixed_token_num,
        },
    )


class _MockTokenizerManager:
    """Minimal mock of the generate_request stream loop the ASR paths drive.

    ``transcript`` is one decode result, or a list yielded one per request.
    ``fail=True`` raises inside the stream to exercise the failure path.
    """

    def __init__(self, transcript=None, *, windowed=False, fail=False):
        self._transcripts = list(transcript) if isinstance(transcript, list) else None
        self._transcript = None if self._transcripts is not None else transcript
        self._fail = fail
        self.requests = []
        self.tokenizer = Mock(
            encode=lambda text, add_special_tokens=False: text.split(),
            decode=lambda token_ids, **kwargs: " ".join(token_ids),
        )
        if windowed:
            self.mm_processor = Mock(
                resolve_audio_encoder_window_config=lambda *args: _WINDOW_CONFIG
            )

    def generate_request(self, adapted_request, raw_request=None, **kwargs):
        self.requests.append(adapted_request)
        self.internal_mm_processor_kwargs = kwargs.get("internal_mm_processor_kwargs")
        fail = self._fail
        transcript = (
            self._transcripts.pop(0)
            if self._transcripts is not None
            else self._transcript
        )

        async def gen():
            if fail:
                raise ValueError("synthetic failure")
            if transcript is not None:
                yield {"text": transcript}

        return gen()


def _websocket():
    return Mock(send_text=AsyncMock(), close=AsyncMock())


def _run(coro):
    return get_or_create_event_loop().run_until_complete(coro)


def _server_args(max_buffer_seconds=60):
    return SimpleNamespace(asr_max_buffer_seconds=max_buffer_seconds)


def _state(**kwargs):
    params = dict(chunk_size_sec=1.0, unfixed_chunk_num=2, unfixed_token_num=2)
    params.update(kwargs)
    return StreamingASRState(**params)


def _chunk(state, transcript, *, is_last=False):
    """Drive one cumulative request and return the emitted delta."""
    tm = _MockTokenizerManager(transcript)
    out = _run(
        process_asr_chunk(
            tokenizer_manager=tm,
            adapter=_adapter(),
            state=state,
            audio_data=_AUDIO,
            sampling_params={},
            is_last=is_last,
        )
    )
    return tm, out


class TestRequestPrompts(CustomTestCase):
    def _windowed_processor(self, transcripts, **cache_overrides):
        cache = {
            "decoder_prefix_max_tokens": 3,
            "decoder_prefix_holdback_words": 1,
            "max_audio_context_windows": 2,
        }
        cache.update(cache_overrides)
        tm = _MockTokenizerManager(transcripts, windowed=True)
        processor = RealtimeASRProcessor(
            tm, _adapter(windowed_config=cache), _server_args()
        )
        state = processor.create_state()
        state.transcript.emitted_text = "one two three"
        state.transcript.chunk_index = 5
        return tm, processor, state

    def test_cumulative_prompt_carries_committed_text(self):
        state = _state(emitted_text="hello", chunk_index=5)
        tm, _ = _chunk(state, "hello world foo")
        self.assertEqual(tm.requests[0].text, "PROMPT:hello")

    def test_windowed_path_emits_only_the_agreed_suffix(self):
        """The windowed path prompts with committed text and decodes only the
        continuation, holding a word back until two decodes agree on it."""
        tm, processor, state = self._windowed_processor(
            ["four five", "four five six", "five six seven"]
        )
        sampling_params = {"max_new_tokens": 256}

        state.audio.append_pcm(b"\x01\x00\x02\x00")
        delta = _run(
            processor.process(state, is_last=False, sampling_params=sampling_params)
        )
        self.assertEqual(delta, "")
        self.assertEqual(tm.requests[0].text, "PROMPT:one two three")
        # Intermediate requests are capped to the new audio, not the full budget.
        self.assertEqual(tm.requests[0].sampling_params["max_new_tokens"], 32)

        state.audio.append_pcm(b"\x03\x00\x04\x00")
        delta = _run(
            processor.process(state, is_last=False, sampling_params=sampling_params)
        )
        self.assertEqual(delta, "four")
        self.assertEqual(state.transcript.pending_suffix, "five six")

        state.audio.append_pcm(b"\x05\x00\x06\x00")
        delta = _run(
            processor.process(state, is_last=True, sampling_params=sampling_params)
        )
        self.assertEqual(delta, "five six seven")
        # The final commit may replay the prefix, so it keeps the full budget.
        self.assertEqual(tm.requests[-1].sampling_params["max_new_tokens"], 256)
        self.assertEqual(
            state.transcript.emitted_text, "one two three four five six seven"
        )

    def test_windowed_request_carries_the_resolved_geometry(self):
        tm, processor, state = self._windowed_processor(["four five"])
        state.audio.append_pcm(b"\x01\x00\x02\x00")
        _run(processor.process(state, is_last=False, sampling_params={}))

        self.assertEqual(
            tm.internal_mm_processor_kwargs["audio_encoder_window_config"],
            _WINDOW_CONFIG,
        )

    def test_suffix_budget_covers_new_audio_plus_pending_words(self):
        self.assertEqual(
            decoder_suffix_token_budget(6400, 32000, " ".join(["word"] * 30)), 64
        )

    def test_preview_decoder_suffix_does_not_mutate_state(self):
        """Preview runs before the commit decision, so it has to stay pure."""
        state = _state(pending_suffix="old pending", emitted_text="done")
        before = state.__dict__.copy()
        state.preview_decoder_suffix("new boundary", holdback_words=1)
        self.assertEqual(state.__dict__, before)


class TestTranscriptReconciliation(CustomTestCase):
    def test_word_holdback_and_case_insensitive_prefix(self):
        state = _state(
            unfixed_chunk_num=0, unfixed_token_num=1, confirmed_text="hello world"
        )
        self.assertEqual(_chunk(state, "hello worldly test tail")[1], "worldly test")

        state = _state(confirmed_text="hello world", emitted_text="hello world")
        self.assertEqual(_chunk(state, "Hello world again", is_last=True)[1], "again")

    def test_cjk_char_holdback_then_growth(self):
        state = _state(unfixed_token_num=1)
        self.assertEqual(_chunk(state, "你好世界")[1], "你好世")
        _chunk(state, "你好世界好")
        self.assertEqual(state.emitted_text, "你好世界")

    def test_cjk_repetition_is_withheld_until_finalize(self):
        state = _state(unfixed_token_num=1)
        _chunk(state, "你好世界")
        repeated = "你好世界" * 10
        self.assertEqual(state.update(repeated), "")
        state.full_transcript = repeated
        state.finalize()
        self.assertEqual(state.emitted_text, repeated)

    def test_cjk_finalize_trims_punctuation_only_overlap(self):
        state = _state(emitted_text="你好，世界。你好")
        state.full_transcript = "你好。世界。你好世界"
        self.assertEqual(state.finalize(), "世界")

    def test_no_whitespace_cjk_cannot_use_the_decoder_prefix_path(self):
        state = _state()
        self.assertFalse(
            is_cjk_no_whitespace(
                state.preview_decoder_suffix(
                    "no，no。", holdback_words=1
                ).pending_suffix
                or ""
            )
        )
        self.assertTrue(
            is_cjk_no_whitespace(
                state.preview_decoder_suffix("你好。", holdback_words=1).pending_suffix
                or ""
            )
        )
        state.emitted_text = "已经发送"
        self.assertFalse(state.can_start_decoder_prefix())

    def test_dedupe_by_word_only_trims_a_verbatim_prefix(self):
        for emitted, candidate, expected in (
            (
                "he hoped there would be stew for dinner turnips",
                "turnips and carrots and bruised",
                ("and carrots and bruised", True),
            ),
            (
                "one two three four five six",
                "x y three four five six seven",
                ("x y three four five six seven", False),
            ),
            (
                "alpha beta",
                "fresh dinner—turnips text",
                ("fresh dinner—turnips text", False),
            ),
        ):
            with self.subTest(candidate=candidate):
                self.assertEqual(_dedupe_by_word(emitted, candidate), expected)


class TestWindowedConfigGuard(CustomTestCase):
    def _conn(self, *, server_args=None, tokenizer_manager=None, **adapter_kwargs):
        return RealtimeConnection(
            object(),
            tokenizer_manager or object(),
            _adapter(**adapter_kwargs),
            server_args or _server_args(),
        )

    def _windowed_conn(self, *, buffer_seconds=120, **adapter_kwargs):
        kwargs = dict(
            model_sample_rate=1,
            min_audio_sec=60.0,
            unfixed_token_num=5,
            windowed_config=_WINDOWED_ADAPTER_CONFIG,
        )
        kwargs.update(adapter_kwargs)
        return self._conn(
            server_args=_server_args(buffer_seconds),
            tokenizer_manager=SimpleNamespace(
                mm_processor=Mock(
                    resolve_audio_encoder_window_config=lambda *args: _WINDOW_CONFIG
                ),
                tokenizer=Mock(
                    encode=lambda text, add_special_tokens=False: text.split(),
                    decode=lambda token_ids, **kwargs: " ".join(token_ids),
                ),
            ),
            **kwargs,
        )

    def test_windowed_mode_needs_a_buffer_cap_above_the_gate(self):
        """The item cap is the long-audio opt-in: while the cap does not exceed
        the activation gate, no admissible item can select the windowed mode."""
        conn = self._windowed_conn(buffer_seconds=60)
        # The session rejects appends past the cap, so the largest admissible
        # item is exactly max_buffer_bytes — still not past the gate.
        conn.asr_state.audio.append_pcm(bytes(conn.asr_processor.max_buffer_bytes))
        self.assertFalse(conn.asr_processor._plan(conn.asr_state, False).windowed)

        conn = self._windowed_conn(buffer_seconds=120)
        conn.asr_state.audio.append_pcm(bytes(122))
        self.assertTrue(conn.asr_processor._plan(conn.asr_state, False).windowed)

    def test_first_windowed_request_covers_all_audio_and_bounds_the_prefix(self):
        conn = self._windowed_conn()
        conn.asr_state.audio.append_pcm(bytes(122))
        conn.asr_state.audio.accepted_offset_bytes = 122
        conn.asr_state.audio.attempted_offset_bytes = 122
        conn.asr_state.transcript.emitted_text = "one two three four five"

        intermediate = conn.asr_processor._plan(conn.asr_state, False)
        final = conn.asr_processor._plan(conn.asr_state, True)

        self.assertEqual(intermediate.start_offset_bytes, 0)
        self.assertEqual(final.start_offset_bytes, 0)
        # decoder_prefix_max_tokens=3 keeps only the newest three words.
        self.assertEqual(intermediate.decoder_prefix, "three four five")

    def test_rolling_request_drops_windows_the_prefix_already_represents(self):
        conn = self._windowed_conn()
        conn.asr_state.audio.append_pcm(bytes(122))
        conn.asr_state.transcript.emitted_text = "one two three four five"
        conn.asr_state.windowed_started = True
        conn.asr_state.audio.append_pcm(bytes(78))
        conn.asr_state.audio.accepted_offset_bytes = 192
        conn.asr_state.audio.attempted_offset_bytes = 192

        rolling = conn.asr_processor._plan(conn.asr_state, False)

        self.assertEqual(rolling.start_offset_bytes, 128)

    def test_windowed_mode_is_off_when_disabled_or_unconfigured(self):
        conn = self._windowed_conn()
        conn.asr_state.audio.append_pcm(bytes(122))
        conn.asr_state.windowed_disabled = True
        self.assertFalse(conn.asr_processor._plan(conn.asr_state, False).windowed)

        conn = self._windowed_conn(windowed_config={})
        self.assertIsNone(conn.asr_processor._windowed_policy)

    def test_processor_survives_a_tokenizer_manager_without_a_mm_processor(self):
        # An adapter without windowed keys must not touch the tokenizer manager.
        conn = self._conn(model_sample_rate=16000, min_audio_sec=45.0)
        self.assertIsNone(conn.asr_processor._windowed_policy)


class TestRealtimeBufferLifecycle(CustomTestCase):
    def _conn(self, tokenizer_manager):
        conn = RealtimeConnection(
            _websocket(), tokenizer_manager, _adapter(), _server_args()
        )
        conn.config.sampling_params = {}
        return conn

    def _prime_state(self, conn, pcm=bytes(range(12)), offset=8):
        conn.asr_state.audio.append_pcm(pcm)
        conn.asr_state.audio.attempted_offset_bytes = offset
        conn.asr_state.audio.accepted_offset_bytes = offset
        conn.asr_state.transcript.chunk_index = 2
        conn.asr_state.transcript.emitted_text = "alpha"

    def test_failed_inference_keeps_audio_and_reset_clears_offsets(self):
        tokenizer_manager = _MockTokenizerManager(fail=True)
        conn = self._conn(tokenizer_manager)
        self._prime_state(conn)

        with self.assertLogs(level="WARNING"):
            ok = _run(conn._run_inference(is_last=False))

        self.assertFalse(ok)
        self.assertEqual(bytes(conn.asr_state.audio.data), bytes(range(12)))
        self.assertEqual(conn.asr_state.audio.accepted_offset_bytes, 8)
        # Append-time failure closes the socket with 1011 (internal error).
        conn.websocket.close.assert_awaited_with(code=1011)

        conn._reset_inference_state()

        self.assertEqual(conn.asr_state.audio.data, bytearray())
        self.assertEqual(conn.asr_state.audio.received_bytes, 0)
        self.assertEqual(conn.asr_state.audio.attempted_offset_bytes, 0)
        self.assertEqual(conn.asr_state.audio.accepted_offset_bytes, 0)


if __name__ == "__main__":
    unittest.main()
