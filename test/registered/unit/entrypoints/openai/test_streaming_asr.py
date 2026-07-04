"""Focused CPU tests for realtime ASR slicing."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.entrypoints.openai.realtime.session import (
    RealtimeConnection,
)
from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
    _dedupe_by_word,
    process_asr_chunk,
)
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeAdapter:
    prompt_template = "PROMPT:"

    def postprocess_text(self, text: str) -> str:
        return text


class _MockTokenizerManager:
    def __init__(self, transcript):
        self._transcripts = list(transcript) if isinstance(transcript, list) else None
        self._transcript = None if self._transcripts is not None else transcript
        self.requests = []

    def generate_request(self, adapted_request, raw_request=None):
        self.requests.append(adapted_request)
        transcript = (
            self._transcripts.pop(0)
            if self._transcripts is not None
            else self._transcript
        )

        async def gen():
            if transcript is not None:
                yield {"text": transcript}

        return gen()


class _FailingTokenizerManager:
    def __init__(self):
        self.requests = []

    def generate_request(self, adapted_request, raw_request=None):
        self.requests.append(adapted_request)

        async def gen():
            raise ValueError("synthetic failure")
            yield  # pragma: no cover

        return gen()


class _FakeWebSocket:
    def __init__(self):
        self.sent = []
        self.closed_code = None

    async def send_text(self, text):
        self.sent.append(text)

    async def close(self, code):
        self.closed_code = code


def _run(coro):
    return get_or_create_event_loop().run_until_complete(coro)


_AUDIO = np.zeros(1600, dtype=np.float32)


class TestProcessAsrChunk(CustomTestCase):
    def _state(self, **kwargs):
        params = dict(chunk_size_sec=1.0, unfixed_chunk_num=2, unfixed_token_num=2)
        params.update(kwargs)
        return StreamingASRState(**params)

    def _chunk(self, state, transcript, is_last=False, **kwargs):
        tm = _MockTokenizerManager(transcript)
        out = _run(
            process_asr_chunk(
                tokenizer_manager=tm,
                adapter=_FakeAdapter(),
                state=state,
                audio_data=_AUDIO,
                sampling_params={},
                is_last=is_last,
                **kwargs,
            )
        )
        return tm, out

    def test_cumulative_and_sliced_prompt_paths(self):
        state = self._state()
        state.emitted_text = "hello"
        state.chunk_index = 5
        tm, _ = self._chunk(state, "hello world foo")
        self.assertEqual(tm.requests[0].text, "PROMPT:hello")

        state = self._state()
        tm, out = self._chunk(
            state,
            "alpha beta gamma",
            is_last=True,
            prompt="PROMPT:",
            dedupe_against="alpha",
        )
        self.assertEqual(tm.requests[0].text, "PROMPT:")
        self.assertEqual(out, "beta gamma")

    def test_transcript_reconciliation_and_cjk_deltas(self):
        state = self._state(
            unfixed_chunk_num=0, unfixed_token_num=1, confirmed_text="hello world"
        )
        _, out = self._chunk(state, "hello worldly test tail")
        self.assertEqual(out, "worldly test")

        state = self._state(confirmed_text="hello world", emitted_text="hello world")
        _, out = self._chunk(state, "Hello world again", is_last=True)
        self.assertEqual(out, "again")

        state = self._state(unfixed_token_num=1)
        _, out = self._chunk(state, "你好世界")
        self.assertEqual(out, "你好世")

        self._chunk(state, "你好世界好")
        self.assertEqual(state.emitted_text, "你好世界")

        # A sliced window's CJK text is final on arrival: char holdback would
        # silently drop its tail once the next window replaces the state.
        state = self._state(emitted_text="latin prefix")
        _, out = self._chunk(
            state, "你好世界", prompt="PROMPT:", dedupe_against="latin prefix"
        )
        self.assertEqual(out, "你好世界")

        # A clause-ending "。" at a sliced boundary must not block the exact
        # re-emitted-prefix trim when the next window extends the clause.
        state = self._state(emitted_text="甚至出现交易几乎停。")
        _, out = self._chunk(
            state,
            "甚至出现交易几乎停滞的情况。",
            prompt="PROMPT:",
            dedupe_against="甚至出现交易几乎停。",
            is_last=True,
        )
        self.assertEqual(out, "滞的情况。")

    def test_dedupe_by_word_only_trims_verbatim_prefix(self):
        self.assertEqual(
            _dedupe_by_word(
                "he hoped there would be stew for dinner turnips",
                "turnips and carrots and bruised",
            ),
            ("and carrots and bruised", True),
        )
        self.assertEqual(
            _dedupe_by_word(
                "one two three four five six", "x y three four five six seven"
            ),
            ("x y three four five six seven", False),
        )
        self.assertEqual(
            _dedupe_by_word("alpha beta", "fresh dinner—turnips text"),
            ("fresh dinner—turnips text", False),
        )


class _SlicingAdapter:
    model_sample_rate = 16000

    def __init__(self, left_overlap_ms, enabled=True, min_audio_sec=45.0):
        self._left_overlap_ms = left_overlap_ms
        self._enabled = enabled
        self._min_audio_sec = min_audio_sec

    @property
    def realtime_slicing_config(self):
        return {
            "enabled": self._enabled,
            "left_overlap_ms": self._left_overlap_ms,
            "min_audio_sec": self._min_audio_sec,
        }

    @property
    def chunked_streaming_config(self):
        return {"chunk_size_sec": 2.0, "unfixed_chunk_num": 2, "unfixed_token_num": 5}


class _RuntimeSlicingAdapter:
    model_sample_rate = 1
    prompt_template = "PROMPT:"

    @property
    def realtime_slicing_config(self):
        return {"enabled": True, "left_overlap_ms": 1000, "min_audio_sec": 0.0}

    @property
    def chunked_streaming_config(self):
        return {"chunk_size_sec": 2.0, "unfixed_chunk_num": 2, "unfixed_token_num": 1}

    def postprocess_text(self, text: str) -> str:
        return text


class TestSlicingConfigGuard(CustomTestCase):
    def _conn(
        self,
        left_overlap_ms,
        enabled=True,
        min_audio_sec=45.0,
        disable_input_slicing=False,
    ):
        server_args = SimpleNamespace(
            asr_max_buffer_seconds=60,
            asr_disable_input_slicing=disable_input_slicing,
        )
        return RealtimeConnection(
            object(),
            object(),
            _SlicingAdapter(left_overlap_ms, enabled, min_audio_sec),
            server_args,
        )

    def test_slicing_config_guard_and_sample_alignment(self):
        self.assertTrue(self._conn(left_overlap_ms=2000).audio.slicing_enabled)
        with self.assertLogs(level="WARNING"):
            self.assertFalse(self._conn(left_overlap_ms=8000).audio.slicing_enabled)
        with self.assertLogs(level="WARNING"):
            self.assertFalse(self._conn(left_overlap_ms=-1).audio.slicing_enabled)
        self.assertFalse(
            self._conn(
                left_overlap_ms=2000, disable_input_slicing=True
            ).audio.slicing_enabled
        )

        class _OddRateAdapter(_SlicingAdapter):
            model_sample_rate = 22050

        server_args = SimpleNamespace(
            asr_max_buffer_seconds=60, asr_disable_input_slicing=False
        )
        conn = RealtimeConnection(
            object(), object(), _OddRateAdapter(left_overlap_ms=250), server_args
        )
        self.assertEqual(conn.audio.left_overlap_bytes % 2, 0)


class TestRealtimePCMCompaction(CustomTestCase):
    def _conn(self, tokenizer_manager, *, adapter=None):
        conn = RealtimeConnection(
            _FakeWebSocket(),
            tokenizer_manager,
            adapter or _RuntimeSlicingAdapter(),
            SimpleNamespace(
                asr_max_buffer_seconds=60,
                asr_disable_input_slicing=False,
            ),
        )
        conn.config.sampling_params = {}
        return conn

    def _prime_sliced_state(self, conn, pcm=bytes(range(12)), offset=8):
        conn.audio.append_pcm(pcm)
        conn.audio.last_scheduled_offset_bytes = offset
        conn.audio.last_inferred_offset_bytes = offset
        conn.audio.state.chunk_index = 2
        conn.audio.state.emitted_text = "alpha"

    def test_sliced_inference_compacts_and_reuses_retained_tail(self):
        tokenizer_manager = _MockTokenizerManager(
            [
                "alpha beta tail",
                "alpha beta gamma",
                "alpha beta gamma delta",
                "delta more",
            ]
        )
        conn = self._conn(tokenizer_manager)
        conn.audio.append_pcm(bytes(range(4)))

        _run(conn._run_inference(is_last=False))
        conn.audio.append_pcm(bytes(range(4, 8)))
        _run(conn._run_inference(is_last=False))

        conn.audio.append_pcm(bytes(range(8, 12)))
        _run(conn._run_inference(is_last=False))

        self.assertEqual(tokenizer_manager.requests[2].text, "PROMPT:")
        expected = (
            np.frombuffer(bytes(range(6, 12)), dtype=np.int16).astype(np.float32)
            / 32768.0
        )
        np.testing.assert_array_equal(
            tokenizer_manager.requests[2].audio_data, expected
        )
        self.assertEqual(bytes(conn.audio.pcm_buffer), bytes(range(6, 12)))

        conn.audio.append_pcm(bytes(range(12, 16)))
        _run(conn._run_inference(is_last=False))

        expected = (
            np.frombuffer(bytes(range(10, 16)), dtype=np.int16).astype(np.float32)
            / 32768.0
        )
        np.testing.assert_array_equal(
            tokenizer_manager.requests[3].audio_data, expected
        )

    def test_skipped_window_cursor_semantics_and_final_commit(self):
        tokenizer_manager = _MockTokenizerManager("unused")
        conn = self._conn(tokenizer_manager)
        self._prime_sliced_state(conn, np.zeros(8, dtype=np.int16).tobytes())

        ok = _run(conn._run_inference(is_last=False))

        self.assertTrue(ok)
        self.assertEqual(len(tokenizer_manager.requests), 0)
        self.assertEqual(conn.audio.last_scheduled_offset_bytes, 16)
        self.assertEqual(conn.audio.last_inferred_offset_bytes, 8)

        tokenizer_manager = _MockTokenizerManager(["alpha beta gamma"])
        conn = self._conn(tokenizer_manager)
        voiced = np.full(6, 12000, dtype=np.int16).tobytes()
        silent = np.zeros(4, dtype=np.int16).tobytes()
        self._prime_sliced_state(conn, voiced + silent, offset=16)

        ok = _run(conn._run_inference(is_last=True))

        self.assertTrue(ok)
        self.assertEqual(len(tokenizer_manager.requests), 1)
        self.assertGreater(len(tokenizer_manager.requests[0].audio_data), 0)
        self.assertEqual(conn.audio.last_inferred_offset_bytes, 20)

    def test_unsafe_boundary_defers_but_commit_emits(self):
        tokenizer_manager = _MockTokenizerManager(["zulu beta gamma"])
        conn = self._conn(tokenizer_manager)
        samples = np.zeros(6, dtype=np.int16)
        samples[3] = 12000
        self._prime_sliced_state(conn, samples.tobytes())

        ok = _run(conn._run_inference(is_last=False))

        self.assertTrue(ok)
        self.assertEqual(len(tokenizer_manager.requests), 1)
        self.assertEqual(conn.item.emitted_deltas, [])
        self.assertEqual(conn.audio.last_scheduled_offset_bytes, 12)
        self.assertEqual(conn.audio.last_inferred_offset_bytes, 8)
        self.assertEqual(conn.audio.pcm_buffer_base_offset_bytes, 0)
        self.assertEqual(len(conn.audio.pcm_buffer), 12)
        self.assertEqual(conn.audio.state.chunk_index, 2)

        tokenizer_manager = _MockTokenizerManager(["zulu beta gamma"])
        conn = self._conn(tokenizer_manager)
        samples = np.zeros(6, dtype=np.int16)
        samples[3] = 12000
        self._prime_sliced_state(conn, samples.tobytes())

        ok = _run(conn._run_inference(is_last=True))

        self.assertTrue(ok)
        self.assertEqual(len(tokenizer_manager.requests), 1)
        self.assertNotEqual(conn.item.emitted_deltas, [])
        self.assertEqual(conn.audio.last_inferred_offset_bytes, 12)

    def test_failed_sliced_inference_does_not_compact_and_reset_clears_offsets(self):
        tokenizer_manager = _FailingTokenizerManager()
        conn = self._conn(tokenizer_manager)
        self._prime_sliced_state(conn)

        with self.assertLogs(level="WARNING"):
            ok = _run(conn._run_inference(is_last=False))

        self.assertFalse(ok)
        self.assertEqual(bytes(conn.audio.pcm_buffer), bytes(range(12)))
        self.assertEqual(conn.audio.last_inferred_offset_bytes, 8)
        # Append-time failure closes the socket with 1011 (internal error).
        self.assertEqual(conn.websocket.closed_code, 1011)

        conn._reset_inference_state()

        self.assertEqual(conn.audio.pcm_buffer, bytearray())
        self.assertEqual(conn.audio.total_pcm_bytes_received, 0)
        self.assertEqual(conn.audio.last_scheduled_offset_bytes, 0)
        self.assertEqual(conn.audio.last_inferred_offset_bytes, 0)

    def test_deferred_sentence_punctuation_flush_and_drop(self):
        conn = self._conn(_MockTokenizerManager("unused"))

        _run(
            conn._emit_transcription_delta(
                "Hello there.", defer_trailing_sentence_punctuation=True
            )
        )
        self.assertEqual("".join(conn.item.emitted_deltas), "Hello there")
        self.assertEqual(conn.item.pending_sentence_punctuation, ".")

        # Lowercase continuation: the deferred '.' was a false sentence end.
        _run(
            conn._emit_transcription_delta(
                "and more.", defer_trailing_sentence_punctuation=True
            )
        )
        self.assertEqual("".join(conn.item.emitted_deltas), "Hello there and more")

        # Non-lowercase continuation flushes the pending punctuation first.
        _run(conn._emit_transcription_delta("Next"))
        self.assertEqual(
            "".join(conn.item.emitted_deltas), "Hello there and more. Next"
        )

        # Commit-time flush emits whatever is still pending.
        conn.item.pending_sentence_punctuation = "?"
        _run(conn._flush_pending_sentence_punctuation())
        self.assertEqual(
            "".join(conn.item.emitted_deltas), "Hello there and more. Next?"
        )

        # CJK ender is deferred; a same-script CJK continuation drops it so the
        # boundary re-transcription can extend the clause seamlessly.
        conn = self._conn(_MockTokenizerManager("unused"))
        _run(
            conn._emit_transcription_delta(
                "甚至出现交易几乎停。", defer_trailing_sentence_punctuation=True
            )
        )
        self.assertEqual(conn.item.pending_sentence_punctuation, "。")
        _run(conn._emit_transcription_delta("滞的情况。"))
        self.assertEqual(
            "".join(conn.item.emitted_deltas), "甚至出现交易几乎停滞的情况。"
        )


if __name__ == "__main__":
    unittest.main()
