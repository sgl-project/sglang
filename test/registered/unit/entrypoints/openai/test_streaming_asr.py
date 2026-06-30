"""Unit tests for the realtime ASR slicing path.

Drives the shared ``process_asr_chunk`` entry point with a mocked
``TokenizerManager`` (same style as ``test_serving_transcription`` /
``test_serving_embedding``) across the real scenarios: the cumulative (M1) and
sliced (M2) inference paths, word-level output dedupe, the no-overlap and
empty-response edges, last-chunk finalize, and word reconciliation -- plus the
``RealtimeConnection`` guard that decides whether slicing turns on and the
rolling PCM compaction invariants for sliced realtime ASR.
"""

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
    """Records the request and yields one synthetic transcript (or nothing, when
    ``transcript`` is None, to exercise the empty-response path)."""

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

    def test_cumulative_path_injects_prefix_and_skips_dedupe(self):
        # prompt=None -> prompt_template + get_prefix_text(), no dedupe (M1).
        state = self._state()
        state.emitted_text = "hello"
        state.chunk_index = 5  # past unfixed_chunk_num, so the prefix is live
        tm, _ = self._chunk(state, "hello world foo")
        self.assertEqual(tm.requests[0].text, "PROMPT:hello")
        self.assertEqual(state.full_transcript, "hello world foo")

    def test_slicing_path_uses_bare_prompt_and_dedupes(self):
        # Bare prompt (no prefix injection); dedupe trims the word that overlaps
        # the committed tail (M2).
        state = self._state()
        tm, _ = self._chunk(
            state, "beta gamma", prompt="PROMPT:", dedupe_against="alpha beta"
        )
        self.assertEqual(tm.requests[0].text, "PROMPT:")
        self.assertEqual(state.full_transcript, "gamma")

    def test_slicing_path_keeps_non_overlapping_candidate(self):
        # No overlap with the committed tail -> nothing is trimmed.
        state = self._state()
        self._chunk(state, "gamma delta", prompt="PROMPT:", dedupe_against="alpha beta")
        self.assertEqual(state.full_transcript, "gamma delta")

    def test_last_chunk_dedupes_then_finalizes(self):
        # The final chunk dedupes against the committed tail, then finalize()
        # emits the remaining text.
        state = self._state()
        _, out = self._chunk(
            state,
            "alpha beta gamma",
            is_last=True,
            prompt="PROMPT:",
            dedupe_against="alpha",
        )
        self.assertEqual(out, "beta gamma")
        self.assertEqual(state.full_transcript, "beta gamma")

    def test_extended_word_emits_whole_word_not_fragment(self):
        # "world" re-transcribed as "worldly" must emit "worldly", not "ly"
        # (regression guard for the removed char-level startswith fast path).
        state = self._state(
            unfixed_chunk_num=0, unfixed_token_num=1, confirmed_text="hello world"
        )
        _, out = self._chunk(state, "hello worldly test tail")
        self.assertEqual(out, "worldly test")

    def test_first_word_recasing_emits_append_only_tail(self):
        # Cumulative snapshots can revise capitalization or punctuation in
        # already-emitted words. Do not re-emit the whole transcript when the
        # exact common prefix drops to zero.
        state = self._state(
            unfixed_chunk_num=0,
            unfixed_token_num=1,
            confirmed_text="hello world",
            emitted_text="hello world",
        )
        _, out = self._chunk(state, "Hello world again tail")
        self.assertEqual(out, "again")

        state = self._state(
            confirmed_text="hello world", emitted_text="hello world"
        )
        _, out = self._chunk(state, "Hello world again", is_last=True)
        self.assertEqual(out, "again")

    def test_cjk_text_emits_incremental_delta(self):
        state = self._state(unfixed_token_num=1)
        _, out = self._chunk(state, "你好世界")
        self.assertEqual(out, "你好世")

    def test_adjacent_cjk_deltas_join_without_spaces(self):
        state = self._state(unfixed_token_num=1)
        self._chunk(state, "你好世界")
        _, out = self._chunk(state, "你好世界好")
        self.assertEqual(out, "界")
        self.assertEqual(state.emitted_text, "你好世界")

    def test_empty_model_response_emits_nothing(self):
        # No model output -> empty delta, no state mutation, no crash.
        state = self._state()
        _, out = self._chunk(state, None)
        self.assertEqual(out, "")
        self.assertEqual(state.full_transcript, "")


class _SlicingAdapter:
    """Minimal adapter exposing only what RealtimeConnection.__init__ reads."""

    model_sample_rate = 16000

    def __init__(self, left_overlap_ms, enabled=True):
        self._left_overlap_ms = left_overlap_ms
        self._enabled = enabled

    @property
    def realtime_slicing_config(self):
        return {
            "enabled": self._enabled,
            "left_overlap_ms": self._left_overlap_ms,
            "min_audio_sec": 16.0,
        }

    @property
    def chunked_streaming_config(self):
        # 2s chunks, 2 unfixed chunks -> 4s unfixed window.
        return {"chunk_size_sec": 2.0, "unfixed_chunk_num": 2, "unfixed_token_num": 5}


class _RuntimeSlicingAdapter:
    """Small byte geometry for exercising RealtimeConnection._run_inference."""

    model_sample_rate = 1
    prompt_template = "PROMPT:"

    @property
    def realtime_slicing_config(self):
        return {"enabled": True, "left_overlap_ms": 1000, "min_audio_sec": 0.0}

    @property
    def chunked_streaming_config(self):
        # 1 Hz * 2 bytes/sample * 2s = 4-byte chunks; 1s overlap = 2 bytes.
        return {"chunk_size_sec": 2.0, "unfixed_chunk_num": 2, "unfixed_token_num": 1}

    def postprocess_text(self, text: str) -> str:
        return text


class TestSlicingEnabledGuard(CustomTestCase):
    def _conn(self, left_overlap_ms, enabled=True):
        server_args = SimpleNamespace(
            asr_max_buffer_seconds=60, asr_disable_input_slicing=False
        )
        return RealtimeConnection(
            object(), object(), _SlicingAdapter(left_overlap_ms, enabled), server_args
        )

    def test_enabled_only_when_overlap_fits_unfixed_window(self):
        # 2s overlap fits the 4s window -> slicing on; 8s overlap makes the
        # dedupe target unreachable -> guard falls back to cumulative.
        self.assertTrue(self._conn(left_overlap_ms=2000).audio.slicing_enabled)
        self.assertFalse(self._conn(left_overlap_ms=8000).audio.slicing_enabled)

    def test_disabled_when_adapter_opts_out(self):
        # enabled=False (the base-adapter default) -> never slices.
        self.assertFalse(
            self._conn(left_overlap_ms=2000, enabled=False).audio.slicing_enabled
        )


class TestSlicingSampleAlignment(CustomTestCase):
    def test_left_overlap_bytes_snapped_to_whole_sample(self):
        # 22050 Hz * 250ms = 11025 bytes raw (odd); an odd slice start would
        # split an int16 frame and raise in pcm_to_float_samples. Must snap
        # down to a whole 2-byte sample.
        class _OddRateAdapter(_SlicingAdapter):
            model_sample_rate = 22050

        server_args = SimpleNamespace(
            asr_max_buffer_seconds=60, asr_disable_input_slicing=False
        )
        conn = RealtimeConnection(
            object(), object(), _OddRateAdapter(left_overlap_ms=250), server_args
        )
        self.assertEqual(conn.audio.left_overlap_bytes % 2, 0)
        self.assertEqual(conn.audio.left_overlap_bytes, 11024)
        self.assertTrue(conn.audio.slicing_enabled)


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

    def _prime_sliced_state(self, conn):
        conn.audio.append_pcm(bytes(range(12)))
        conn.audio.last_inference_offset_bytes = 8
        conn.audio.last_sliced_buffer_end_bytes = 8
        conn.audio.state.chunk_index = 2
        conn.audio.state.emitted_text = "alpha"

    def test_first_sliced_inference_uses_cumulative_anchor_and_compacts(self):
        tokenizer_manager = _MockTokenizerManager(
            ["alpha beta tail", "alpha beta gamma", "alpha beta gamma delta"]
        )
        conn = self._conn(tokenizer_manager)
        conn.audio.append_pcm(bytes(range(4)))

        self.assertTrue(_run(conn._run_inference(is_last=False)))
        self.assertEqual(conn.audio.last_sliced_buffer_end_bytes, 4)

        conn.audio.append_pcm(bytes(range(4, 8)))
        self.assertTrue(_run(conn._run_inference(is_last=False)))
        self.assertEqual(conn.audio.last_sliced_buffer_end_bytes, 8)

        conn.audio.append_pcm(bytes(range(8, 12)))
        ok = _run(conn._run_inference(is_last=False))

        self.assertTrue(ok)
        self.assertEqual(tokenizer_manager.requests[2].text, "PROMPT:")
        expected = (
            np.frombuffer(bytes(range(6, 12)), dtype=np.int16).astype(np.float32)
            / 32768.0
        )
        np.testing.assert_array_equal(
            tokenizer_manager.requests[2].audio_data, expected
        )
        self.assertEqual(conn.audio.last_sliced_buffer_end_bytes, 12)
        self.assertEqual(conn.audio.last_inference_offset_bytes, 12)
        self.assertEqual(conn.audio.pcm_buffer_base_offset_bytes, 6)
        self.assertEqual(bytes(conn.audio.pcm_buffer), bytes(range(6, 12)))

    def test_next_sliced_inference_uses_retained_tail_after_compaction(self):
        tokenizer_manager = _MockTokenizerManager(
            ["alpha beta gamma", "alpha beta gamma delta"]
        )
        conn = self._conn(tokenizer_manager)
        self._prime_sliced_state(conn)

        self.assertTrue(_run(conn._run_inference(is_last=False)))
        conn.audio.append_pcm(bytes(range(12, 16)))
        self.assertTrue(_run(conn._run_inference(is_last=False)))

        expected = (
            np.frombuffer(bytes(range(10, 16)), dtype=np.int16).astype(np.float32)
            / 32768.0
        )
        np.testing.assert_array_equal(
            tokenizer_manager.requests[1].audio_data, expected
        )

    def test_failed_sliced_inference_does_not_compact(self):
        tokenizer_manager = _FailingTokenizerManager()
        conn = self._conn(tokenizer_manager)
        self._prime_sliced_state(conn)

        ok = _run(conn._run_inference(is_last=False))

        self.assertFalse(ok)
        self.assertEqual(conn.websocket.closed_code, 1011)
        self.assertEqual(conn.audio.pcm_buffer_base_offset_bytes, 0)
        self.assertEqual(bytes(conn.audio.pcm_buffer), bytes(range(12)))
        self.assertEqual(conn.audio.last_sliced_buffer_end_bytes, 8)
        self.assertEqual(conn.audio.last_inference_offset_bytes, 8)

    def test_reset_clears_absolute_offsets(self):
        conn = self._conn(_MockTokenizerManager("unused"))
        conn.audio.append_pcm(bytes(range(12)))
        conn.audio.last_sliced_buffer_end_bytes = 12
        conn.audio.last_inference_offset_bytes = 12
        conn.audio.compact_after_sliced_inference()

        conn._reset_inference_state()

        self.assertEqual(conn.audio.pcm_buffer, bytearray())
        self.assertEqual(conn.audio.pcm_buffer_base_offset_bytes, 0)
        self.assertEqual(conn.audio.total_pcm_bytes_received, 0)
        self.assertEqual(conn.audio.last_inference_offset_bytes, 0)
        self.assertEqual(conn.audio.last_sliced_buffer_end_bytes, 0)

    def test_cumulative_flip_after_compaction_does_not_crash(self):
        # A within-item sliced->cumulative flip: a slice compacts the buffer
        # (base advances), then emitted_text becomes CJK-no-whitespace so
        # get_prefix_text() returns "" and the next inference takes the
        # cumulative branch (slice_start_global=0 < base). The local start must
        # clamp to 0 instead of going negative and crashing the live session.
        tokenizer_manager = _MockTokenizerManager(["alpha beta gamma", "你好世界"])
        conn = self._conn(tokenizer_manager)
        self._prime_sliced_state(conn)

        # First inference slices + compacts: base advances past 0.
        self.assertTrue(_run(conn._run_inference(is_last=False)))
        self.assertEqual(conn.audio.pcm_buffer_base_offset_bytes, 6)

        # Force the flip: more audio arrives and the transcript so far is now
        # CJK-no-whitespace, disabling slicing for the next call.
        conn.audio.append_pcm(bytes(range(16, 20)))
        conn.audio.state.emitted_text = "你好世界"

        ok = _run(conn._run_inference(is_last=False))

        # Before the clamp fix this raised (negative slice start), surfaced as a
        # spurious inference failure that closed the live WS with 1011.
        self.assertTrue(ok)
        self.assertIsNone(conn.websocket.closed_code)
        self.assertEqual(len(tokenizer_manager.requests), 2)
        # Cumulative slice spans the whole retained buffer: global [6, 16) ->
        # local [0, 10) -> 5 int16 samples.
        self.assertEqual(len(tokenizer_manager.requests[1].audio_data), 5)


if __name__ == "__main__":
    unittest.main()
