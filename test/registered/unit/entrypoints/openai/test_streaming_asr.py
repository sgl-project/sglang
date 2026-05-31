"""Unit tests for the realtime ASR slicing path: process_asr_chunk's prompt
override + dedupe, update() reconciliation, and the dedupe / PCM helpers."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import io
import unittest
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from sglang.srt.entrypoints.openai.realtime.session import (
    RealtimeConnection,
    _pcm_to_float_samples,
    _slice_pcm_from,
)
from sglang.srt.entrypoints.openai.streaming_asr import (
    StreamingASRState,
    dedupe_overlap,
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
    """Yields one synthetic transcript and records the request, so tests can
    assert on the prompt that was sent."""

    def __init__(self, transcript: str):
        self._transcript = transcript
        self.requests = []

    def generate_request(self, adapted_request, raw_request=None):
        self.requests.append(adapted_request)
        transcript = self._transcript

        async def gen():
            yield {"text": transcript}

        return gen()


def _run(coro):
    return get_or_create_event_loop().run_until_complete(coro)


_AUDIO = np.zeros(1600, dtype=np.float32)


class TestProcessAsrChunkSlicing(CustomTestCase):
    def _state(self):
        return StreamingASRState(
            chunk_size_sec=1.0, unfixed_chunk_num=2, unfixed_token_num=2
        )

    def test_cumulative_path_injects_prefix_and_skips_dedupe(self):
        # prompt=None -> prompt_template + get_prefix_text(), no dedupe.
        state = self._state()
        state.emitted_text = "hello"
        state.chunk_index = 5  # past unfixed_chunk_num, so the prefix is live
        tm = _MockTokenizerManager("hello world foo")

        _run(
            process_asr_chunk(
                tokenizer_manager=tm,
                adapter=_FakeAdapter(),
                state=state,
                audio_data=_AUDIO,
                sampling_params={},
                is_last=False,
            )
        )

        self.assertEqual(tm.requests[0].text, "PROMPT:hello")
        self.assertEqual(state.full_transcript, "hello world foo")

    def test_slicing_path_uses_bare_prompt_and_dedupes(self):
        # Bare prompt (no prefix injection); dedupe trims the overlapping word.
        state = self._state()
        tm = _MockTokenizerManager("beta gamma")

        _run(
            process_asr_chunk(
                tokenizer_manager=tm,
                adapter=_FakeAdapter(),
                state=state,
                audio_data=_AUDIO,
                sampling_params={},
                is_last=False,
                prompt="PROMPT:",
                dedupe_against="alpha beta",
            )
        )

        self.assertEqual(tm.requests[0].text, "PROMPT:")
        self.assertEqual(state.full_transcript, "gamma")

    def test_is_last_dedupes_then_finalizes(self):
        # The final chunk also dedupes before finalize().
        state = self._state()
        tm = _MockTokenizerManager("alpha beta gamma")

        out = _run(
            process_asr_chunk(
                tokenizer_manager=tm,
                adapter=_FakeAdapter(),
                state=state,
                audio_data=_AUDIO,
                sampling_params={},
                is_last=True,
                prompt="PROMPT:",
                dedupe_against="alpha",
            )
        )

        self.assertEqual(state.full_transcript, "beta gamma")
        self.assertEqual(out, "beta gamma")


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
        # 2s chunks, 2 unfixed chunks -> 4s (=4000ms) unfixed window.
        return {"chunk_size_sec": 2.0, "unfixed_chunk_num": 2, "unfixed_token_num": 5}


class TestSlicingEnabledGuard(CustomTestCase):
    """RealtimeConnection.__init__ guard: slicing only turns on when opted in
    AND the left overlap fits inside the unfixed-chunk window. Pure CPU — no
    GPU, tokenizer, or websocket I/O is touched in __init__."""

    def _conn(self, left_overlap_ms, enabled=True):
        server_args = SimpleNamespace(asr_max_buffer_seconds=60)
        return RealtimeConnection(
            object(), object(), _SlicingAdapter(left_overlap_ms, enabled), server_args
        )

    def test_overlap_within_unfixed_window_enables_slicing(self):
        # 2s overlap < 4s unfixed window -> slicing turns on.
        self.assertTrue(self._conn(left_overlap_ms=2000).audio.slicing_enabled)

    def test_overlap_exceeding_unfixed_window_disables_slicing(self):
        # 8s overlap >= 4s window: the dedupe target is unreachable, so the
        # guard disables slicing and falls back to cumulative inference.
        self.assertFalse(self._conn(left_overlap_ms=8000).audio.slicing_enabled)

    def test_opt_out_disables_slicing(self):
        # enabled=False: never slices regardless of overlap.
        self.assertFalse(
            self._conn(left_overlap_ms=2000, enabled=False).audio.slicing_enabled
        )


class TestStreamingASRStateUpdate(CustomTestCase):
    def test_extended_word_emits_whole_word_not_fragment(self):
        # "world" re-transcribed as "worldly" must emit "worldly", not "ly"
        # (regression guard for the removed char-level startswith fast path).
        state = StreamingASRState(
            chunk_size_sec=1.0,
            unfixed_chunk_num=0,
            unfixed_token_num=1,
            confirmed_text="hello world",
        )
        self.assertEqual(state.update("hello worldly test tail"), "worldly test")

    def test_clean_append_emits_only_new_words(self):
        state = StreamingASRState(
            chunk_size_sec=1.0,
            unfixed_chunk_num=0,
            unfixed_token_num=1,
            confirmed_text="hello",
        )
        self.assertEqual(state.update("hello world tail"), "world")


class TestDedupeOverlap(CustomTestCase):
    def test_full_candidate_overlap_returns_empty(self):
        self.assertEqual(dedupe_overlap("hello world", "hello world"), "")

    def test_em_dash_and_case_normalized_during_match(self):
        # Trailing em dash and case are stripped before matching.
        self.assertEqual(
            dedupe_overlap("stew for dinner—", "Dinner: turnips"), "turnips"
        )

    def test_cjk_char_level_fallback(self):
        # No whitespace -> word-level can't match -> CJK char-level fallback.
        self.assertEqual(dedupe_overlap("你好世界", "世界今天很好"), "今天很好")

    def test_long_history_matches_only_committed_suffix(self):
        # A large unrelated prefix must not change the suffix match.
        committed = " ".join(["old"] * 1000 + ["a", "b", "c"])
        self.assertEqual(dedupe_overlap(committed, "b c d"), "d")


class TestPcmHelpers(CustomTestCase):
    def test_pcm_to_float_matches_soundfile_round_trip(self):
        # Direct PCM16->float stays bit-equal to the legacy PCM->WAV->sf.read.
        rng = np.random.default_rng(42)
        ints = rng.integers(-32768, 32768, size=4096, dtype=np.int16)

        direct = _pcm_to_float_samples(ints.tobytes())

        buf = io.BytesIO()
        sf.write(buf, ints, 16000, format="WAV")
        buf.seek(0)
        round_trip, _ = sf.read(buf)

        np.testing.assert_array_equal(direct, round_trip)

    def test_slice_out_of_bounds_start_raises(self):
        # Out-of-range start raises rather than returning a wrong window.
        with self.assertRaises(ValueError):
            _slice_pcm_from(b"abcdef", -1)
        with self.assertRaises(ValueError):
            _slice_pcm_from(b"abcdef", 7)


if __name__ == "__main__":
    unittest.main()
