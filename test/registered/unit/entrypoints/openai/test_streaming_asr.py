"""Unit tests for the realtime ASR slicing path.

Drives the shared ``process_asr_chunk`` entry point with a mocked
``TokenizerManager`` (same style as ``test_serving_transcription`` /
``test_serving_embedding``) across the real scenarios: the cumulative (M1) and
sliced (M2) inference paths, output dedupe for Latin and CJK, the no-overlap and
empty-response edges, last-chunk finalize, and word reconciliation -- plus the
``RealtimeConnection`` guard that decides whether slicing turns on.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.entrypoints.openai.realtime.session import RealtimeConnection
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
        self._transcript = transcript
        self.requests = []

    def generate_request(self, adapted_request, raw_request=None):
        self.requests.append(adapted_request)
        transcript = self._transcript

        async def gen():
            if transcript is not None:
                yield {"text": transcript}

        return gen()


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


class TestSlicingEnabledGuard(CustomTestCase):
    def _conn(self, left_overlap_ms, enabled=True):
        server_args = SimpleNamespace(asr_max_buffer_seconds=60)
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


if __name__ == "__main__":
    unittest.main()
