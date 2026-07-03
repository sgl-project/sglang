"""Unit tests for the realtime ASR slicing path.

Drives the shared ``process_asr_chunk`` entry point and
``RealtimeConnection._run_inference`` with lightweight mocked model components.
The cases stay focused on the request paths and offset invariants that are most
likely to regress.
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
    _apply_sliced_dedupe,
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
    """Records the request and yields one synthetic transcript."""

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
        state.chunk_index = 5  # past unfixed_chunk_num, so the prefix is live
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

    def test_cumulative_early_revision_does_not_drop_new_words(self):
        # Regression: on the cumulative (HTTP SSE / pre-gate WS) path, when a
        # snapshot recases word 0 and shifts the confirmed word count (here the
        # model drops "nation"), the reconciler must not skip a genuinely-new
        # word. The old len(old_words) tail-slice dropped "and" between "up"
        # and "live" -- the "missing words" the reviewer saw.
        state = self._state(unfixed_chunk_num=0, unfixed_token_num=5)
        snapshots = [
            "The nation will rise",
            "The nation will rise up and",
            "The nation will rise up and live out the true",
            "the will rise up and live out the true meaning of its creed today",
        ]
        for snap in snapshots:
            self._chunk(state, snap)
        self._chunk(
            state,
            "the will rise up and live out the true meaning of its creed today done",
            is_last=True,
        )
        emitted = state.emitted_text.split()
        # Every distinct new word from the stream must survive (no drop).
        for word in ("rise", "up", "and", "live", "out", "meaning", "creed", "done"):
            self.assertIn(word, emitted, f"dropped new word: {word!r}")

        state = self._state(
            chunk_size_sec=2.0,
            unfixed_chunk_num=0,
            unfixed_token_num=0,
            confirmed_text=" ".join(f"w{i}" for i in range(35)),
            emitted_text=" ".join(f"w{i}" for i in range(35)),
        )
        _, out = self._chunk(
            state,
            f"{state.emitted_text} {state.emitted_text} fresh tail",
        )
        self.assertEqual(out, "fresh tail")

    def test_cjk_char_to_word_flip_does_not_duplicate(self):
        # Regression: a CJK->Latin code-switch flips the spaceless char path to
        # the whitespace word path; str.split collapses the prior CJK run into
        # one token, so the word reconciler must not re-emit the already-emitted
        # CJK prefix ("我今天很...我今天很...").
        state = StreamingASRState(
            chunk_size_sec=2.0, unfixed_chunk_num=2, unfixed_token_num=5
        )
        state.update("我今天很高兴认识你", cumulative=True)  # char path
        state.update("我今天很高兴认识你 Anna", cumulative=True)  # word path (flip)
        state.full_transcript = "我今天很高兴认识你 Anna 你叫什么名字"
        state.finalize(cumulative=True)
        self.assertEqual(state.emitted_text, "我今天很高兴认识你 Anna 你叫什么名字")

        repeat_state = StreamingASRState(
            chunk_size_sec=2.0, unfixed_chunk_num=2, unfixed_token_num=0
        )
        repeat_state.update("你好", cumulative=True)
        repeat_state.update("你好你好", cumulative=True)
        self.assertEqual(repeat_state.emitted_text, "你好你好")

    def test_sliced_path_emits_deduped_window_without_token_holdback(self):
        # Regression: on the sliced path (cumulative=False) dedupe_overlap is the
        # only intended dedup. Token-count holdback can drop words when their
        # audio falls outside the next overlap, so the deduped window is emitted.
        state = StreamingASRState(
            chunk_size_sec=2.0, unfixed_chunk_num=2, unfixed_token_num=2
        )
        state.confirmed_text = "the dog saw"
        state.emitted_text = "the dog saw"
        deduped = dedupe_overlap("the dog saw", "saw the cat ran up")
        self.assertEqual(deduped, "the cat ran up")
        state.update(deduped, cumulative=False)
        self.assertEqual(state.emitted_text, "the dog saw the cat ran up")

    def test_dedupe_overlap_only_trims_verbatim_prefix(self):
        # Regression (reviewer missing-word class): the sliced-overlap dedupe must
        # trim ONLY a verbatim normalized *prefix* of the candidate that matches
        # the committed tail; it must never delete unmatched leading words or
        # rewrite text when nothing overlaps.
        # (a) a genuine verbatim overlap prefix is trimmed.
        self.assertEqual(
            dedupe_overlap(
                "he hoped there would be stew for dinner turnips",
                "turnips and carrots and bruised",
            ),
            "and carrots and bruised",
        )
        # (b) a phrase that recurs later in the candidate must NOT drag the
        #     genuine new leading words out with it (the old fuzzy leading-skip
        #     deleted "x y" here).
        self.assertEqual(
            dedupe_overlap(
                "one two three four five six", "x y three four five six seven"
            ),
            "x y three four five six seven",
        )
        # (c) no overlap -> the candidate is returned byte-for-byte, never
        #     re-tokenized or rewritten (em/en dashes are preserved).
        self.assertEqual(
            dedupe_overlap("alpha beta", "fresh dinner—turnips text"),
            "fresh dinner—turnips text",
        )

    def test_sliced_dedupe_safety_verdict(self):
        # The default sliced dedupe reports whether the overlap trim is provably
        # safe. A voiced overlap the exact-prefix dedupe cannot match (reworded)
        # is UNVERIFIED (caller defers); the same non-match over a SILENT overlap
        # is genuinely new content and is verified (safe to emit). A verbatim
        # overlap is trimmed and verified.
        voiced = np.full(16000, 0.1, dtype=np.float32)
        silent = np.zeros(16000, dtype=np.float32)
        # reworded/unmatched over VOICED overlap -> nothing deleted, NOT verified.
        text, verified = _apply_sliced_dedupe(
            "alpha beta", "gamma delta epsilon", voiced, 16000, 0.5
        )
        self.assertEqual(text, "gamma delta epsilon")
        self.assertFalse(verified)
        # same unmatched candidate over a SILENT overlap -> new content, verified.
        text, verified = _apply_sliced_dedupe(
            "alpha beta", "gamma delta epsilon", silent, 16000, 0.5
        )
        self.assertEqual(text, "gamma delta epsilon")
        self.assertTrue(verified)
        # a verbatim overlap prefix is trimmed and verified.
        text, verified = _apply_sliced_dedupe(
            "one two three", "three four five", voiced, 16000, 0.5
        )
        self.assertEqual(text, "four five")
        self.assertTrue(verified)


class _SlicingAdapter:
    """Minimal adapter exposing only what RealtimeConnection.__init__ reads."""

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
        self.assertFalse(self._conn(left_overlap_ms=8000).audio.slicing_enabled)
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

    def _prime_sliced_state(self, conn):
        conn.audio.append_pcm(bytes(range(12)))
        conn.audio.last_scheduled_offset_bytes = 8
        conn.audio.last_inferred_offset_bytes = 8
        conn.audio.last_sliced_buffer_end_bytes = 8
        conn.audio.state.chunk_index = 2
        conn.audio.state.emitted_text = "alpha"

    def test_first_sliced_inference_uses_cumulative_anchor_and_compacts(self):
        tokenizer_manager = _MockTokenizerManager(
            ["alpha beta tail", "alpha beta gamma", "alpha beta gamma delta"]
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

    def test_next_sliced_inference_uses_retained_tail_after_compaction(self):
        tokenizer_manager = _MockTokenizerManager(
            ["alpha beta gamma", "alpha beta gamma delta"]
        )
        conn = self._conn(tokenizer_manager)
        self._prime_sliced_state(conn)

        _run(conn._run_inference(is_last=False))
        conn.audio.append_pcm(bytes(range(12, 16)))
        _run(conn._run_inference(is_last=False))

        expected = (
            np.frombuffer(bytes(range(10, 16)), dtype=np.int16).astype(np.float32)
            / 32768.0
        )
        np.testing.assert_array_equal(
            tokenizer_manager.requests[1].audio_data, expected
        )

    def test_skipped_window_cursor_semantics_and_final_commit(self):
        # Mid-stream skip advances only scheduling; final commit still infers
        # pending audio by widening to retained voiced PCM.
        tokenizer_manager = _MockTokenizerManager("unused")
        conn = self._conn(tokenizer_manager)
        conn.audio.append_pcm(np.zeros(8, dtype=np.int16).tobytes())  # 16 bytes silence
        conn.audio.last_scheduled_offset_bytes = 8
        conn.audio.last_inferred_offset_bytes = 8
        conn.audio.last_sliced_buffer_end_bytes = 8
        conn.audio.state.chunk_index = 2
        conn.audio.state.emitted_text = "alpha"

        ok = _run(conn._run_inference(is_last=False))

        self.assertTrue(ok)
        self.assertEqual(len(tokenizer_manager.requests), 0)  # skipped, no model call
        self.assertEqual(conn.audio.last_scheduled_offset_bytes, 16)  # advanced
        self.assertEqual(conn.audio.last_inferred_offset_bytes, 8)  # NOT advanced
        self.assertEqual(conn.audio.last_sliced_buffer_end_bytes, 8)  # anchor frozen

        tokenizer_manager = _MockTokenizerManager(["alpha beta gamma"])
        conn = self._conn(tokenizer_manager)
        voiced = np.full(6, 12000, dtype=np.int16).tobytes()  # 12 bytes, loud
        silent = np.zeros(4, dtype=np.int16).tobytes()  # 8 bytes, silence
        conn.audio.append_pcm(voiced + silent)  # total = 20
        conn.audio.last_scheduled_offset_bytes = 12
        conn.audio.last_inferred_offset_bytes = 12
        conn.audio.last_sliced_buffer_end_bytes = 16  # bounded window [14,20]=silent
        conn.audio.state.chunk_index = 2
        conn.audio.state.emitted_text = "alpha"

        ok = _run(conn._run_inference(is_last=True))

        self.assertTrue(ok)
        # The model WAS called (final commit did a real inference, not a skip).
        self.assertEqual(len(tokenizer_manager.requests), 1)
        self.assertGreater(len(tokenizer_manager.requests[0].audio_data), 0)
        self.assertEqual(conn.audio.last_inferred_offset_bytes, 20)  # consumed

    def test_unsafe_boundary_defers_but_commit_emits(self):
        # Voiced unmatched overlap is deferred mid-stream without compacting, but
        # commit always emits so pending audio is never dropped.
        tokenizer_manager = _MockTokenizerManager(["zulu beta gamma"])  # no "alpha"
        conn = self._conn(tokenizer_manager)
        samples = np.zeros(6, dtype=np.int16)
        samples[3] = 12000  # loud int16 at byte offset 6 == the overlap region
        conn.audio.append_pcm(samples.tobytes())  # 12 bytes
        conn.audio.last_scheduled_offset_bytes = 8
        conn.audio.last_inferred_offset_bytes = 8
        conn.audio.last_sliced_buffer_end_bytes = 8
        conn.audio.state.chunk_index = 2
        conn.audio.state.emitted_text = "alpha"

        ok = _run(conn._run_inference(is_last=False))

        self.assertTrue(ok)
        self.assertEqual(len(tokenizer_manager.requests), 1)  # model ran...
        self.assertEqual(conn.item.emitted_deltas, [])  # ...but nothing emitted
        self.assertEqual(conn.audio.last_scheduled_offset_bytes, 12)  # advanced
        self.assertEqual(conn.audio.last_inferred_offset_bytes, 8)  # NOT advanced
        self.assertEqual(conn.audio.last_sliced_buffer_end_bytes, 8)  # anchor frozen
        self.assertEqual(conn.audio.pcm_buffer_base_offset_bytes, 0)  # NOT compacted
        self.assertEqual(len(conn.audio.pcm_buffer), 12)  # audio recoverable
        self.assertEqual(conn.audio.state.chunk_index, 2)  # not ingested into state

        tokenizer_manager = _MockTokenizerManager(["zulu beta gamma"])
        conn = self._conn(tokenizer_manager)
        samples = np.zeros(6, dtype=np.int16)
        samples[3] = 12000
        conn.audio.append_pcm(samples.tobytes())
        conn.audio.last_scheduled_offset_bytes = 8
        conn.audio.last_inferred_offset_bytes = 8
        conn.audio.last_sliced_buffer_end_bytes = 8
        conn.audio.state.chunk_index = 2
        conn.audio.state.emitted_text = "alpha"

        ok = _run(conn._run_inference(is_last=True))

        self.assertTrue(ok)
        self.assertEqual(len(tokenizer_manager.requests), 1)  # model ran
        self.assertNotEqual(conn.item.emitted_deltas, [])  # emitted (not deferred)
        self.assertEqual(conn.audio.last_inferred_offset_bytes, 12)  # consumed

    def test_failed_sliced_inference_does_not_compact(self):
        tokenizer_manager = _FailingTokenizerManager()
        conn = self._conn(tokenizer_manager)
        self._prime_sliced_state(conn)

        ok = _run(conn._run_inference(is_last=False))

        self.assertFalse(ok)
        self.assertEqual(bytes(conn.audio.pcm_buffer), bytes(range(12)))
        self.assertEqual(conn.audio.last_sliced_buffer_end_bytes, 8)

    def test_reset_clears_absolute_offsets(self):
        conn = self._conn(_MockTokenizerManager("unused"))
        conn.audio.append_pcm(bytes(range(12)))
        conn.audio.last_sliced_buffer_end_bytes = 12
        conn.audio.last_scheduled_offset_bytes = 12
        conn.audio.last_inferred_offset_bytes = 12
        conn.audio.compact_after_sliced_inference()

        conn._reset_inference_state()

        self.assertEqual(conn.audio.pcm_buffer, bytearray())
        self.assertEqual(conn.audio.total_pcm_bytes_received, 0)
        self.assertEqual(conn.audio.last_sliced_buffer_end_bytes, 0)
        self.assertEqual(conn.audio.last_scheduled_offset_bytes, 0)
        self.assertEqual(conn.audio.last_inferred_offset_bytes, 0)

    def test_sliced_boundary_sentence_punctuation_is_deferred(self):
        conn = self._conn(_MockTokenizerManager("unused"))

        _run(
            conn._emit_transcription_delta(
                "I have a dream.",
                defer_trailing_sentence_punctuation=True,
            )
        )
        _run(
            conn._emit_transcription_delta(
                "that one day",
                defer_trailing_sentence_punctuation=True,
            )
        )
        self.assertEqual(
            "".join(conn.item.emitted_deltas), "I have a dream that one day"
        )

        conn = self._conn(_MockTokenizerManager("unused"))
        _run(
            conn._emit_transcription_delta(
                "its creed.",
                defer_trailing_sentence_punctuation=True,
            )
        )
        _run(
            conn._emit_transcription_delta(
                "He hoped",
                defer_trailing_sentence_punctuation=True,
            )
        )
        self.assertEqual("".join(conn.item.emitted_deltas), "its creed. He hoped")


if __name__ == "__main__":
    unittest.main()
