import unittest

from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)


class TestSamplingBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_max_new_tokens_zero_rejected(self):
        self.server.execute_script(self._script_max_new_tokens_zero_rejected)

    @staticmethod
    def _script_max_new_tokens_zero_rejected(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=0)
        for _ in range(DEFAULT_MAX_STEPS):
            if r.req is not None and isinstance(r.req.finished_reason, FINISH_ABORT):
                return
            if r.finished:
                return
            yield
        raise AssertionError("max_new_tokens=0 should fast-fail with an error_message")

    def test_max_new_tokens_one_long_chunked(self):
        self.server.execute_script(self._script_max_new_tokens_one_long_chunked)

    @staticmethod
    def _script_max_new_tokens_one_long_chunked(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=1)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.req.output_ids) == 1, (
            f"max_new_tokens=1 must produce exactly 1 token, got "
            f"{len(r.req.output_ids)}"
        )

    def test_max_new_tokens_1000_long_chunked(self):
        self.server.execute_script(self._script_max_new_tokens_1000_long_chunked)

    @staticmethod
    def _script_max_new_tokens_1000_long_chunked(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=1000,
            ignore_eos=True,
        )
        yield from run_until(r, lambda h: h.finished, max_steps=2000)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.req.output_ids) == 1000, (
            f"ignore_eos=True + max_new_tokens=1000 must produce 1000 "
            f"output tokens; got {len(r.req.output_ids)}"
        )

    def test_greedy_chunked_deterministic(self):
        self.server.execute_script(self._script_greedy_chunked_deterministic)

    @staticmethod
    def _script_greedy_chunked_deterministic(t: ScriptedContext):
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=8,
            temperature=0.0,
            ignore_eos=True,
        )
        yield from run_until_finished(r1)
        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=8,
            temperature=0.0,
            ignore_eos=True,
        )
        yield from run_until_finished(r2)
        assert (
            r1.req.output_ids == r2.req.output_ids
        ), f"greedy non-determinism: {r1.req.output_ids} != {r2.req.output_ids}"
        assert len(r1.req.output_ids) == 8
        assert r1.chunks_done >= 2 and r2.chunks_done >= 2

    def test_return_logprob_chunked(self):
        self.server.execute_script(self._script_return_logprob_chunked)

    @staticmethod
    def _script_return_logprob_chunked(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            return_logprob=True,
            ignore_eos=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert r.logprobs is not None
        assert len(r.logprobs) == 4

    def test_ignore_eos_chunked(self):
        self.server.execute_script(self._script_ignore_eos_chunked)

    @staticmethod
    def _script_ignore_eos_chunked(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=16, ignore_eos=True
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.req.output_ids) == 16
        assert r.req.finished_reason == "length", (
            f"ignore_eos=True must finish via length cap; got " f"{r.req.finished_reason!r}"
        )

    def test_stop_str_chunked(self):
        self.server.execute_script(self._script_stop_str_chunked)

    @staticmethod
    def _script_stop_str_chunked(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=512, stop=["xyz"]
        )
        yield from run_until(r, lambda h: h.finished, max_steps=2000)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.req.output_ids) < 512
        assert r.req.finished_reason in ("stop", "length"), (
            f"finish_reason must be populated for chunked stop-str runs; "
            f"got {r.req.finished_reason!r}"
        )

    def test_high_temperature_chunked(self):
        self.server.execute_script(self._script_high_temperature_chunked)

    @staticmethod
    def _script_high_temperature_chunked(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            temperature=2.0,
            ignore_eos=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.req.output_ids) == 4

    def test_greedy_two_sequential_reqs(self):
        self.server.execute_script(self._script_greedy_two_sequential_reqs)

    @staticmethod
    def _script_greedy_two_sequential_reqs(t: ScriptedContext):
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            temperature=0.0,
            ignore_eos=True,
        )
        yield from run_until_finished(r1)
        out_a = list(r1.req.output_ids)

        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            temperature=0.0,
            ignore_eos=True,
        )
        yield from run_until_finished(r2)
        assert list(r2.req.output_ids) == out_a
        assert len(out_a) == 4
        assert r1.chunks_done >= 2 and r2.chunks_done >= 2

    def test_greedy_chunked_with_radix_hit(self):
        self.server.execute_script(self._script_greedy_chunked_with_radix_hit)

    @staticmethod
    def _script_greedy_chunked_with_radix_hit(t: ScriptedContext):
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            temperature=0.0,
            ignore_eos=True,
        )
        yield from run_until_finished(r1)
        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            temperature=0.0,
            ignore_eos=True,
        )
        yield from run_until_finished(r2)
        assert list(r1.req.output_ids) == list(r2.req.output_ids)
        assert len(r1.req.output_ids) == 4
        assert r2.req.cached_tokens > 0, (
            f"r2 must hit r1's radix prefix; got cached_tokens=" f"{r2.req.cached_tokens}"
        )

    def test_return_logprob_top_logprobs_chunked(self):
        self.server.execute_script(self._script_return_logprob_top_logprobs_chunked)

    @staticmethod
    def _script_return_logprob_top_logprobs_chunked(t: ScriptedContext):
        top_k = 5
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            return_logprob=True,
            top_logprobs_num=top_k,
            ignore_eos=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert r.logprobs is not None
        top = r.logprobs.output_token_top_logprobs_val
        assert len(top) == 4, (
            f"top logprobs must be reported once per output token; "
            f"got {len(top)} entries for 4 tokens"
        )
        for step_entries in top:
            assert len(step_entries) == top_k, (
                f"each step must carry exactly top_logprobs_num={top_k} "
                f"entries; got {len(step_entries)}"
            )

    def test_multiple_stop_strs_chunked(self):
        self.server.execute_script(self._script_multiple_stop_strs_chunked)

    @staticmethod
    def _script_multiple_stop_strs_chunked(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=64,
            stop=["a", "b", "c"],
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.req.output_ids) < 64, (
            f"a 64-token decode budget over common stop chars must stop "
            f"early; got {len(r.req.output_ids)} tokens"
        )
        assert r.req.finished_reason == "stop", (
            f"finish_reason must reflect a stop-string match; got "
            f"{r.req.finished_reason!r}"
        )

    def test_stop_token_ids_chunked(self):
        self.server.execute_script(self._script_stop_token_ids_chunked)

    @staticmethod
    def _script_stop_token_ids_chunked(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=64,
            stop_token_ids=[2, 3],
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert r.req.finished_reason in ("stop", "length"), (
            f"finish_reason must be set after chunked decode; got "
            f"{r.req.finished_reason!r}"
        )
        if r.req.finished_reason == "stop":
            assert r.req.output_ids[-1] in (2, 3), (
                f"finish_reason=stop but last token {r.req.output_ids[-1]} "
                f"is not in stop_token_ids"
            )
        else:
            assert len(r.req.output_ids) == 64

    def test_min_new_tokens_chunked(self):
        self.server.execute_script(self._script_min_new_tokens_chunked)

    @staticmethod
    def _script_min_new_tokens_chunked(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=16, min_new_tokens=4
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.req.output_ids) >= 4, (
            f"min_new_tokens=4 must produce at least 4 tokens; got "
            f"{len(r.req.output_ids)}"
        )
        assert len(r.req.output_ids) <= 16

    def test_repetition_penalty_chunked(self):
        self.server.execute_script(self._script_repetition_penalty_chunked)

    @staticmethod
    def _script_repetition_penalty_chunked(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            repetition_penalty=1.2,
            ignore_eos=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.req.output_ids) == 4

    def test_explicit_rid_chunked(self):
        self.server.execute_script(self._script_explicit_rid_chunked)

    @staticmethod
    def _script_explicit_rid_chunked(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            rid="custom-rid-1",
            ignore_eos=True,
        )
        yield from run_until_finished(r)
        assert r.rid == "custom-rid-1"
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.req.output_ids) == 2

    def test_default_sampling_chunked(self):
        self.server.execute_script(self._script_default_sampling_chunked)

    @staticmethod
    def _script_default_sampling_chunked(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            ignore_eos=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.req.output_ids) == 4

    def test_greedy_chunked(self):
        self.server.execute_script(self._script_greedy_chunked)

    @staticmethod
    def _script_greedy_chunked(t: ScriptedContext):
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            temperature=0.0,
            ignore_eos=True,
        )
        yield from run_until_finished(r1)
        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            temperature=0.0,
            ignore_eos=True,
        )
        yield from run_until_finished(r2)
        assert r1.finished and r2.finished
        assert r1.chunks_done >= 2 and r2.chunks_done >= 2
        assert len(r1.req.output_ids) == 4 and len(r2.req.output_ids) == 4
        assert list(r1.req.output_ids) == list(r2.req.output_ids), (
            f"greedy chunked must be deterministic; "
            f"{r1.req.output_ids} != {r2.req.output_ids}"
        )

    def test_high_temperature_short(self):
        self.server.execute_script(self._script_high_temperature_short)

    @staticmethod
    def _script_high_temperature_short(t: ScriptedContext):
        r = t.start_req(
            prompt_len=16,
            max_new_tokens=4,
            temperature=1.8,
            ignore_eos=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0, (
            f"short prompt must not trigger chunked path; got "
            f"chunks_done={r.chunks_done}"
        )
        assert len(r.req.output_ids) == 4

    def test_low_temperature_short(self):
        self.server.execute_script(self._script_low_temperature_short)

    @staticmethod
    def _script_low_temperature_short(t: ScriptedContext):
        r = t.start_req(
            prompt_len=16,
            max_new_tokens=4,
            temperature=0.1,
            ignore_eos=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0
        assert len(r.req.output_ids) == 4

    def test_default_top_p(self):
        self.server.execute_script(self._script_default_top_p)

    @staticmethod
    def _script_default_top_p(t: ScriptedContext):
        r = t.start_req(
            prompt_len=16,
            max_new_tokens=4,
            top_p=0.95,
            ignore_eos=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0
        assert len(r.req.output_ids) == 4

    def test_default_top_k(self):
        self.server.execute_script(self._script_default_top_k)

    @staticmethod
    def _script_default_top_k(t: ScriptedContext):
        r = t.start_req(
            prompt_len=16,
            max_new_tokens=4,
            top_k=50,
            ignore_eos=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0
        assert len(r.req.output_ids) == 4

    def test_combined_sampling_chunked(self):
        self.server.execute_script(self._script_combined_sampling_chunked)

    @staticmethod
    def _script_combined_sampling_chunked(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            ignore_eos=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.req.output_ids) == 4

    def test_default_sampling_short(self):
        self.server.execute_script(self._script_default_sampling_short)

    @staticmethod
    def _script_default_sampling_short(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2, ignore_eos=True)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0
        assert len(r.req.output_ids) == 2

    def test_sampling_diversity_two_reqs(self):
        self.server.execute_script(self._script_sampling_diversity_two_reqs)

    @staticmethod
    def _script_sampling_diversity_two_reqs(t: ScriptedContext):
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            temperature=1.0,
            ignore_eos=True,
        )
        yield from run_until_finished(r1)
        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            temperature=1.0,
            ignore_eos=True,
        )
        yield from run_until_finished(r2)
        assert r1.finished and r2.finished
        assert r1.chunks_done >= 2 and r2.chunks_done >= 2
        assert len(r1.req.output_ids) == 4 and len(r2.req.output_ids) == 4

    def test_chunked_logprob_input_accumulates_across_chunks(self):
        self.server.execute_script(
            self._script_chunked_logprob_input_accumulates_across_chunks
        )

    @staticmethod
    def _script_chunked_logprob_input_accumulates_across_chunks(t: ScriptedContext):
        prompt_len = VERY_LONG_PROMPT_LEN
        r = t.start_req(
            prompt_len=prompt_len,
            max_new_tokens=4,
            return_logprob=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert (
            r.chunks_done >= 2
        ), f"prompt should span multiple chunks, got chunks_done={r.chunks_done}"
        assert r.logprobs is not None
        input_lp = r.logprobs.input_token_logprobs_val
        assert len(input_lp) == prompt_len - 1, (
            f"expected {prompt_len - 1} input logprobs (one per token after "
            f"the first), got {len(input_lp)}"
        )

    def test_logprob_start_len_inside_chunk_2(self):
        self.server.execute_script(self._script_logprob_start_len_inside_chunk_2)

    @staticmethod
    def _script_logprob_start_len_inside_chunk_2(t: ScriptedContext):
        prompt_len = 4 * DEFAULT_CHUNK_SIZE
        start_len = DEFAULT_CHUNK_SIZE + 50
        r = t.start_req(
            prompt_len=prompt_len,
            max_new_tokens=4,
            return_logprob=True,
            logprob_start_len=start_len,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert (
            r.chunks_done >= 3
        ), f"prompt should span 3+ chunks, got chunks_done={r.chunks_done}"
        assert r.logprobs is not None
        input_lp = r.logprobs.input_token_logprobs_val
        assert len(input_lp) == prompt_len - start_len, (
            f"expected {prompt_len - start_len} input logprobs for tokens "
            f">= logprob_start_len={start_len}, got {len(input_lp)}"
        )

    def test_chunked_streaming_no_mid_chunk_output(self):
        self.server.execute_script(self._script_chunked_streaming_no_mid_chunk_output)

    @staticmethod
    def _script_chunked_streaming_no_mid_chunk_output(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            stream=True,
        )
        yield from run_until(r, lambda h: h.chunks_done >= 1)
        assert (
            r.stream_events == []
        ), f"stream output must be suppressed mid-chunk, got {r.stream_events!r}"
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.stream_events) >= 1, (
            f"stream events expected after last chunk completes, "
            f"got {r.stream_events!r}"
        )

    def test_finish_reason_value_eos_vs_length_chunked(self):
        self.server.execute_script(
            self._script_finish_reason_value_eos_vs_length_chunked
        )

    @staticmethod
    def _script_finish_reason_value_eos_vs_length_chunked(t: ScriptedContext):

        r_eos = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=999,
            ignore_eos=False,
        )
        yield from run_until_finished(r_eos, max_steps=2000)
        assert r_eos.finished
        assert (
            r_eos.chunks_done >= 2
        ), f"scenario 1 should chunk; got chunks_done={r_eos.chunks_done}"
        assert r_eos.req.finished_reason == "stop", (
            f"ignore_eos=False + max_new_tokens=999 chunked must finish via "
            f"EOS (stop); got {r_eos.req.finished_reason!r}"
        )

        r_length = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            ignore_eos=True,
        )
        yield from run_until_finished(r_length)
        assert r_length.finished
        assert (
            r_length.chunks_done >= 2
        ), f"scenario 2 should chunk; got chunks_done={r_length.chunks_done}"
        assert r_length.req.finished_reason == "length", (
            f"ignore_eos=True + max_new_tokens=4 chunked must finish via "
            f"length cap; got {r_length.req.finished_reason!r}"
        )

    def test_seed_chunked_bit_identical_runs(self):
        self.server.execute_script(self._script_seed_chunked_bit_identical_runs)

    @staticmethod
    def _script_seed_chunked_bit_identical_runs(t: ScriptedContext):
        seed = 12345
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=32,
            temperature=0.8,
            seed=seed,
        )
        yield from run_until_finished(r1)
        out1 = list(r1.req.output_ids)

        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=32,
            temperature=0.8,
            seed=seed,
        )
        yield from run_until_finished(r2)
        out2 = list(r2.req.output_ids)

        assert r1.chunks_done >= 2 and r2.chunks_done >= 2, (
            f"both runs should be chunked, got chunks_done="
            f"{r1.chunks_done}, {r2.chunks_done}"
        )
        assert out1 == out2, (
            f"same seed + same prompt + chunked must be bit-identical: "
            f"{out1} != {out2}"
        )


if __name__ == "__main__":
    unittest.main()
