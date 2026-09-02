import unittest

from sglang.srt.managers.schedule_batch import FINISH_LENGTH, FINISH_MATCHED_TOKEN
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)


class TestSamplingBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_max_new_tokens_zero_prefill_only(self):
        self.server.execute_script(self._script_max_new_tokens_zero_prefill_only)

    @staticmethod
    def _script_max_new_tokens_zero_prefill_only(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=0)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.req.output_ids) == 1, (
            f"max_new_tokens=0 finishes on the prefill chunk with one sampled "
            f"token; got {len(r.req.output_ids)}"
        )
        decode_records = [
            rec
            for rec in t._scheduler_hook._batch_log
            if r.rid in rec.rids and rec.mode == "decode"
        ]
        assert len(decode_records) == 0, (
            f"max_new_tokens=0 must run zero decode forwards; got "
            f"{len(decode_records)}"
        )

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
        assert r.req.logprob is not None
        assert len(r.req.logprob.output_token_logprobs_val) == 4

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
        assert isinstance(r.req.finished_reason, FINISH_LENGTH), (
            f"ignore_eos=True must finish via length cap; got "
            f"{r.req.finished_reason!r}"
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
        assert r.req.logprob is not None
        top = r.req.logprob.output_top_logprobs_val
        assert len(top) == 4, (
            f"top logprobs must be reported once per output token; "
            f"got {len(top)} entries for 4 tokens"
        )
        for step_entries in top:
            assert len(step_entries) == top_k, (
                f"each step must carry exactly top_logprobs_num={top_k} "
                f"entries; got {len(step_entries)}"
            )

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

    def test_default_sampling_short(self):
        self.server.execute_script(self._script_default_sampling_short)

    @staticmethod
    def _script_default_sampling_short(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2, ignore_eos=True)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0
        assert len(r.req.output_ids) == 2

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
            logprob_start_len=0,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert (
            r.chunks_done >= 2
        ), f"prompt should span multiple chunks, got chunks_done={r.chunks_done}"
        assert r.req.logprob is not None
        input_lp = r.req.logprob.input_token_logprobs_val
        assert len(input_lp) == prompt_len, (
            f"expected {prompt_len} input logprobs (one per prompt token), "
            f"got {len(input_lp)}"
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
        assert r.req.logprob is not None
        input_lp = r.req.logprob.input_token_logprobs_val
        assert len(input_lp) == prompt_len - start_len, (
            f"expected {prompt_len - start_len} input logprobs for tokens "
            f">= logprob_start_len={start_len}, got {len(input_lp)}"
        )

    def test_finish_reason_value_eos_vs_length_chunked(self):
        self.server.execute_script(
            self._script_finish_reason_value_eos_vs_length_chunked
        )

    @staticmethod
    def _script_finish_reason_value_eos_vs_length_chunked(t: ScriptedContext):
        probe = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=1,
            ignore_eos=True,
            prompt_token=7,
            temperature=0.0,
        )
        yield from run_until_finished(probe)
        assert probe.finished
        first_token = probe.req.output_ids[0]

        for _ in range(40):
            if t.is_fully_idle:
                break
            yield
        t.flush_cache()
        yield

        r_eos = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=999,
            ignore_eos=False,
            prompt_token=7,
            stop_token_ids=[first_token],
            temperature=0.0,
        )
        yield from run_until_finished(r_eos, max_steps=2000)
        assert r_eos.finished
        assert (
            r_eos.chunks_done >= 2
        ), f"scenario 1 should chunk; got chunks_done={r_eos.chunks_done}"
        assert isinstance(r_eos.req.finished_reason, FINISH_MATCHED_TOKEN), (
            f"a stop token the model deterministically produces under greedy must "
            f"finish via the matched-token path; got {r_eos.req.finished_reason!r}"
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
        assert isinstance(r_length.req.finished_reason, FINISH_LENGTH), (
            f"ignore_eos=True + max_new_tokens=4 chunked must finish via "
            f"length cap; got {r_length.req.finished_reason!r}"
        )


if __name__ == "__main__":
    unittest.main()
