"""Sampling parameter scripted tests for chunked prefill.

Covers the A.2 series from the expansion plan plus parametrised
combinations of greedy / stochastic / EOS-handling / stop-str /
return-logprob with chunked prefill.

Many cases here invoke ``start_req`` with sampling kwargs that the
v0 ``ScriptedRuntime`` API does not yet accept (temperature, top_p,
top_k, ignore_eos, stop, return_logprob, rid). Those are listed in
expansion plan §6 as P0/P1 wishlist items.

Also covers B.4 series from the expansion plan plus parametric fan-out
across (default / greedy / high temperature / top-k / top-p) ×
(short / chunked prompt).
"""

import unittest

from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.testcase import ScriptedRuntimeTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)


class TestSamplingBasic(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_max_new_tokens_zero_rejected(self):
        """Max_new_tokens = 0: engine should reject the req with a sampling validation error."""
        self.runtime.run(self._script_max_new_tokens_zero_rejected)

    @staticmethod
    def _script_max_new_tokens_zero_rejected(t: ScriptedRuntime):
        # max_new_tokens = 0: engine should reject the req with a sampling
        # validation error.
        # NEW API NEEDED: start_req should propagate sampling validation
        # errors back to the caller as ReqHandle.error_message.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=0)
        for _ in range(DEFAULT_MAX_STEPS):
            if r.error_message is not None:
                return
            if r.finished:
                return
            yield
        raise AssertionError("max_new_tokens=0 should fast-fail with an error_message")

    def test_max_new_tokens_one_long_chunked(self):
        """Max_new_tokens = 1 with a long chunked prompt: completes after 1 decode and emits exactly 1 token."""
        self.runtime.run(self._script_max_new_tokens_one_long_chunked)

    @staticmethod
    def _script_max_new_tokens_one_long_chunked(t: ScriptedRuntime):
        # max_new_tokens = 1 with a long chunked prompt: completes after 1 decode.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=1)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.output_tokens) == 1, (
            f"max_new_tokens=1 must produce exactly 1 token, got "
            f"{len(r.output_tokens)}"
        )

    def test_max_new_tokens_1000_long_chunked(self):
        """Max_new_tokens = 1000: long decode after chunked prefill, exact length."""
        self.runtime.run(self._script_max_new_tokens_1000_long_chunked)

    @staticmethod
    def _script_max_new_tokens_1000_long_chunked(t: ScriptedRuntime):
        # max_new_tokens = 1000 over a chunked prompt: the decode phase
        # must produce exactly 1000 tokens (or finish via natural EOS
        # below the cap).
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=1000,
            ignore_eos=True,
        )
        yield from run_until(r, lambda h: h.finished, max_steps=2000)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.output_tokens) == 1000, (
            f"ignore_eos=True + max_new_tokens=1000 must produce 1000 "
            f"output tokens; got {len(r.output_tokens)}"
        )

    def test_greedy_chunked_deterministic(self):
        """Temperature = 0 (greedy) + chunked: same prompt gives same output."""
        self.runtime.run(self._script_greedy_chunked_deterministic)

    @staticmethod
    def _script_greedy_chunked_deterministic(t: ScriptedRuntime):
        # temperature = 0 (greedy) + chunked: same prompt gives same output.
        # NEW API NEEDED: start_req(..., temperature=) — sampling kwarg passthrough.
        # NEW API NEEDED: r.output_tokens — list[int] of generated tokens.
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
            r1.output_tokens == r2.output_tokens
        ), f"greedy non-determinism: {r1.output_tokens} != {r2.output_tokens}"
        assert len(r1.output_tokens) == 8
        assert r1.chunks_done >= 2 and r2.chunks_done >= 2

    def test_return_logprob_chunked(self):
        """Return_logprob = True + chunked: logprob array length matches output."""
        self.runtime.run(self._script_return_logprob_chunked)

    @staticmethod
    def _script_return_logprob_chunked(t: ScriptedRuntime):
        # return_logprob = True + chunked: logprob array length matches output.
        # NEW API NEEDED: start_req(..., return_logprob=True).
        # NEW API NEEDED: r.logprobs — list (or None when not requested).
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
        """Ignore_eos = True + early EOS production + chunked: still runs to max_new_tokens; doesn't shortcut on EOS."""
        self.runtime.run(self._script_ignore_eos_chunked)

    @staticmethod
    def _script_ignore_eos_chunked(t: ScriptedRuntime):
        # ignore_eos = True + early EOS production + chunked: still runs to
        # max_new_tokens; doesn't shortcut on EOS.
        # NEW API NEEDED: start_req(..., ignore_eos=True).
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=16, ignore_eos=True
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.output_tokens) == 16
        assert r.finish_reason == "length", (
            f"ignore_eos=True must finish via length cap; got " f"{r.finish_reason!r}"
        )

    def test_stop_str_chunked(self):
        """Stop=["xyz"] + chunked: stops at stop_str, doesn't reach max_new_tokens."""
        self.runtime.run(self._script_stop_str_chunked)

    @staticmethod
    def _script_stop_str_chunked(t: ScriptedRuntime):
        # stop=["xyz"] + chunked: stops at stop_str, doesn't reach max_new_tokens.
        # NEW API NEEDED: start_req(..., stop=["..."]).
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=512, stop=["xyz"]
        )
        yield from run_until(r, lambda h: h.finished, max_steps=2000)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.output_tokens) < 512
        assert r.finish_reason in ("stop", "length"), (
            f"finish_reason must be populated for chunked stop-str runs; "
            f"got {r.finish_reason!r}"
        )

    def test_high_temperature_chunked(self):
        """High temperature + chunked: produces exactly max_new_tokens, multi-chunk."""
        self.runtime.run(self._script_high_temperature_chunked)

    @staticmethod
    def _script_high_temperature_chunked(t: ScriptedRuntime):
        # High temperature + chunked: stable output across multi-chunk
        # prefill. ignore_eos guarantees the length cap is hit so the
        # output count is deterministic.
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            temperature=2.0,
            ignore_eos=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.output_tokens) == 4

    def test_greedy_two_sequential_reqs(self):
        """Greedy + chunked, 2 sequential reqs — verify second matches first."""
        self.runtime.run(self._script_greedy_two_sequential_reqs)

    @staticmethod
    def _script_greedy_two_sequential_reqs(t: ScriptedRuntime):
        # Greedy + chunked, 2 sequential reqs — verify second matches first
        # in tokens AND length.
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            temperature=0.0,
            ignore_eos=True,
        )
        yield from run_until_finished(r1)
        out_a = list(r1.output_tokens)

        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            temperature=0.0,
            ignore_eos=True,
        )
        yield from run_until_finished(r2)
        assert list(r2.output_tokens) == out_a
        assert len(out_a) == 4
        assert r1.chunks_done >= 2 and r2.chunks_done >= 2

    def test_greedy_chunked_with_radix_hit(self):
        """Greedy + chunked + radix prefix hit: r2 hits cache and matches r1's tokens."""
        self.runtime.run(self._script_greedy_chunked_with_radix_hit)

    @staticmethod
    def _script_greedy_chunked_with_radix_hit(t: ScriptedRuntime):
        # Greedy + chunked + radix prefix hit. r2 reuses r1's KV via the
        # radix cache, so cached_tokens > 0 and outputs match.
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
        assert list(r1.output_tokens) == list(r2.output_tokens)
        assert len(r1.output_tokens) == 4
        assert r2.cached_tokens > 0, (
            f"r2 must hit r1's radix prefix; got cached_tokens=" f"{r2.cached_tokens}"
        )

    def test_return_logprob_top_logprobs_chunked(self):
        """Return_logprob + top_logprobs_num + chunked: per-step top_logprobs length == top_logprobs_num."""
        self.runtime.run(self._script_return_logprob_top_logprobs_chunked)

    @staticmethod
    def _script_return_logprob_top_logprobs_chunked(t: ScriptedRuntime):
        # return_logprob + top_logprobs_num + chunked. With ignore_eos
        # the decode reaches the length cap, so output_token_top_logprobs
        # must be present for every emitted token, each with exactly
        # top_logprobs_num entries.
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
        """Multiple stop strings + chunked: stops at one of the stop strings before max_new_tokens."""
        self.runtime.run(self._script_multiple_stop_strs_chunked)

    @staticmethod
    def _script_multiple_stop_strs_chunked(t: ScriptedRuntime):
        # Multiple stop strings + chunked. Output must NOT reach the
        # length cap (one of the stop strings should trigger first); if
        # it does reach the cap, the stop matcher silently failed.
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=64,
            stop=["a", "b", "c"],
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.output_tokens) < 64, (
            f"a 64-token decode budget over common stop chars must stop "
            f"early; got {len(r.output_tokens)} tokens"
        )
        assert r.finish_reason == "stop", (
            f"finish_reason must reflect a stop-string match; got "
            f"{r.finish_reason!r}"
        )

    def test_stop_token_ids_chunked(self):
        """Stop_token_ids + chunked: finish_reason indicates stop when one of the explicit token ids is sampled, otherwise length."""
        self.runtime.run(self._script_stop_token_ids_chunked)

    @staticmethod
    def _script_stop_token_ids_chunked(t: ScriptedRuntime):
        # stop_token_ids + chunked. With max_new_tokens=64 the decode
        # may either hit one of the stop token ids (finish_reason ==
        # "stop") or hit the length cap. In both cases the finish_reason
        # must be populated AND consistent with output content.
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=64,
            stop_token_ids=[2, 3],
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert r.finish_reason in ("stop", "length"), (
            f"finish_reason must be set after chunked decode; got "
            f"{r.finish_reason!r}"
        )
        if r.finish_reason == "stop":
            # Stop-token branch: the last emitted token must be one of
            # the configured stop ids.
            assert r.output_tokens[-1] in (2, 3), (
                f"finish_reason=stop but last token {r.output_tokens[-1]} "
                f"is not in stop_token_ids"
            )
        else:
            # Length branch: must have produced exactly max_new_tokens.
            assert len(r.output_tokens) == 64

    def test_min_new_tokens_chunked(self):
        """Min_new_tokens > 0 + chunked + ignore_eos forced by minimum."""
        self.runtime.run(self._script_min_new_tokens_chunked)

    @staticmethod
    def _script_min_new_tokens_chunked(t: ScriptedRuntime):
        # min_new_tokens > 0 + chunked + ignore_eos forced by minimum.
        # NEW API NEEDED: start_req(..., min_new_tokens=).
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=16, min_new_tokens=4
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.output_tokens) >= 4, (
            f"min_new_tokens=4 must produce at least 4 tokens; got "
            f"{len(r.output_tokens)}"
        )
        assert len(r.output_tokens) <= 16

    def test_repetition_penalty_chunked(self):
        """Repetition_penalty + chunked: emits exactly max_new_tokens."""
        self.runtime.run(self._script_repetition_penalty_chunked)

    @staticmethod
    def _script_repetition_penalty_chunked(t: ScriptedRuntime):
        # repetition_penalty + chunked. ignore_eos pins length so we
        # know exactly how many tokens to expect; the chunked path must
        # not silently truncate or extend.
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            repetition_penalty=1.2,
            ignore_eos=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.output_tokens) == 4

    def test_explicit_rid_chunked(self):
        """Explicit rid + chunked: handle uses given rid."""
        self.runtime.run(self._script_explicit_rid_chunked)

    @staticmethod
    def _script_explicit_rid_chunked(t: ScriptedRuntime):
        # Explicit rid + chunked: handle uses given rid, chunked path
        # runs cleanly, output length matches max_new_tokens.
        # NEW API NEEDED: start_req(..., rid="custom-rid").
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
        assert len(r.output_tokens) == 2

    def test_default_sampling_chunked(self):
        """All defaults + chunked: emits exactly max_new_tokens (ignore_eos)."""
        self.runtime.run(self._script_default_sampling_chunked)

    @staticmethod
    def _script_default_sampling_chunked(t: ScriptedRuntime):
        # All defaults + chunked: deterministic length via ignore_eos.
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=4,
            ignore_eos=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.output_tokens) == 4

    def test_greedy_chunked(self):
        """Greedy (temperature=0) sampling over chunked prompt: exact length + bit-identical re-run."""
        self.runtime.run(self._script_greedy_chunked)

    @staticmethod
    def _script_greedy_chunked(t: ScriptedRuntime):
        # Greedy sampling must be deterministic, so a second run with
        # the same prompt yields the same tokens. Together with the
        # length / chunks_done invariants this catches stochastic leakage.
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
        assert len(r1.output_tokens) == 4 and len(r2.output_tokens) == 4
        assert list(r1.output_tokens) == list(r2.output_tokens), (
            f"greedy chunked must be deterministic; "
            f"{r1.output_tokens} != {r2.output_tokens}"
        )

    def test_high_temperature_short(self):
        """Short prompt with high temperature (1.8): no chunking, exact length."""
        self.runtime.run(self._script_high_temperature_short)

    @staticmethod
    def _script_high_temperature_short(t: ScriptedRuntime):
        # Short prompt below chunk_size — chunked path must NOT engage.
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
        assert len(r.output_tokens) == 4

    def test_low_temperature_short(self):
        """Short prompt with low temperature (0.1): no chunking, exact length."""
        self.runtime.run(self._script_low_temperature_short)

    @staticmethod
    def _script_low_temperature_short(t: ScriptedRuntime):
        r = t.start_req(
            prompt_len=16,
            max_new_tokens=4,
            temperature=0.1,
            ignore_eos=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0
        assert len(r.output_tokens) == 4

    def test_default_top_p(self):
        """Top_p=0.95 sampling on a short prompt: no chunking, exact length."""
        self.runtime.run(self._script_default_top_p)

    @staticmethod
    def _script_default_top_p(t: ScriptedRuntime):
        r = t.start_req(
            prompt_len=16,
            max_new_tokens=4,
            top_p=0.95,
            ignore_eos=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0
        assert len(r.output_tokens) == 4

    def test_default_top_k(self):
        """Top_k=50 sampling on a short prompt: no chunking, exact length."""
        self.runtime.run(self._script_default_top_k)

    @staticmethod
    def _script_default_top_k(t: ScriptedRuntime):
        r = t.start_req(
            prompt_len=16,
            max_new_tokens=4,
            top_k=50,
            ignore_eos=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0
        assert len(r.output_tokens) == 4

    def test_combined_sampling_chunked(self):
        """All sampling knobs on at once over a chunked prompt: exact length."""
        self.runtime.run(self._script_combined_sampling_chunked)

    @staticmethod
    def _script_combined_sampling_chunked(t: ScriptedRuntime):
        # All sampling knobs on at once over a chunked prompt.
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
        assert len(r.output_tokens) == 4

    def test_default_sampling_short(self):
        """Default sampling parameters on a short prompt: no chunking, exact length."""
        self.runtime.run(self._script_default_sampling_short)

    @staticmethod
    def _script_default_sampling_short(t: ScriptedRuntime):
        r = t.start_req(prompt_len=8, max_new_tokens=2, ignore_eos=True)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0
        assert len(r.output_tokens) == 2

    def test_sampling_diversity_two_reqs(self):
        """Two reqs with same prompt and non-greedy temp: both exact-length, may differ in tokens."""
        self.runtime.run(self._script_sampling_diversity_two_reqs)

    @staticmethod
    def _script_sampling_diversity_two_reqs(t: ScriptedRuntime):
        # Two reqs with same prompt and non-greedy temp: outputs may
        # differ (diversity), but both must hit exact length.
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
        assert len(r1.output_tokens) == 4 and len(r2.output_tokens) == 4

    def test_chunked_logprob_input_accumulates_across_chunks(self):
        """Return_logprob + multi-chunk prompt: input logprobs accumulate across chunks to cover the whole prompt."""
        self.runtime.run(self._script_chunked_logprob_input_accumulates_across_chunks)

    @staticmethod
    def _script_chunked_logprob_input_accumulates_across_chunks(t: ScriptedRuntime):
        # Guards _apply_chunked_prefill_logprobs in
        # batch_result_processor.py (lines 274-285, 451-482). Each middle
        # chunk should append its input logprobs incrementally so that, at
        # finish time, the accumulated input_token_logprobs_val covers the
        # whole prompt (minus the standard leading-token offset).
        # NEW API NEEDED: start_req(..., return_logprob=True).
        # NEW API NEEDED: r.logprobs.input_token_logprobs_val — list of
        # input-token logprobs accumulated across chunks.
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
        # The first input token has no preceding context so it is not scored;
        # everything after it should have a logprob entry accumulated across
        # all chunks.
        assert len(input_lp) == prompt_len - 1, (
            f"expected {prompt_len - 1} input logprobs (one per token after "
            f"the first), got {len(input_lp)}"
        )

    def test_logprob_start_len_inside_chunk_2(self):
        """Logprob_start_len that falls inside the 2nd chunk: only tokens >= start_len have logprobs."""
        self.runtime.run(self._script_logprob_start_len_inside_chunk_2)

    @staticmethod
    def _script_logprob_start_len_inside_chunk_2(t: ScriptedRuntime):
        # Logprob_start_len positioned inside the
        # 2nd chunk (chunk_size + 50 with a 4*chunk_size prompt) exercises
        # the chunked logprob-start alignment in
        # _apply_chunked_prefill_logprobs: only tokens at index >=
        # logprob_start_len should produce input logprobs.
        # NEW API NEEDED: start_req(..., return_logprob=True,
        # logprob_start_len=).
        # NEW API NEEDED: r.logprobs.input_token_logprobs_val.
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
        # Only tokens at positions >= start_len contribute input logprobs.
        assert len(input_lp) == prompt_len - start_len, (
            f"expected {prompt_len - start_len} input logprobs for tokens "
            f">= logprob_start_len={start_len}, got {len(input_lp)}"
        )

    def test_chunked_streaming_no_mid_chunk_output(self):
        """Stream=True + chunked: no output events fire until the last chunk finishes."""
        self.runtime.run(self._script_chunked_streaming_no_mid_chunk_output)

    @staticmethod
    def _script_chunked_streaming_no_mid_chunk_output(t: ScriptedRuntime):
        # Guards the skip_stream_req branch in
        # batch_result_processor.py: while a req is in the middle of
        # chunked prefill, stream_output must suppress its emission until
        # the final chunk lands.
        # NEW API NEEDED: start_req(..., stream=True).
        # NEW API NEEDED: ReqHandle.stream_events list capturing each
        # delivered stream chunk so the test can observe ordering.
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
        """Chunked req's finish_reason matches EOS (stop) vs length cap (length) per sampling kwargs."""
        self.runtime.run(self._script_finish_reason_value_eos_vs_length_chunked)

    # output-state contract: a chunked req that decodes to its
    # natural EOS should report finish_reason == "stop"; one capped by
    # max_new_tokens under ignore_eos should report "length". Both code
    # paths must produce a populated finish_reason after a chunked
    # prefill — pre-fix the chunked path could leave it None.
    @staticmethod
    def _script_finish_reason_value_eos_vs_length_chunked(t: ScriptedRuntime):
        # NEW API NEEDED: r.finish_reason — the engine-reported reason
        # string ("stop" | "length" | "abort" | None until finalized).
        # NEW API NEEDED: start_req(..., ignore_eos=).

        # Scenario 1: long chunked req with ignore_eos=False and a
        # generous max_new_tokens — decode should hit EOS naturally.
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
        assert r_eos.finish_reason == "stop", (
            f"ignore_eos=False + max_new_tokens=999 chunked must finish via "
            f"EOS (stop); got {r_eos.finish_reason!r}"
        )

        # Scenario 2: long chunked req with ignore_eos=True and a tiny
        # max_new_tokens — decode is forced to the length cap.
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
        assert r_length.finish_reason == "length", (
            f"ignore_eos=True + max_new_tokens=4 chunked must finish via "
            f"length cap; got {r_length.finish_reason!r}"
        )

    def test_seed_chunked_bit_identical_runs(self):
        """Same seed + same prompt + chunked, run twice sequentially: identical output tokens."""
        self.runtime.run(self._script_seed_chunked_bit_identical_runs)

    @staticmethod
    def _script_seed_chunked_bit_identical_runs(t: ScriptedRuntime):
        # Seeded sampling over a chunked prompt
        # must be reproducible: running the same prompt + same seed twice
        # within one engine should yield identical output tokens despite
        # the chunked-prefill scheduling.
        # NEW API NEEDED: start_req(..., seed=, temperature=).
        seed = 12345
        r1 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=32,
            temperature=0.8,
            seed=seed,
        )
        yield from run_until_finished(r1)
        out1 = list(r1.output_tokens)

        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=32,
            temperature=0.8,
            seed=seed,
        )
        yield from run_until_finished(r2)
        out2 = list(r2.output_tokens)

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
