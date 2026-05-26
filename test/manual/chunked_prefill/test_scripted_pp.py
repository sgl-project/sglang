"""PP × chunked: naive ScriptedRuntime smoke.

Submit one long-prompt request that must be chunked across at least
two scheduler iterations, with ``pp_size=2`` so the chunked req crosses
microbatch boundaries.

Asserts the request reaches ``finished`` and went through >= 2 chunks.
Does not attempt to reproduce the 309b6dc last-chunk-in-flight race —
that lives in ``test_scripted_regression_309b6dc.py``.

Requires 2 GPUs. ScriptedRuntime must support ``pp_size > 1`` (see
wishlist §4 P2 (12)).
"""

import unittest
from typing import Any, Dict

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST, CustomTestCase


def _pp_engine_kwargs(*, pp_size: int = 2, **overrides: Any) -> Dict[str, Any]:
    return base_engine_kwargs(
        model_path=DEFAULT_MODEL_NAME_FOR_TEST,
        pp_size=pp_size,
        **overrides,
    )


class TestScriptedPP(CustomTestCase):
    def test_naive_pp_chunked(self):
        """PP × chunked: naive ScriptedRuntime smoke."""
        execute_scripted_runtime(
            self._script_naive_pp_chunked,
            **_pp_engine_kwargs(enable_dynamic_chunking=True),
        )

    @staticmethod
    def _script_naive_pp_chunked(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished
        # VERY_LONG_PROMPT_LEN / DEFAULT_CHUNK_SIZE chunks expected.
        assert r.chunks_done >= 2, f"expected >=2 chunks, got {r.chunks_done}"

    def test_pp_chunked_no_double_finalize(self):
        """PP=2 chunked req must finalize exactly once across microbatches."""
        execute_scripted_runtime(
            self._script_pp_chunked_no_double_finalize,
            **_pp_engine_kwargs(),
        )

    # PP cross-mb _handle_finished_req must not double-finalize:
    # a chunked req visible in mb_a + mb_other was finalized twice pre-fix.
    @staticmethod
    def _script_pp_chunked_no_double_finalize(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished
        assert r.finish_event_count == 1, (
            f"chunked req must finalize once across microbatches, "
            f"got finish_event_count={r.finish_event_count}"
        )

    def test_pp_abort_during_inflight_chunk(self):
        """PP=2 abort on chunked req in both mb_other and waiting_queue dedups cleanly."""
        execute_scripted_runtime(
            self._script_pp_abort_during_inflight_chunk,
            **_pp_engine_kwargs(),
        )

    # PP abort_request must dedup across batch_rids:
    # a single req can sit in mb_other.reqs and waiting_queue, abort once.
    @staticmethod
    def _script_pp_abort_during_inflight_chunk(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.abort(r)
        yield
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        assert r.finish_event_count <= 1

    def test_pp_last_chunk_cross_mb_kv_correctness(self):
        """PP=2 last-chunk-in-flight across microbatches must not corrupt decode KV."""
        execute_scripted_runtime(
            self._script_pp_last_chunk_cross_mb_kv_correctness,
            **_pp_engine_kwargs(),
        )

    # PP last-chunk cross-mb KV correctness:
    # gsm8k 70B accuracy 0.66 -> 0.77 after fix; decode KV positions must
    # be written correctly when the last chunk lands in the alternate mb.
    @staticmethod
    def _script_pp_last_chunk_cross_mb_kv_correctness(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=8)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        # max_new_tokens > 1 forces decode after the last chunk — pre-fix
        # the cross-mb KV was silently corrupted; surface via output length.
        assert len(r.output_tokens) == 8, (
            f"decode must produce all 8 tokens cleanly, got "
            f"len(output_tokens)={len(r.output_tokens)}"
        )

    def test_pp_multi_microbatch_chunks_done_aggregation(self):
        """PP=2 single chunked req across 4+ chunks aggregates chunks_done correctly."""
        execute_scripted_runtime(
            self._script_pp_multi_microbatch_chunks_done_aggregation,
            **_pp_engine_kwargs(),
        )

    # PP=2 chunks_done aggregation across mbs — single chunked req
    # spans 4+ chunks; chunks_done increments observed by ReqHandle must
    # reflect the union of per-mb progress.
    @staticmethod
    def _script_pp_multi_microbatch_chunks_done_aggregation(t: ScriptedRuntime):
        # 4 * DEFAULT_CHUNK_SIZE forces >=4 chunks distributed across mbs.
        r = t.start_req(prompt_len=4 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.chunks_done >= 4, (
            f"PP=2 multi-chunk req should aggregate >=4 chunks_done across "
            f"microbatches, got {r.chunks_done}"
        )
        assert r.finish_event_count == 1

    def test_pp_size_4_chunked_completes(self):
        """PP=4 long chunked req completes with no microbatch residue."""
        execute_scripted_runtime(
            self._script_pp_size_4_chunked_completes,
            **_pp_engine_kwargs(pp_size=4),
        )

    # PP=4 chunked completion — verifies the cross-mb bookkeeping
    # scales beyond pp_size=2; long chunked req must not leak.
    @staticmethod
    def _script_pp_size_4_chunked_completes(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.chunks_done >= 4
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_pp_two_chunked_one_per_mb_simultaneous(self):
        """PP=2 with one chunked req per microbatch — both complete; per-mb in-flight bounded."""
        execute_scripted_runtime(
            self._script_pp_two_chunked_one_per_mb_simultaneous,
            **_pp_engine_kwargs(),
        )

    # PP=2, one chunked per mb — chunked_in_flight_count must stay
    # <=1 per mb but the global count may reach 2.
    @staticmethod
    def _script_pp_two_chunked_one_per_mb_simultaneous(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        # Observe in-flight count while both chunking.
        for _ in range(50):
            if r1.is_chunking and r2.is_chunking:
                assert t.chunked_in_flight_count <= 2, (
                    f"global chunked_in_flight_count exceeds pp_size, got "
                    f"{t.chunked_in_flight_count}"
                )
                break
            yield
        yield from run_until_all_finished(handles=[r1, r2], max_steps=800)
        assert r1.finished and r2.finished

    def test_pp_retract_chunked_in_middle_mb(self):
        """PP=2 retract of chunked req mid-mb cleans cross-mb exclude set."""
        execute_scripted_runtime(
            self._script_pp_retract_chunked_in_middle_mb,
            **_pp_engine_kwargs(),
        )

    # PP=2 mid-mb chunked retract — exclude set must drop the
    # cross-mb chunked_req reference so it does not re-enter the batch.
    @staticmethod
    def _script_pp_retract_chunked_in_middle_mb(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.force_retract(r)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_pp_chunked_req_to_exclude_pp_context(self):
        """PP last_batch.chunked_req stale pointer must be excluded from new batch."""
        execute_scripted_runtime(
            self._script_pp_chunked_req_to_exclude_pp_context,
            **_pp_engine_kwargs(),
        )

    # PP-path exclude set — last_batch.chunked_req can be stale
    # under PP; must not re-enter the batch on the next admission.
    @staticmethod
    def _script_pp_chunked_req_to_exclude_pp_context(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        # After abort the chunked_req pointer goes stale across mbs.
        t.abort(r)
        # Submit a fresh req — must admit cleanly without re-picking up
        # the stale chunked_req pointer.
        r2 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=2)
        yield from run_until_finished(r2, max_steps=400)
        assert r2.finished

    def test_pp_split_prefill_chunked_no_merge_assert(self):
        """PP=2 + pdmux split-prefill + chunked must not trip merge_batch assert."""
        execute_scripted_runtime(
            self._script_pp_split_prefill_chunked_no_merge_assert,
            **_pp_engine_kwargs(enable_pdmux=True),
        )

    # pdmux + chunked — split-prefill filter must
    # exclude chunked reqs so merge_batch assert does not fire.
    @staticmethod
    def _script_pp_split_prefill_chunked_no_merge_assert(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=800)
        # Concrete observable signal that pdmux + chunked actually exercised
        # the merge_batch defense path: engine survived AND chunking happened.
        assert (
            r.finished
        ), "engine died before req finished — merge_batch assert may have tripped"
        assert r.chunks_done >= 2, (
            f"pdmux + chunked path must produce >=2 chunks to exercise "
            f"split_prefill_batch filter; got chunks_done={r.chunks_done}"
        )
        # NEW API NEEDED: t.engine_stats() should expose a
        # ``merge_batch_assert_violations`` counter so the test can directly
        # probe the defense path b-36ec1d7269 widened. Until then the
        # engine-survived + chunks-done assertions above are the strongest
        # behavior-level signal we can give.
        stats = t.engine_stats()
        if "merge_batch_assert_violations" in stats:
            assert stats["merge_batch_assert_violations"] == 0, (
                f"merge_batch assert tripped under pdmux + chunked: "
                f"{stats['merge_batch_assert_violations']}"
            )

    def test_pp_dynamic_chunking_predictor(self):
        """PP=2 + dynamic chunking — last_chunked_prefill_size set per iter by predictor."""
        execute_scripted_runtime(
            self._script_pp_dynamic_chunking_predictor,
            **_pp_engine_kwargs(enable_dynamic_chunking=True),
        )

    # dynamic chunking under PP — predictor must populate
    # last_chunked_prefill_size every iter the chunked req is in flight.
    @staticmethod
    def _script_pp_dynamic_chunking_predictor(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        observed_non_none = False
        for _ in range(400):
            if r.is_chunking:
                stats = t.engine_stats
                if stats.last_chunked_prefill_size is not None:
                    observed_non_none = True
                    assert stats.last_chunked_prefill_size > 0
            if r.finished:
                break
            yield
        assert r.finished
        assert (
            observed_non_none
        ), "dynamic chunking predictor never produced a non-None size"


if __name__ == "__main__":
    unittest.main()
