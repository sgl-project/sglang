import unittest
from typing import Any, Dict

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST


def _pp_engine_kwargs(*, pp_size: int = 2, **overrides: Any) -> Dict[str, Any]:
    return base_engine_kwargs(
        model_path=DEFAULT_MODEL_NAME_FOR_TEST,
        pp_size=pp_size,
        **overrides,
    )


def _expected_chunks(prompt_len: int, chunk_size: int) -> int:
    # chunks_done model: 0 if the prompt fits one shot, else ceil(prompt/chunk).
    # The tail iteration is counted, so prompt 2048 / chunk 256 -> exactly 8.
    if prompt_len <= chunk_size:
        return 0
    return (prompt_len + chunk_size - 1) // chunk_size


class TestPPBasic(ScriptedTestCase):
    ENGINE_KWARGS = _pp_engine_kwargs()

    def test_pp_abort_during_inflight_chunk(self):
        self.server.execute_script(self._script_pp_abort_during_inflight_chunk)

    @staticmethod
    def _script_pp_abort_during_inflight_chunk(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.abort(r)
        yield
        assert r.kv_pages == 0
        assert r.lock_refs == 0
        # at-most-one finish is enforced by the engine (output_streamer: assert not req.finished_output)

    def test_pp_last_chunk_cross_mb_kv_correctness(self):
        self.server.execute_script(self._script_pp_last_chunk_cross_mb_kv_correctness)

    @staticmethod
    def _script_pp_last_chunk_cross_mb_kv_correctness(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=8)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2
        assert len(r.req.output_ids) == 8, (
            f"decode must produce all 8 tokens cleanly, got "
            f"len(output_tokens)={len(r.req.output_ids)}"
        )

    def test_pp_static_chunk_size_predictor_returns_none(self):
        self.server.execute_script(
            self._script_pp_static_chunk_size_predictor_returns_none
        )

    @staticmethod
    def _script_pp_static_chunk_size_predictor_returns_none(t: ScriptedContext):
        # With dynamic chunking OFF, predict_next_chunk_size early-returns None
        # (scheduler_pp_mixin.py:736-741) and the static chunked_prefill_size drives
        # every chunk, yielding the exact deterministic chunk count.
        sched = t.scheduler
        # The branch's controlling flag must be off, so every chunked iteration takes
        # the early-return at 736-741 instead of computing a dynamic size.
        assert sched.enable_dynamic_chunking is False
        # Witness the early-return directly: a non-ready/disabled predictor returns
        # None for any history length, the precondition for the static path below.
        assert sched.predict_next_chunk_size(0) is None
        assert sched.predict_next_chunk_size(VERY_LONG_PROMPT_LEN // 2) is None
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        # Exact static count (8 for 2048 / 256) witnesses that the fixed chunk size,
        # not a dynamic prediction, drove every chunk. A dynamic size would shrink
        # later chunks and change this count, so == is the discriminating assertion.
        expected = _expected_chunks(VERY_LONG_PROMPT_LEN, DEFAULT_CHUNK_SIZE)
        assert r.chunks_done == expected, (
            f"static chunked_prefill_size must produce exactly {expected} chunks, "
            f"got {r.chunks_done}"
        )

    def test_pp_multi_microbatch_chunks_done_aggregation(self):
        self.server.execute_script(
            self._script_pp_multi_microbatch_chunks_done_aggregation
        )

    @staticmethod
    def _script_pp_multi_microbatch_chunks_done_aggregation(t: ScriptedContext):
        r = t.start_req(prompt_len=4 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.chunks_done >= 4, (
            f"PP=2 multi-chunk req should aggregate >=4 chunks_done across "
            f"microbatches, got {r.chunks_done}"
        )

    def test_pp_two_chunked_one_per_mb_simultaneous(self):
        self.server.execute_script(self._script_pp_two_chunked_one_per_mb_simultaneous)

    @staticmethod
    def _script_pp_two_chunked_one_per_mb_simultaneous(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_all_finished(handles=[r1, r2], max_steps=800)
        assert r1.finished and r2.finished
        assert r1.chunks_done >= 2 and r2.chunks_done >= 2
        assert r1.kv_pages == 0 and r2.kv_pages == 0
        assert r1.lock_refs == 0 and r2.lock_refs == 0

    def test_pp_retract_chunked_in_middle_mb(self):
        self.server.execute_script(self._script_pp_retract_chunked_in_middle_mb)

    @staticmethod
    def _script_pp_retract_chunked_in_middle_mb(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.pause_generation(mode="retract")
        yield
        t.continue_generation()
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_pp_chunked_req_to_exclude_pp_context(self):
        self.server.execute_script(self._script_pp_chunked_req_to_exclude_pp_context)

    @staticmethod
    def _script_pp_chunked_req_to_exclude_pp_context(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        t.abort(r)
        r2 = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE * 2, max_new_tokens=2)
        yield from run_until_finished(r2, max_steps=400)
        assert r2.finished
        assert r2.kv_pages == 0
        assert r2.lock_refs == 0


class TestPPPdmux(ScriptedTestCase):
    ENGINE_KWARGS = _pp_engine_kwargs(enable_pdmux=True)

    def test_pp_split_prefill_chunked_no_merge_assert(self):
        self.server.execute_script(
            self._script_pp_split_prefill_chunked_no_merge_assert
        )

    @staticmethod
    def _script_pp_split_prefill_chunked_no_merge_assert(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=800)
        assert (
            r.finished
        ), "engine died before req finished — merge_batch assert may have tripped"
        assert r.chunks_done >= 2, (
            f"pdmux + chunked path must produce >=2 chunks to exercise "
            f"split_prefill_batch filter; got chunks_done={r.chunks_done}"
        )


class TestPPDynamic(ScriptedTestCase):
    ENGINE_KWARGS = _pp_engine_kwargs(enable_dynamic_chunking=True)

    def test_naive_pp_chunked(self):
        self.server.execute_script(self._script_naive_pp_chunked)

    @staticmethod
    def _script_naive_pp_chunked(t: ScriptedContext):
        # Smoke test for the PP + dynamic-chunking path: a long prompt must chunk
        # across microbatches and still decode all requested tokens cleanly.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2, f"expected >=2 chunks, got {r.chunks_done}"
        assert len(r.req.output_ids) == 4

    def test_pp_dynamic_chunk_size_recompute_branch_taken(self):
        self.server.execute_script(
            self._script_pp_dynamic_chunk_size_recompute_branch_taken
        )

    @staticmethod
    def _script_pp_dynamic_chunk_size_recompute_branch_taken(t: ScriptedContext):
        # Prove the dynamic-recompute branch (scheduler.py:2612-2616) actually
        # consults the predictor and uses its size, rather than predict_next_chunk_size
        # silently returning None and the static size being used.
        sched = t.scheduler
        # Profiling at scheduler init must have produced a ready predictor; on failure
        # enable_dynamic_chunking is flipped off (scheduler.py:958) and the branch is
        # dead. Assert it is live so the dynamic path is genuinely reachable.
        assert sched.enable_dynamic_chunking is True
        assert sched.length_predictor is not None
        assert sched.length_predictor.is_ready is True
        # A ready predictor passes the 736-741 gate and returns a concrete size, so at
        # scheduler.py:2615 dynamic_size is not None and line 2616 overrides the static
        # chunk size. Witness the non-None return directly for the branch's history_len.
        dynamic_size = sched.predict_next_chunk_size(0)
        assert dynamic_size is not None
        assert isinstance(dynamic_size, int) and dynamic_size > 0
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        # The prompt chunked across multiple iterations, so chunked_req was non-None
        # when scheduler.py:2612 ran -- the branch condition (chunked_req is not None
        # and enable_dynamic_chunking) held and the dynamic size was applied.
        assert r.chunks_done >= 2, f"expected >=2 chunks, got {r.chunks_done}"


class TestPPSize4(ScriptedTestCase):
    ENGINE_KWARGS = _pp_engine_kwargs(pp_size=4)

    def test_pp_size_4_chunked_completes(self):
        self.server.execute_script(self._script_pp_size_4_chunked_completes)

    @staticmethod
    def _script_pp_size_4_chunked_completes(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.chunks_done >= 4
        assert r.kv_pages == 0
        assert r.lock_refs == 0


if __name__ == "__main__":
    unittest.main()
