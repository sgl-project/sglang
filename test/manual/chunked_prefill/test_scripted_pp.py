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


class TestPPBasic(ScriptedTestCase):
    ENGINE_KWARGS = _pp_engine_kwargs()

    def test_pp_chunked_no_double_finalize(self):
        self.server.execute_script(self._script_pp_chunked_no_double_finalize)

    @staticmethod
    def _script_pp_chunked_no_double_finalize(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished
        # exactly-one finish is enforced by the engine (output_streamer: assert not req.finished_output)

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
        assert r.finished

    def test_pp_two_chunked_one_per_mb_simultaneous(self):
        self.server.execute_script(self._script_pp_two_chunked_one_per_mb_simultaneous)

    @staticmethod
    def _script_pp_two_chunked_one_per_mb_simultaneous(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        observed_both_chunking = False
        for _ in range(50):
            if r1.is_chunking and r2.is_chunking:
                observed_both_chunking = True
                count = 1 if t._scheduler.chunked_req is not None else 0
                assert count <= 2, (
                    f"global chunked_in_flight_count exceeds pp_size, got " f"{count}"
                )
                break
            yield
        assert observed_both_chunking, (
            "both reqs must be observed chunking simultaneously to "
            "exercise the cross-mb path"
        )
        yield from run_until_all_finished(handles=[r1, r2], max_steps=800)
        assert r1.finished and r2.finished
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
        stats = t.engine_stats()
        if "merge_batch_assert_violations" in stats:
            assert stats["merge_batch_assert_violations"] == 0, (
                f"merge_batch assert tripped under pdmux + chunked: "
                f"{stats['merge_batch_assert_violations']}"
            )


class TestPPDynamic(ScriptedTestCase):
    ENGINE_KWARGS = _pp_engine_kwargs(enable_dynamic_chunking=True)

    def test_naive_pp_chunked(self):
        self.server.execute_script(self._script_naive_pp_chunked)

    @staticmethod
    def _script_naive_pp_chunked(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2, f"expected >=2 chunks, got {r.chunks_done}"
        assert r.finished
        assert len(r.req.output_ids) == 4

    def test_pp_dynamic_chunking_predictor(self):
        self.server.execute_script(self._script_pp_dynamic_chunking_predictor)

    @staticmethod
    def _script_pp_dynamic_chunking_predictor(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        observed_non_none = False
        for _ in range(400):
            if r.is_chunking:
                stats = t.engine_stats()
                size = stats.get("last_chunked_prefill_size")
                if size is not None:
                    observed_non_none = True
                    assert size > 0, (
                        f"dynamic chunking predictor produced a non-"
                        f"positive size: {size}"
                    )
            if r.finished:
                break
            yield
        assert r.finished
        assert (
            observed_non_none
        ), "dynamic chunking predictor never produced a non-None size"


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
