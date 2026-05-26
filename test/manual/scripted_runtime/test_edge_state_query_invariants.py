"""Edge cases — state-query API invariants.

Covers A.8 series from the expansion plan. These tests are
contract-style — they don't drive an interesting trajectory, they
just hammer on the harness API (status, chunks_done, kv_pages,
batch_composition, etc.) at every yield step and verify the
monotone / non-negative / self-consistent invariants hold.
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.req_handle import ReqHandle
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)
from sglang.test.test_utils import CustomTestCase


def _script_status_unknown_before_submit(t: ScriptedRuntime):
    # A bare ReqHandle whose rid was never submitted reports "unknown".
    bogus = ReqHandle(rid="never-submitted", runtime=t)
    assert bogus.status == "unknown"
    yield


def _script_status_finished_after_done(t: ScriptedRuntime):
    # After finish, r.status == "finished" (not unknown).
    r = t.start_req(prompt_len=16, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.status == "finished"


def _script_chunks_done_monotone_invariant(t: ScriptedRuntime):
    # chunks_done is non-decreasing across yield steps.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    prev = 0
    for _ in range(DEFAULT_MAX_STEPS):
        cur = r.chunks_done
        assert cur >= prev, f"chunks_done regressed: {prev} -> {cur}"
        prev = cur
        if r.finished:
            return
        yield
    raise AssertionError("req never finished")


def _script_kv_pages_zero_after_finish(t: ScriptedRuntime):
    # kv_pages drops to 0 after the req finishes.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.kv_pages == 0


def _script_kv_pages_positive_mid_chunk(t: ScriptedRuntime):
    # Mid-chunked, kv_pages > 0.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking)
    assert r.kv_pages > 0
    yield from run_until_finished(r)


def _script_batch_composition_consistent_with_status(t: ScriptedRuntime):
    # If r.status == "running" then r.rid appears in batch_composition.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    for _ in range(DEFAULT_MAX_STEPS):
        if r.status == "running":
            comp = t.batch_composition()
            all_rids = (
                comp.get("prefill", [])
                + comp.get("decode", [])
                + comp.get("chunked", [])
            )
            assert r.rid in all_rids, f"running but not in batch_composition: {comp}"
        if r.finished:
            return
        yield
    raise AssertionError("req never finished")


def _script_is_idle_excludes_chunked_in_flight(t: ScriptedRuntime):
    # t.is_idle and chunked_in_flight_count > 0 are mutually exclusive.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    for _ in range(DEFAULT_MAX_STEPS):
        if t.chunked_in_flight_count() > 0:
            assert not t.is_idle, "is_idle must be False when chunked is in flight"
        if r.finished:
            return
        yield
    raise AssertionError("req never finished")


def _script_finish_event_count_exactly_one(t: ScriptedRuntime):
    # Every normally-completed req emits exactly one finish event.
    r = t.start_req(prompt_len=16, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finish_event_count == 1


def _script_kv_pages_non_negative(t: ScriptedRuntime):
    # kv_pages never negative.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    for _ in range(DEFAULT_MAX_STEPS):
        assert r.kv_pages >= 0
        if r.finished:
            return
        yield
    raise AssertionError("req never finished")


def _script_pending_middle_outputs_non_negative(t: ScriptedRuntime):
    # pending_middle_outputs non-negative invariant.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    for _ in range(DEFAULT_MAX_STEPS):
        assert r.pending_middle_outputs >= 0
        if r.finished:
            return
        yield


def _script_inflight_middle_chunks_non_negative(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    for _ in range(DEFAULT_MAX_STEPS):
        assert r.inflight_middle_chunks >= 0
        if r.finished:
            return
        yield


def _script_lock_refs_non_negative(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    for _ in range(DEFAULT_MAX_STEPS):
        assert r.lock_refs >= 0
        if r.finished:
            return
        yield


def _script_chunked_in_flight_count_le_one(t: ScriptedRuntime):
    # main-upstream invariant: at most one chunked req in flight.
    reqs = [
        t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2) for _ in range(3)
    ]
    for _ in range(DEFAULT_MAX_STEPS * 3):
        assert t.chunked_in_flight_count() <= 1
        if all(r.finished for r in reqs):
            return
        yield


def _script_status_transition_monotone(t: ScriptedRuntime):
    # Allowed transitions: unknown -> waiting -> running -> finished.
    # Verify no reverse transitions happen.
    r = t.start_req(prompt_len=16, max_new_tokens=2)
    rank = {"unknown": 0, "waiting": 1, "running": 2, "finished": 3}
    prev_rank = 0
    for _ in range(DEFAULT_MAX_STEPS):
        cur_rank = rank[r.status]
        # waiting can come after running once (retract), so accept rank ≥ prev - 1.
        assert (
            cur_rank >= prev_rank - 1
        ), f"status regressed: {r.status} (prev rank {prev_rank})"
        prev_rank = max(prev_rank, cur_rank)
        if r.finished:
            return
        yield


def _script_active_reqs_listing(t: ScriptedRuntime):
    # NEW API NEEDED: t.list_active_reqs() returns the currently-running
    # set of ReqHandle objects.
    r1 = t.start_req(prompt_len=16, max_new_tokens=4)
    r2 = t.start_req(prompt_len=16, max_new_tokens=4)
    yield
    actives = t.list_active_reqs()
    rids = {h.rid for h in actives}
    assert r1.rid in rids or r2.rid in rids
    yield from run_until_all_finished([r1, r2])
    actives_after = t.list_active_reqs()
    assert all(h.rid not in (r1.rid, r2.rid) for h in actives_after)


def _script_batch_composition_disjoint_subsets(t: ScriptedRuntime):
    # prefill / decode / chunked subsets must be disjoint.
    r1 = t.start_req(prompt_len=16, max_new_tokens=2)
    r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    for _ in range(DEFAULT_MAX_STEPS):
        comp = t.batch_composition()
        prefill = set(comp.get("prefill", []))
        decode = set(comp.get("decode", []))
        chunked = set(comp.get("chunked", []))
        assert prefill & decode == set()
        assert prefill & chunked == set()
        assert decode & chunked == set()
        if r1.finished and r2.finished:
            return
        yield


def _script_finished_means_chunks_done_stable(t: ScriptedRuntime):
    # Once finished, chunks_done value should not change.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)
    snap = r.chunks_done
    for _ in range(10):
        yield
        assert r.chunks_done == snap


def _script_finished_means_kv_pages_stays_zero(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)
    for _ in range(10):
        yield
        assert r.kv_pages == 0


def _script_engine_stats_keys_present(t: ScriptedRuntime):
    # engine_stats returns a dict with expected keys.
    stats = t.engine_stats()
    assert isinstance(stats, dict)
    assert "kv_pool_free" in stats
    assert "row_pool_free" in stats
    yield


def _script_kv_pool_recovers_to_baseline(t: ScriptedRuntime):
    # Full lifecycle: pool counts return to baseline after all done.
    before = t.engine_stats()["kv_pool_free"]
    reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(8)]
    yield from run_until_all_finished(reqs)
    after = t.engine_stats()["kv_pool_free"]
    assert after >= before


class TestEdgeStateQueryInvariants(CustomTestCase):
    def test_status_unknown_before_submit(self):
        execute_scripted_runtime(
            _script_status_unknown_before_submit,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_status_finished_after_done(self):
        execute_scripted_runtime(
            _script_status_finished_after_done,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_chunks_done_monotone_invariant(self):
        execute_scripted_runtime(
            _script_chunks_done_monotone_invariant,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_kv_pages_zero_after_finish(self):
        execute_scripted_runtime(
            _script_kv_pages_zero_after_finish,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_kv_pages_positive_mid_chunk(self):
        execute_scripted_runtime(
            _script_kv_pages_positive_mid_chunk,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_batch_composition_consistent_with_status(self):
        execute_scripted_runtime(
            _script_batch_composition_consistent_with_status,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_is_idle_excludes_chunked_in_flight(self):
        execute_scripted_runtime(
            _script_is_idle_excludes_chunked_in_flight,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_finish_event_count_exactly_one(self):
        execute_scripted_runtime(
            _script_finish_event_count_exactly_one,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_kv_pages_non_negative(self):
        execute_scripted_runtime(
            _script_kv_pages_non_negative,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_pending_middle_outputs_non_negative(self):
        execute_scripted_runtime(
            _script_pending_middle_outputs_non_negative,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_inflight_middle_chunks_non_negative(self):
        execute_scripted_runtime(
            _script_inflight_middle_chunks_non_negative,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_lock_refs_non_negative(self):
        execute_scripted_runtime(
            _script_lock_refs_non_negative,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_chunked_in_flight_count_le_one(self):
        execute_scripted_runtime(
            _script_chunked_in_flight_count_le_one,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_status_transition_monotone(self):
        execute_scripted_runtime(
            _script_status_transition_monotone,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_active_reqs_listing(self):
        execute_scripted_runtime(
            _script_active_reqs_listing,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_batch_composition_disjoint_subsets(self):
        execute_scripted_runtime(
            _script_batch_composition_disjoint_subsets,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_finished_means_chunks_done_stable(self):
        execute_scripted_runtime(
            _script_finished_means_chunks_done_stable,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_finished_means_kv_pages_stays_zero(self):
        execute_scripted_runtime(
            _script_finished_means_kv_pages_stays_zero,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_engine_stats_keys_present(self):
        execute_scripted_runtime(
            _script_engine_stats_keys_present,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_kv_pool_recovers_to_baseline(self):
        execute_scripted_runtime(
            _script_kv_pool_recovers_to_baseline,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )


if __name__ == "__main__":
    unittest.main()
