import unittest

from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.http_server import ScriptedHttpServer
from sglang.test.scripted_runtime.req_handle import ScriptedReqHandle
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    advance_to_decode_step,
    advance_to_nth_chunk,
    base_engine_kwargs,
    run_until_finished,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=420, stage="base-b", runner_config="1-gpu-small")


_CHUNK_SIZE = 64
_LONG_PROMPT_LEN = 4 * _CHUNK_SIZE - 3
_SHORT_PROMPT_LEN = 16
_DECODE_MAX_NEW_TOKENS = 8

_ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=_CHUNK_SIZE)


def _script_noop(t: ScriptedContext):
    yield


class TestScriptedRuntimeCore(ScriptedTestCase):
    ENGINE_KWARGS = _ENGINE_KWARGS

    def test_start_req_auto_rid_and_finishes(self):
        self.server.execute_script(self._script_start_req_auto_rid_and_finishes)

    @staticmethod
    def _script_start_req_auto_rid_and_finishes(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=4)
        assert r.rid.startswith("scripted-"), f"unexpected auto rid {r.rid!r}"
        yield from run_until_finished(r)
        assert r.finished, "auto-rid req did not finish"

    def test_start_req_explicit_rid(self):
        self.server.execute_script(self._script_start_req_explicit_rid)

    @staticmethod
    def _script_start_req_explicit_rid(t: ScriptedContext):
        r = t.start_req(
            prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=2, rid="explicit-rid-test"
        )
        assert r.rid == "explicit-rid-test", f"explicit rid not honored: {r.rid!r}"
        yield
        assert (
            t.find_req_by_rid("explicit-rid-test") is not None
        ), "explicit rid not visible to the scheduler after one step"

    def test_find_req_by_rid_hit_and_miss(self):
        self.server.execute_script(self._script_find_req_by_rid_hit_and_miss)

    @staticmethod
    def _script_find_req_by_rid_hit_and_miss(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=4)
        yield
        found = t.find_req_by_rid(r.rid)
        assert (
            found is not None and found.rid == r.rid
        ), f"find_req_by_rid missed the live rid {r.rid!r}"
        assert (
            t.find_req_by_rid("no-such-rid") is None
        ), "find_req_by_rid returned a req for an unknown rid"
        yield from run_until_finished(r)

    def test_is_finished_reflects_completion(self):
        self.server.execute_script(self._script_is_finished_reflects_completion)

    @staticmethod
    def _script_is_finished_reflects_completion(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=4)
        yield
        assert not t.is_finished(r.rid), "is_finished True while the req still runs"
        yield from run_until_finished(r)
        assert t.is_finished(r.rid), "is_finished False after the req completed"

    def test_req_handle_req_property(self):
        self.server.execute_script(self._script_req_handle_req_property)

    @staticmethod
    def _script_req_handle_req_property(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=4)
        yield
        assert (
            r.req is not None and r.req.rid == r.rid
        ), f"handle.req did not resolve to the live req for {r.rid!r}"
        bogus = ScriptedReqHandle(rid="no-such-rid", context=t)
        assert bogus.req is None, "handle.req returned a req for an unknown rid"
        yield from run_until_finished(r)

    def test_req_handle_finished_property(self):
        self.server.execute_script(self._script_req_handle_finished_property)

    @staticmethod
    def _script_req_handle_finished_property(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=4)
        yield
        assert not r.finished, "handle.finished True while the req still runs"
        yield from run_until_finished(r)
        assert r.finished, "handle.finished False after the req completed"

    def test_is_chunking_true_mid_prefill_false_after(self):
        self.server.execute_script(
            self._script_is_chunking_true_mid_prefill_false_after
        )

    @staticmethod
    def _script_is_chunking_true_mid_prefill_false_after(t: ScriptedContext):
        r = t.start_req(prompt_len=_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from advance_to_nth_chunk(r, 1)
        assert r.is_chunking, "handle.is_chunking False during multi-chunk prefill"
        assert t.is_chunking(
            r.rid
        ), "context.is_chunking False during multi-chunk prefill"
        assert not t.is_chunking(
            "no-such-rid"
        ), "context.is_chunking True for an unknown rid"
        yield from run_until_finished(r)
        assert not r.is_chunking, "handle.is_chunking still True after finish"
        assert not t.is_chunking(r.rid), "context.is_chunking still True after finish"

    def test_pause_retract_parks_in_waiting_queue_then_resumes(self):
        self.server.execute_script(self._script_pause_retract_parks_then_resumes)

    @staticmethod
    def _script_pause_retract_parks_then_resumes(t: ScriptedContext):
        r = t.start_req(
            prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=_DECODE_MAX_NEW_TOKENS
        )
        yield from advance_to_decode_step(r, 1)
        req = r.req
        assert req is not None, "req vanished before pause(retract)"
        frozen = len(req.output_ids)

        t.pause_generation(mode="retract")
        yield

        req = r.req
        assert (
            req is not None and req in t._scheduler.waiting_queue
        ), f"pause(retract) did not park the req in waiting_queue; found {req!r}"

        for _ in range(3):
            yield
            req = r.req
            assert req is not None and len(req.output_ids) == frozen, (
                f"paused(retract) engine advanced the req: "
                f"{len(req.output_ids) if req is not None else None} != {frozen}"
            )

        t.continue_generation()
        yield from run_until_finished(r)
        assert r.finished, "req did not finish after pause(retract)/continue"

    def test_pause_in_place_freezes_then_resumes(self):
        self.server.execute_script(self._script_pause_in_place_freezes_then_resumes)

    @staticmethod
    def _script_pause_in_place_freezes_then_resumes(t: ScriptedContext):
        r = t.start_req(
            prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=_DECODE_MAX_NEW_TOKENS
        )
        yield from advance_to_decode_step(r, 1)
        req = r.req
        assert req is not None, "req vanished before pause(in_place)"
        frozen = len(req.output_ids)

        t.pause_generation(mode="in_place")
        yield

        req = r.req
        assert (
            req is not None and req not in t._scheduler.waiting_queue
        ), f"pause(in_place) should not retract the req to waiting_queue; found {req!r}"

        for _ in range(3):
            yield
            req = r.req
            assert req is not None and len(req.output_ids) == frozen, (
                f"paused(in_place) engine advanced the req: "
                f"{len(req.output_ids) if req is not None else None} != {frozen}"
            )

        t.continue_generation()
        yield from run_until_finished(r)
        assert r.finished, "req did not finish after pause(in_place)/continue"

    def test_continue_generation_with_torch_empty_cache(self):
        self.server.execute_script(
            self._script_continue_generation_with_torch_empty_cache
        )

    @staticmethod
    def _script_continue_generation_with_torch_empty_cache(t: ScriptedContext):
        r = t.start_req(
            prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=_DECODE_MAX_NEW_TOKENS
        )
        yield from advance_to_decode_step(r, 1)
        t.pause_generation(mode="retract")
        yield
        t.continue_generation(torch_empty_cache=True)
        yield from run_until_finished(r)
        assert r.finished, "req did not finish after continue(torch_empty_cache=True)"

    def test_abort_all_finishes_running_req(self):
        self.server.execute_script(self._script_abort_all_finishes_running_req)

    @staticmethod
    def _script_abort_all_finishes_running_req(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=64)
        yield from advance_to_decode_step(r, 1)
        assert not r.finished, "req finished before abort_all could act"

        t.abort_all()
        for _ in range(8):
            yield
            if r.finished:
                break
        assert r.finished, "req did not finish after abort_all"

    def test_flush_cache_clears_radix_tree(self):
        self.server.execute_script(self._script_flush_cache_clears_radix_tree)

    @staticmethod
    def _script_flush_cache_clears_radix_tree(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        for _ in range(3):
            yield
        assert t.get_all_node_hit_counts(), "expected radix nodes after a finished req"

        t.flush_cache()
        yield
        assert (
            not t.get_all_node_hit_counts()
        ), "flush_cache did not clear the radix tree"

    def test_get_all_node_hit_counts_increments_on_cache_hit(self):
        self.server.execute_script(self._script_hit_counts_increment_on_cache_hit)

    @staticmethod
    def _script_hit_counts_increment_on_cache_hit(t: ScriptedContext):
        t.flush_cache()
        yield

        r1 = t.start_req(prompt_len=_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r1)
        for _ in range(3):
            yield
        counts_before = t.get_all_node_hit_counts()
        assert counts_before, "expected radix nodes after the first req"
        max_before = max(counts_before.values())

        r2 = t.start_req(prompt_len=_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r2)
        counts_after = t.get_all_node_hit_counts()
        assert max(counts_after.values()) > max_before, (
            f"identical prompt did not bump any node hit_count: "
            f"before={max_before} after={max(counts_after.values())}"
        )

    def test_get_all_node_lock_refs_held_during_run_released_after(self):
        self.server.execute_script(self._script_lock_refs_held_then_released)

    @staticmethod
    def _script_lock_refs_held_then_released(t: ScriptedContext):
        t.flush_cache()
        yield

        r = t.start_req(
            prompt_len=_LONG_PROMPT_LEN, max_new_tokens=_DECODE_MAX_NEW_TOKENS
        )
        yield from advance_to_decode_step(r, 1)
        lock_refs = t.get_all_node_lock_refs()
        assert (
            lock_refs and max(lock_refs.values()) >= 1
        ), f"expected a locked radix node while the req runs; got {lock_refs}"

        yield from run_until_finished(r)
        for _ in range(3):
            yield
        released = t.get_all_node_lock_refs()
        assert released and all(
            ref == 0 for ref in released.values()
        ), f"radix nodes still locked after the req finished: {released}"

    def test_start_req_ignore_eos_runs_full_length(self):
        self.server.execute_script(self._script_ignore_eos_runs_full_length)

    @staticmethod
    def _script_ignore_eos_runs_full_length(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=6, ignore_eos=True)
        yield from run_until_finished(r)
        req = r.req
        assert req is not None, "finished req vanished before its output could be read"
        assert (
            len(req.output_ids) == 6
        ), f"ignore_eos must decode the full length; got {list(req.output_ids)!r}"

    def test_start_req_priority_is_propagated(self):
        self.server.execute_script(self._script_priority_is_propagated)

    @staticmethod
    def _script_priority_is_propagated(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=4, priority=7)
        yield from advance_to_decode_step(r, 1)
        req = r.req
        assert req is not None and req.priority == 7, (
            f"priority not propagated to the scheduler req; "
            f"got {None if req is None else req.priority}"
        )
        yield from run_until_finished(r)

    def test_start_req_dp_rank_zero_accepted(self):
        self.server.execute_script(self._script_dp_rank_zero_accepted)

    @staticmethod
    def _script_dp_rank_zero_accepted(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=4, dp_rank=0)
        yield from run_until_finished(r)
        assert r.finished, "dp_rank=0 req did not finish"

    def test_abort_single_handle_finishes_with_abort_reason(self):
        self.server.execute_script(self._script_abort_single_handle)

    @staticmethod
    def _script_abort_single_handle(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=64)
        yield from advance_to_decode_step(r, 1)
        assert not r.finished, "req finished before abort could act"

        t.abort(r)
        saw_abort_reason = False
        for _ in range(16):
            yield
            req = r.req
            if req is not None and isinstance(req.finished_reason, FINISH_ABORT):
                saw_abort_reason = True
            if r.finished:
                break
        assert r.finished, "single-handle abort did not finish the req"
        assert saw_abort_reason, "aborted req never carried a FINISH_ABORT reason"

    def test_abort_single_handle_leaves_other_reqs_running(self):
        self.server.execute_script(self._script_abort_single_handle_targeted)

    @staticmethod
    def _script_abort_single_handle_targeted(t: ScriptedContext):
        keep = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=64)
        victim = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=64)
        yield from advance_to_decode_step(keep, 1)

        t.abort(victim)
        for _ in range(16):
            yield
            if victim.finished:
                break
        assert victim.finished, "targeted abort did not finish the victim req"
        assert not keep.finished, "targeted abort wrongly finished the other req"

        t.abort(keep)
        yield from run_until_finished(keep)

    def test_list_active_reqs_contains_live_req(self):
        self.server.execute_script(self._script_list_active_reqs)

    @staticmethod
    def _script_list_active_reqs(t: ScriptedContext):
        assert t.list_active_reqs() == [], "no active reqs expected when idle"
        r = t.start_req(
            prompt_len=_SHORT_PROMPT_LEN,
            max_new_tokens=_DECODE_MAX_NEW_TOKENS,
            ignore_eos=True,
        )
        yield from advance_to_decode_step(r, 1)
        actives = t.list_active_reqs()
        assert any(req.rid == r.rid for req in actives), (
            f"live req {r.rid!r} missing from active reqs "
            f"{[req.rid for req in actives]!r}"
        )
        yield from run_until_finished(r)
        for _ in range(5):
            yield
        assert t.list_active_reqs() == [], "active reqs should drain to empty after finish"

    def test_empty_script_returns_immediately(self):
        self.server.execute_script(self._script_empty_return)

    @staticmethod
    def _script_empty_return(t: ScriptedContext):
        if False:
            yield
        return

    def test_failing_script_surfaces_and_session_survives(self):
        with self.assertRaises(AssertionError) as ctx:
            self.server.execute_script(self._script_assertion_failure)
        self.assertIn("boom", str(ctx.exception))
        self.server.execute_script(self._script_minimal_ok)

    @staticmethod
    def _script_assertion_failure(t: ScriptedContext):
        yield
        assert False, "boom"

    @staticmethod
    def _script_minimal_ok(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=2)
        yield
        yield

    def test_runtime_error_in_script_surfaces_to_caller(self):
        with self.assertRaises(AssertionError) as ctx:
            self.server.execute_script(self._script_runtime_error)
        err_text = str(ctx.exception)
        self.assertIn("RuntimeError", err_text)
        self.assertIn("simulated runtime error", err_text)

    @staticmethod
    def _script_runtime_error(t: ScriptedContext):
        yield
        raise RuntimeError("simulated runtime error")

    def test_non_generator_script_rejected(self):
        with self.assertRaises(AssertionError) as ctx:
            self.server.execute_script(self._script_not_a_generator)
        err_text = str(ctx.exception)
        self.assertIn("TypeError", err_text)
        self.assertIn("NoneType", err_text)

    @staticmethod
    def _script_not_a_generator(t: ScriptedContext):
        return None


class TestScriptedRuntimeSession(CustomTestCase):

    def test_shutdown_is_idempotent(self):
        session = ScriptedHttpServer.start(**_ENGINE_KWARGS)
        session.shutdown()
        session.shutdown()
        assert session._shutdown_done is True

    def test_dirty_session_refuses_to_run(self):
        session = ScriptedHttpServer.start(**_ENGINE_KWARGS)
        try:
            session._dirty = "test dirty"
            with self.assertRaises(RuntimeError) as ctx:
                session.execute_script(_script_noop)
            assert "test dirty" in str(ctx.exception)
        finally:
            session.shutdown()


if __name__ == "__main__":
    unittest.main()
