import unittest

from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.http_server import ScriptedHttpServer
from sglang.test.scripted_runtime.req_handle import ScriptedReqHandle
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    advance_to_decode_step,
    advance_to_nth_chunk,
    base_engine_kwargs,
    exhaust_row_pool,
    run_until_finished,
    warmup_radix,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=460, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=460, stage="stage-b", runner_config="1-gpu-small-amd")


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
        assert r.req is not None, "req vanished before pause(retract)"

        t.pause_generation(mode="retract")
        yield

        req = r.req
        assert (
            req is not None and req in t.scheduler.waiting_queue
        ), f"pause(retract) did not park the req in waiting_queue; found {req!r}"

        frozen = len(req.output_ids)
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
            req is not None and req not in t.scheduler.waiting_queue
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
        assert (
            t.list_active_reqs() == []
        ), "active reqs should drain to empty after finish"

    def test_kv_pages_held_during_run_released_after(self):
        self.server.execute_script(self._script_kv_pages_set_then_released)

    @staticmethod
    def _script_kv_pages_set_then_released(t: ScriptedContext):
        r = t.start_req(
            prompt_len=_LONG_PROMPT_LEN,
            max_new_tokens=_DECODE_MAX_NEW_TOKENS,
            ignore_eos=True,
        )
        yield from advance_to_decode_step(r, 1)
        assert r.kv_pages > 0, f"expected kv_pages>0 mid-run; got {r.kv_pages}"
        yield from run_until_finished(r)
        assert r.kv_pages == 0, f"kv_pages should be 0 after finish; got {r.kv_pages}"

    def test_engine_stats_tracks_kv_pool(self):
        self.server.execute_script(self._script_engine_stats_tracks_kv)

    @staticmethod
    def _script_engine_stats_tracks_kv(t: ScriptedContext):
        stats = t.engine_stats()
        for key in ("kv_pool_free", "req_pool_free", "req_pool_total", "page_size"):
            assert key in stats, f"engine_stats missing {key!r}: {stats!r}"
        baseline_free = stats["kv_pool_free"]
        assert baseline_free > 0
        r = t.start_req(
            prompt_len=_LONG_PROMPT_LEN,
            max_new_tokens=_DECODE_MAX_NEW_TOKENS,
            ignore_eos=True,
        )
        yield from advance_to_decode_step(r, 1)
        during_free = t.engine_stats()["kv_pool_free"]
        assert during_free < baseline_free, (
            f"kv_pool_free should drop while a req holds KV; "
            f"baseline={baseline_free} during={during_free}"
        )
        yield from run_until_finished(r)
        for _ in range(5):
            yield
        t.flush_cache()
        yield
        after_free = t.engine_stats()["kv_pool_free"]
        assert (
            after_free >= during_free
        ), f"kv_pool_free should recover after finish; during={during_free} after={after_free}"

    def test_lock_refs_held_during_run_released_after(self):
        self.server.execute_script(self._script_lock_refs_held_then_released)

    @staticmethod
    def _script_lock_refs_held_then_released(t: ScriptedContext):
        t.flush_cache()
        yield
        r = t.start_req(
            prompt_len=_LONG_PROMPT_LEN,
            max_new_tokens=_DECODE_MAX_NEW_TOKENS,
            ignore_eos=True,
        )
        yield from advance_to_decode_step(r, 1)
        assert (
            r.lock_refs >= 1
        ), f"radix lock_ref must be held mid-run; got {r.lock_refs}"
        yield from run_until_finished(r)
        assert (
            r.lock_refs == 0
        ), f"lock_refs must be released after finish; got {r.lock_refs}"

    def test_batch_composition_shape_and_disjoint(self):
        self.server.execute_script(self._script_batch_composition)

    @staticmethod
    def _script_batch_composition(t: ScriptedContext):
        r = t.start_req(prompt_len=_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from advance_to_nth_chunk(r, 1)
        comp = t.batch_composition()
        assert set(comp) == {
            "prefill",
            "decode",
            "chunked",
            "running",
        }, f"unexpected batch_composition keys: {comp!r}"
        assert (
            r.rid in comp["chunked"]
        ), f"chunked req must be in 'chunked'; got {comp!r}"
        prefill, decode, chunked = (
            set(comp["prefill"]),
            set(comp["decode"]),
            set(comp["chunked"]),
        )
        assert (
            prefill.isdisjoint(decode)
            and prefill.isdisjoint(chunked)
            and decode.isdisjoint(chunked)
        ), f"prefill/decode/chunked subsets must be disjoint; got {comp!r}"
        yield from run_until_finished(r)
        assert (
            t.batch_composition()["chunked"] == []
        ), "no chunked req should remain after the req finishes"

    def test_chunks_done_zero_for_unchunked_prompt(self):
        self.server.execute_script(self._script_chunks_done_zero)

    @staticmethod
    def _script_chunks_done_zero(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=2, ignore_eos=True)
        yield from run_until_finished(r)
        assert (
            r.chunks_done == 0
        ), f"prompt <= chunk must not chunk; got {r.chunks_done}"

    def test_chunks_done_counts_two_chunks(self):
        self.server.execute_script(self._script_chunks_done_two)

    @staticmethod
    def _script_chunks_done_two(t: ScriptedContext):
        r = t.start_req(prompt_len=_CHUNK_SIZE + 2, max_new_tokens=2, ignore_eos=True)
        yield from run_until_finished(r)
        assert (
            r.chunks_done == 2
        ), f"chunk_size+2 prompt -> 2 chunks; got {r.chunks_done}"

    def test_chunks_done_scales_with_prompt(self):
        self.server.execute_script(self._script_chunks_done_five)

    @staticmethod
    def _script_chunks_done_five(t: ScriptedContext):
        r = t.start_req(prompt_len=5 * _CHUNK_SIZE, max_new_tokens=2, ignore_eos=True)
        yield from run_until_finished(r)
        assert (
            r.chunks_done == 5
        ), f"5*chunk_size prompt -> 5 chunks; got {r.chunks_done}"

    def test_is_idle_reflects_engine_activity(self):
        self.server.execute_script(self._script_is_idle_reflects_activity)

    @staticmethod
    def _script_is_idle_reflects_activity(t: ScriptedContext):
        assert t.is_idle, "engine should be idle at script start"
        r = t.start_req(
            prompt_len=_SHORT_PROMPT_LEN,
            max_new_tokens=_DECODE_MAX_NEW_TOKENS,
            ignore_eos=True,
        )
        yield from advance_to_decode_step(r, 1)
        assert not t.is_idle, "is_idle True while a req is decoding"
        yield from run_until_finished(r)
        for _ in range(5):
            yield
        assert t.is_idle, "is_idle False after the req drained"
        assert t.is_fully_idle, "is_fully_idle False after the req drained"

    def test_status_transitions_running_to_finished(self):
        self.server.execute_script(self._script_status_transitions)

    @staticmethod
    def _script_status_transitions(t: ScriptedContext):
        assert (
            t.status("no-such-rid") == "unknown"
        ), "status of a never-seen rid must be 'unknown'"
        r = t.start_req(
            prompt_len=_SHORT_PROMPT_LEN,
            max_new_tokens=_DECODE_MAX_NEW_TOKENS,
            ignore_eos=True,
        )
        yield from advance_to_decode_step(r, 1)
        assert (
            r.status == "running"
        ), f"decoding req status should be running; got {r.status!r}"
        yield from run_until_finished(r)
        assert (
            r.status == "finished"
        ), f"completed req status should be finished; got {r.status!r}"

    def test_last_batch_forward_mode_extend_then_decode(self):
        self.server.execute_script(self._script_last_batch_forward_mode)

    @staticmethod
    def _script_last_batch_forward_mode(t: ScriptedContext):
        r = t.start_req(
            prompt_len=_LONG_PROMPT_LEN,
            max_new_tokens=_DECODE_MAX_NEW_TOKENS,
            ignore_eos=True,
        )
        yield from advance_to_nth_chunk(r, 1)
        assert t.last_batch_forward_mode in ("EXTEND", "MIXED"), (
            f"mid-prefill batch should be an extend mode; "
            f"got {t.last_batch_forward_mode!r}"
        )
        yield from advance_to_decode_step(r, 1)
        assert (
            t.last_batch_forward_mode == "DECODE"
        ), f"decode batch mode should be DECODE; got {t.last_batch_forward_mode!r}"
        yield from run_until_finished(r)

    def test_remaining_prompt_tokens_shrinks_to_zero(self):
        self.server.execute_script(self._script_remaining_prompt_tokens)

    @staticmethod
    def _script_remaining_prompt_tokens(t: ScriptedContext):
        r = t.start_req(prompt_len=_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from advance_to_nth_chunk(r, 1)
        rem = r.remaining_prompt_tokens
        assert 0 < rem < _LONG_PROMPT_LEN, (
            f"mid-prefill remaining_prompt_tokens should be partial; "
            f"got {rem} (prompt={_LONG_PROMPT_LEN})"
        )
        yield from run_until_finished(r)
        assert (
            r.remaining_prompt_tokens == 0
        ), f"finished req should have 0 remaining; got {r.remaining_prompt_tokens}"

    def test_evict_radix_full_clears_tree_and_rejects_prefix(self):
        self.server.execute_script(self._script_evict_radix)

    @staticmethod
    def _script_evict_radix(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        for _ in range(3):
            yield
        assert t.get_all_node_hit_counts(), "expected radix nodes after a finished req"

        t.evict_radix(prefix_tokens=None)
        yield
        assert (
            not t.get_all_node_hit_counts()
        ), "evict_radix(prefix_tokens=None) did not clear the radix tree"

        rejected = False
        try:
            t.evict_radix(prefix_tokens=[1, 2, 3])
        except AssertionError:
            rejected = True
        assert (
            rejected
        ), "evict_radix must reject a non-None prefix (only full evict supported)"

    def test_warmup_radix_populates_prefix(self):
        self.server.execute_script(self._script_warmup_radix_populates_prefix)

    @staticmethod
    def _script_warmup_radix_populates_prefix(t: ScriptedContext):
        t.flush_cache()
        yield
        yield from warmup_radix(t, [1] * (2 * _CHUNK_SIZE))

        r = t.start_req(prompt_len=2 * _CHUNK_SIZE + 1, max_new_tokens=1)
        yield from run_until_finished(r)
        assert (
            r.req is not None
        ), "finished req vanished before cached_tokens could be read"
        assert r.req.cached_tokens > 0, (
            f"req with the warmed prefix should hit the radix cache; "
            f"got cached_tokens={r.req.cached_tokens}"
        )

    def test_exhaust_kv_creates_pressure_and_release_restores(self):
        self.server.execute_script(self._script_exhaust_kv_round_trip)

    @staticmethod
    def _script_exhaust_kv_round_trip(t: ScriptedContext):
        stats = t.engine_stats()
        page = stats["page_size"]
        baseline = stats["kv_pool_free"]
        assert (
            baseline > 4 * page
        ), f"need KV headroom to test pressure; baseline={baseline}"

        t.exhaust_kv(leave_pages=2)
        pressured = t.engine_stats()["kv_pool_free"]
        assert (
            pressured < baseline
        ), f"exhaust_kv must reduce free KV; pressured={pressured} baseline={baseline}"
        assert (
            pressured <= 3 * page
        ), f"exhaust_kv(leave_pages=2) left too much free KV; got {pressured} (page={page})"

        t._release_exhausted_pools()
        restored = t.engine_stats()["kv_pool_free"]
        assert (
            restored == baseline
        ), f"release must restore the full pool; restored={restored} baseline={baseline}"
        yield

    def test_exhaust_row_pool_leaves_requested_free_rows(self):
        self.server.execute_script(self._script_exhaust_row_pool)

    @staticmethod
    def _script_exhaust_row_pool(t: ScriptedContext):
        avail = t.engine_stats()["req_pool_free"]
        assert avail >= 5, f"need free rows to test; avail={avail}"
        target_free = avail - 3

        yield from exhaust_row_pool(t, leave_rows=target_free)

        free_after = t.engine_stats()["req_pool_free"]
        assert (
            free_after <= target_free
        ), f"exhaust_row_pool should leave <= {target_free} free rows; got {free_after}"
        assert (
            free_after < avail
        ), f"exhaust_row_pool did not consume any rows; avail={avail} after={free_after}"

    def test_forward_ct_advances_once_per_yield(self):
        self.server.execute_script(self._script_forward_ct_advances_once_per_yield)

    @staticmethod
    def _script_forward_ct_advances_once_per_yield(t: ScriptedContext):
        sched = t.scheduler

        before_no_yield = sched.forward_ct
        r = t.start_req(
            prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=128, ignore_eos=True
        )
        assert (
            sched.forward_ct == before_no_yield
        ), f"forward_ct moved without a yield: {before_no_yield} -> {sched.forward_ct}"

        yield from advance_to_decode_step(r, 1)

        for n in (1, 3, 5):
            before = sched.forward_ct
            for _ in range(n):
                yield
            advanced = sched.forward_ct - before
            assert advanced == n, (
                f"forward_ct advanced by {advanced} over {n} yields "
                f"(before={before} after={sched.forward_ct})"
            )

        t.abort_all()
        for _ in range(8):
            yield
            if r.finished:
                break

    def test_empty_script_returns_immediately(self):
        self.server.execute_script(self._script_empty_return)

    @staticmethod
    def _script_empty_return(t: ScriptedContext):
        yield from ()

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
        t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=2)
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
