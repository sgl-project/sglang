import unittest

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
        assert (
            len(r.output_tokens) == 6
        ), f"ignore_eos must decode the full length; got {r.output_tokens!r}"
        assert all(isinstance(tok, int) for tok in r.output_tokens)

    def test_start_req_priority_is_propagated(self):
        self.server.execute_script(self._script_priority_is_propagated)

    @staticmethod
    def _script_priority_is_propagated(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=4, priority=7)
        yield from advance_to_decode_step(r, 1)
        req = r.req
        assert (
            req is not None and req.priority == 7
        ), f"priority not propagated to the scheduler req; got {None if req is None else req.priority}"
        yield from run_until_finished(r)

    def test_start_req_dp_rank_zero_accepted(self):
        self.server.execute_script(self._script_dp_rank_zero_accepted)

    @staticmethod
    def _script_dp_rank_zero_accepted(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=4, dp_rank=0)
        yield from run_until_finished(r)
        assert r.finished, "dp_rank=0 req did not finish"

    def test_token_count_handles_are_consistent(self):
        self.server.execute_script(self._script_token_counts_consistent)

    @staticmethod
    def _script_token_counts_consistent(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=4, ignore_eos=True)
        yield from advance_to_decode_step(r, 2)
        assert (
            r.num_input_tokens == _SHORT_PROMPT_LEN
        ), f"num_input_tokens={r.num_input_tokens}, expected {_SHORT_PROMPT_LEN}"
        assert r.fill_ids_len > 0, "fill_ids_len should be positive during the run"
        assert r.total_tokens == r.num_input_tokens + len(r.output_tokens), (
            f"total_tokens={r.total_tokens} != "
            f"input({r.num_input_tokens}) + output({len(r.output_tokens)})"
        )
        yield from run_until_finished(r)
        assert len(r.output_tokens) == 4

    def test_row_idx_set_during_run_released_after_finish(self):
        self.server.execute_script(self._script_row_idx_set_then_released)

    @staticmethod
    def _script_row_idx_set_then_released(t: ScriptedContext):
        r = t.start_req(
            prompt_len=_SHORT_PROMPT_LEN,
            max_new_tokens=_DECODE_MAX_NEW_TOKENS,
            ignore_eos=True,
        )
        yield from advance_to_decode_step(r, 1)
        assert (
            r.row_idx is not None and r.row_idx >= 0
        ), f"expected a row idx while the req runs; got {r.row_idx!r}"
        yield from run_until_finished(r)
        assert (
            r.row_idx is None
        ), f"row_idx should be released after finish; got {r.row_idx!r}"

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

    def test_finish_reason_length_has_no_error_message(self):
        self.server.execute_script(self._script_finish_reason_length_no_error)

    @staticmethod
    def _script_finish_reason_length_no_error(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=3, ignore_eos=True)
        assert (
            r.finish_reason is None
        ), "finish_reason should be None before the req finishes"
        yield from run_until_finished(r)
        reason = r.finish_reason
        assert (
            reason is not None and reason.get("type") == "length"
        ), f"unexpected finish_reason {reason!r}"
        assert (
            r.error_message is None
        ), f"a length finish must not carry an error message; got {r.error_message!r}"

    def test_error_message_set_when_aborted(self):
        self.server.execute_script(self._script_error_message_on_abort)

    @staticmethod
    def _script_error_message_on_abort(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=64)
        yield from advance_to_decode_step(r, 1)
        t.abort(r)
        message = None
        for _ in range(16):
            yield
            current = r.error_message
            if current is not None:
                message = current
            if r.finished:
                break
        assert r.finished, "aborted req did not finish"
        assert message == "Aborted", f"expected abort message 'Aborted'; got {message!r}"

    def test_extend_input_len_positive_during_chunked_prefill(self):
        self.server.execute_script(self._script_extend_input_len_during_chunk)

    @staticmethod
    def _script_extend_input_len_during_chunk(t: ScriptedContext):
        r = t.start_req(prompt_len=_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from advance_to_nth_chunk(r, 1)
        assert (
            r.extend_input_len > 0
        ), f"extend_input_len should be positive during chunked prefill; got {r.extend_input_len}"
        yield from run_until_finished(r)

    def test_misc_handle_fields_readable_and_cleared(self):
        self.server.execute_script(self._script_misc_handle_fields)

    @staticmethod
    def _script_misc_handle_fields(t: ScriptedContext):
        r = t.start_req(
            prompt_len=_SHORT_PROMPT_LEN,
            max_new_tokens=_DECODE_MAX_NEW_TOKENS,
            ignore_eos=True,
        )
        yield from advance_to_decode_step(r, 1)
        assert r.kv_committed_len >= 0
        assert r.cached_tokens >= 0
        assert r.cached_tokens_snapshot == r.cached_tokens
        assert r.prefix_indices_len >= 0
        assert r.host_hit_length >= 0
        assert r.inflight_middle_chunks >= 0
        assert r.spec_verify_count >= 0
        assert r.lora_path is None, f"no LoRA configured; got {r.lora_path!r}"
        yield from run_until_finished(r)
        for _ in range(5):
            yield
        assert r.req is None, "req should be gone after finish + drain"
        assert r.kv_committed_len == 0
        assert r.cached_tokens == 0
        assert r.prefix_indices_len == 0
        assert r.spec_verify_count == 0
        assert r.lora_path is None

    def test_cached_tokens_and_last_node_on_radix_hit(self):
        self.server.execute_script(self._script_cached_tokens_radix_hit)

    @staticmethod
    def _script_cached_tokens_radix_hit(t: ScriptedContext):
        t.flush_cache()
        yield
        r1 = t.start_req(prompt_len=_LONG_PROMPT_LEN, max_new_tokens=2, ignore_eos=True)
        yield from run_until_finished(r1)
        for _ in range(3):
            yield
        r2 = t.start_req(prompt_len=_LONG_PROMPT_LEN, max_new_tokens=2, ignore_eos=True)
        yield from advance_to_decode_step(r2, 1)
        assert (
            r2.cached_tokens > 0
        ), f"identical prompt should hit the radix prefix; cached_tokens={r2.cached_tokens}"
        assert (
            r2.last_node_id is not None
        ), "a radix-hitting req should expose its matched last_node id"
        yield from run_until_finished(r2)

    def test_chunked_req_rid_and_in_flight_count(self):
        self.server.execute_script(self._script_chunked_req_rid_and_count)

    @staticmethod
    def _script_chunked_req_rid_and_count(t: ScriptedContext):
        assert t.get_chunked_req_rid() is None, "no chunked req expected when idle"
        assert t.chunked_in_flight_count() == 0
        r = t.start_req(prompt_len=_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from advance_to_nth_chunk(r, 1)
        assert (
            t.get_chunked_req_rid() == r.rid
        ), f"chunked rid mismatch: {t.get_chunked_req_rid()!r} != {r.rid!r}"
        assert t.chunked_in_flight_count() == 1
        yield from run_until_finished(r)
        assert t.get_chunked_req_rid() is None
        assert t.chunked_in_flight_count() == 0

    def test_running_and_batch_rids_track_live_req(self):
        self.server.execute_script(self._script_running_batch_rids)

    @staticmethod
    def _script_running_batch_rids(t: ScriptedContext):
        assert t.running_rids() == [], "running_rids should be empty when idle"
        assert t.batch_rids() == []
        assert t.batch_size() == 0
        assert t.waiting_rids() == []
        r = t.start_req(
            prompt_len=_SHORT_PROMPT_LEN,
            max_new_tokens=_DECODE_MAX_NEW_TOKENS,
            ignore_eos=True,
        )
        yield from advance_to_decode_step(r, 1)
        assert r.rid in t.running_rids(), f"{r.rid!r} not in {t.running_rids()!r}"
        assert r.rid in t.batch_rids()
        assert t.batch_size() >= 1
        yield from run_until_finished(r)
        for _ in range(5):
            yield
        assert t.running_rids() == [], "running_rids should drain to empty after finish"
        assert t.batch_size() == 0

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
        assert any(
            req.rid == r.rid for req in actives
        ), f"live req {r.rid!r} missing from active reqs {[req.rid for req in actives]!r}"
        yield from run_until_finished(r)
        for _ in range(5):
            yield
        assert t.list_active_reqs() == [], "active reqs should drain to empty after finish"

    def test_lock_refs_snapshot_matches_node_lock_refs(self):
        self.server.execute_script(self._script_lock_refs_snapshot)

    @staticmethod
    def _script_lock_refs_snapshot(t: ScriptedContext):
        t.flush_cache()
        yield
        r = t.start_req(
            prompt_len=_LONG_PROMPT_LEN,
            max_new_tokens=_DECODE_MAX_NEW_TOKENS,
            ignore_eos=True,
        )
        yield from advance_to_decode_step(r, 1)
        assert (
            t.lock_refs_snapshot() == t.get_all_node_lock_refs()
        ), "lock_refs_snapshot must mirror get_all_node_lock_refs"
        yield from run_until_finished(r)
        for _ in range(3):
            yield
        released = t.lock_refs_snapshot()
        assert released and all(
            ref == 0 for ref in released.values()
        ), f"radix nodes still locked after finish: {released}"

    def test_abort_single_handle_finishes_req(self):
        self.server.execute_script(self._script_abort_single_handle)

    @staticmethod
    def _script_abort_single_handle(t: ScriptedContext):
        r = t.start_req(prompt_len=_SHORT_PROMPT_LEN, max_new_tokens=64)
        yield from advance_to_decode_step(r, 1)
        assert not r.finished, "req finished before abort could act"
        t.abort(r)
        for _ in range(16):
            yield
            if r.finished:
                break
        assert r.finished, "single-handle abort did not finish the req"

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
        t.flush_cache()
        for _ in range(5):
            yield
        after_free = t.engine_stats()["kv_pool_free"]
        assert (
            after_free >= during_free
        ), f"kv_pool_free should recover after finish; during={during_free} after={after_free}"

    def test_row_pool_used_tracks_reqs(self):
        self.server.execute_script(self._script_row_pool_used_tracks_reqs)

    @staticmethod
    def _script_row_pool_used_tracks_reqs(t: ScriptedContext):
        for _ in range(3):
            yield
        baseline = t.row_pool_used()
        r = t.start_req(
            prompt_len=_SHORT_PROMPT_LEN,
            max_new_tokens=_DECODE_MAX_NEW_TOKENS,
            ignore_eos=True,
        )
        yield from advance_to_decode_step(r, 1)
        during = t.row_pool_used()
        assert (
            during >= baseline + 1
        ), f"row pool should grow with a live req; baseline={baseline} during={during}"
        yield from run_until_finished(r)
        for _ in range(8):
            yield
        after = t.row_pool_used()
        assert after == baseline, f"row pool leaked: baseline={baseline} after={after}"

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
