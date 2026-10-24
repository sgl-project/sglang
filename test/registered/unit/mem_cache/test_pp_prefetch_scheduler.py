"""Readiness-first PP gate with native queue ownership and CPU cache/pool doubles."""

import logging
import unittest
from array import array
from dataclasses import replace
from enum import Enum, auto
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.pp_cache_test_support import (
    Allocator,
    RequestDouble,
    RowPool,
    cycle,
    functions,
    m,
    method,
    request,
    settle,
    stage,
    tick,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class GateTests(CustomTestCase):
    def stages(self, size=3, caps=None):
        return [stage(i, size, caps or {"r": 1024 - i * 64}) for i in range(size)]

    def assert_released(self, stages):
        for runtime in stages:
            self.assertFalse(runtime.entries)
            self.assertTrue(
                all(
                    v == 0
                    for refs in runtime.s.tree_cache.refs.values()
                    for v in refs.values()
                )
            )

    def test_cancelled_entry_keeps_gate_active_until_commit_is_forwarded(self):
        stages = self.stages()
        cycle(stages)
        prepared, _ = tick(stages)
        root = stages[0]
        root.consume_return(prepared)
        retire = method(
            "managers/scheduler.py",
            "Scheduler",
            "_release_aborted_request",
            {"CacheRequestOutcome": NS(ABORT="abort")},
        )
        for gate in stages:
            req = gate.s.waiting_queue.pop()
            gate.s.tree_cache.finish = Mock()
            retire(gate.s, req)
        self.assertFalse(root.entries)
        self.assertFalse(root.s.waiting_queue)
        self.assertTrue(root.has_pending_work())
        forwarded, _ = cycle(stages)
        self.assertEqual(forwarded.commit, prepared.prepare)
        self.assertFalse(root.has_pending_work())
        self.assert_released(stages)

    def test_native_attempt_retirement_allows_a_fresh_watch(self):
        stages = self.stages()
        cycle(stages)
        old_keys = [gate.entries["r"].key for gate in stages]
        owners = {gate.s.waiting_queue[0]: gate for gate in stages}
        retire = method(
            "managers/scheduler.py",
            "Scheduler",
            "_release_aborted_request",
            {"CacheRequestOutcome": NS(ABORT="abort")},
        )
        advance = method(
            "managers/schedule_batch.py",
            "Req",
            "advance_cache_request_handle",
            {"dataclasses": NS(replace=replace)},
        )

        def release_kv(req, cache, **kwargs):
            # Exercise the actual producer's ordering, not an invented mutation
            # while WATCH is live. This does not enable optimistic PP serving.
            self.assertNotIn(req.rid, owners[req].entries)
            self.assertEqual(req.cache_request_handle.attempt_id, 0)
            cache.finish.assert_called_once_with(req.cache_request_handle, "abort")

        requeue = method(
            "disaggregation/prefill.py",
            "SchedulerDisaggregationPrefillMixin",
            "optimistic_release_and_requeue",
            {
                "get_disagg": lambda: NS(
                    optimistic_prefill_attempts=2,
                    disaggregation_decode_allocation_policy="default",
                ),
                "_uses_write_through_cache": lambda cache: True,
                "release_kv_cache": release_kv,
                "array": array,
                "logger": logging.getLogger(__name__),
            },
        )
        for gate in stages:
            s = gate.s
            req = s.waiting_queue.pop()
            s.tree_cache.finish = Mock()
            s._release_aborted_request = lambda req, s=s: retire(s, req)
            s.clear_pending_chunk_send = Mock()
            s.metrics_reporter = NS(enable_metrics=False)
            s.processed_tokens_counter = 0
            req.advance_cache_request_handle = lambda req=req: advance(req)
            req.reset_for_retract = Mock()
            req.skip_radix_cache_insert = True
            req.prefill_attempt_count = 1
            requeue(s, req)
            self.assertEqual(s.waiting_queue, [req])
            self.assertFalse(gate.entries)
        fresh, _ = cycle(stages)
        self.assertEqual(fresh.watch[0].attempt_id, 1)
        self.assertNotEqual(fresh.watch[0], old_keys[0])
        cycle(stages)
        self.assertEqual(cycle(stages)[1], [(("r", 896, 1025),)] * 3)
        self.assert_released(stages)

    def test_readiness_records_native_storage_wait_once(self):
        """Bypassing native selection must not lose or restart the cache wait timer."""
        gate = stage(0, 3, {"r": 512})
        req = gate.s.waiting_queue[0]
        req.time_stats.enable_metrics = True
        req.time_stats.metrics_collector = Mock()
        req.time_stats.set_wait_queue_entry_time(ts=1.0)
        gate._watch_waiting()
        gate.s.tree_cache.waiting.add("r")
        now = [10.0]
        with patch("sglang.srt.utils.poll_timer.time", NS(perf_counter=lambda: now[0])):
            self.assertEqual(gate._report_readiness(), ())
            now[0] = 13.0
            gate.s.tree_cache.waiting.clear()
            self.assertEqual(gate._report_readiness()[0][1], 1)
            self.assertEqual(req.time_stats.storage_prefetch_wait_s, 3.0)
            now[0] = 20.0
            self.assertEqual(gate._report_readiness(), ())
            self.assertEqual(req.time_stats.storage_prefetch_wait_s, 3.0)

    def test_all_ready_then_finalize_min_then_native_selection(self):
        stages = self.stages()
        first, batches = cycle(stages)
        self.assertEqual(batches, [()] * 3)
        self.assertEqual(len(first.watch), 1)
        self.assertEqual(first.ready[0][1], 7)
        self.assertTrue(all(not r.s.tree_cache.finalized for r in stages))
        prepared, batches = cycle(stages)
        self.assertEqual(prepared.prepare[0].cap, 896)
        self.assertEqual(batches, [()] * 3)
        self.assertTrue(all(r.s.tree_cache.finalized == ["r"] for r in stages))
        _, batches = cycle(stages)
        self.assertEqual(batches, [(("r", 896, 1025),)] * 3)
        self.assert_released(stages)

    def test_pending_only_reports_deltas_without_termination_or_reservation(self):
        stages = self.stages()
        stages[-1].s.tree_cache.waiting.add("r")
        first, _ = cycle(stages)
        self.assertEqual(first.ready[0][1], 3)
        identities = [runtime.entries["r"] for runtime in stages]
        for _ in range(8):
            envelope, batches = cycle(stages)
            self.assertEqual(envelope.ready, ())
            self.assertEqual(envelope.prepare, ())
            self.assertFalse(any(batches))
            for runtime, entry in zip(stages, identities):
                self.assertIs(runtime.entries["r"], entry)
                self.assertEqual(entry.phase, m.Phase.PENDING)
                self.assertEqual(entry.credit, 0)
                self.assertIsNone(entry.lease)
                self.assertEqual(runtime.s.tree_cache.finalized, [])
        stages[-1].s.tree_cache.waiting.clear()
        ready, _ = cycle(stages)
        self.assertEqual(ready.ready[0][1], 4)
        cycle(stages)
        self.assertEqual(cycle(stages)[1], [(("r", 896, 1025),)] * 3)

    def test_cap_is_measured_after_finalize_not_at_readiness(self):
        stages = self.stages()
        cycle(stages)
        for runtime in stages:
            runtime.s.tree_cache.caps["r"] = 128 + 64 * runtime.rank
        prepared, _ = cycle(stages)
        self.assertEqual(prepared.prepare[0].cap, 128)
        self.assertEqual(cycle(stages)[1], [(("r", 128, 1025),)] * 3)

    def test_zero_prefix_is_a_valid_cold_certificate(self):
        stages = self.stages(caps={"r": 0})
        cycle(stages)
        prepared, _ = cycle(stages)
        self.assertEqual(prepared.prepare[0].cap, 0)
        self.assertEqual(cycle(stages)[1], [(("r", 0, 1025),)] * 3)

    def test_capacity_nack_releases_same_round_before_retry(self):
        stages = self.stages()
        stages[-1].s.req_to_token_pool = RowPool(0)
        cycle(stages)
        failed, _ = tick(stages)
        self.assertIsNone(failed.prepare[0].cap)
        old_lease = stages[0].entries["r"].lease
        stages[0].consume_return(failed)
        self.assertTrue(old_lease.released)
        stages[-1].s.req_to_token_pool = RowPool(1)
        newer, _ = tick(stages)
        self.assertGreater(newer.prepare[0].generation, failed.prepare[0].generation)
        self.assertIsNone(newer.commit[0].cap)
        for runtime in stages[1:]:
            runtime.consume_return(failed)
            self.assertEqual(
                runtime.entries["r"].generation, newer.prepare[0].generation
            )
        settle(stages, newer)
        self.assertEqual(cycle(stages)[1], [(("r", 896, 1025),)] * 3)
        self.assert_released(stages)

    def test_small_pool_credit_does_not_hide_the_only_native_row(self):
        stages = self.stages()
        for runtime in stages:
            runtime.s.req_to_token_pool = RowPool(1)
        cycle(stages)
        cycle(stages)
        for runtime in stages:
            self.assertEqual(runtime.s.req_to_token_pool.available_size(), 1)
            self.assertEqual(
                runtime.s.token_to_kv_pool_allocator.available_size(), 65536
            )
            self.assertGreater(runtime.entries["r"].credit, 1025)
        self.assertEqual(cycle(stages)[1], [(("r", 896, 1025),)] * 3)

    def test_unselected_certificate_retains_credit_without_allocator_churn(self):
        stages = self.stages(caps={"a": 512, "b": 256})
        for _ in range(4):
            cycle(stages, limit=0)
        owners = [r.entries["a"] for r in stages]
        for runtime in stages:
            runtime.s.token_to_kv_pool_allocator.alloc = Mock(
                side_effect=AssertionError("placeholder allocation")
            )
            runtime.s.token_to_kv_pool_allocator.free_segment = Mock(
                side_effect=AssertionError("placeholder free")
            )
            runtime.s.req_to_token_pool.alloc_rows = Mock(
                side_effect=AssertionError("placeholder row")
            )
        for _ in range(5):
            cycle(stages, limit=0)
        for runtime, owner in zip(stages, owners):
            self.assertIs(runtime.entries["a"], owner)
            self.assertIsNotNone(owner.lease)
            self.assertGreater(owner.credit, 0)
        _, layouts = cycle(stages, limit=1)
        self.assertEqual(layouts, [(("a", 512, 1025),)] * 3)
        self.assertTrue(all(r.entries["b"].credit > 0 for r in stages))

    def test_unready_request_does_not_starve_ready_request_or_reorder_queue(self):
        stages = self.stages(caps={"slow": 512, "fast": 256})
        for runtime in stages:
            runtime.s.tree_cache.waiting.add("slow")
        for _ in range(3):
            _, batches = cycle(stages)
        self.assertEqual(batches, [(("fast", 256, 1025),)] * 3)
        self.assertTrue(
            all([r.rid for r in s.s.waiting_queue] == ["slow"] for s in stages)
        )
        self.assertTrue(all(s.entries["slow"].phase == m.Phase.PENDING for s in stages))

    def test_capacity_window_is_bounded(self):
        stages = self.stages(caps={f"r{i}": 512 for i in range(20)})
        for runtime in stages:
            runtime.window = 2
        for _ in range(12):
            cycle(stages, limit=0)
        self.assertEqual(sum(e.credit > 0 for e in stages[0].entries.values()), 2)
        self.assertEqual(len(stages[0].entries), 20)
        self.assertEqual(stages[0].s.tree_cache.finalized, ["r0", "r1"])

    def test_new_arrivals_cannot_starve_a_capacity_retry(self):
        stages = self.stages()
        stages[-1].s.req_to_token_pool = RowPool(0)
        cycle(stages)
        cycle(stages)  # A real capacity NACK, not a readiness retry.
        stages[-1].s.req_to_token_pool = RowPool(1)
        computed = []
        for i in range(4):
            for runtime in stages:
                runtime.s.waiting_queue.append(request(f"new{i}"))
                runtime.s.tree_cache.caps[f"new{i}"] = 0
            _, layouts = cycle(stages)
            computed.extend(row[0] for row in layouts[0])
        self.assertEqual(computed[0], "r")

    def test_ready_waiter_is_stable_when_native_selection_defers(self):
        stages = self.stages()
        cycle(stages)
        prepared, _ = cycle(stages)
        for _ in range(4):
            cycle(stages, limit=0)
            stages[0].consume_return(prepared)
        self.assertEqual(stages[0].entries["r"].phase, m.Phase.CERTIFIED)
        self.assertEqual(cycle(stages)[1], [(("r", 896, 1025),)] * 3)

    def test_native_selection_owns_waiting_queue_and_new_arrivals(self):
        stages = self.stages()
        cycle(stages)
        cycle(stages)
        root = stages[0]
        waiting = root.s.waiting_queue
        selection = root.prepare_selection()
        waiting.append(request("new"))
        root.s.tree_cache.caps["new"] = 0
        self.assertEqual([req.rid for req in selection.candidates], ["r"])
        root.finish_selection(None)
        self.assertIs(root.s.waiting_queue, waiting)
        self.assertEqual([r.rid for r in waiting], ["r", "new"])

    def test_retirement_ignores_late_results_and_reused_rid(self):
        for boundary in ("pending", "prepared", "certified"):
            with self.subTest(boundary=boundary):
                stages = self.stages()
                old, _ = cycle(stages)
                if boundary != "pending":
                    old, _ = tick(stages)
                    if boundary == "certified":
                        settle(stages, old)
                old_key = stages[0].entries["r"].key
                for runtime in stages:
                    req = runtime.s.waiting_queue.pop()
                    runtime.release(req)  # The native retirement hook.
                    runtime.s.waiting_queue.append(request("r"))
                newer, _ = cycle(stages)
                self.assertNotEqual(newer.watch[0], old_key)
                for runtime in stages:
                    runtime.consume_return(old)
                    self.assertEqual(runtime.entries["r"].key, newer.watch[0])
                cycle(stages)
                self.assertEqual(cycle(stages)[1], [(("r", 896, 1025),)] * 3)
                self.assert_released(stages)

    def test_only_current_prepare_result_is_committed_and_forwarded_once(self):
        for cap in (None, 0, 512):
            with self.subTest(cap=cap):
                stages = self.stages()
                cycle(stages)
                update, _ = tick(stages)
                root = stages[0]
                result = replace(update.prepare[0], cap=cap)
                for stale in (
                    replace(result, request=replace(result.request, rid="missing")),
                    replace(result, request=replace(result.request, sequence=-1)),
                    replace(result, generation=-1),
                ):
                    root.consume_return(m.AdmissionUpdate(prepare=(stale,)))
                    self.assertEqual(root.commits, [])
                    self.assertEqual(root.entries["r"].phase, m.Phase.PREPARING)
                returned = m.AdmissionUpdate(prepare=(result,))
                root.consume_return(returned)
                root.consume_return(returned)
                self.assertEqual(root.commits, [result])
                entry = root.entries["r"]
                if cap is None:
                    self.assertEqual(entry.phase, m.Phase.PENDING)
                    self.assertIsNone(entry.lease)
                    self.assertEqual(entry.credit, 0)
                else:
                    self.assertEqual(entry.phase, m.Phase.CERTIFIED)
                    self.assertEqual(entry.cap, cap)

    def test_delayed_bootstrap_is_native_only_no_early_finalization(self):
        stages = self.stages()
        peer = stages[-1]
        req = peer.s.waiting_queue.pop()
        peer.s.disagg_prefill_bootstrap_queue.queue.append(req)
        for _ in range(5):
            cycle(stages)
        self.assertIs(peer.entries["r"].req, req)
        self.assertTrue(all(not r.s.tree_cache.finalized for r in stages))
        peer.s.disagg_prefill_bootstrap_queue.queue.remove(req)
        peer.s.waiting_queue.append(req)  # Native bootstrap completed.
        cycle(stages)
        cycle(stages)
        self.assertEqual(cycle(stages)[1], [(("r", 896, 1025),)] * 3)

    def test_withheld_credits_are_passed_to_native_budget(self):
        stages = self.stages(caps={"a": 512, "b": 256, "c": 128})
        first, _ = cycle(stages)
        old, _ = tick(stages)  # a prepared, no return yet.
        self.assertGreater(stages[0].prepare_selection().reserved_tokens, 0)
        self.assertFalse(stages[0].prepare_selection().candidates)
        stages[0].consume_return(old)
        selection = stages[0].prepare_selection()
        self.assertEqual([r.rid for r in selection.candidates], ["a"])
        self.assertEqual(selection.reserved_tokens, 0)

    def test_gpu_capacity_and_chunk_runway_bound_new_credits(self):
        runtime = stage(0, 2, {"r": 512, "other": 0})
        runtime.s.token_to_kv_pool_allocator = Allocator(1800)
        runtime.s.chunked_req = NS(origin_input_ids=list(range(1025)), extend_end=512)
        # 1216 credit + 641 runway does not fit: no mutation or placeholder alloc.
        self.assertEqual(runtime._reserve_capacity(runtime.s.waiting_queue[0]), 0)
        runtime.s.chunked_req.extend_end = 768
        credit = runtime._reserve_capacity(runtime.s.waiting_queue[0])
        self.assertEqual(credit, 1216)
        runtime._watch_waiting()
        runtime.entries["r"].credit = credit
        self.assertEqual(runtime._reserve_capacity(runtime.s.waiting_queue[1]), 0)
        self.assertEqual(runtime.s.token_to_kv_pool_allocator.available_size(), 1800)

    def test_skew_delayed_returns_and_capacity_changes_make_progress(self):
        for pp in (2, 3, 4, 8, 16):
            with self.subTest(pp=pp):
                stages = [
                    stage(i, pp, {f"r{j}": 512 for j in range(12)}) for i in range(pp)
                ]
                returns, completed = [], []
                for beat in range(180):
                    for when, envelope in list(returns):
                        if when == beat:
                            settle(stages, envelope)
                            returns.remove((when, envelope))
                    for rank, runtime in enumerate(stages):
                        runtime.s.tree_cache.waiting = {
                            f"r{j}" for j in range(12) if beat < (j + rank) % 7
                        }
                    envelope, batches = tick(stages, limit=2)
                    self.assertTrue(all(b == batches[0] for b in batches))
                    completed.extend(row[0] for row in batches[0])
                    returns.append((beat + 2, envelope))
                    self.assertLessEqual(
                        sum(
                            e.phase != m.Phase.PENDING
                            for e in stages[0].entries.values()
                        ),
                        8,
                    )
                    if len(completed) == 12:
                        break
                self.assertCountEqual(completed, [f"r{i}" for i in range(12)])
                self.assert_released(stages)

    def test_credits_survive_real_pool_consumption_and_skewed_release(self):
        for pp in (2, 3, 4, 8):
            with self.subTest(pp=pp):
                stages = [
                    stage(i, pp, {f"r{j}": 512 for j in range(16)}) for i in range(pp)
                ]
                for runtime in stages:
                    runtime.s.allocate_native_batch = True
                    runtime.s.token_to_kv_pool_allocator = Allocator(4096)
                    runtime.s.req_to_token_pool = RowPool(2)
                retirements, completed = [], []
                for beat in range(160):
                    for when, runtime, tokens, rows in list(retirements):
                        if when <= beat:
                            runtime.s.token_to_kv_pool_allocator.free(tokens)
                            runtime.s.req_to_token_pool.free_rows(rows)
                            retirements.remove((when, runtime, tokens, rows))
                    waiting = [
                        {req.rid: req for req in r.s.waiting_queue} for r in stages
                    ]
                    _, layouts = cycle(stages, limit=2)
                    self.assertTrue(all(layout == layouts[0] for layout in layouts))
                    completed.extend(row[0] for row in layouts[0])
                    for runtime, requests, layout in zip(stages, waiting, layouts):
                        for rid, *_ in layout:
                            tokens, rows = requests[rid].test_allocation
                            retirements.append(
                                (beat + 3 + runtime.rank % 3, runtime, tokens, rows)
                            )
                        held = sum(e.credit for e in runtime.entries.values())
                        self.assertGreaterEqual(
                            runtime.s.token_to_kv_pool_allocator.available_size(), held
                        )
                        self.assertGreaterEqual(
                            runtime.s.req_to_token_pool.available_size(),
                            sum(e.credit > 0 for e in runtime.entries.values()),
                        )
                    if len(completed) == 16 and not retirements:
                        break
                self.assertCountEqual(completed, [f"r{j}" for j in range(16)])
                self.assert_released(stages)
                for runtime in stages:
                    self.assertEqual(
                        runtime.s.token_to_kv_pool_allocator.available_size(), 4096
                    )
                    self.assertEqual(runtime.s.req_to_token_pool.available_size(), 2)


class NativeChunkReplayTests(CustomTestCase):
    def bind_native_selection(self, gate, policy_name):
        # Exercise the actual Scheduler selection body and chunk/budget methods.
        # Only GPU batch allocation, request/cache objects and metrics are doubles.
        scope = dict(
            Enum=Enum,
            auto=auto,
            CacheAwarePolicy=NS(SHORTEST_PREFILL_FIRST="shortest-prefill-first"),
            _ceil_div=lambda a, b: -(-a // b),
            CLIP_MAX_NEW_TOKENS=4096,
        )
        policy_methods = {"shortest_prefill_chunk_limit", "_shortest_prefill_work"}
        functions(
            "managers/schedule_policy.py", policy_methods, scope, cls="SchedulePolicy"
        )
        Policy = type("Policy", (), {name: scope[name] for name in policy_methods})
        policy = Policy()
        policy.policy = policy_name
        policy.calc_priority = Mock()  # One fixed candidate; no sorting ambiguity.
        policy.shortest_prefill_chunk_limit = Mock(
            wraps=policy.shortest_prefill_chunk_limit
        )
        functions("managers/schedule_policy.py", {"AddReqResult"}, scope)
        functions("mem_cache/prefill_budget.py", {"PrefillBudget"}, scope)
        adder_methods = {
            "add_chunked_req",
            "add_one_req",
            "_update_prefill_budget",
            "_swa_new_tokens",
            "ceil_paged_tokens",
            "budget_state",
        }
        functions(
            "managers/schedule_policy.py", adder_methods, scope, cls="PrefillAdder"
        )
        Adder = type("Adder", (), {name: scope[name] for name in adder_methods})
        s = gate.s

        def make_adder(
            page,
            cache,
            allocator,
            running,
            ratio,
            input_budget,
            chunk_budget,
            *args,
            **kwargs,
        ):
            memory = scope["PrefillBudget"](
                NS(page_size=page, available_size=allocator.available_size),
                NS(supports_mamba=lambda: False, evictable_size=lambda: 0),
            )
            memory.total_offset += kwargs["reserved_tokens"]
            memory.current_offset += kwargs["reserved_tokens"]
            adder = Adder()
            vars(adder).update(
                dllm_config=None,
                memory_budget=memory,
                tree_cache=cache,
                rem_chunk_tokens=chunk_budget,
                rem_input_tokens=input_budget,
                page_size=page,
                per_req_token_overhead=page,
                kv_shard_granule=0,
                prefill_delayer_single_pass=None,
                chunked_req_limit=None,
                exact_chunk_fill=False,
                can_run_list=[],
                new_chunked_req=None,
                preempt_list=[],
                prefill_max_requests=kwargs["prefill_max_requests"],
                log_hit_tokens=0,
                log_input_tokens=0,
                log_replay_tokens=0,
                _kv_shard_reserve_scratch=lambda **kw: True,
                _mamba_gap_budget_for_req=lambda req, **kw: 0,
                _request_slot_reserve_for_req=lambda req, **kw: 0,
                rem_mamba_slots=None,
                _account_prefill_cache_admission=Mock(),
            )
            return adder

        class Chunk(RequestDouble):
            @property
            def extend_len(self):
                return self.extend_end - self.prefix_len

        vars(s).update(
            policy=policy,
            chunked_req=Chunk(
                rid="R",
                kv=NS(holds_kv=True),
                prefix_len=1024,
                extend_end=1024,
                output_ids=[],
                origin_input_ids=list(range(4096)),
                full_untruncated_fill_ids=list(range(4096)),
                sampling_params=NS(max_new_tokens=32),
                retracted_stain=False,
                token_indices_to_pool=None,
                inflight_middle_chunks=0,
                init_next_round_input=lambda: None,
            ),
            grammar_manager=NS(has_waiting_grammars=lambda: False),
            enable_priority_preemption=False,
            is_hybrid_swa=False,
            min_free_slots_delayer=None,
            running_batch=NS(reqs=[], batch_is_full=False, is_empty=lambda: True),
            processed_tokens_counter=0,
            get_num_allocatable_reqs=lambda *a, **kw: 64,
            get_prefill_adder_class=lambda: make_adder,
            chunked_prefill_size=1024,
            dynamic_chunk_sizer=None,
            tp_worker=NS(model_runner=NS(attn_backend=NS(), prefill_aware_swa=False)),
            new_token_ratio_tracker=NS(current=1),
            max_prefill_tokens=1024,
            is_mixed_chunk=False,
            priority_scheduling_preemption_threshold=0,
            max_prefill_bs=64,
            max_running_requests=64,
            dllm_config=None,
            enable_lora=False,
            disaggregation_mode="prefill",
            truncation_align_size=None,
            enable_hicache_storage=True,
            enable_hierarchical_cache=True,
            enable_overlap=False,
            model_config=NS(),
            spec_algorithm=None,
            enable_priority_scheduling=False,
            hisparse_coordinator=None,
            load_inquirer=NS(_get_num_pending_tokens=lambda **kw: 0),
        )
        s.tree_cache.buffer_pipeline = None
        s.tree_cache.storage_prefetch_retries = NS(cancel=Mock())
        s.tree_cache.ready_to_load_host_cache = lambda: 0
        for req in s.waiting_queue:
            req.output_ids = []
            req.num_matched_prefix_tokens = 512
            req.token_indices_to_pool = None
        scope.update(
            TEST_RETRACT=False,
            get_schedule=lambda: NS(prefill_max_requests=1),
            DisaggregationMode=NS(PREFILL="prefill"),
            set_time_batch=Mock(),
            ScheduleBatch=NS(
                init_new=lambda reqs, *a, **kw: NS(reqs=reqs, prepare_for_extend=Mock())
            ),
            PrefillStats=NS(from_adder=lambda *a, **kw: None),
        )
        raw = method(
            "managers/scheduler.py", "Scheduler", "_get_new_batch_prefill_raw", scope
        )

        def select(running_batch, *, admission=None):
            batch, running = raw(s, None, running_batch, admission)
            return NS(batch_to_run=batch, running_batch=running)

        s.get_new_batch_prefill = select

    def test_peers_replay_chunk_end_without_the_unselected_queue_context(self):
        for policy_name, expected_end in (
            ("shortest-prefill-first", 1472),
            ("fcfs", 2048),
        ):
            with self.subTest(policy=policy_name):
                stages = [stage(rank, 3, {"Q": 512}) for rank in range(3)]
                for _ in range(3):
                    cycle(stages, limit=0)
                for gate in stages:
                    self.bind_native_selection(gate, policy_name)
                update = None
                for gate in stages:
                    if update is not None:
                        gate.receive_forward(update)
                    candidates = gate.prepare_selection().candidates
                    self.assertEqual(
                        [req.rid for req in candidates], ["Q"] if gate.rank == 0 else []
                    )
                    plan, update = gate.select_batch(gate.s.running_batch)
                    self.assertEqual(
                        m.batch_layout(plan.batch_to_run), (("R", 1024, expected_end),)
                    )
                    self.assertEqual(update.layout, (("R", 1024, expected_end),))
                    self.assertEqual([req.rid for req in gate.s.waiting_queue], ["Q"])
                    self.assertEqual(gate.entries["Q"].phase, m.Phase.CERTIFIED)
                    self.assertEqual(
                        gate.s.policy.shortest_prefill_chunk_limit.call_count,
                        int(gate.rank == 0),
                    )
                    plan.batch_to_run.prepare_for_extend.assert_called_once()


class NativeBoundaryTests(CustomTestCase):
    def test_pending_gate_keeps_idle_safety_without_skipping_housekeeping(self):
        runtime = stage(0, 3, {"r": 512})
        s = runtime.s
        runtime._watch_waiting()
        mode = NS(PREFILL="prefill", DECODE="decode", NULL="null")
        clock = NS(monotonic=lambda: 100.0, monotonic_ns=lambda: 1000, sleep=Mock())
        native_idle = method(
            "managers/scheduler.py",
            "Scheduler",
            "is_fully_idle",
            {
                "DisaggregationMode": mode,
                "get_memory": lambda: NS(hicache_host_memory_mode="cache"),
            },
        )
        housekeeping = method(
            "managers/scheduler.py",
            "Scheduler",
            "on_idle",
            {
                "scheduler_stage_method": lambda stage: lambda fn: fn,
                "SCHEDULER_STAGE_IDLE": "idle",
                "LOAD_STALL_REFRESH_S": 1,
                "DisaggregationMode": mode,
                "time": clock,
            },
        )
        vars(s).update(
            running_batch=NS(is_empty=lambda: True),
            last_batch=None,
            dllm_manager=NS(any_staging_reqs=lambda: False),
            enable_overlap=False,
            _pp_microbatches_drained=lambda: True,
            _engine_paused=True,
            disaggregation_mode=mode.PREFILL,
            disagg_prefill_inflight_queue=[],
            grammar_manager=NS(grammar_queue=[]),
            enable_hisparse=False,
            enable_hierarchical_cache=True,
            enable_lmcache=False,
            enable_hicache_storage=True,
            dp_attn_adapter=NS(drop_sync_wait_carry=Mock()),
            maybe_send_health_check_signal=Mock(),
            metrics_reporter=NS(record_scheduler_active=Mock()),
            _last_stall_publish_ts=0,
            publish_load_snapshot=Mock(return_value="snapshot"),
            load_publisher=NS(publish_load_stat=Mock()),
            load_inquirer=NS(get_loads=Mock()),
        )
        vars(s.tree_cache).update(
            ongoing_write_through={},
            ongoing_load_back={},
            ongoing_backup={},
            enable_storage=True,
        )
        s.is_fully_idle = lambda **kw: native_idle(s, **kw)
        self.assertTrue(runtime.has_pending_work())
        self.assertFalse(s.is_fully_idle())
        self.assertFalse(s.is_fully_idle(ignore_waiting=True))
        housekeeping(s)
        s.metrics_reporter.record_scheduler_active.assert_called_once_with(1000)
        s.publish_load_snapshot.assert_called_once_with(force=True)
        s.load_publisher.publish_load_stat.assert_called_once_with(
            s.load_inquirer.get_loads, force=True, snapshot="snapshot"
        )
        clock.sleep.assert_called_once_with(0)
        runtime.release(s.waiting_queue[0])
        self.assertFalse(runtime.has_pending_work())
        self.assertFalse(s.is_fully_idle())  # Native waiting-queue guard still applies.
        self.assertTrue(s.is_fully_idle(ignore_waiting=True))
        s.waiting_queue.clear()
        self.assertTrue(s.is_fully_idle())

    def test_abort_dispatch_is_native_and_gate_cleanup_is_in_retirement(self):
        abort = method(
            "managers/scheduler.py",
            "Scheduler",
            "abort_request",
            {
                "DisaggregationMode": NS(PREFILL="prefill", DECODE="decode"),
                "get_parallel": lambda: NS(pp_size=3),
                "prepare_abort": Mock(),
                "_make_abort_req": lambda req, **kw: req,
                "FINISH_ABORT": lambda: "aborted",
                "logger": logging.getLogger(__name__),
            },
        )
        retire = method(
            "managers/scheduler.py",
            "Scheduler",
            "_release_aborted_request",
            {"CacheRequestOutcome": NS(ABORT="abort")},
        )
        stages = [stage(i, 3, {"r": 512, "keep": 128}) for i in range(3)]
        cycle(stages)
        old, _ = tick(stages)
        for runtime in stages:
            s = runtime.s
            s.tree_cache.finish = Mock()
            s._release_aborted_request = lambda req, s=s: retire(s, req)
            s.chunked_req = None
            s.mm_receiver = None
            s.dllm_config = None
            s.disaggregation_mode = "prefill"
            s.disagg_prefill_inflight_queue = []
            s.grammar_manager = NS(abort_requests=Mock())
            s.beam_coordinator = NS(retire_group=Mock())
            s.ipc_channels = NS(send_to_tokenizer=NS(send_output=Mock()))
            s.release_aborted_prefill_waiting_req = Mock()
            s.collect_inflight_reqs = lambda: []
            abort(
                s,
                NS(rid="r", abort_all=False, finished_reason=None, abort_message=None),
            )
            self.assertEqual([r.rid for r in s.waiting_queue], ["keep"])
            self.assertNotIn("r", runtime.entries)
            s.tree_cache.finish.assert_called_once()
            s.release_aborted_prefill_waiting_req.assert_called_once()
            runtime.consume_return(old)
            self.assertNotIn("r", runtime.entries)
        cycle(stages)
        _, batches = cycle(stages)
        self.assertEqual(batches, [(("keep", 128, 1025),)] * 3)

    def test_real_native_row_gate_sees_credit_as_available(self):
        runtime = stage(0, 3, {"r": 512})
        runtime.s.req_to_token_pool = RowPool(1)
        runtime.s.running_batch = NS(reqs=[])
        runtime.s.beam_coordinator = NS(pending_member_rows=lambda batch: 0)
        slots = method(
            "managers/scheduler.py",
            "Scheduler",
            "get_num_allocatable_reqs",
            {"get_parallel": lambda: NS(pp_max_micro_batch_size=8)},
        )
        runtime._watch_waiting()
        entry = runtime.entries["r"]
        entry.ready_mask = 7
        runtime._prepare([m.Prepare(entry.key, 1, 1024)])
        self.assertEqual(slots(runtime.s, 0), 1)
        self.assertEqual(entry.credit, 1216)

    def test_real_budget_excludes_withheld_tokens_without_touching_pool(self):
        scope = functions("mem_cache/prefill_budget.py", {"PrefillBudget"}, {})
        allocator = NS(page_size=64, available_size=lambda: 4096)
        cache = NS(supports_mamba=lambda: False, evictable_size=lambda: 0)
        budget = scope["PrefillBudget"](allocator, cache)
        budget.total_offset += 1216
        budget.current_offset += 1216
        self.assertEqual(budget.remaining_current, 2880)
        self.assertEqual(budget.remaining_total, 2880)
        self.assertEqual(allocator.available_size(), 4096)


if __name__ == "__main__":
    unittest.main()
