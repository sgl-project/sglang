"""Focused regressions for the additional PP gate, using native queue ownership."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from sglang.srt.managers import pp_prefetch_scheduler as m
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.pp_cache_prefix import CachePrefixLease
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
from sglang.srt.utils.poll_timer import PollTimer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import published_topology

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def stage(rank, cap, *, rows=8):
    req = Mock(rid="r", origin_input_ids=list(range(1025)))
    req.cache_request_handle = CacheRequestHandle("r", 0)
    req.sampling_params.max_new_tokens = 1
    cache = Mock(page_size=64)
    cache.prefetch_ready_to_finalize.return_value = [True]
    cache.get_admission_storage_span.return_value = None

    def match(tree, *, max_cache_prefix_tokens, **kw):
        req.prefix_len = min(cap, max_cache_prefix_tokens)
        req.host_hit_length = 0
        req.last_node = req.best_match_node = req.prefix_len

    req.init_next_round_input = match
    s = NS(
        waiting_queue=[req],
        disagg_prefill_bootstrap_queue=NS(queue=[]),
        tree_cache=cache,
        page_size=64,
        chunked_req=None,
        req_to_token_pool=NS(available_size=lambda: rows),
        token_to_kv_pool_allocator=NS(available_size=lambda: 65536),
    )
    return m.PPAdmissionGate(s, parallel=NS(pp_rank=rank, pp_size=3))


def beat(stages):
    update = None
    for gate in stages:
        if update is not None:
            gate.receive_forward(update)
        update = gate.finish_selection(None)
    stages[0].consume_return(update)
    return update


class AdmissionTests(unittest.TestCase):
    def stages(self):
        return [stage(i, cap, rows=1) for i, cap in enumerate((1024, 960, 1024))]

    def test_peer_replays_pp0_chunk_limit_without_its_waiting_candidates(self):
        class LimitObserved(Exception):
            pass

        for layout, candidates in ((None, [object()]), ((("R", 1024, 1472),), [])):
            with self.subTest(layout=layout), published_topology(role="scheduler"):
                adder = Mock(rem_chunk_tokens=1024)
                adder.add_chunked_req.side_effect = LimitObserved
                s = Mock(
                    waiting_queue=[],
                    enable_priority_preemption=False,
                    is_hybrid_swa=False,
                    dynamic_chunk_sizer=None,
                    min_free_slots_delayer=None,
                    is_mixed_chunk=False,
                    max_prefill_bs=8,
                    chunked_req=Mock(rid="R"),
                )
                s.grammar_manager.has_waiting_grammars.return_value = False
                s.get_num_allocatable_reqs.return_value = 8
                s.get_prefill_adder_class.return_value = lambda *a, **kw: adder
                s.policy.shortest_prefill_chunk_limit.return_value = 448
                selection = NS(
                    candidates=candidates, reserved_tokens=0, expected_layout=layout
                )
                with self.assertRaises(LimitObserved):
                    Scheduler._get_new_batch_prefill_raw(
                        s, None, Mock(reqs=[], batch_is_full=False), selection
                    )
                self.assertEqual(adder.chunked_req_limit, 448)
                self.assertEqual(
                    s.policy.shortest_prefill_chunk_limit.call_count,
                    int(layout is None),
                )

    def test_readiness_then_full_prefix_min_and_native_handoff(self):
        gates = self.stages()
        gates[2].s.tree_cache.prefetch_ready_to_finalize.return_value = [False]
        self.assertEqual(beat(gates).ready[0][1], 3)
        self.assertFalse(beat(gates).prepare)
        for g in gates:
            g.s.tree_cache.finalize_prefetch.assert_not_called()
            self.assertEqual(g.entries["r"].credit, 0)
        gates[2].s.tree_cache.prefetch_ready_to_finalize.return_value = [True]
        beat(gates)
        self.assertEqual(beat(gates).prepare[0].cap, 960)
        beat(gates)
        for g in gates:
            self.assertEqual(g.entries["r"].lease.cap, 960)
            self.assertEqual(g.entries["r"].phase, m.Phase.CERTIFIED)
            queue = g.s.waiting_queue
            req = queue.pop()  # Native selection removes its selected request.
            selection = m.PrefillAdmissionSelection({req.rid: g.entries[req.rid]})
            self.assertEqual(selection.prepare_request(req, g.s.tree_cache), 960)
            self.assertEqual(req.prefix_len, 960)
            g.finish_selection(NS(reqs=[req]))
            self.assertNotIn("r", g.entries)
            self.assertIs(g.s.waiting_queue, queue)
            self.assertFalse(queue)

    def test_capacity_nack_and_cancel_release_only_gate_resources(self):
        gates = self.stages()
        gates[2].s.req_to_token_pool.available_size = lambda: 0
        beat(gates)
        self.assertIsNone(beat(gates).prepare[0].cap)
        beat(gates)
        for g in gates:
            req = g.s.waiting_queue[0]
            g.release(req)
            self.assertFalse(g.entries)
            self.assertEqual(g.s.waiting_queue, [req])
        gates = self.stages()
        beat(gates)
        beat(gates)
        old = gates[0].entries["r"]
        gates[0].release(old.req)
        self.assertFalse(gates[0]._commit(m.Prepare(old.key, old.generation, 960)))

    def test_device_only_lease_does_not_touch_backup_host_reference(self):
        gate = stage(0, 1024)
        req, cache = gate.s.waiting_queue[0], gate.s.tree_cache
        hold = CachePrefixLease.acquire(cache, req, 1024)
        cache.inc_host_lock_ref.assert_not_called()
        hold = hold.restrict(960)
        self.assertEqual(hold.cap, 960)
        hold.release()
        hold.release()
        cache.dec_host_lock_ref.assert_not_called()
        self.assertEqual(cache.unlock.call_count, 2)

    def test_wait_timer_stops_at_local_readiness_once(self):
        g = stage(0, 1024)
        stats = SchedulerReqTimeStats(
            storage_prefetch_wait_s=0.0, _storage_prefetch_wait=PollTimer()
        )
        g.s.waiting_queue[0].time_stats = stats
        g._watch_waiting()
        with patch("time.perf_counter", return_value=10.0) as clock:
            g.s.tree_cache.prefetch_ready_to_finalize.return_value = [False]
            g._report_readiness()
            clock.return_value = 13.0
            g.s.tree_cache.prefetch_ready_to_finalize.return_value = [True]
            g._report_readiness()
            clock.return_value = 20.0
            g._report_readiness()
        self.assertEqual(stats.storage_prefetch_wait_s, 3.0)
        self.assertNotEqual(g.entries["r"].ready_mask, g.all_ready)


if __name__ == "__main__":
    unittest.main()
