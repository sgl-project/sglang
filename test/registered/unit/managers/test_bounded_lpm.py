import argparse
import unittest
from unittest.mock import patch

from sglang.srt.managers.schedule_policy import CacheAgnosticPolicy, SchedulePolicy
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
    move,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import FakeReq
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def policy(name="lpm", disabled=False, enabled=True):
    cache = RadixCache.create_simulated()
    cache.disable = disabled
    p = SchedulePolicy(name, cache, False, False, False, bounded_lpm_enabled=enabled)
    p._should_start_bounded_batch = lambda waiting: True
    return p


def requests(n):
    return [FakeReq(rid=str(i), num_matched_prefix_tokens=i % 3) for i in range(n)]


class TestBoundedLPM(CustomTestCase):
    def setUp(self):
        reset_context()
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)
        patcher = patch("sglang.srt.managers.schedule_policy.touch_waiting_prefix")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_threshold_and_stable_sort(self):
        for n in (0, 1, 63, 64, 65, 127, 128, 129, 512):
            with self.subTest(n=n):
                p, q = policy(), requests(n)
                with patch.object(
                    p, "_compute_prefix_matches", return_value=set()
                ) as match:
                    result = p.calc_priority(q)
                k = 64 if n > 128 else n
                self.assertEqual(
                    result,
                    sorted(q[:k], key=lambda r: -r.num_matched_prefix_tokens) + q[k:],
                )
                self.assertEqual(p._bounded_active, n > 128)
                self.assertEqual(p.waiting_queue_prefix_matched(result), n <= 128)
                self.assertTrue(all(p.allows_admission(r) for r in result[:k]))
                self.assertTrue(all(not p.allows_admission(r) for r in result[k:]))
                self.assertEqual(len(match.call_args.args[0]), k)

    def test_drain_below_threshold_then_restore_short_lpm(self):
        p, q = policy(), requests(150)
        with patch.object(p, "_compute_prefix_matches", return_value=set()) as match:
            p.calc_priority(q)
            p.retire_waiting_members(q[:30])
            p.calc_priority(q[30:])  # 120 requests, 34 committed members
            self.assertEqual(
                set(map(id, match.call_args.args[0])), set(map(id, q[30:64]))
            )
            self.assertFalse(p.allows_admission(q[64]))
            p.retire_waiting_members(q[30:64])
            p.calc_priority(q[64:])
            self.assertFalse(p._bounded_active)
            self.assertEqual(len(match.call_args.args[0]), 86)
            self.assertTrue(p.waiting_queue_prefix_matched(q[64:]))

    def test_short_long_short_long_transitions(self):
        p, q = policy(), requests(200)
        with patch.object(p, "_compute_prefix_matches", return_value=set()):
            p.calc_priority(q[:128])
            self.assertFalse(p._bounded_active)
            p.calc_priority(q[:129])
            self.assertTrue(p._bounded_active)
            p.calc_priority([])
            self.assertFalse(p._bounded_active)
            p.calc_priority(q)
            self.assertTrue(p._bounded_active)
            self.assertEqual(len(p._bounded_members), 64)

    def test_member_rematches_and_newcomer_does_not_replace_it(self):
        p, q = policy(), requests(150)
        with patch.object(p, "_compute_prefix_matches", return_value=set()):
            p.calc_priority(q)
            q[0].num_matched_prefix_tokens = 100
            result = p.calc_priority(q[:2] + q[64:])
            self.assertIs(result[0], q[0])
            self.assertEqual(len(p._bounded_members), 2)
            self.assertFalse(p.allows_admission(q[64]))

    def test_departure_before_retraction_without_priority_call(self):
        for operation in ("move", "bulk_move"):
            with self.subTest(operation=operation):
                p, q = policy(), requests(150)
                waiting = IndexedReqList(
                    ListKind.WAITING, on_leave=p.retire_waiting_members
                )
                running = IndexedReqList(ListKind.DECODING)
                move(q, waiting)
                with patch.object(p, "_compute_prefix_matches", return_value=set()):
                    waiting.reorder(p.calc_priority(waiting))
                    a = waiting[0]
                    if operation == "move":
                        move([a], running)
                    else:
                        move(list(waiting), running)
                    move([a], waiting)
                    self.assertFalse(p.allows_admission(a))

    def test_scheduler_installs_departure_hook(self):
        from types import SimpleNamespace

        from sglang.srt.managers.scheduler import Scheduler

        s = Scheduler.__new__(Scheduler)
        s.schedule_policy = "lpm"
        s.policy = policy()
        s.tree_cache = s.policy.tree_cache
        with patch(
            "sglang.srt.managers.scheduler.get_parallel",
            return_value=SimpleNamespace(tp_size=1),
        ):
            s.init_running_status()
        s.enable_hierarchical_cache = False
        s.enable_priority_scheduling = False
        s.schedule_low_priority_values_first = False
        s.dllm_config = None
        s.init_schedule_policy()
        s.policy._should_start_bounded_batch = lambda waiting: True
        self.assertIsNone(s.waiting_queue._on_leave)
        q = requests(150)
        move(q, s.waiting_queue)
        with patch.object(s.policy, "_compute_prefix_matches", return_value=set()):
            s.policy.calc_priority(s.waiting_queue)
        running = IndexedReqList(ListKind.DECODING)
        move([q[0]], running)
        move([q[0]], s.waiting_queue)
        self.assertFalse(s.policy.allows_admission(q[0]))
        move(list(s.waiting_queue), running)
        s.policy.calc_priority(s.waiting_queue)
        self.assertIsNone(s.waiting_queue._on_leave)

    def test_dllm_legacy_fallback_and_other_policies(self):
        q = requests(200)
        for p in (
            policy(enabled=False),
            policy("hrrn"),
            policy("fcfs"),
            policy(disabled=True),
        ):
            with patch.object(p, "_compute_prefix_matches") as match:
                self.assertEqual(p.calc_priority(q), q)
                match.assert_not_called()
            self.assertFalse(p._bounded_active)
            self.assertEqual(list(p.admission_candidates(q)), q)
            self.assertTrue(all(p.allows_admission(r) for r in q))
            self.assertEqual(p._determine_active_policy(q), CacheAgnosticPolicy.FCFS)

    def test_cli_removes_separate_policy_and_flag(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = ServerArgs.from_cli_args(
            parser.parse_args(["--model", "dummy", "--schedule-policy", "lpm"])
        )
        self.assertEqual(args.schedule_policy, "lpm")
        self.assertFalse(hasattr(args, "lpm_window_size"))
        with self.assertRaises(SystemExit):
            parser.parse_args(["--model", "dummy", "--schedule-policy", "bounded-lpm"])

    def test_matching_failure_propagates(self):
        p = policy()
        with patch.object(
            p, "_compute_prefix_matches", side_effect=RuntimeError("expected")
        ):
            with self.assertRaisesRegex(RuntimeError, "expected"):
                p.calc_priority(requests(129))


if __name__ == "__main__":
    unittest.main()
