import argparse
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    SchedulePolicy,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def policy(k=0, name="lpm", disabled=False, priority=False):
    cache = RadixCache.create_simulated()
    cache.disable = disabled
    return SchedulePolicy(name, cache, False, priority, False, lpm_window_size=k)


class TestBoundedLPM(CustomTestCase):
    def test_window_boundaries_and_stable_ties(self):
        for k in (1, 16, 32, 64, 128):
            for n in sorted({0, 1, k - 1, k, k + 1, 128, 129, 512}):
                with self.subTest(k=k, n=n):
                    p = policy(k)
                    q = [
                        SimpleNamespace(rid=str(i), num_matched_prefix_tokens=i % 3)
                        for i in range(n)
                    ]
                    before = q.copy()
                    with patch.object(
                        p, "_compute_prefix_matches", return_value=set()
                    ) as match:
                        q = p.calc_priority(q)
                    self.assertEqual(
                        q[:k],
                        sorted(before[:k], key=lambda r: -r.num_matched_prefix_tokens),
                    )
                    self.assertEqual(q[k:], before[k:])
                    self.assertEqual(sorted(map(id, q)), sorted(map(id, before)))
                    self.assertEqual(match.call_args.args[0], q[:k])
                    self.assertFalse(p.waiting_queue_prefix_matched(q))

    def test_input_iterable_is_not_reordered(self):
        p = policy(2)
        original = [
            SimpleNamespace(rid=str(i), num_matched_prefix_tokens=i) for i in range(4)
        ]
        with patch.object(p, "_compute_prefix_matches", return_value=set()):
            result = p.calc_priority(iter(original))
        self.assertEqual([r.rid for r in original], ["0", "1", "2", "3"])
        self.assertEqual([r.rid for r in result], ["1", "0", "2", "3"])

    def test_default_fallback_and_other_policy(self):
        p = policy()
        self.assertEqual(p._determine_active_policy([None] * 128), CacheAwarePolicy.LPM)
        self.assertEqual(
            p._determine_active_policy([None] * 129), CacheAgnosticPolicy.FCFS
        )
        self.assertEqual(
            policy(name="hrrn")._determine_active_policy([None] * 129),
            CacheAgnosticPolicy.FCFS,
        )

    def test_deprioritized_remains_inside_window(self):
        p = policy(2)
        q = [
            SimpleNamespace(rid=str(i), num_matched_prefix_tokens=100 - i)
            for i in range(4)
        ]
        with patch.object(p, "_compute_prefix_matches", return_value={"0"}):
            q = p.calc_priority(q)
        self.assertEqual([r.rid for r in q], ["1", "0", "2", "3"])

    def test_disabled_cache_delegates_fcfs(self):
        p = policy(32, disabled=True)
        self.assertEqual(p.policy, CacheAgnosticPolicy.FCFS)
        q = [SimpleNamespace(rid="a"), SimpleNamespace(rid="b")]
        with (
            patch.object(p, "_compute_prefix_matches") as match,
            patch.object(p, "_touch_waiting_prefixes") as touch,
        ):
            q = p.calc_priority(q)
            match.assert_not_called()
            touch.assert_called_once()
        self.assertEqual([r.rid for r in q], ["a", "b"])

    def test_validation(self):
        for k, name, priority in [
            (-1, "lpm", False),
            (32, "fcfs", False),
            (32, "hrrn", False),
            (32, "lpm", True),
        ]:
            with (
                self.subTest(k=k, name=name, priority=priority),
                self.assertRaises(ValueError),
            ):
                policy(k, name, priority=priority)

    def test_server_args_reject_invalid_windows(self):
        for kwargs, message in [
            ({"lpm_window_size": -1}, "must be non-negative"),
            ({"lpm_window_size": 32, "schedule_policy": "fcfs"}, "requires"),
            (
                {"lpm_window_size": 32, "enable_priority_scheduling": True},
                "does not support priority scheduling",
            ),
        ]:
            with self.subTest(kwargs=kwargs):
                values = {"model_path": "dummy", "schedule_policy": "lpm", **kwargs}
                with self.assertRaisesRegex(ValueError, message):
                    ServerArgs(**values).check_server_args()

    def test_cli(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        a = ServerArgs.from_cli_args(parser.parse_args(["--model", "dummy"]))
        self.assertEqual(a.lpm_window_size, 0)
        a = ServerArgs.from_cli_args(
            parser.parse_args(
                [
                    "--model",
                    "dummy",
                    "--schedule-policy",
                    "lpm",
                    "--lpm-window-size",
                    "64",
                ]
            )
        )
        self.assertEqual(a.lpm_window_size, 64)

    def test_real_in_batch_matching_and_tail(self):
        def req(i, tokens):
            return Req(str(i), "", array("q", tokens), SamplingParams())

        p = policy(2)
        a = req(0, list(range(100)))
        b = req(1, list(range(100)))
        c = req(2, list(range(200, 300)))
        p.calc_priority([a, b, c])
        self.assertFalse(p.waiting_queue_prefix_matched([a, b, c]))
        # A fully covered window follows ordinary LPM, including duplicate-prefix deferral.
        qa = [
            req(i, list(range(100)) if i < 2 else list(range(200, 300)))
            for i in range(3)
        ]
        qb = [
            req(i, list(range(100)) if i < 2 else list(range(200, 300)))
            for i in range(3)
        ]
        qa = policy(64).calc_priority(qa)
        qb = policy(0).calc_priority(qb)
        self.assertEqual([r.rid for r in qa], [r.rid for r in qb])

    def test_fast_match_tail_is_not_reported_as_matched(self):
        p = policy(32)
        with patch.object(
            p.tree_cache, "supports_fast_match_prefix", return_value=True
        ):
            self.assertFalse(p.waiting_queue_prefix_matched([None] * 129))

    def test_matching_exception_propagates(self):
        p = policy(32)
        with patch.object(
            p, "_compute_prefix_matches", side_effect=RuntimeError("expected")
        ):
            with self.assertRaisesRegex(RuntimeError, "expected"):
                p.calc_priority([])


if __name__ == "__main__":
    unittest.main()
