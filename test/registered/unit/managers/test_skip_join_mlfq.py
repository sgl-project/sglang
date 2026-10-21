"""Unit tests for managers/skip_join_mlfq.py: placement, demotion, promotion,
re-pricing, the prefill gate and the online cost model, with no server."""

import time
import unittest
from itertools import count

from sglang.srt.managers.skip_join_mlfq import (
    STARVATION_CHECK_INTERVAL,
    STARVATION_S,
    OnlinePrefillCostModel,
    SkipJoinMlfq,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_rids = count()


class _Req:
    """The request fields the policy reads and writes."""

    def __init__(self, prompt_len: int, cached: int = 0):
        self.rid = f"r{next(_rids)}"
        self.origin_input_ids = [0] * prompt_len
        self.num_matched_prefix_tokens = cached
        self.priority = None
        self.mlfq_level = None
        self.mlfq_runs = 0
        self.mlfq_service_s = 0.0
        self.mlfq_level_entry_time = 0.0
        self.mlfq_last_service_time = 0.0


def _policy(cost_mode: str = "cache-aware") -> SkipJoinMlfq:
    return SkipJoinMlfq(cost_mode=cost_mode)


class TestPlacement(CustomTestCase):
    def test_cache_aware_prices_the_uncached_suffix(self):
        # A returning turn: 5,000 tokens of which 4,500 are cached. Seed cost:
        # 0.36 s for the whole prompt (level 6), 0.045 s for the suffix (level 3).
        prompt, aware = _Req(5000, cached=4500), _Req(5000, cached=4500)
        _policy("prompt").order([prompt])
        _policy("cache-aware").order([aware])
        self.assertEqual((prompt.mlfq_level, aware.mlfq_level), (6, 3))
        # The level is the request priority, higher values first.
        self.assertEqual((prompt.priority, aware.priority), (-6, -3))

    def test_level_is_the_log2_of_cost_over_quantum(self):
        # 60,000 uncached tokens predict 4.3 s: level 9 at a 10 ms base quantum,
        # level 7 at 40 ms. A prompt under the quantum (138 tokens, 9.9 ms) is level 0.
        fresh, short = _Req(60000), _Req(138)
        _policy().order([fresh, short])
        self.assertEqual((fresh.mlfq_level, short.mlfq_level), (9, 0))
        wide = _Req(60000)
        SkipJoinMlfq(cost_mode="cache-aware", base_quantum_s=0.040).order([wide])
        self.assertEqual(wide.mlfq_level, 7)

    def test_order_is_level_then_level_entry_time(self):
        q = _policy("prompt")
        long_first, short_old, short_new = _Req(8000), _Req(256), _Req(256)
        queue = [long_first, short_old]
        q.order(queue)
        q.charge([], 0.0, now=0.5)  # the clock advances only through charged batches
        queue.append(short_new)
        q.order(queue)
        self.assertEqual(queue, [short_old, short_new, long_first])

    def test_in_batch_prefix_siblings_go_to_the_back_for_one_ordering(self):
        q = _policy()
        first, second, other = _Req(8000), _Req(8000), _Req(256)
        queue = [first, second, other]
        # The in-batch prefix set marks the second sibling: the first prefills
        # their shared prefix, the second waits one round instead of duplicating it.
        q.order(queue, temporary_deprioritized={second.rid})
        self.assertEqual(queue, [other, first, second])
        q.order(queue)  # no longer marked: level order, entry time breaks the tie
        self.assertEqual(queue, [other, first, second])
        self.assertEqual(first.mlfq_level, second.mlfq_level)


class TestServiceAndClock(CustomTestCase):
    def test_demotion_after_the_level_quantum(self):
        q = _policy()
        req = _Req(5000, cached=4500)
        q.order([req])
        self.assertEqual(req.mlfq_level, 3)  # 80 ms quantum
        q.charge([req], 0.050, now=1.0)
        self.assertEqual(req.mlfq_level, 3)
        self.assertAlmostEqual(req.mlfq_service_s, 0.050)
        q.charge([req], 0.040, now=1.1)
        self.assertEqual(
            (req.mlfq_level, req.priority, req.mlfq_service_s), (4, -4, 0.0)
        )

    def test_state_follows_the_charged_clock_not_wall_time(self):
        # Every TP rank must reach the same levels from the same charged
        # (elapsed, now) sequence, however much real time passes between calls.
        def run(sleep_s):
            q = _policy("prompt")
            long_req, short_req = _Req(8000), _Req(256)
            queue = [long_req, short_req]
            q.order(queue)
            for i in range(5):
                time.sleep(sleep_s)
                q.charge([short_req], 0.040, now=0.1 * (i + 1))
                q.order(queue)
            return [
                (r.mlfq_level, r.mlfq_level_entry_time, r.mlfq_last_service_time)
                for r in queue
            ]

        self.assertEqual(run(0.0), run(0.002))

    def test_starvation_promotion_follows_the_charged_clock(self):
        # A request last served at clock 0 is promoted to level 0 once a charged
        # batch moves the clock past STARVATION_S, at the next check, not before.
        for now, promoted in [(STARVATION_S - 1, False), (STARVATION_S + 0.5, True)]:
            with self.subTest(now=now):
                q = _policy("prompt")
                starved = _Req(8000)
                q.order([starved])
                self.assertGreater(starved.mlfq_level, 0)
                q.charge([], 0.0, now=now)
                for _ in range(STARVATION_CHECK_INTERVAL):
                    q.order([starved])
                self.assertEqual(starved.mlfq_level == 0, promoted)
                self.assertEqual(starved.priority == 0, promoted)

    def test_waiting_request_is_repriced_only_down_and_only_before_it_runs(self):
        q = _policy()
        sibling = _Req(8000)  # placed before its shared prefix is cached
        q.order([sibling])
        placed, entry = sibling.mlfq_level, sibling.mlfq_level_entry_time
        self.assertGreater(placed, 3)
        # Another request prefilled the shared 7.5k prefix: the next ordering re-prices it.
        sibling.num_matched_prefix_tokens = 7500
        q.order([sibling])
        self.assertLess(sibling.mlfq_level, placed)
        self.assertEqual(sibling.priority, -sibling.mlfq_level)
        self.assertEqual(sibling.mlfq_level_entry_time, entry)
        repriced = sibling.mlfq_level
        # The cache no longer holds it: the level does not go back up.
        sibling.num_matched_prefix_tokens = 0
        q.order([sibling])
        self.assertEqual(sibling.mlfq_level, repriced)
        # Once it has run, charging owns the level: a full cache hit (a retracted
        # request resuming from kept KV) no longer re-prices it.
        q.charge([sibling], 0.040, now=1.0)
        sibling.num_matched_prefix_tokens = 8000
        q.order([sibling])
        self.assertEqual(sibling.mlfq_level, repriced)
        # A starvation promotion to level 0 is never undone by re-pricing.
        waiting = _Req(8000)
        q.order([waiting])
        q.charge([], 0.0, now=1.0 + STARVATION_S + 1)
        for _ in range(STARVATION_CHECK_INTERVAL):
            q.order([waiting])
        self.assertEqual(waiting.mlfq_level, 0)


class TestPrefillGate(CustomTestCase):
    def test_gate_level_is_the_most_urgent_running_level(self):
        def at(level):
            req = _Req(1)
            req.mlfq_level = level
            return req

        gate = SkipJoinMlfq.prefill_gate_level
        self.assertEqual(gate([at(2), at(7)]), 2)  # a level-2 request is mid-answer
        self.assertEqual(gate([at(None), at(4)]), 4)  # an unplaced request has no say
        self.assertIsNone(gate([]))  # nothing running: prefill whatever waits


class TestOnlineCostModel(CustomTestCase):
    def test_starts_from_the_seed_and_learns_the_scale(self):
        model = OnlinePrefillCostModel()
        # Seed: about 0.6 s for an 8k chunk with no prefix, 1.3 s on a 32k prefix.
        self.assertTrue(0.5 < model.predict_s(uncached=8192, cached=0) < 0.7)
        self.assertTrue(1.2 < model.predict_s(uncached=8192, cached=32768) < 1.5)

        # A GPU twice as slow: batches with doubled times; the fit must follow.
        def truth(rows):
            seed = OnlinePrefillCostModel()
            return 2 * sum(seed.predict_s(uncached=u, cached=c) for u, c in rows)

        batches = [
            [(8192, 0)],
            [(2048, 32768), (1024, 0)],
            [(8192, 98304)],
            [(512, 4096)] * 4,
            [(4096, 16384)],
            [(1, 20000)] * 8 + [(2048, 0)],
        ]
        for _ in range(5):
            for rows in batches:
                model.observe(rows, truth(rows))
        for u, c in [(8192, 0), (2048, 32768), (512, 98304)]:
            want = truth([(u, c)])
            self.assertLess(
                abs(model.predict_s(uncached=u, cached=c) - want), 0.15 * want
            )
        model.observe([(8192, 0)], 500.0)  # an implausible time is ignored
        self.assertEqual(model.observations, 30)


if __name__ == "__main__":
    unittest.main()
