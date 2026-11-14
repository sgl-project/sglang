import unittest
from array import array
from unittest.mock import patch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    SchedulePolicy,
)
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_req(rid, origin_input_text, origin_input_ids, sampling_params=None, **kwargs):
    if sampling_params is None:
        sampling_params = SamplingParams()
    return Req(
        rid,
        origin_input_text,
        array("q", origin_input_ids),
        sampling_params,
        **kwargs,
    )


class TestSchedulePolicyHRRN(CustomTestCase):
    def test_calc_priority_hrrn(self):
        """HRRN sorts by response ratio (waited_tokens / uncached_tokens).

        With three fresh reqs (waited_tokens = 0 for all), the ratio is 0 / uncached for each,
        and stable sort keeps original order among equal keys -- effectively pure SUF via the
        rid tie-breaker is avoided, so we set a non-zero arrival snapshot on some reqs to exercise
        the aging half of the formula.
        """
        tree_cache = RadixCache.create_simulated()

        # r_short: small uncached, just arrived (waited=0)
        # r_long:  large uncached, just arrived (waited=0)
        # r_aged:  medium uncached, arrived long ago (waited>>0)
        r_short = _make_req("short", "a", [1])
        r_long = _make_req("long", "a" * 10, list(range(10)))
        r_aged = _make_req("aged", "a" * 3, [1, 2, 3])

        # Fresh reqs arrived with the counter still at 0.
        r_short.arrival_processed_tokens = 0
        r_long.arrival_processed_tokens = 0
        r_aged.arrival_processed_tokens = 0

        waiting_queue = [r_long, r_aged, r_short]

        policy = SchedulePolicy(
            policy="hrrn",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        # processed_tokens=1000 -> waited_tokens is 1000 for every req here.
        # Ratios: short = 1000/1 = 1000; aged = 1000/3 ~= 333; long = 1000/10 = 100.
        # Highest ratio first -> short, aged, long.
        waiting_queue = policy.calc_priority(waiting_queue, processed_tokens=1000)

        self.assertEqual(waiting_queue[0].rid, "short")
        self.assertEqual(waiting_queue[1].rid, "aged")
        self.assertEqual(waiting_queue[2].rid, "long")

    def test_calc_priority_hrrn_aging_overtakes_short(self):
        """A long request that has waited enough should overtake a
        just-arrived short request."""
        tree_cache = RadixCache.create_simulated()

        r_long_old = _make_req("long_old", "a" * 100, list(range(100)))
        r_short_new = _make_req("short_new", "a", [1])

        # long_old arrived at counter=0 and has been waiting; short_new
        # just arrived (its snapshot equals the current counter).
        r_long_old.arrival_processed_tokens = 0
        r_short_new.arrival_processed_tokens = 100000

        waiting_queue = [r_short_new, r_long_old]

        policy = SchedulePolicy(
            policy="hrrn",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        # processed_tokens=100000 -> long_old.waited = 100000, ratio = 100000/100 = 1000.
        #                            short_new.waited = 0, ratio = 0.
        # long_old should now be first.
        waiting_queue = policy.calc_priority(waiting_queue, processed_tokens=100000)

        self.assertEqual(waiting_queue[0].rid, "long_old")
        self.assertEqual(waiting_queue[1].rid, "short_new")

    def test_calc_priority_hrrn_cached_length_affects_order(self):
        """Cached prefix length shortens uncached, so reqs with the same input
        length can sort differently by HRRN. Also verifies rid tie-break for
        equal-ratio reqs.

        Uses _sort_by_hrrn directly to bypass the prefix-match pass inside
        calc_priority, which would overwrite num_matched_prefix_tokens.
        """
        r_more_cached = _make_req("a", "x" * 100, list(range(100)))
        r_less_cached = _make_req("b", "x" * 100, list(range(100)))
        r_tie = _make_req("c", "x" * 100, list(range(100)))

        # r_more_cached: 90 cached -> uncached = 10 -> ratio = 1000 / 10 = 100.
        # r_less_cached:  0 cached -> uncached = 100 -> ratio = 1000 / 100 = 10.
        # r_tie:          0 cached -> uncached = 100 -> ratio = 1000 / 100 = 10 (ties with r_less_cached; rid "b" < "c").
        r_more_cached.num_matched_prefix_tokens = 90
        r_less_cached.num_matched_prefix_tokens = 0
        r_tie.num_matched_prefix_tokens = 0
        r_more_cached.arrival_processed_tokens = 0
        r_less_cached.arrival_processed_tokens = 0
        r_tie.arrival_processed_tokens = 0

        waiting_queue = [r_tie, r_less_cached, r_more_cached]
        SchedulePolicy._sort_by_hrrn(waiting_queue, set(), processed_tokens=1000)

        self.assertEqual(waiting_queue[0].rid, "a")
        self.assertEqual(waiting_queue[1].rid, "b")
        self.assertEqual(waiting_queue[2].rid, "c")


class TestShortestPrefillFirst(CustomTestCase):
    def setUp(self):
        self.policy = SchedulePolicy(
            policy="shortest-prefill-first",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )

    def make_req(self, rid, uncached, *, cached=0, arrived=0):
        req = _make_req(rid, "", list(range(uncached + cached)))
        req.full_untruncated_fill_ids = req.origin_input_ids[:]
        req.num_matched_prefix_tokens = cached
        req.prefix_len = cached
        req.time_stats.wait_queue_entry_time = arrived
        return req

    def test_calc_priority_uses_uncached_work(self):
        cached = self.make_req("cached", 16, cached=4096)
        short = self.make_req("short", 32)
        long = self.make_req("long", 1024)
        queue = [long, short, cached]
        with patch.object(self.policy, "_compute_prefix_matches", return_value=set()):
            queue = self.policy.calc_priority(queue)
        self.assertEqual([req.rid for req in queue], ["cached", "short", "long"])

    def test_equal_work_uses_arrival_time(self):
        older = self.make_req("z", 32, arrived=1)
        newer = self.make_req("a", 32, arrived=2)
        queue = [newer, older]
        self.policy._sort_by_shortest_prefill(queue, set(), processed_tokens=0)
        self.assertEqual(queue, [older, newer])

    def test_duplicate_prefix_is_deprioritized(self):
        duplicate = self.make_req("duplicate", 1)
        other = self.make_req("other", 1024)
        queue = [duplicate, other]
        with patch.object(
            self.policy, "_compute_prefix_matches", return_value={duplicate.rid}
        ):
            queue = self.policy.calc_priority(queue)
        self.assertEqual(queue, [other, duplicate])

    def test_retracted_output_is_part_of_uncached_work(self):
        replay = self.make_req("replay", 16, cached=1024)
        replay.output_ids.extend([0] * 64)
        short = self.make_req("short", 32)
        queue = [replay, short]
        self.policy._sort_by_shortest_prefill(queue, set(), processed_tokens=0)
        self.assertEqual(queue, [short, replay])

    def test_chunk_limit_reserves_complete_short_prefills(self):
        continuation = self.make_req("continuation", 16384)
        waiting = [self.make_req("a", 512), self.make_req("b", 1024)]
        self.assertEqual(
            self.policy.shortest_prefill_chunk_limit(
                continuation, waiting, 4096, 256, processed_tokens=0
            ),
            2560,
        )

    def test_reservation_rounds_to_pages_and_keeps_continuation_progress(self):
        continuation = self.make_req("continuation", 16384)
        self.assertEqual(
            self.policy.shortest_prefill_chunk_limit(
                continuation,
                [self.make_req("short", 257)],
                4096,
                256,
                processed_tokens=0,
            ),
            3584,
        )
        self.assertEqual(
            self.policy.shortest_prefill_chunk_limit(
                continuation,
                [self.make_req("short", 3840)],
                4096,
                256,
                processed_tokens=0,
            ),
            256,
        )

    def test_no_reservation_when_request_cannot_fit_or_is_not_shorter(self):
        continuation = self.make_req("continuation", 8192)
        for waiting, budget in [
            ([], 4096),
            ([self.make_req("same", 8192)], 4096),
            ([self.make_req("too-large", 4096)], 4096),
            ([self.make_req("short", 1)], 256),
        ]:
            with self.subTest(budget=budget, waiting=[req.rid for req in waiting]):
                self.assertIsNone(
                    self.policy.shortest_prefill_chunk_limit(
                        continuation, waiting, budget, 256, processed_tokens=0
                    )
                )

    def test_other_policy_keeps_normal_chunk_limit(self):
        self.policy.policy = CacheAwarePolicy.HRRN
        self.assertIsNone(
            self.policy.shortest_prefill_chunk_limit(
                self.make_req("continuation", 8192),
                [self.make_req("short", 512)],
                4096,
                256,
                processed_tokens=0,
            )
        )

    def test_staged_storage_prefix_counts_as_cached(self):
        # A buffer-mode storage hit stays outside the tree until admission.
        staged = self.make_req("staged", 4096, cached=512)
        short = self.make_req("short", 1024)
        tree_cache = self.policy.tree_cache
        staged_end = {staged.cache_request_handle: 4600}
        with (
            patch.object(
                tree_cache, "can_poll_prefetch_progress", return_value=True
            ) as can_poll,
            patch.object(tree_cache, "poll_prefetch_progress", create=True) as poll,
            patch.object(
                tree_cache,
                "staged_storage_prefix_end",
                side_effect=lambda handle: staged_end.get(handle, 0),
            ),
        ):
            queue = [short, staged]
            self.policy.poll_storage_prefetches(queue)
            self.policy._sort_by_shortest_prefill(queue, set(), processed_tokens=0)
            self.assertEqual(queue, [staged, short])
            self.assertEqual(
                self.policy.shortest_prefill_chunk_limit(
                    self.make_req("continuation", 16384),
                    [staged],
                    4096,
                    256,
                    processed_tokens=0,
                ),
                3840,
            )
            # Neither a stop policy that polling would cut short nor another
            # schedule policy polls.
            can_poll.return_value = False
            self.policy.poll_storage_prefetches(queue)
            can_poll.return_value = True
            self.policy.policy = CacheAgnosticPolicy.FCFS
            self.policy.poll_storage_prefetches(queue)
        poll.assert_called_once_with(
            [short.cache_request_handle, staged.cache_request_handle]
        )

    def test_aging_promotes_waiting_long_requests(self):
        policy = SchedulePolicy(
            policy="shortest-prefill-first",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
            shortest_prefill_aging_factor=0.5,
        )
        old_long = self.make_req("old_long", 1000)
        new_short = self.make_req("new_short", 10)
        new_short.arrival_processed_tokens = 3000
        queue = [new_short, old_long]
        # Rank = uncached work - 0.5 * prefill tokens processed while waiting.
        policy._sort_by_shortest_prefill(queue, set(), processed_tokens=1000)
        self.assertEqual(queue, [new_short, old_long])
        policy._sort_by_shortest_prefill(queue, set(), processed_tokens=3000)
        self.assertEqual(queue, [old_long, new_short])

        continuation = self.make_req("continuation", 16384)
        self.assertEqual(
            policy.shortest_prefill_chunk_limit(
                continuation, [new_short], 4096, 256, processed_tokens=3000
            ),
            3840,
        )
        fresh_short = self.make_req("fresh_short", 10)
        fresh_short.arrival_processed_tokens = 40000
        self.assertIsNone(
            policy.shortest_prefill_chunk_limit(
                continuation, [fresh_short], 4096, 256, processed_tokens=40000
            )
        )

    def test_invalid_aging_factor_is_rejected(self):
        for schedule_policy, factor in [
            ("shortest-prefill-first", -0.5),
            ("shortest-prefill-first", float("nan")),
            ("shortest-prefill-first", float("inf")),
            ("fcfs", 0.5),
        ]:
            with self.subTest(policy=schedule_policy, factor=factor):
                with self.assertRaises(ValueError):
                    SchedulePolicy(
                        policy=schedule_policy,
                        tree_cache=RadixCache.create_simulated(),
                        enable_hierarchical_cache=True,
                        enable_priority_scheduling=False,
                        schedule_low_priority_values_first=False,
                        shortest_prefill_aging_factor=factor,
                    )


if __name__ == "__main__":
    unittest.main()
