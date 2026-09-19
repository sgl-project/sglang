import unittest
from array import array
from unittest.mock import patch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
        policy.calc_priority(waiting_queue, processed_tokens=1000)

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
        policy.calc_priority(waiting_queue, processed_tokens=100000)

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
        self.policy = self.make_policy("shortest-prefill-first")

    def make_policy(self, name, **kwargs):
        return SchedulePolicy(
            policy=name,
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
            **kwargs,
        )

    def make_req(self, rid, uncached, *, cached=0, arrived=0):
        req = _make_req(rid, "", list(range(uncached + cached)))
        req.full_untruncated_fill_ids = req.origin_input_ids[:]
        req.num_matched_prefix_tokens = cached
        req.prefix_indices = list(range(cached))
        req.time_stats.wait_queue_entry_time = arrived
        return req

    def test_calc_priority_uses_uncached_work(self):
        cached = self.make_req("cached", 16, cached=4096)
        short = self.make_req("short", 32)
        long = self.make_req("long", 1024)
        queue = [long, short, cached]
        with patch.object(self.policy, "_compute_prefix_matches", return_value=set()):
            self.policy.calc_priority(queue)
        self.assertEqual([req.rid for req in queue], ["cached", "short", "long"])

    def test_equal_work_uses_arrival_time(self):
        older = self.make_req("z", 32, arrived=1)
        newer = self.make_req("a", 32, arrived=2)
        queue = [newer, older]
        self.policy._sort_by_shortest_prefill(queue, set())
        self.assertEqual(queue, [older, newer])

    def test_duplicate_prefix_is_deprioritized(self):
        duplicate = self.make_req("duplicate", 1)
        other = self.make_req("other", 1024)
        queue = [duplicate, other]
        with patch.object(
            self.policy, "_compute_prefix_matches", return_value={duplicate.rid}
        ):
            self.policy.calc_priority(queue)
        self.assertEqual(queue, [other, duplicate])

    def test_retracted_output_is_part_of_uncached_work(self):
        replay = self.make_req("replay", 16, cached=1024)
        replay.output_ids.extend([0] * 64)
        short = self.make_req("short", 32)
        queue = [replay, short]
        self.policy._sort_by_shortest_prefill(queue, set())
        self.assertEqual(queue, [short, replay])

    def test_chunk_limit_reserves_complete_short_prefills(self):
        continuation = self.make_req("continuation", 16384)
        waiting = [self.make_req("a", 512), self.make_req("b", 1024)]
        self.assertEqual(
            self.policy.prefill_interleaving_chunk_limit(
                continuation, waiting, 4096, 256
            ),
            2560,
        )

    def test_reservation_rounds_to_pages_and_keeps_continuation_progress(self):
        continuation = self.make_req("continuation", 16384)
        self.assertEqual(
            self.policy.prefill_interleaving_chunk_limit(
                continuation, [self.make_req("short", 257)], 4096, 256
            ),
            3584,
        )
        self.assertEqual(
            self.policy.prefill_interleaving_chunk_limit(
                continuation, [self.make_req("short", 3840)], 4096, 256
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
                    self.policy.prefill_interleaving_chunk_limit(
                        continuation, waiting, budget, 256
                    )
                )

    def test_other_policy_keeps_normal_chunk_limit(self):
        self.policy = self.make_policy("hrrn")
        self.assertIsNone(
            self.policy.prefill_interleaving_chunk_limit(
                self.make_req("continuation", 8192),
                [self.make_req("short", 512)],
                4096,
                256,
            )
        )

    def test_interleaving_disable_preserves_queue_and_full_chunk(self):
        for name in ("hrrn", "shortest-prefill-first"):
            with self.subTest(policy=name):
                policy = self.make_policy(name, disable_prefill_interleaving=True)
                waiting = [self.make_req("short", 512)]
                original = waiting[:]
                self.assertIsNone(
                    policy.prefill_interleaving_chunk_limit(
                        self.make_req("continuation", 8192), waiting, 4096, 256
                    )
                )
                self.assertEqual(waiting, original)

    def test_policy_selection_and_minimum_allocation(self):
        # HRRN must scan beyond a large head and retain priority among fitting waiters.
        for name, expected_limit, expected_order in (
            ("hrrn", 1024, ["first-fit", "last-fit", "large", "middle"]),
            (
                "shortest-prefill-first",
                None,
                ["large", "first-fit", "middle", "last-fit"],
            ),
        ):
            with self.subTest(policy=name):
                policy = self.make_policy(
                    name,
                    enable_prefill_interleaving=True,
                    prefill_interleaving_min_continuation_tokens=1024,
                )
                waiting = [
                    self.make_req(rid, n)
                    for rid, n in (
                        ("large", 8192),
                        ("first-fit", 257),
                        ("middle", 3072),
                        ("last-fit", 2560),
                    )
                ]
                limit = policy.prefill_interleaving_chunk_limit(
                    self.make_req("continuation", 16384), waiting, 4096, 256
                )
                self.assertEqual(limit, expected_limit)
                self.assertEqual([r.rid for r in waiting], expected_order)

    def test_interleaving_scan_cap_preserves_unscanned_requests(self):
        continuation = self.make_req("continuation", 8192)
        for large_count in (126, 127, 128):
            with self.subTest(large_count=large_count):
                policy = self.make_policy("hrrn", enable_prefill_interleaving=True)
                large = [self.make_req(str(i), 4096) for i in range(large_count)]
                small = [self.make_req("first", 256), self.make_req("second", 256)]
                waiting = large + small
                selected_count = min(2, 128 - large_count)
                limit = policy.prefill_interleaving_chunk_limit(
                    continuation, waiting, 4096, 256
                )
                self.assertEqual(
                    limit, 4096 - selected_count * 256 if selected_count else None
                )
                self.assertEqual(
                    waiting, small[:selected_count] + large + small[selected_count:]
                )

        waiting = [self.make_req(str(i), 1) for i in range(130)]
        original = waiting[:]
        self.assertEqual(
            self.policy.prefill_interleaving_chunk_limit(
                continuation, waiting, 4096, 1
            ),
            4096 - 128,
        )
        self.assertEqual(waiting, original)

    def test_minimum_and_shorter_than_continuation_are_independent(self):
        for name in ("hrrn", "shortest-prefill-first"):
            with self.subTest(policy=name):
                policy = self.make_policy(
                    name,
                    enable_prefill_interleaving=True,
                    prefill_interleaving_min_continuation_tokens=1024,
                )
                waiting = [self.make_req("waiter", 3072)]
                self.assertEqual(
                    policy.prefill_interleaving_chunk_limit(
                        self.make_req("continuation", 8192), waiting, 4096, 256
                    ),
                    1024,
                )
                self.assertIsNone(
                    policy.prefill_interleaving_chunk_limit(
                        self.make_req("continuation", 8192), waiting, 1024, 256
                    )
                )
                limit = policy.prefill_interleaving_chunk_limit(
                    self.make_req("continuation", 2048), waiting, 4096, 256
                )
                self.assertEqual(limit, 1024 if name == "hrrn" else None)

    def test_hrrn_aging_orders_fitting_requests_before_reservation(self):
        policy = self.make_policy(
            "hrrn",
            enable_prefill_interleaving=True,
            prefill_interleaving_min_continuation_tokens=1024,
        )
        old = self.make_req("old", 3072)
        old.arrival_processed_tokens = 0
        new = self.make_req("new", 1024)
        new.origin_input_ids = array("q", range(20000, 21024))
        new.full_untruncated_fill_ids = new.origin_input_ids[:]
        new.arrival_processed_tokens = 100000
        waiting = [new, old]
        policy.calc_priority(waiting, processed_tokens=100000)
        self.assertEqual(waiting, [old, new])
        self.assertEqual(
            policy.prefill_interleaving_chunk_limit(
                self.make_req("continuation", 16384), waiting, 4096, 256
            ),
            1024,
        )
        self.assertEqual(waiting, [old, new])

    def test_policy_default_minimum_and_explicit_override(self):
        for name, minimum, budget, waiter, expected in (
            ("hrrn", None, 16384, 8192, 8192),
            ("hrrn", None, 4096, 3072, None),
            ("hrrn", None, 1280, 512, 768),
            ("hrrn", None, 256, 1, None),
            ("hrrn", 256, 4096, 3840, 256),
            ("shortest-prefill-first", None, 4096, 3840, 256),
            ("shortest-prefill-first", 2048, 4096, 3072, None),
        ):
            with self.subTest(policy=name, minimum=minimum, budget=budget):
                policy = self.make_policy(
                    name,
                    enable_prefill_interleaving=True,
                    prefill_interleaving_min_continuation_tokens=minimum,
                )
                self.assertEqual(
                    policy.prefill_interleaving_chunk_limit(
                        self.make_req("continuation", 32768),
                        [self.make_req("waiter", waiter)],
                        budget,
                        256,
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
