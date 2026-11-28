import unittest
from array import array
from unittest.mock import patch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import (
    CacheAgnosticPolicy,
    CacheAwarePolicy,
    SchedulePolicy,
)
from sglang.srt.managers.shortest_prefill_load import ShortestPrefillLoadScaling
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.runtime_context import get_context
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
        # The chunk reservation reads the encoder replay flag.
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
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

    def test_shared_prefix_holds_the_longer_request_not_the_shorter(self):
        # long and short share 32 tokens, the dedupe threshold.
        shared = list(range(32))
        long = _make_req("long", "", shared + list(range(1000, 5000)))
        other = _make_req("other", "", list(range(20000, 26000)))
        short = _make_req("short", "", shared + list(range(9000, 9393)))
        for arrived, req in enumerate([long, other, short]):
            req.full_untruncated_fill_ids = req.origin_input_ids[:]
            req.time_stats.wait_queue_entry_time = arrived
        queue = self.policy.calc_priority([long, other, short])
        self.assertEqual([req.rid for req in queue], ["short", "other", "long"])

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


class TestAdmissionMatchedPrefixTokens(CustomTestCase):
    """The admission re-match must record num_matched_prefix_tokens.

    Cache-agnostic policies (fcfs, lof, random, routing-key) skip the
    sort-time match unless the tree supports fast matching, so this is the only
    place their requests record it; a request left at 0 makes load snapshots and
    cache-hit reporting read zero reuse (#43094).
    """

    PREFIX = list(range(64))

    def _cache_with_prefix(self):
        tree_cache = RadixCache.create_simulated()
        tree_cache.insert(InsertParams(key=RadixKey(token_ids=array("q", self.PREFIX))))
        return tree_cache

    def test_warm_req_records_match_length(self):
        req = _make_req("warm", "", self.PREFIX + [7, 8, 9])
        req.init_next_round_input(self._cache_with_prefix())
        self.assertEqual(req.num_matched_prefix_tokens, len(self.PREFIX))

    def test_cold_req_records_zero(self):
        req = _make_req("cold", "", [100 + i for i in range(80)])
        req.init_next_round_input(self._cache_with_prefix())
        self.assertEqual(req.num_matched_prefix_tokens, 0)

    def test_rematch_after_retraction_records_again(self):
        # Retraction zeroes the field; the next admission must record the
        # reuse again rather than keep the reset value.
        tree_cache = self._cache_with_prefix()
        req = _make_req("warm", "", self.PREFIX + [7, 8, 9])
        req.init_next_round_input(tree_cache)
        req.reset_for_retract()
        self.assertEqual(req.num_matched_prefix_tokens, 0)
        req.init_next_round_input(tree_cache)
        self.assertEqual(req.num_matched_prefix_tokens, len(self.PREFIX))


class TestShortestPrefillLoadScaling(CustomTestCase):
    make_req = TestShortestPrefillFirst.make_req

    def setUp(self):
        # The chunk reservation reads the encoder replay flag.
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def make_policy(self, **scaling):
        return SchedulePolicy(
            policy="shortest-prefill-first",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
            shortest_prefill_load_scaling=resolve(chunked_prefill_size=4096, **scaling),
        )

    def sort(self, policy, queue, chunked_req=None):
        with patch.object(policy, "_compute_prefix_matches", return_value=set()):
            return policy.calc_priority(queue, chunked_req=chunked_req)

    def sorted_rids(self, policy, queue):
        return [req.rid for req in self.sort(policy, queue)]

    def test_unset_values_derive_and_explicit_zeros_stay(self):
        self.assertIsNone(resolve(chunked_prefill_size=32768, halving_chunks=0))
        derived = resolve(chunked_prefill_size=32768, halving_chunks=2.5)
        self.assertEqual(
            (
                derived.halving_tokens,
                derived.decay_start_tokens,
                derived.budget_min_tokens,
                derived.budget_doubling_tokens,
                derived.budget_full_backlog,
            ),
            (81920, 163840, 4096, 81920, 245760),
        )
        zeros = resolve(
            chunked_prefill_size=32768,
            halving_chunks=2.5,
            decay_start_chunks=0,
            budget_min_ratio=0,
        )
        self.assertEqual(zeros.decay_start_tokens, 0)
        self.assertIsNone(zeros.token_budget(0, page_size=1, align=None))

    def test_needs_chunked_prefill(self):
        with self.assertRaises(ValueError):
            resolve(chunked_prefill_size=None, halving_chunks=1)

    def test_mid_size_request_moves_to_arrival_class_as_backlog_grows(self):
        policy = self.make_policy(halving_chunks=0.5, decay_start_chunks=3.5)
        queue = [
            self.make_req("long-new", 4200, arrived=5),
            self.make_req("mid", 2500, arrived=4),
            self.make_req("short", 500, arrived=3),
            self.make_req("long-old", 6000, arrived=1),
        ]
        # Up to the decay start the boundary is the chunk size, so only
        # chunk-sized requests go by arrival.
        self.assertEqual(
            self.sorted_rids(policy, queue), ["short", "mid", "long-old", "long-new"]
        )
        self.assertEqual(policy.shortest_prefill_backlog, 13200)
        self.assertEqual(policy.shortest_prefill_class_boundary, 4096)
        # Half a chunk past the decay start, the boundary halves.
        queue.append(self.make_req("newcomer", 3184, arrived=6))
        self.assertEqual(
            self.sorted_rids(policy, queue),
            ["short", "long-old", "mid", "long-new", "newcomer"],
        )
        self.assertEqual(policy.shortest_prefill_backlog, 16384)
        self.assertEqual(policy.shortest_prefill_class_boundary, 2048)

    def test_backlog_counts_what_the_continuing_chunk_has_left(self):
        policy = self.make_policy(halving_chunks=0.5, decay_start_chunks=2)
        continuation = self.make_req("continuation", 6000, cached=2000)
        continuation.prefix_len = 4000
        self.sort(
            policy,
            [self.make_req("a", 3000), self.make_req("b", 5288)],
            chunked_req=continuation,
        )
        self.assertEqual(policy.shortest_prefill_backlog, 12288)
        self.assertEqual(policy.shortest_prefill_class_boundary, 1024)

    def test_backlog_skips_confirmed_storage_hit_until_the_prefetch_drops(self):
        policy = self.make_policy(halving_chunks=0.5, decay_start_chunks=2)
        stored = self.make_req("stored", 12000, cached=1000, arrived=1)
        short = self.make_req("short", 500, arrived=2)
        spans = {stored.cache_request_handle: (0, 12000)}
        with patch.object(
            policy.tree_cache,
            "waiting_storage_span",
            side_effect=lambda handle: spans.get(handle),
        ):
            # The hit beyond the device match is not backlog; the sort still
            # ranks the request by its full work.
            self.assertEqual(
                self.sorted_rids(policy, [stored, short]), ["short", "stored"]
            )
            self.assertEqual(policy.shortest_prefill_backlog, 1500)
            self.assertEqual(policy.shortest_prefill_class_boundary, 4096)
            # A revoked or failed prefetch drops its span: full work next pass.
            spans.clear()
            self.assertEqual(
                self.sorted_rids(policy, [stored, short]), ["short", "stored"]
            )
            self.assertEqual(policy.shortest_prefill_backlog, 12500)
            self.assertAlmostEqual(
                policy.shortest_prefill_class_boundary, 4096 * 2 ** (-4308 / 2048)
            )

    def test_reservation_stops_at_the_first_arrival_ordered_request(self):
        # 14768 tokens left in the chunk and 1616 waiting: two halvings of 2 chunks.
        policy = self.make_policy(halving_chunks=2, decay_start_chunks=0)
        continuation = self.make_req("continuation", 14768)
        queue = [
            self.make_req("mid", 1104, arrived=1),
            self.make_req("small", 512, arrived=2),
        ]
        queue = self.sort(policy, queue, chunked_req=continuation)
        self.assertEqual(policy.shortest_prefill_class_boundary, 1024)
        self.assertEqual(
            policy.shortest_prefill_chunk_limit(
                continuation, queue, 4096, 256, processed_tokens=0
            ),
            3584,
        )
        # Without the boundary, the mid-size request fits beside the small one.
        self.assertEqual(
            self.make_policy(halving_chunks=0).shortest_prefill_chunk_limit(
                continuation, queue, 4096, 256, processed_tokens=0
            ),
            2304,
        )

    def test_token_budget_scales_with_backlog_and_rounds_to_page_and_alignment(self):
        for min_ratio, doubling_chunks, page_size, align, backlog, budget in (
            # An empty queue gets the minimum, rounded down to a page.
            (0.24, 1, 64, None, 0, 960),
            (0.24, 1, 64, None, 4096, 1920),
            # No cap once the budget reaches the chunk size.
            (0.24, 0.001, 64, None, 2000, None),
            # Never below one page, or one alignment unit.
            (0.005, 1, 64, None, 0, 64),
            (0.25, 1, 64, 1536, 0, 1536),
        ):
            with self.subTest(min_ratio=min_ratio, align=align, backlog=backlog):
                scaling = resolve(
                    chunked_prefill_size=4096,
                    halving_chunks=1,
                    budget_min_ratio=min_ratio,
                    budget_doubling_chunks=doubling_chunks,
                )
                self.assertEqual(
                    scaling.token_budget(backlog, page_size=page_size, align=align),
                    budget,
                )


def resolve(**kwargs):
    return ShortestPrefillLoadScaling.resolve(
        **{
            "decay_start_chunks": None,
            "budget_min_ratio": None,
            "budget_doubling_chunks": None,
            **kwargs,
        }
    )


if __name__ == "__main__":
    unittest.main()
