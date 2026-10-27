import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import CacheAwarePolicy, SchedulePolicy
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
            self.policy.shortest_prefill_chunk_limit(continuation, waiting, 4096, 256),
            2560,
        )

    def test_reservation_rounds_to_pages_and_keeps_continuation_progress(self):
        continuation = self.make_req("continuation", 16384)
        self.assertEqual(
            self.policy.shortest_prefill_chunk_limit(
                continuation, [self.make_req("short", 257)], 4096, 256
            ),
            3584,
        )
        self.assertEqual(
            self.policy.shortest_prefill_chunk_limit(
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
                    self.policy.shortest_prefill_chunk_limit(
                        continuation, waiting, budget, 256
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
            )
        )


class TestPrefixWeave(CustomTestCase):
    def setUp(self):
        self.policy = SchedulePolicy(
            policy="prefix-weave",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
            chunked_prefill_size=64,
            max_prefill_tokens=128,
        )

    @staticmethod
    def make_req(rid, *, routing_key=None, prefix_len=0):
        req = _make_req(
            rid,
            "",
            list(range(10)),
            routing_key=routing_key or f"key-{rid}",
        )
        req.prefix_len = prefix_len
        return req

    def test_tier_precedence_overrides_cached_prefix_length(self):
        """A lower tier must not jump ahead because its prefix_len is larger.

        PrefixWeave combines global-cache reuse with marginal in-batch overlap.
        Reversing either comparison silently changes which requests are admitted
        first, even though every request still receives a valid numeric score.
        """
        tier_1 = self.make_req("tier-1", prefix_len=1)
        tier_2 = self.make_req("tier-2", prefix_len=9)
        tier_3 = self.make_req("tier-3", prefix_len=8)
        unclassified = self.make_req("unclassified", prefix_len=10)
        queue = [unclassified, tier_3, tier_2, tier_1]

        global_cached = {
            "tier-1": 8,
            "tier-2": 8,
            "tier-3": 1,
            "unclassified": 1,
        }
        batch_overlap = {
            "tier-1": 0,
            "tier-2": 9,
            "tier-3": 0,
            "unclassified": 2,
        }

        with (
            patch(
                "sglang.srt.managers.schedule_policy.match_kv_cache",
                side_effect=lambda req, tree_cache, token_ids=None: SimpleNamespace(
                    device_prefix_len=global_cached[req.rid]
                ),
            ),
            patch.object(
                self.policy,
                "_match_batch_prefix_for_req",
                side_effect=lambda req, token_ids=None: SimpleNamespace(
                    device_prefix_len=batch_overlap[req.rid]
                ),
            ),
            patch.object(self.policy, "_insert_inbatch_tree"),
        ):
            self.policy.calc_priority(queue, max_req_num=4, token_budget=128)

        self.assertEqual(
            [req.rid for req in queue],
            ["tier-1", "tier-2", "tier-3", "unclassified"],
        )
        self.assertEqual(self.policy.classified_reqs, {"tier-1", "tier-2", "tier-3"})

        self.policy.update_classified_reqs([tier_1, tier_3])
        self.assertEqual(self.policy.classified_reqs, {"tier-2"})

    def test_full_classification_budget_keeps_existing_order(self):
        """Previously classified work filling the request budget is not reshuffled.

        This preserves the admission decision across scheduler passes and keeps
        the policy's calculation counter limited to actual classifications.
        """
        first = self.make_req("first")
        second = self.make_req("second")
        queue = [first, second]
        self.policy.classified_reqs = {first.rid}

        with patch(
            "sglang.srt.managers.schedule_policy.match_kv_cache",
            return_value=SimpleNamespace(device_prefix_len=0),
        ):
            self.policy.calc_priority(queue, max_req_num=1, token_budget=128)

        self.assertEqual(queue, [first, second])
        self.assertEqual(self.policy.classified_reqs, {first.rid})
        self.assertEqual(self.policy.calc_count, 0)

    def test_finished_routing_length_is_reused_for_next_turn(self):
        """The latest completed length seeds the next turn's cache estimate.

        The completion path and the next scheduling pass are separate call sites;
        losing their routing-key handoff degrades multi-turn cache prioritization
        without causing a request failure.
        """
        finished = self.make_req("finished", routing_key="conversation")
        finished.output_ids.extend([10, 11])
        self.policy.update_finished_reqs_info(finished)
        finished.output_ids.append(12)
        self.policy.update_finished_reqs_info(finished)

        next_turn = self.make_req("next", routing_key="conversation")
        with patch(
            "sglang.srt.managers.schedule_policy.match_kv_cache"
        ) as global_match:
            calculated = self.policy._prefix_weave_phase0_presort(
                [next_turn], max_req_num=1, max_token=128
            )

        self.assertTrue(calculated)
        self.assertEqual(self.policy.finished_req_lengths["conversation"], 13)
        self.assertEqual(self.policy.estimated_cached_tokens["conversation"], 13)
        global_match.assert_not_called()


if __name__ == "__main__":
    unittest.main()
