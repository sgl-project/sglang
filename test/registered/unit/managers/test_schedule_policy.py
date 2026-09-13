import unittest
from array import array

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


if __name__ == "__main__":
    unittest.main()
