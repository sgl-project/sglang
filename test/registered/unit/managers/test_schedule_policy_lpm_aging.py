"""Unit tests for LPM anti-starvation aging (--lpm-aging-tokens-per-pass).

Without aging, LPM sorts purely by matched-prefix length, so a cache-cold
request is displaced indefinitely by a continuous stream of cache-hot
arrivals (unbounded starvation). With aging, each pass spent waiting adds a
configurable token bonus to the effective match length, bounding the wait.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


def _req(rid: str, matched: int, waiting_passes: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        rid=rid,
        num_matched_prefix_tokens=matched,
        lpm_waiting_passes=waiting_passes,
    )


class TestLpmAgingSort(CustomTestCase):
    def test_no_aging_preserves_pure_match_order(self):
        """aging=0 (default): match length alone decides; waiting passes are ignored."""
        cold = _req("cold", matched=0, waiting_passes=1000)
        hot = _req("hot", matched=100_000, waiting_passes=0)
        queue = [cold, hot]
        SchedulePolicy._sort_by_longest_prefix(queue, set())
        self.assertEqual([r.rid for r in queue], ["hot", "cold"])

    def test_aging_bounds_starvation(self):
        """With aging, a cold request eventually outranks fresh hot arrivals."""
        aging = 4096
        hot_match = 100_000
        # One pass short of overtaking: still behind.
        almost = _req("cold", matched=0, waiting_passes=hot_match // aging - 1)
        hot = _req("hot", matched=hot_match, waiting_passes=0)
        queue = [almost, hot]
        SchedulePolicy._sort_by_longest_prefix(queue, set(), aging)
        self.assertEqual(queue[0].rid, "hot")

        # Enough passes accumulated: the cold request sorts first.
        aged = _req("cold", matched=0, waiting_passes=hot_match // aging + 1)
        queue = [aged, _req("hot", matched=hot_match, waiting_passes=0)]
        SchedulePolicy._sort_by_longest_prefix(queue, set(), aging)
        self.assertEqual(queue[0].rid, "cold")

    def test_aging_applies_to_effective_match_not_order(self):
        """Aging composes with match length rather than overriding it."""
        aging = 1000
        a = _req("a", matched=5_000, waiting_passes=2)  # effective 7_000
        b = _req("b", matched=6_000, waiting_passes=0)  # effective 6_000
        queue = [b, a]
        SchedulePolicy._sort_by_longest_prefix(queue, set(), aging)
        self.assertEqual([r.rid for r in queue], ["a", "b"])

    def test_deprioritized_requests_stay_last_despite_aging(self):
        """In-batch deprioritized requests are exempt from aging (one-pass setback
        by design; next pass they match the group leader's cached prefix)."""
        aging = 4096
        dep = _req("dep", matched=0, waiting_passes=10_000)
        hot = _req("hot", matched=10, waiting_passes=0)
        queue = [dep, hot]
        SchedulePolicy._sort_by_longest_prefix(queue, set(), aging_tokens_per_pass=aging)
        self.assertEqual([r.rid for r in queue], ["dep", "hot"])
        SchedulePolicy._sort_by_longest_prefix(queue, {"dep"}, aging)
        self.assertEqual([r.rid for r in queue], ["hot", "dep"])

    def test_missing_pass_attr_defaults_to_zero(self):
        """Requests that never went through an aging-enabled pass sort as unaged."""
        no_attr = SimpleNamespace(rid="fresh", num_matched_prefix_tokens=50)
        aged = _req("aged", matched=0, waiting_passes=1)
        queue = [no_attr, aged]
        SchedulePolicy._sort_by_longest_prefix(queue, set(), 100)
        self.assertEqual([r.rid for r in queue], ["aged", "fresh"])


class TestServerArgsDefault(CustomTestCase):
    def test_default_is_disabled(self):
        from sglang.srt.server_args import ServerArgs

        self.assertEqual(ServerArgs.lpm_aging_tokens_per_pass, 0)


if __name__ == "__main__":
    unittest.main()
