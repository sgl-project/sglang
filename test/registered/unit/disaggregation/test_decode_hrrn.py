import unittest
from types import SimpleNamespace as NS

from sglang.srt.disaggregation import decode_hrrn as module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def entry(rid, tokens, fraction=1):
    return NS(
        req=NS(
            rid=rid, origin_input_ids=range(tokens), prefill_uncached_fraction=fraction
        )
    )


class TestDecodeHrrn(unittest.TestCase):
    def test_short_cached_work_wins_after_equal_aging(self):
        q = module.DecodeHrrn()
        items = [entry("long", 1000), entry("cached", 1000, 0.01), entry("seed", 100)]
        q.order(items)
        q.admitted("seed")
        items = items[:2]
        q.order(items)
        self.assertEqual([x.req.rid for x in items], ["cached", "long"])

    def test_new_arrival_does_not_reset_old_age(self):
        q = module.DecodeHrrn()
        old = entry("old", 100)
        items = [old, entry("seed", 1000)]
        q.order(items)
        q.admitted("seed")
        items = [entry("new", 1), old]
        q.order(items)
        self.assertEqual(items[0].req.rid, "old")

    def test_cancel_cleanup_and_deterministic_ties(self):
        a, b = module.DecodeHrrn(), module.DecodeHrrn()
        items = [entry("z", 10), entry("a", 10)]
        left, right = list(items), list(items)
        a.order(left)
        b.order(right)
        self.assertEqual([x.req.rid for x in left], ["z", "a"])
        a.order(left[1:])
        self.assertNotIn("z", a.waiting)
        self.assertEqual([x.req.rid for x in left], [x.req.rid for x in right])

    def test_invalid_estimates_are_cold(self):
        for value in (None, "bad", "nan", "inf", "-1", "1.1"):
            self.assertEqual(module.parse_uncached_fraction(value), 1)
        self.assertEqual(module.parse_uncached_fraction("0.25"), 0.25)

    def test_fully_cached_cost_is_nonzero(self):
        q = module.DecodeHrrn()
        q.order([entry("cached", 10000, 0)])
        self.assertEqual(q.waiting["cached"][1], 1)

    def test_rankings_identical_under_repeated_cycles(self):
        a, b = module.DecodeHrrn(), module.DecodeHrrn()
        for cycle in range(100):
            left = [
                entry(str(i), (i + 1) * 101, 0.1 if i % 2 else 1)
                for i in range(cycle, cycle + 32)
            ]
            right = list(left)
            a.order(left)
            b.order(right)
            self.assertEqual([x.req.rid for x in left], [x.req.rid for x in right])
            for item in left[:3]:
                a.admitted(item.req.rid)
                b.admitted(item.req.rid)


if __name__ == "__main__":
    unittest.main()
