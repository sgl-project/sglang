"""Unit tests for /v1/loads helpers: _compute_aggregate and _loads_dict_factory."""

import unittest

from sglang.srt.entrypoints.v1_loads import _compute_aggregate, _loads_dict_factory
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


_INT_AGG_KEYS = (
    "total_running_reqs",
    "total_waiting_reqs",
    "total_reqs",
    "total_used_tokens",
    "total_tokens",
)
_FLOAT_AGG_KEYS = ("avg_token_usage", "avg_throughput", "avg_utilization")


def _load(
    *,
    dp_rank=0,
    running=0,
    waiting=0,
    used=0,
    total=0,
    token_usage=0.0,
    throughput=0.0,
    utilization=0.0,
):
    return {
        "dp_rank": dp_rank,
        "num_running_reqs": running,
        "num_waiting_reqs": waiting,
        "num_used_tokens": used,
        "num_total_tokens": total,
        "token_usage": token_usage,
        "gen_throughput": throughput,
        "utilization": utilization,
    }


class TestComputeAggregate(CustomTestCase):
    def test_empty_input_zero_for_every_key(self):
        agg = _compute_aggregate([])
        for k in _INT_AGG_KEYS:
            with self.subTest(key=k):
                self.assertEqual(agg[k], 0)
        for k in _FLOAT_AGG_KEYS:
            with self.subTest(key=k):
                self.assertEqual(agg[k], 0.0)

    def test_single_dp_rank_sums(self):
        agg = _compute_aggregate([_load(running=4, waiting=2, used=100, total=150)])
        self.assertEqual(agg["total_running_reqs"], 4)
        self.assertEqual(agg["total_waiting_reqs"], 2)
        self.assertEqual(agg["total_reqs"], 6)
        self.assertEqual(agg["total_used_tokens"], 100)
        self.assertEqual(agg["total_tokens"], 150)

    def test_multi_dp_rank_sums(self):
        agg = _compute_aggregate(
            [
                _load(dp_rank=0, running=3, waiting=1, used=50, total=70),
                _load(dp_rank=1, running=5, waiting=2, used=80, total=100),
                _load(dp_rank=2, running=0, waiting=4, used=0, total=40),
            ]
        )
        self.assertEqual(agg["total_running_reqs"], 8)
        self.assertEqual(agg["total_waiting_reqs"], 7)
        self.assertEqual(agg["total_reqs"], 15)
        self.assertEqual(agg["total_used_tokens"], 130)
        self.assertEqual(agg["total_tokens"], 210)

    def test_averages_over_dp_count(self):
        agg = _compute_aggregate(
            [
                _load(token_usage=0.6, throughput=100.0, utilization=0.5),
                _load(token_usage=0.8, throughput=200.0, utilization=0.7),
            ]
        )
        self.assertAlmostEqual(agg["avg_token_usage"], 0.7)
        self.assertAlmostEqual(agg["avg_throughput"], 150.0)
        self.assertAlmostEqual(agg["avg_utilization"], 0.6)

    def test_total_tokens_differs_from_total_used_tokens(self):
        # Regression: total_tokens sums num_total_tokens, NOT num_used_tokens.
        agg = _compute_aggregate([_load(used=10, total=30), _load(used=20, total=45)])
        self.assertEqual(agg["total_used_tokens"], 30)
        self.assertEqual(agg["total_tokens"], 75)


class TestLoadsDictFactory(CustomTestCase):
    def test_filters_none_values(self):
        d = _loads_dict_factory(
            [("dp_rank", 0), ("memory", None), ("num_used_tokens", 42)]
        )
        self.assertEqual(d, {"dp_rank": 0, "num_used_tokens": 42})

    def test_strips_timestamp_key(self):
        d = _loads_dict_factory(
            [("dp_rank", 1), ("timestamp", 1234.5), ("num_used_tokens", 0)]
        )
        self.assertNotIn("timestamp", d)
        self.assertEqual(d["dp_rank"], 1)

    def test_keeps_zero_values(self):
        d = _loads_dict_factory([("num_running_reqs", 0), ("token_usage", 0.0)])
        self.assertEqual(d, {"num_running_reqs": 0, "token_usage": 0.0})


if __name__ == "__main__":
    unittest.main()
