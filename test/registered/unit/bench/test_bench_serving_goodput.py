"""Unit tests for the goodput bench_serving reports under --goodput."""

import argparse
import unittest

from sglang.benchmark.goodput import calculate_goodput
from sglang.benchmark.serving import RequestFuncOutput, _goodput_slo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DURATION_S = 10.0


def _output(ttft, latency, output_len, success=True):
    return RequestFuncOutput(
        success=success, ttft=ttft, latency=latency, output_len=output_len
    )


# ttft / tpot / e2el in ms: 100 / 100 / 900, 500 / 100 / 900, 200 / 200 / 2200,
# 200 / 0 / 200 (a single token), then one failed request.
OUTPUTS = [
    _output(ttft=0.1, latency=0.9, output_len=9),
    _output(ttft=0.5, latency=0.9, output_len=5),
    _output(ttft=0.2, latency=2.2, output_len=11),
    _output(ttft=0.2, latency=0.2, output_len=1),
    _output(ttft=0.0, latency=0.0, output_len=0, success=False),
]


class TestBenchServingGoodput(CustomTestCase):
    def test_request_is_good_only_if_it_meets_every_listed_slo(self):
        all_three = {"ttft": 3 / 5, "tpot": 3 / 5, "e2el": 3 / 5}
        # (--goodput entries, good requests, attainment of each listed SLO alone)
        cases = [
            (["ttft:300", "tpot:150", "e2el:1000"], 2, all_three),
            (["ttft:300"], 3, {"ttft": 3 / 5}),
            # Dividing by output_len instead would give 88.9 and 80 ms, and pass.
            (["tpot:95"], 1, {"tpot": 1 / 5}),
        ]
        for entries, num_good, by_metric in cases:
            with self.subTest(entries=entries):
                goodput = calculate_goodput(
                    outputs=OUTPUTS,
                    goodput_slos=dict(map(_goodput_slo, entries)),
                    dur_s=DURATION_S,
                )
                self.assertEqual(goodput.request_goodput, num_good / DURATION_S)
                self.assertEqual(goodput.slo_attainment, num_good / len(OUTPUTS))
                self.assertEqual(goodput.slo_attainment_by_metric, by_metric)

    def test_invalid_slo_entries_are_rejected(self):
        for entry in ["itl:10", "ttft:0"]:
            with self.subTest(entry=entry):
                with self.assertRaises(argparse.ArgumentTypeError):
                    _goodput_slo(entry)


if __name__ == "__main__":
    unittest.main()
