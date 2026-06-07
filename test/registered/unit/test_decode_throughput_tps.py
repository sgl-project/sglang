"""Unit tests for the per-request decode-throughput (new TPS) definition.

The client SLO bar is a per-request decode throughput of >= 30 tok/s, defined as
``output_tokens / (e2e_latency - ttft)`` — output tokens over the post-first-token
decode wall-time. These tests pin that formula, prove it is DISTINCT from the
legacy ``output_tokens / e2e`` definition, and cross-lock the two implementations
(the shipped ``sglang.bench_serving`` one and the dependency-free copy in the
closed-batch probe ``development/loop7/perf_closed_batch.py``) so they cannot
drift apart.

    python -m pytest test/registered/unit/test_decode_throughput_tps.py -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

from sglang.bench_serving import decode_throughput_tps


def _load_perf_closed_batch():
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "development",
        "loop7",
        "perf_closed_batch.py",
    )
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("_perf_closed_batch", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_perf_closed_batch"] = mod  # register before exec (importlib contract)
    spec.loader.exec_module(mod)
    return mod


class TestDecodeThroughputTps(unittest.TestCase):
    def test_basic_formula(self):
        # 100 tokens, 5.0s e2e, 1.0s ttft -> 100 / (5 - 1) = 25 tok/s.
        self.assertAlmostEqual(decode_throughput_tps(100, 5.0, 1.0), 25.0)

    def test_meets_30_tps_bar(self):
        # 512 OSL, e2e 8.0s, ttft 1.0s -> 512 / 7 ~= 73.1 tok/s (>= 30).
        self.assertGreaterEqual(decode_throughput_tps(512, 8.0, 1.0), 30.0)

    def test_distinct_from_old_definition(self):
        # The old (rejected) definition output_tokens / e2e double-counts the
        # prefill/TTFT time in the denominator and reports a LOWER rate. They
        # must differ whenever ttft > 0.
        ct, e2e, ttft = 200, 10.0, 4.0
        new = decode_throughput_tps(ct, e2e, ttft)  # 200 / 6 ~= 33.3
        old = ct / e2e  # 200 / 10 = 20.0
        self.assertNotAlmostEqual(new, old)
        self.assertGreater(new, old)  # excluding TTFT raises the decode rate

    def test_zero_when_no_decode_window(self):
        # ttft == e2e (single-token / non-streaming): unmeasurable, not zero-rate.
        self.assertEqual(decode_throughput_tps(50, 3.0, 3.0), 0.0)
        # ttft > e2e (clock noise): also unmeasurable.
        self.assertEqual(decode_throughput_tps(50, 3.0, 3.5), 0.0)

    def test_zero_when_no_tokens(self):
        self.assertEqual(decode_throughput_tps(0, 5.0, 1.0), 0.0)

    def test_cross_locked_with_perf_probe(self):
        # The probe keeps a stdlib-only copy; it must compute the SAME value as
        # the shipped helper for a spread of inputs.
        mod = _load_perf_closed_batch()
        cases = [
            (100, 5.0, 1.0),
            (512, 8.0, 1.0),
            (1, 2.0, 1.9),
            (256, 4.0, 4.0),  # zero window -> both 0.0
            (0, 5.0, 1.0),  # no tokens -> both 0.0
        ]
        for ct, e2e, ttft in cases:
            with self.subTest(ct=ct, e2e=e2e, ttft=ttft):
                self.assertAlmostEqual(
                    decode_throughput_tps(ct, e2e, ttft),
                    mod._decode_tps(ct, e2e, ttft),
                )


if __name__ == "__main__":
    unittest.main()
