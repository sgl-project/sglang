import importlib.util
import unittest
from datetime import datetime
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PATH = (
    Path(__file__).resolve().parents[4] / "benchmark/engine_steps/bench_engine_steps.py"
)
_spec = importlib.util.spec_from_file_location("bench_engine_steps", _PATH)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


class TestStepWindow(CustomTestCase):
    def test_window_spans_whole_steps(self):
        # 30 steps of 100 ms with two completions each; a 1 s stall precedes step 3,
        # and step 10 reports late, so its group merges into step 11's and the step
        # count must come from items.
        times = []
        for k in range(30):
            end = 0.03 + 0.1 * k + (1.0 if k >= 3 else 0.0) + (0.098 if k == 10 else 0)
            times += [end, end + 0.0005]
        groups = bench.step_groups(times, 2)
        self.assertEqual(len(groups), 29)
        self.assertEqual(sum(c for _, c in groups), 60)
        t0, t1, reqs, group_ms = bench.group_window(groups, 0.2, 0.9)
        self.assertEqual((reqs, len(group_ms)), (42, 20))
        self.assertAlmostEqual(sum(group_ms), (t1 - t0) * 1000)
        self.assertAlmostEqual((t1 - t0) / (reqs / 2), 0.1)

    def test_no_whole_step_fails(self):
        with self.assertRaises(SystemExit):
            bench.group_window([(1.0, 10)], 0.2, 0.9)


class TestLogSteps(CustomTestCase):
    def test_prefill_and_decode_lines(self):
        t0 = datetime(2026, 10, 22, 10, 0, 5).timestamp()
        stamp = "[2026-10-22 10:00:0{} {}] "
        prefill = (
            "Prefill batch, #new-seq: 2, #new-token: {}, #cached-token: 253952, "
            "#running-req: 0, cuda graph: False, input throughput (token/s): {}"
        )
        decode = (
            "Decode batch, #running-req: 64, token usage: 0.50, {}cuda graph: True, "
            "gen throughput (token/s): {}"
        )
        # A line counts when its interval, from the rank's previous line, is inside
        # the window; whole-second stamps may run up to a second late.
        lines = [
            stamp.format(3, "DP0 TP0 EP0") + prefill.format(8192, "10000.00"),
            stamp.format(5, "DP0 TP0 EP0") + prefill.format(8192, "10000.00"),
            stamp.format(5, "DP0 TP0 EP0") + prefill.format(8192, "40960.00"),
            stamp.format(5, "DP1 TP1 EP1") + prefill.format(8192, "20000.00"),
            stamp.format(6, "DP1 TP1 EP1") + prefill.format(8192, "50000.00"),
            stamp.format(6, "DP0 TP0 EP0") + prefill.format(4096, "40960.00"),
            "[2026-10-22 10:00:06.100 DP2 TP2 EP2] " + prefill.format(8192, "1.00"),
            "[2026-10-22 10:00:06.200 DP2 TP2 EP2] " + prefill.format(8192, "81920.00"),
            stamp.format(9, "DP0 TP0 EP0") + prefill.format(8192, "10000.00"),
            stamp.format(5, "DP0 TP0 EP0") + decode.format("", "1000.00"),
            stamp.format(5, "DP0 TP0 EP0")
            + decode.format("accept len: 2.50, block accept len: 3.00, ", "4000.00"),
            stamp.format(6, "TP0") + decode.format("", "3200.00"),
        ]
        got = bench.log_steps(lines, "prefill", t0, t0 + 2, 8192)
        self.assertEqual(got["ms_per_step_per_rank"], {0: 200.0, 1: 163.84, 2: 100.0})
        self.assertEqual(got["sizes_seen"], [4096, 8192])
        self.assertAlmostEqual(got["coverage"], 0.101 / 2)  # rank 2: 6.1 s to 6.201 s
        got = bench.log_steps(lines, "decode", t0, t0 + 2, 64)
        self.assertEqual(got["ms_per_step_per_rank"].keys(), {0})
        self.assertAlmostEqual(got["ms_per_step_per_rank"][0], (40.0 + 20.0) / 2)
        self.assertEqual(got["accept_len"], 2.5)
        got = bench.log_steps(lines, "decode", t0, t0 + 2, 63)
        self.assertEqual((got["ms_per_step_per_rank"], got["sizes_seen"]), ({}, [64]))

    def test_prefill_lines_at_step_ends(self):
        # Step ends at 10:00:05.000, .100, .200 and .300; each rank logs within 1 ms.
        t0 = datetime(2026, 10, 22, 10, 0, 5).timestamp()
        line = (
            "[2026-10-22 10:00:05.{:03d} DP{} TP0 EP0] Prefill batch, #new-seq: 2, "
            "#new-token: 8192, #cached-token: 0, input throughput (token/s): 1.00"
        )
        lines = [line.format(ms, r) for ms in (0, 100, 200, 301) for r in (0, 1)]
        got = bench.log_steps(lines, "prefill", t0, t0 + 0.3, 8192, tol=0.02)
        self.assertTrue(got["complete"])
        self.assertAlmostEqual(got["ms_per_step_per_rank"][1], 301 / 3, places=3)
        got = bench.log_steps(lines[:-1], "prefill", t0, t0 + 0.3, 8192, tol=0.02)
        self.assertFalse(got["complete"])  # rank 1 has no line at the last step end


if __name__ == "__main__":
    unittest.main()
