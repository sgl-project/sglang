import asyncio
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PATH = (
    Path(__file__).resolve().parents[4] / "benchmark/engine_steps/bench_engine_steps.py"
)
_spec = importlib.util.spec_from_file_location("bench_engine_steps", _PATH)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


class TestStepWindow(unittest.TestCase):
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

    def test_empty_or_invalid_window_fails(self):
        for groups, lo, hi in (([], 0.2, 0.9), ([(1, 1)], 0.9, 0.2)):
            with self.assertRaises(SystemExit):
                bench.group_window(groups, lo, hi)


class TestLogSteps(unittest.TestCase):
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

    def test_cross_check_requires_all_ranks_and_fixed_batch(self):
        def check(ranks, sizes=(64,), coverage=0.9, blurred=False):
            return bench.check_logs(
                dict(
                    ms_per_step_per_rank=ranks,
                    sizes_seen=list(sizes),
                    complete=coverage >= 0.8,
                ),
                {"ms_per_step": 100},
                64,
                2,
                blurred,
            )

        self.assertTrue(check({0: 100, 1: 100})["agree"])
        self.assertFalse(check({0: 100})["agree"])
        self.assertFalse(check({0: 100, 1: 100}, coverage=0.7)["agree"])
        self.assertFalse(check({0: 100, 1: 100}, sizes=(32, 64))["agree"])
        self.assertFalse(check({0: 90, 1: 90})["agree"])
        self.assertFalse(check({}, sizes=(), coverage=0)["agree"])
        self.assertFalse(check({0: 100, 1: 100}, blurred=True)["agree"])


class TestWorkloadGuards(unittest.TestCase):
    def test_gpu_denominator(self):
        self.assertEqual(bench.gpu_count(dict(tp_size=4, dp_size=1), 1), 4)
        self.assertEqual(bench.gpu_count(dict(tp_size=2, dp_size=4), 4), 8)
        self.assertEqual(
            bench.gpu_count(dict(tp_size=8, pp_size=2, dp_size=1, attn_dp_size=4), 4),
            16,
        )
        self.assertEqual(
            bench.gpu_count(
                dict(tp_size=8, pp_size=2, dp_size=4, enable_dp_attention=True), 4
            ),
            16,
        )
        self.assertEqual(bench.gpu_count({}, 4, explicit=8), 8)
        with self.assertRaises(SystemExit):
            bench.gpu_count({}, 4)
        with self.assertRaises(SystemExit):
            bench.gpu_count(dict(tp_size=8, dp_size=1, attn_dp_size=4), 2)

    def test_cache_requires_exact_suffix(self):
        self.assertEqual(
            bench.check_cache(dict(prompt_tokens=128, cached_tokens=96), 32), 32
        )
        for cached in (0, 112):
            with self.assertRaisesRegex(RuntimeError, "expected 32"):
                bench.check_cache(dict(prompt_tokens=128, cached_tokens=cached), 32)

    def test_transport_closes_if_guard_fails(self):
        transport = Mock()
        argv = ["bench", "prefill", "--dp", "1", "--base-url", "http://localhost"]
        with (
            patch.object(sys, "argv", argv),
            patch.object(bench, "HttpTransport", return_value=transport),
            patch.object(bench, "run_benchmark", side_effect=RuntimeError("guard")),
        ):
            with self.assertRaisesRegex(RuntimeError, "guard"):
                bench.main()
        transport.finish.assert_called_once()

    def test_cli_rejects_invalid_workload(self):
        common = ["bench", "prefill", "--dp", "1", "--base-url", "http://localhost"]
        for extra in (
            ["--new-tokens", "0"],
            ["--num-gpus", "0"],
            ["--timeout", "nan"],
            ["--context", "1"],
            ["--tokens-per-step", "1000"],
            ["--lo", "0.9", "--hi", "0.2"],
        ):
            with patch.object(sys, "argv", common + extra):
                with self.assertRaises(SystemExit):
                    bench.main()

    def test_prefill_timeout_cancels_pending_requests(self):
        cancelled = []

        class StalledTransport:
            async def generate(self, *args):
                try:
                    await asyncio.Event().wait()
                    yield None
                finally:
                    cancelled.append(True)

        args = SimpleNamespace(tokens_per_step=1, new_tokens=1, dp=1, timeout=0.01)
        with self.assertRaises(asyncio.TimeoutError):
            asyncio.run(
                bench.batch(args, StalledTransport(), lambda rank: [], 2, {}, [], "x")
            )
        self.assertEqual(len(cancelled), 2)

    def test_prefill_record_uses_server_gpu_count(self):
        class Transport:
            def __init__(self):
                self.count = 0
                self.closed = False

            def server_info(self):
                return dict(tp_size=4, dp_size=1)

            def flush_cache(self):
                return None

            async def generate(self, ids, params, rank, stream):
                self.count += 1
                stamp = 1000 + (self.count // 2) * 0.1
                yield (
                    stamp,
                    dict(prompt_tokens=8, cached_tokens=7, prefill_finished_time=stamp),
                )

            def run(self, coro):
                return asyncio.run(coro)

            def finish(self):
                self.closed = True

        transport = Transport()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            argv = [
                "bench",
                "prefill",
                "--dp",
                "1",
                "--base-url",
                "http://localhost",
                "--context",
                "8",
                "--new-tokens",
                "1",
                "--tokens-per-step",
                "2",
                "--requests",
                "80",
                "--warmup-steps",
                "1",
                "--out",
                str(output),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(bench, "HttpTransport", return_value=transport),
                patch.object(bench, "environment", return_value={}),
            ):
                bench.main()
            record = json.loads(output.read_text())
        self.assertTrue(transport.closed)
        self.assertEqual(record["workload"]["num_gpus"], 4)
        self.assertAlmostEqual(record["result"]["tok_s_per_gpu"], 5)

    def test_decode_does_not_count_metadata_only_chunks_as_steps(self):
        class Transport:
            async def generate(self, *args):
                for count in range(1, 10000):
                    meta = dict(
                        prompt_tokens=8, cached_tokens=7, completion_tokens=count
                    )
                    await asyncio.sleep(0.002)
                    yield time.time(), meta
                    await asyncio.sleep(0.002)
                    yield time.time(), meta

        args = SimpleNamespace(
            dp=1,
            batch_per_rank=1,
            new_tokens=1,
            output_len=10000,
            timeout=2,
            warmup_steps=0,
            measure_steps=5,
            profile_steps=0,
        )
        result = asyncio.run(bench.decode(args, Transport(), lambda rank: []))
        self.assertEqual(result["tokens"], 5)


if __name__ == "__main__":
    unittest.main()
