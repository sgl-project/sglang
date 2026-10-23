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
        # The engine's input throughput counts replayed tokens with the new ones.
        replayed = [
            stamp.format(5, "DP0 TP0 EP0") + prefill.format(7936, "10000.00"),
            stamp.format(6, "DP0 TP0 EP0")
            + prefill.format(7936, "40960.00").replace(
                "cuda graph", "#replay-token: 256, cuda graph"
            ),
        ]
        got = bench.log_steps(replayed, "prefill", t0, t0 + 2, 7936)
        self.assertEqual(got["ms_per_step_per_rank"], {0: 200.0})
        # Without a size, the window's most common batch is matched.
        got = bench.log_steps(lines, "prefill", t0, t0 + 2, None)
        self.assertEqual(got["ms_per_step_per_rank"], {0: 200.0, 1: 163.84, 2: 100.0})

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

    def test_prefill_cross_check_takes_the_engine_batch(self):
        # Window edges cut chunked requests, so the measured new tokens per step only
        # approximate the engine's batch: one batch size within 1% of it agrees.
        def check(sizes):
            return bench.check_logs(
                dict(
                    ms_per_step_per_rank={0: 100}, sizes_seen=list(sizes), complete=True
                ),
                {"ms_per_step": 100, "new_tokens_per_step": 31793},
                None,
                1,
                False,
            )

        self.assertTrue(check([31744])["agree"])
        self.assertFalse(check([31744, 32768])["agree"])
        self.assertFalse(check([28672])["agree"])


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

    def test_prefill_steps_short_of_the_budget(self):
        # Steps of two 2-token requests every 100 ms against a 6-token budget, as
        # when replayed tokens share it: steps come from boundaries, not tokens / B.
        class Transport:
            def __init__(self):
                self.count = 0

            def server_info(self):
                return dict(tp_size=1, dp_size=1)

            def flush_cache(self):
                return None

            async def generate(self, ids, params, rank, stream):
                self.count += 1
                stamp = 1000 + (self.count // 2) * 0.1
                yield (
                    stamp,
                    dict(prompt_tokens=9, cached_tokens=7, prefill_finished_time=stamp),
                )

            def run(self, coro):
                return asyncio.run(coro)

            def finish(self):
                pass

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
                "9",
                "--new-tokens",
                "2",
                "--tokens-per-step",
                "6",
                "--requests",
                "80",
                "--warmup-steps",
                "1",
                "--out",
                str(output),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(bench, "HttpTransport", return_value=Transport()),
                patch.object(bench, "environment", return_value={}),
            ):
                bench.main()
            result = json.loads(output.read_text())["result"]
        self.assertAlmostEqual(result["ms_per_step"], 100.0)
        self.assertEqual(result["new_tokens_per_step"], 4)

    def test_prefill_steps_with_offset_ranks(self):
        # Two ranks step every 100 ms, rank 1 45 ms behind rank 0: a step must not
        # split into one group per rank.
        class Transport:
            def __init__(self):
                self.count = [0, 0]

            def server_info(self):
                return dict(tp_size=1, dp_size=2)

            def flush_cache(self):
                return None

            async def generate(self, ids, params, rank, stream):
                self.count[rank] += 1
                stamp = 1000 + (self.count[rank] // 2) * 0.1 + 0.045 * rank
                yield (
                    stamp,
                    dict(prompt_tokens=9, cached_tokens=7, prefill_finished_time=stamp),
                )

            def run(self, coro):
                return asyncio.run(coro)

            def finish(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            argv = ["bench", "prefill", "--dp", "2", "--base-url", "http://localhost"]
            argv += ["--context", "9", "--new-tokens", "2", "--tokens-per-step", "4"]
            argv += ["--requests", "80", "--warmup-steps", "1", "--out", str(output)]
            with (
                patch.object(sys, "argv", argv),
                patch.object(bench, "HttpTransport", return_value=Transport()),
                patch.object(bench, "environment", return_value={}),
            ):
                bench.main()
            result = json.loads(output.read_text())["result"]
        self.assertAlmostEqual(result["ms_per_step"], 100.0)
        self.assertEqual(result["new_tokens_per_step"], 4)

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


class TestPrefillProfiling(unittest.IsolatedAsyncioTestCase):
    def args(self, **changes):
        values = dict(
            tokens_per_step=2,
            new_tokens=1,
            dp=2,
            timeout=2,
            requests=400,
            lo=0.2,
            hi=0.9,
            warmup_steps=1,
            profile_steps=2,
            profile=dict(num_steps=2),
        )
        return SimpleNamespace(**(values | changes))

    def transport(self, args):
        class Transport:
            def __init__(self):
                self.completed = [0] * args.dp
                self.active = 0
                self.calls = 0
                self.stops = 0
                self.ack_started = asyncio.Event()
                self.delay_ack = False
                self.fail_peer = False
                self.fail_start = False

            async def generate(self, ids, params, rank, stream):
                self.active += 1
                try:
                    await asyncio.sleep(0)  # deterministic event-loop interleaving
                    if self.fail_peer and self.ack_started.is_set() and rank == 0:
                        raise ValueError("peer failed")
                    self.completed[rank] += 1
                    stamp = 1000 + self.completed[rank] // 2 * 0.1
                    yield (
                        stamp,
                        dict(
                            prompt_tokens=2,
                            cached_tokens=1,
                            prefill_finished_time=stamp,
                        ),
                    )
                finally:
                    self.active -= 1

            async def start_profile(self, body):
                self.calls += 1
                self.at_trigger = self.completed.copy()
                self.ack_started.set()
                if self.fail_start:
                    raise ValueError("profile start failed")
                if self.delay_ack:
                    await asyncio.Event().wait()
                await asyncio.sleep(0)
                self.at_ack = self.completed.copy()

            async def stop_profile(self):
                self.stops += 1

        return Transport()

    async def test_continuous_supply_and_all_rank_threshold(self):
        args = self.args()
        tp = self.transport(args)
        generate = tp.generate
        slow_rank_gate = asyncio.Event()

        async def unequal(ids, params, rank, stream):
            if rank == 1:
                await slow_rank_gate.wait()
            async for item in generate(ids, params, rank, stream):
                if tp.completed[0] >= 100:
                    slow_rank_gate.set()
                yield item

        tp.generate = unequal
        record = await bench.profile_prefill(args, tp, lambda r: [], {})
        self.assertEqual(tp.calls, 1)
        self.assertGreaterEqual(min(tp.at_trigger), 80)
        self.assertGreater(tp.at_trigger[0], tp.at_trigger[1])
        # The trigger is beyond the entire in-flight limit (17/rank), proving
        # the producer replenished it rather than waiting for a drained burst.
        self.assertGreater(min(tp.at_trigger), 17)
        self.assertGreater(sum(tp.at_ack), sum(tp.at_trigger))
        self.assertEqual(record["final_completions"], [400, 400])
        self.assertEqual(tp.active, 0)

    async def test_insufficient_budget_fails_before_sending(self):
        args = self.args(lo=0.99)
        tp = self.transport(args)
        with self.assertRaisesRegex(SystemExit, "profile needs"):
            await bench.profile_prefill(args, tp, lambda r: [], {})
        self.assertEqual(tp.completed, [0, 0])
        self.assertEqual(tp.calls, 0)

    async def test_peer_failure_cancels_callback_waiting_for_ack(self):
        args = self.args()
        tp = self.transport(args)
        tp.fail_peer = tp.delay_ack = True
        with self.assertRaisesRegex(ValueError, "peer failed"):
            await bench.profile_prefill(args, tp, lambda r: [], {})
        self.assertEqual(tp.active, 0)
        self.assertEqual(tp.stops, 1)
        self.assertFalse(
            [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        )

    async def test_profile_ack_timeout_cleans_up(self):
        args = self.args(timeout=0.03)
        tp = self.transport(args)
        tp.delay_ack = True
        with self.assertRaises(asyncio.TimeoutError):
            await bench.profile_prefill(args, tp, lambda r: [], {})
        self.assertEqual(tp.active, 0)
        self.assertEqual(tp.stops, 1)

    async def test_profile_start_failure_cleans_up(self):
        args = self.args()
        tp = self.transport(args)
        tp.fail_start = True
        with self.assertRaisesRegex(ValueError, "profile start failed"):
            await bench.profile_prefill(args, tp, lambda r: [], {})
        self.assertEqual(tp.active, 0)
        self.assertEqual(tp.stops, 1)

    async def test_profile_does_not_change_timed_result(self):
        plain_args = self.args(profile_steps=0)
        plain = await bench.prefill(
            plain_args, self.transport(plain_args), lambda r: []
        )
        prof_args = self.args()
        prof = await bench.prefill(prof_args, self.transport(prof_args), lambda r: [])
        profile_record = prof.pop("profile")
        self.assertEqual(prof, plain)
        self.assertEqual(profile_record["final_completions"], [400, 400])

    async def test_completions_survive_scoring_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "completions.json"
            args = self.args(profile_steps=0, prefill_completions=str(path))
            with patch.object(
                bench, "group_window", side_effect=SystemExit("rejected")
            ):
                with self.assertRaisesRegex(SystemExit, "rejected"):
                    await bench.prefill(args, self.transport(args), lambda r: [])
            record = json.loads(path.read_text())
            self.assertTrue(record["requests_drained"])
            self.assertEqual(len(record["completions"]), 800)
            self.assertEqual(record["validation"], "not_evaluated")

    async def test_partial_completions_survive_request_failure(self):
        async def fail_batch(args, tp, prompt, count, params, done, label, **kw):
            if label == "measured":
                done.append((1.0, 0.9, 1, 0))
                raise RuntimeError("request failed")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "completions.json"
            args = self.args(prefill_completions=str(path))
            with patch.object(bench, "batch", side_effect=fail_batch):
                with self.assertRaisesRegex(RuntimeError, "request failed"):
                    await bench.prefill(args, None, lambda r: [])
            record = json.loads(path.read_text())
            self.assertFalse(record["requests_drained"])
            self.assertEqual(record["completions"], [[1.0, 0.9, 1, 0]])

    async def test_completion_dump_does_not_change_result(self):
        plain_args = self.args(profile_steps=0)
        plain = await bench.prefill(
            plain_args, self.transport(plain_args), lambda r: []
        )
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(profile_steps=0, prefill_completions=tmp + "/samples.json")
            recorded = await bench.prefill(args, self.transport(args), lambda r: [])
        self.assertEqual(recorded, plain)

    async def test_delayed_ack_cannot_silently_profile_the_drain(self):
        args = self.args()
        tp = self.transport(args)
        start = tp.start_profile

        async def late_ack(body):
            await start(body)
            # Keep serving during the ACK, but exhaust the reserve on one rank.
            while min(tp.completed) < args.requests - 1:
                await asyncio.sleep(0)

        tp.start_profile = late_ack
        with self.assertRaisesRegex(RuntimeError, "profile burst is draining"):
            await bench.profile_prefill(args, tp, lambda r: [], {})
        self.assertEqual(tp.active, 0)
        self.assertEqual(tp.stops, 1)

    async def test_external_cancellation_joins_requests_and_stops_profile(self):
        args = self.args()
        tp = self.transport(args)
        tp.delay_ack = True
        task = asyncio.create_task(bench.profile_prefill(args, tp, lambda r: [], {}))
        await tp.ack_started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(tp.active, 0)
        self.assertEqual(tp.stops, 1)


if __name__ == "__main__":
    unittest.main()
