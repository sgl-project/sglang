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


class TestPromptInputs(unittest.TestCase):
    def inputs(self, order, seed=17):
        class Collected(Exception):
            pass

        seen = [[], []]

        async def skip_warmup(*args, **kwargs):
            pass

        async def collect(args, tp, prompt):
            for rank in order:
                seen[rank].append(prompt(rank))
            raise Collected

        args = SimpleNamespace(
            seed=seed,
            new_tokens=4,
            context=16,
            dp=2,
            num_gpus=2,
            mode="prefill",
            no_flush=False,
            sample_gpu=False,
        )
        tp = SimpleNamespace(
            server_info=lambda: {"tp_size": 2, "dp_size": 1, "attn_dp_size": 2},
            flush_cache=lambda: None,
            run=asyncio.run,
        )
        with (
            patch.object(bench, "batch", skip_warmup),
            patch.object(bench, "prefill", collect),
            self.assertRaises(Collected),
        ):
            bench.run_benchmark(args, tp)
        return seen

    def test_rank_inputs_do_not_depend_on_interleaving(self):
        # Each rank's semaphore can admit its next request first. A shared RNG
        # must not change which suffix that rank receives between A/B runs.
        self.assertEqual(
            self.inputs([0, 1, 0, 1, 0, 1]),
            self.inputs([1, 1, 0, 1, 0, 0]),
        )

    def test_prefix_is_shared_and_suffixes_are_distinct(self):
        seen = self.inputs([0, 1, 0, 1, 0, 1])
        for prompts in seen:
            self.assertTrue(all(p[:12] == prompts[0][:12] for p in prompts))
            self.assertEqual(len({tuple(p[12:]) for p in prompts}), 3)
        self.assertNotEqual(seen[0][0], seen[1][0])
        self.assertNotEqual(seen, self.inputs([0, 1, 0, 1, 0, 1], seed=18))


class TestInFlightDepth(unittest.IsolatedAsyncioTestCase):
    async def test_each_rank_respects_the_explicit_depth(self):
        for steps, per_step in ((8, 1), (32, 1), (8, 8), (32, 8)):
            with self.subTest(steps=steps, per_step=per_step):
                limit = steps * per_step + 1
                release = asyncio.Event()
                saturated = asyncio.Event()
                active, peak = [0, 0], [0, 0]

                class Transport:
                    async def generate(self, ids, params, rank, stream):
                        active[rank] += 1
                        peak[rank] = max(peak[rank], active[rank])
                        if active == [limit, limit]:
                            saturated.set()
                        try:
                            await release.wait()
                            yield 1.0, {"prompt_tokens": 2, "cached_tokens": 1}
                        finally:
                            active[rank] -= 1

                args = SimpleNamespace(
                    tokens_per_step=per_step,
                    new_tokens=1,
                    in_flight_steps=steps,
                    dp=2,
                    timeout=5,
                )
                done = []
                task = asyncio.create_task(
                    bench.batch(
                        args, Transport(), lambda r: [], limit + 3, {}, done, "test"
                    )
                )
                try:
                    await asyncio.wait_for(saturated.wait(), timeout=2)
                    # Pending requests must not acquire either rank's semaphore.
                    await asyncio.sleep(0)
                    self.assertEqual(active, [limit, limit])
                finally:
                    release.set()
                    await task
                self.assertEqual(peak, [limit, limit])
                self.assertEqual(active, [0, 0])
                self.assertEqual(len(done), 2 * (limit + 3))


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
        t0, t1 = bench.group_window(groups, 0.2, 0.9)
        self.assertAlmostEqual(t1 - t0, 21 * 0.1)  # 21 steps; step 10 is merged

    def test_no_whole_step_fails(self):
        with self.assertRaises(SystemExit):
            bench.group_window([(1.0, 10)], 0.2, 0.9)


class TestCompletionDiagnostics(unittest.TestCase):
    def test_raw_written_on_request_or_scoring_failure(self):
        for request_failure in (False, True):
            with (
                self.subTest(request_failure=request_failure),
                tempfile.TemporaryDirectory() as tmp,
            ):
                path = Path(tmp) / "raw.json"
                args = SimpleNamespace(
                    new_tokens=2,
                    tokens_per_step=4,
                    dp=1,
                    requests=2,
                    warmup_steps=0,
                    lo=0.2,
                    hi=0.9,
                    prefill_completions=str(path),
                )

                async def fake_batch(
                    args, tp, prompt, n, params, done, phase, **kwargs
                ):
                    if phase == "measured":
                        done.append((1.01, 1.0, 2, 0))
                        if request_failure:
                            raise RuntimeError("request failed")

                with patch.object(bench, "batch", fake_batch):
                    with self.assertRaises((SystemExit, RuntimeError)):
                        asyncio.run(bench.prefill(args, None, None))
                raw = json.loads(path.read_text())
                self.assertEqual(raw["requests_drained"], not request_failure)
                self.assertEqual(raw["validation"], "not_evaluated")
                self.assertEqual(raw["completions"], [[1.01, 1.0, 2, 0]])


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
        def check(ranks, sizes=(64,), coverage=0.9):
            return bench.check_logs(
                dict(
                    ms_per_step_per_rank=ranks,
                    sizes_seen=list(sizes),
                    complete=coverage >= 0.8,
                ),
                {"ms_per_step": 100},
                64,
                2,
            )

        self.assertTrue(check({0: 100, 1: 100})["agree"])
        self.assertFalse(check({0: 100})["agree"])
        self.assertFalse(check({0: 100, 1: 100}, coverage=0.7)["agree"])
        self.assertFalse(check({0: 100, 1: 100}, sizes=(32, 64))["agree"])
        self.assertFalse(check({0: 90, 1: 90})["agree"])
        self.assertFalse(check({}, sizes=(), coverage=0)["agree"])

    def test_blurred_decode_steps(self):
        self.assertFalse(bench.steps_blurred([100, 95, 105, 70]))
        self.assertTrue(bench.steps_blurred([100, 95, 105, 50]))

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
            ["--in-flight-steps", "0"],
            ["--in-flight-steps", "-1"],
            ["--in-flight-steps", "1.5"],
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

        args = SimpleNamespace(
            tokens_per_step=1, new_tokens=1, in_flight_steps=8, dp=1, timeout=0.01
        )
        with self.assertRaises(asyncio.TimeoutError):
            asyncio.run(
                bench.batch(args, StalledTransport(), lambda rank: [], 2, {}, [], "x")
            )
        self.assertEqual(len(cancelled), 2)

    def run_prefill(self, dp, tp_size, cached, new, budget, offset=0.0):
        """Each rank completes two requests every 100 ms, `offset` s behind the
        previous rank; returns the JSON record and whether the transport closed."""

        class Transport:
            def __init__(self):
                self.count = [0] * dp
                self.closed = False

            def server_info(self):
                return dict(tp_size=tp_size, dp_size=dp)

            def flush_cache(self):
                return None

            async def generate(self, ids, params, rank, stream):
                self.count[rank] += 1
                stamp = 1000 + (self.count[rank] // 2) * 0.1 + offset * rank
                meta = dict(
                    prompt_tokens=cached + new,
                    cached_tokens=cached,
                    prefill_finished_time=stamp,
                )
                yield stamp, meta

            def run(self, coro):
                return asyncio.run(coro)

            def finish(self):
                self.closed = True

        transport = Transport()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            argv = ["bench", "prefill", "--dp", str(dp), "--base-url", "http://x"]
            argv += ["--context", str(cached + new), "--new-tokens", str(new)]
            argv += ["--tokens-per-step", str(budget), "--requests", "80"]
            argv += ["--warmup-steps", "1", "--out", str(output)]
            with (
                patch.object(sys, "argv", argv),
                patch.object(bench, "HttpTransport", return_value=transport),
                patch.object(bench, "environment", return_value={}),
            ):
                bench.main()
            return json.loads(output.read_text()), transport.closed

    def test_prefill_steps_short_of_the_budget(self):
        # Two 2-token requests per step against a 6-token budget, as when replayed
        # tokens share it: steps come from boundaries, not tokens / B.
        record, _ = self.run_prefill(dp=1, tp_size=1, cached=7, new=2, budget=6)
        self.assertAlmostEqual(record["result"]["ms_per_step"], 100.0)
        self.assertEqual(record["result"]["new_tokens_per_step"], 4)

    def test_prefill_steps_with_offset_ranks(self):
        # Rank 1 runs 45 ms behind rank 0: a step must not split into one group per
        # rank.
        record, _ = self.run_prefill(
            dp=2, tp_size=1, cached=7, new=2, budget=4, offset=0.045
        )
        self.assertAlmostEqual(record["result"]["ms_per_step"], 100.0)
        self.assertEqual(record["result"]["new_tokens_per_step"], 4)

    def test_prefill_record_uses_server_gpu_count(self):
        record, closed = self.run_prefill(dp=1, tp_size=4, cached=7, new=1, budget=2)
        self.assertTrue(closed)
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
            in_flight_steps=8,
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
        self.assertEqual(record["trigger_completions_per_rank"], tp.at_trigger)
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

    async def test_profile_does_not_change_timed_result(self):
        plain_args = self.args(profile_steps=0)
        plain = await bench.prefill(
            plain_args, self.transport(plain_args), lambda r: []
        )
        prof_args = self.args()
        prof = await bench.prefill(prof_args, self.transport(prof_args), lambda r: [])
        profile_record = prof.pop("profile")
        self.assertEqual(prof, plain)
        self.assertGreaterEqual(min(profile_record["trigger_completions_per_rank"]), 80)

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


def run_main(transport, *extra):
    """bench.main() over `transport`; returns the JSON record."""
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "result.json"
        argv = ["bench", "prefill", "--dp", "1", "--base-url", "http://localhost"]
        argv += ["--context", "9", "--new-tokens", "2", "--tokens-per-step", "4"]
        argv += ["--requests", "80", "--warmup-steps", "1", "--out", str(output)]
        with (
            patch.object(sys, "argv", argv + list(extra)),
            patch.object(bench, "HttpTransport", return_value=transport),
            patch.object(bench, "environment", return_value={}),
        ):
            bench.main()
        return json.loads(output.read_text())


class LinkerServer:
    """A linker server: every request inserts its prefix into the device tree and
    writes it to the store; a flush empties the device tree but not the store."""

    def __init__(self, prefix_len=7, linker=True, flush_drops_device=True):
        self.prefix_len, self.linker = prefix_len, linker
        self.flush_drops_device = flush_drops_device
        self.device, self.store, self.flushes, self.sent = set(), set(), 0, []
        self.store_limit = None
        self.inputs = []
        self.profile_starts = []

    def server_info(self):
        return dict(
            tp_size=1,
            dp_size=1,
            enable_unified_cache_external_linker=self.linker,
            unified_cache_external_linker_backend="mooncake",
        )

    def flush_cache(self):
        self.flushes += 1
        if self.flush_drops_device:
            self.device.clear()

    async def generate(self, ids, params, rank, stream):
        prefix, n = tuple(ids[: self.prefix_len]), self.prefix_len
        if prefix in self.device:
            cached, tiers = n, dict(device=n, host=0)
        elif prefix in self.store:  # the linker reports an L3 load as a host hit
            cached, tiers = n, dict(device=0, host=n)
        else:
            cached, tiers = 0, None
        self.sent.append((self.flushes, prefix))
        self.inputs.append(tuple(ids))
        self.device.add(prefix)
        if self.store_limit is None or len(self.store) < self.store_limit:
            self.store.add(prefix)
        stamp = 1000 + len(self.sent) // 2 * 0.1
        meta = dict(
            prompt_tokens=len(ids), cached_tokens=cached, prefill_finished_time=stamp
        )
        yield stamp, meta | ({"cached_tokens_details": tiers} if tiers else {})

    def run(self, coro):
        return asyncio.run(coro)

    def finish(self):
        pass

    async def start_profile(self, body):
        self.profile_starts.append(body)

    async def stop_profile(self):
        pass


class TestPrefixFromL3(unittest.TestCase):
    def test_check_cache_requires_a_whole_off_device_prefix(self):
        meta = dict(prompt_tokens=9, cached_tokens=7)
        for tiers in (dict(device=0, host=7), dict(device=0, host=0, storage=7)):
            self.assertEqual(
                bench.check_cache(meta | {"cached_tokens_details": tiers}, 2, True), 2
            )
        # Negative controls: a device hit, a partial restore, no tier report
        for tiers in (dict(device=7, host=0), dict(device=3, host=4), None):
            with self.assertRaisesRegex(RuntimeError, "restored from L3"):
                bench.check_cache(meta | {"cached_tokens_details": tiers}, 2, True)
        # The device-prefix mode does not look at tiers
        self.assertEqual(bench.check_cache(meta, 2), 2)

    def test_every_measured_request_restores_its_own_prefix(self):
        server = LinkerServer()
        record = run_main(server, "--prefix-source", "l3")
        written = [p for f, p in server.sent if f == 1]
        measured = [p for f, p in server.sent if f == 2]
        self.assertEqual(server.flushes, 2)  # before writing, then the device only
        self.assertEqual(len(written), 1 * 2 + 80)  # warmup steps + measured requests
        self.assertEqual(measured, written)  # in order, each sent once
        n = len(written)
        self.assertTrue(
            all(a[-2:] != b[-2:] for a, b in zip(server.inputs[:n], server.inputs[n:]))
        )
        self.assertEqual(len(set(measured)), len(measured))
        self.assertAlmostEqual(record["result"]["ms_per_step"], 100.0)
        self.assertEqual(record["workload"]["prefix_source"], "l3")

    def test_warmup_and_profile_prefixes_are_also_prepared_once(self):
        server = LinkerServer()
        record = run_main(server, "--prefix-source", "l3", "--profile-steps", "3")
        written = [p for f, p in server.sent if f == 1]
        restored = [p for f, p in server.sent if f == 2]
        self.assertEqual(len(written), 2 + 80 + 80)
        self.assertEqual(written, restored)
        self.assertEqual(len(set(restored)), len(restored))
        self.assertEqual(len(server.profile_starts), 1)
        self.assertIn("profile", record["result"])

    def test_decode_mode_rejects_l3_before_transport(self):
        argv = [
            "bench",
            "decode",
            "--dp",
            "1",
            "--base-url",
            "http://localhost",
            "--prefix-source",
            "l3",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(bench, "HttpTransport") as transport,
        ):
            with self.assertRaises(SystemExit):
                bench.main()
            transport.assert_not_called()

    def test_same_seed_reproduces_inputs_with_fresh_stores(self):
        first, second = LinkerServer(), LinkerServer()
        run_main(first, "--prefix-source", "l3", "--seed", "71")
        run_main(second, "--prefix-source", "l3", "--seed", "71")
        self.assertEqual(first.sent, second.sent)
        self.assertEqual(first.inputs, second.inputs)
        third = LinkerServer()
        run_main(third, "--prefix-source", "l3", "--seed", "72")
        self.assertNotEqual(first.sent, third.sent)

    def test_rank_submission_order_does_not_change_l3_inputs(self):
        def collect(order):
            args = SimpleNamespace(
                new_tokens=2,
                prefix_source="l3",
                seed=71,
                dp=2,
                context=9,
                num_gpus=2,
                mode="prefill",
                no_flush=False,
                warmup_steps=1,
                tokens_per_step=4,
                requests=4,
                profile_steps=0,
                sample_gpu=False,
            )
            records = {"write": {0: [], 1: []}, "read": {0: [], 1: []}}
            tp = SimpleNamespace(
                server_info=lambda: dict(
                    tp_size=2,
                    dp_size=1,
                    attn_dp_size=2,
                    enable_unified_cache_external_linker=True,
                    unified_cache_external_linker_backend="mooncake",
                ),
                flush_cache=lambda: None,
                run=asyncio.run,
            )

            async def write(args, tp, prompt, n, params, done, what):
                for rank in order:
                    records["write"][rank] = [prompt(rank) for _ in range(n)]

            async def read(args, tp, prompt):
                for rank in reversed(order):
                    records["read"][rank] = [prompt(rank) for _ in range(6)]
                raise RuntimeError("inputs collected")

            with (
                patch.object(bench, "batch", side_effect=write),
                patch.object(bench, "prefill", side_effect=read),
            ):
                with self.assertRaisesRegex(RuntimeError, "inputs collected"):
                    bench.run_benchmark(args, tp)
            return records

        first, second = collect([0, 1]), collect([1, 0])
        self.assertEqual(first, second)
        for rank in (0, 1):
            for written, restored in zip(first["write"][rank], first["read"][rank]):
                self.assertEqual(written[:-2], restored[:-2])
                self.assertNotEqual(written[-2:], restored[-2:])

    def test_device_hits_fail(self):
        # Negative control: a flush that keeps the device tree serves device hits
        with self.assertRaisesRegex(RuntimeError, "restored from L3"):
            run_main(LinkerServer(flush_drops_device=False), "--prefix-source", "l3")

    def test_prefixes_missing_from_the_store_fail(self):
        server = LinkerServer()
        server.store_limit = 40
        with self.assertRaisesRegex(RuntimeError, "expected 2 uncached"):
            run_main(server, "--prefix-source", "l3")

    def test_other_linker_backend_fails_before_sending(self):
        server = LinkerServer()
        info = server.server_info() | {"unified_cache_external_linker_backend": "other"}
        with patch.object(server, "server_info", return_value=info):
            with self.assertRaisesRegex(SystemExit, "Mooncake"):
                run_main(server, "--prefix-source", "l3")
        self.assertEqual(server.sent, [])

    def test_failed_device_flush_fails_before_restore(self):
        server = LinkerServer()
        flush = server.flush_cache

        def fail_second_flush():
            if server.flushes == 1:
                return "still busy"
            return flush()

        with patch.object(server, "flush_cache", side_effect=fail_second_flush):
            with self.assertRaisesRegex(SystemExit, "device cache"):
                run_main(server, "--prefix-source", "l3")
        self.assertTrue(server.sent)
        self.assertTrue(all(f == 1 for f, _ in server.sent))

    def test_server_without_the_linker_fails_before_sending(self):
        server = LinkerServer(linker=False)
        with self.assertRaisesRegex(SystemExit, "external-linker"):
            run_main(server, "--prefix-source", "l3")
        self.assertEqual(server.sent, [])

    def test_unsupported_modes_fail_before_generation(self):
        for flags in (
            ("--prefix-source", "other"),
            ("--prefix-source", "l3", "--no-flush"),
            ("--prefix-source", "l3", "--context", "2"),
        ):
            server = LinkerServer()
            with self.assertRaises(SystemExit):
                run_main(server, *flags)
            self.assertEqual(server.sent, [])

    def test_storage_tier_cannot_fake_a_partial_prefix(self):
        for tiers in (
            {"device": 0, "host": 6},
            {"device": 0, "storage": 6},
            {"host": 7},
        ):
            with self.assertRaisesRegex(RuntimeError, "restored from L3"):
                bench.check_cache(
                    dict(prompt_tokens=9, cached_tokens=7, cached_tokens_details=tiers),
                    2,
                    True,
                )

    def test_in_flight_depth_is_recorded_and_default_is_unchanged(self):
        default = run_main(LinkerServer())
        explicit = run_main(LinkerServer(), "--in-flight-steps", "32")
        self.assertEqual(default["workload"]["in_flight_steps"], 8)
        self.assertEqual(explicit["workload"]["in_flight_steps"], 32)
        self.assertEqual(default["result"], explicit["result"])

    def test_device_mode_keeps_one_shared_prefix(self):
        server = LinkerServer()
        run_main(server)
        self.assertEqual(server.flushes, 1)
        self.assertEqual(len({p for _, p in server.sent}), 1)


if __name__ == "__main__":
    unittest.main()
