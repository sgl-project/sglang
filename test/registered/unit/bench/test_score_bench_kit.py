"""CPU-only workload, mocked HTTP accounting and load-generator regressions."""

import asyncio
import gc
import json
import math
import time
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import aiohttp
import msgspec

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.score_bench_kit import (
    Workload,
    build_requests,
    make_workload,
    run_load,
    validate_response,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def response_for(workload):
    row = [1 / len(workload.labels)] * len(workload.labels)
    item = [row] * workload.anchors_per_item if workload.anchors_per_item else row
    return {"scores": [item] * workload.items}


class TestScoreWorkload(CustomTestCase):
    def test_shared_tokenizer_validation(self):
        class Tokenizer:
            vocab_size = 248000
            all_special_ids = (100, 248047)
            tokens = {" yes": [9542], " no": [874], "<|object_ref_start|>": [248047]}

            def encode(self, text, **kwargs):
                return self.tokens[text]

            def __len__(self):
                return 248048

        tokenizer = Tokenizer()
        workload = make_workload(mode="setwise", tokenizer=tokenizer)
        self.assertEqual(workload.labels, (9542, 874))
        self.assertNotIn(100, workload.content_ids)
        tokenizer.all_special_ids = (9542,)
        with self.assertRaisesRegex(ValueError, "ordinary"):
            make_workload(mode="pointwise", tokenizer=tokenizer)
        tokenizer.all_special_ids = ()
        tokenizer.tokens = {**tokenizer.tokens, " yes": [248001]}
        with self.assertRaisesRegex(ValueError, "base-vocabulary"):
            make_workload(mode="pointwise", tokenizer=tokenizer)
        tokenizer.tokens = {**tokenizer.tokens, " yes": [1, 2]}
        with self.assertRaisesRegex(ValueError, "one ordinary"):
            make_workload(mode="pointwise", tokenizer=tokenizer)

    def test_matched_seeded_pairs_and_grouping(self):
        for mode in ("pointwise", "setwise"):
            workload = Workload(mode)
            requests = build_requests(workload, 33)
            with self.subTest(mode=mode):
                for count in (1, 16, 17):
                    self.assertEqual(requests[:count], build_requests(workload, count))
                self.assertEqual(
                    requests,
                    build_requests(
                        msgspec.structs.replace(workload, mode=mode + "-mis"), 33
                    ),
                )
                self.assertEqual(len({tuple(r["query"]) for r in requests[:16]}), 16)
                self.assertEqual(requests[16:], requests[:17])
                request = requests[0]
                self.assertEqual(len(request["query"]), 120)
                self.assertEqual(len(request["items"]), 2 if mode == "setwise" else 10)
                for item in request["items"]:
                    self.assertEqual(len(item), 905 if mode == "setwise" else 180)
                    self.assertEqual(
                        item.count(workload.anchor_id), workload.anchors_per_item
                    )
                validate_response(response_for(workload), workload)
        for mode in ("setwise", "setwise-mis"):
            with self.assertRaisesRegex(ValueError, "even"):
                Workload(mode, candidates=3)

    def test_content_exclusions_and_score_validation(self):
        workload = Workload(
            "pointwise",
            candidates=1,
            token_ids=(100, 101, 874, 9542, 9999, 248047),
            special_token_ids=(100,),
        )
        request = build_requests(workload, 1)[0]
        self.assertEqual(set(request["query"] + request["items"][0]), {101})
        for scores in (
            [],
            [[1]],
            [[True, 0]],
            [[math.nan, 0]],
            [[1.1, -0.1]],
            [[0, 0]],
            [[10**1000, 0]],
        ):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                validate_response({"scores": scores}, workload)
        with self.assertRaisesRegex(ValueError, "anchor"):
            validate_response({"scores": [[], []]}, Workload("setwise"))
        validate_response(
            {"scores": [[0.001, 0.002]]},
            msgspec.structs.replace(workload, apply_softmax=False),
        )


class TestScoreLoad(CustomTestCase, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.workload = Workload("pointwise", query_tokens=2, candidate_tokens=2)
        self.delay, self.failure, self.calls = 0, None, 0
        self.concurrent = self.peak = 0
        self.offline = False
        self.sessions_open = 0
        self.connectors = []

        @asynccontextmanager
        async def session(*, connector, timeout):
            self.connectors.append(connector)
            self.sessions_open += 1

            @asynccontextmanager
            async def post(url, *, json):
                if self.offline:
                    raise aiohttp.ClientConnectionError("Test transport disconnected")
                self.calls += 1
                failure = self.failure if self.calls > 1 else None
                self.concurrent += 1
                self.peak = max(self.peak, self.concurrent)
                try:
                    async with asyncio.timeout(timeout.total):
                        await asyncio.sleep(
                            0.04 if failure == "timeout" else self.delay
                        )

                        async def read_json():
                            if failure == "json":
                                raise ValueError("Malformed JSON response")
                            return (
                                {"scores": []}
                                if failure == "response"
                                else response_for(self.workload)
                            )

                        yield SimpleNamespace(
                            status=503 if failure == "http" else 200, json=read_json
                        )
                finally:
                    self.concurrent -= 1

            try:
                yield SimpleNamespace(post=post)
            finally:
                await connector.close()
                self.sessions_open -= 1

        client = patch(
            "sglang.test.kits.score_bench_kit.aiohttp.ClientSession",
            side_effect=session,
        )
        client.start()
        self.addCleanup(client.stop)
        self.url = "http://score.test"

    async def probe(self, **overrides):
        options = dict(
            qps=100,
            num_requests=12,
            warmup=1,
            min_samples=12,
            min_duration_s=0.1,
            throughput_tolerance=0.3,
            max_backlog_growth=20,
        )
        options.update(overrides)
        return await run_load(self.url, self.workload, **options)

    async def test_open_loop_and_intended_arrival_accounting(self):
        self.delay = 0.03
        result = await self.probe()
        self.assertTrue(result["passed"], result["reasons"])
        self.assertGreater(self.peak, 1)
        self.assertEqual(result["successful"], 12)
        self.assertGreater(result["latency_ms"]["p99"], 25)
        self.assertGreater(result["drain_s"], 0)
        self.assertGreater(result["backlog"]["at_window_end"], 0)
        self.assertLess(result["completed_in_window"], 12)
        for index, record in enumerate(result["records"]):
            self.assertAlmostEqual(record["scheduled_s"], index / 100)
            self.assertAlmostEqual(
                record["latency_ms"],
                (record["finished_s"] - record["scheduled_s"]) * 1000,
                places=2,
            )
        json.dumps(result, allow_nan=False)

    async def test_complete_server_failures_are_valid_but_never_pass(self):
        for failure in ("http", "response", "json", "timeout"):
            self.failure, self.calls = failure, 0
            result = await self.probe(timeout_s=0.01)
            with self.subTest(failure=failure):
                self.assertTrue(result["valid"], result["reasons"])
                self.assertFalse(result["passed"])
                self.assertEqual(
                    result["errors"],
                    {("response" if failure == "json" else failure): 12},
                )
                self.assertEqual(result["completed"], 12)
                self.assertEqual(result["failed"], 12)
                self.assertIsNone(result["latency_ms"]["p99"])

    async def test_slow_successful_requests_fail_each_performance_guard(self):
        self.delay = 0.04
        result = await self.probe(
            slo_ms=5,
            throughput_tolerance=0.01,
            max_backlog_growth=0.1,
            max_drain_s=0.001,
        )
        self.assertTrue(result["valid"], result["reasons"])
        self.assertEqual(result["successful"], 12)
        self.assertEqual(result["failed"], 0)
        self.assertFalse(result["passed"])
        self.assertEqual(
            set(result["reasons"]),
            {"p99_slo", "throughput_shortfall", "backlog_growth", "excessive_drain"},
        )

    async def test_sentinel_validation_and_warmup_fail_closed(self):
        sentinel = build_requests(self.workload, 13)[1]
        result = await self.probe(
            response_validator=lambda payload, data: payload != sentinel
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], {"sentinel": 1})
        self.assertFalse(result["passed"])
        result = await self.probe(response_validator=lambda payload, data: False)
        self.assertEqual(result["reasons"], ["warmup_failed"])
        self.assertEqual(result["warmup"]["errors"], {"sentinel": 1})
        self.assertEqual(result["not_scheduled"], 12)
        self.offline = True
        result = await self.probe()
        self.assertEqual(result["warmup"]["errors"], {"transport": 1})
        self.assertIsNone(result["latency_ms"]["p99"])

    async def test_generator_cap_starvation_and_offered_floors(self):
        self.delay = 0.04
        result = await self.probe(qps=1000, max_in_flight=1)
        self.assertIn("generator_overload", result["reasons"])
        self.assertEqual(result["completed"] + result["errors"]["overload"], 12)
        self.assertLessEqual(result["backlog"]["peak"], 1)
        self.delay = 0
        calls = 0

        def block_client(payload, data):
            nonlocal calls
            calls += 1
            if calls == 2:
                time.sleep(0.08)

        result = await self.probe(response_validator=block_client)
        self.assertIn("generator_starvation", result["reasons"])
        result = await self.probe(min_samples=13, min_duration_s=1)
        self.assertFalse(result["valid"])
        self.assertIn("insufficient_samples", result["reasons"])
        self.assertIn("insufficient_duration", result["reasons"])
        with self.assertRaises(ValueError):
            await self.probe(max_dispatch_lag_s=0.051)

    async def test_gc_pause_restore_overlap_and_error(self):
        enabled = gc.isenabled()
        observed = []

        def validate(payload, data):
            observed.append(gc.isenabled())
            cycle = []
            cycle.append(cycle)

        try:
            await asyncio.gather(
                self.probe(response_validator=validate),
                self.probe(qps=50, response_validator=validate),
            )
            self.assertEqual(set(observed), {False})
            self.assertEqual(gc.isenabled(), enabled)
            gc.disable()
            with self.assertRaises(ValueError):
                await self.probe(qps=0)
            self.assertFalse(gc.isenabled())
            gc.enable()
            with self.assertRaises(ValueError):
                await self.probe(qps=0)
            self.assertTrue(gc.isenabled())
        finally:
            gc.enable() if enabled else gc.disable()

    async def test_cancelling_active_load_cleans_up_and_restores_gc(self):
        enabled = gc.isenabled()
        self.delay = 0.1
        task = asyncio.create_task(
            self.probe(num_requests=100, min_samples=100, min_duration_s=1)
        )
        try:
            for _ in range(100):
                if self.calls >= 3 and self.concurrent >= 2:
                    break
                await asyncio.sleep(0.005)
            else:
                self.fail("Load did not reach concurrent measured requests")
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(self.concurrent, 0)
            self.assertEqual(self.sessions_open, 0)
            self.assertTrue(all(connector.closed for connector in self.connectors))
            self.assertEqual(gc.isenabled(), enabled)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            gc.enable() if enabled else gc.disable()


if __name__ == "__main__":
    unittest.main()
