"""Live-server abort regression tests (Qwen3-0.6B is sufficient).

SGLANG_ABORT_TEST_URL=http://127.0.0.1:30000 python \
    test/manual/entrypoints/http_server/test_abort_request_isolation.py -v

Run against a warmed-up server with strict idle memory checks enabled.
No server processes are started or stopped by this module.
"""

import asyncio
import json
import os
import unittest
import uuid

import httpx


class TestAbortRequestIsolation(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            base_url=os.environ.get("SGLANG_ABORT_TEST_URL", "http://127.0.0.1:30000"),
            timeout=120,
            trust_env=False,
        )
        response = await self.client.get("/health")
        self.assertEqual(response.status_code, 200, "server must finish warmup first")

    async def asyncTearDown(self):
        await self.client.aclose()

    async def abort(self, **payload):
        response = await self.client.post("/abort_request", json=payload)
        self.assertEqual(response.status_code, 200, response.text)

    async def generate(self, rid, ready, *, n=1, tokens=512):
        result = {}
        async with self.client.stream(
            "POST",
            "/generate",
            json={
                "text": "Tell a long story.",
                "rid": rid,
                "stream": True,
                "sampling_params": {
                    "max_new_tokens": tokens,
                    "ignore_eos": True,
                    "n": n,
                },
            },
        ) as response:
            self.assertEqual(response.status_code, 200)
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                self.assertNotIn("error", chunk, chunk)
                result[chunk.get("index", 0)] = chunk["meta_info"]
                if len(result) == n:
                    ready.set()
        return result

    async def assert_idle_and_reusable(self, rid=None):
        # Reuse the server's pool invariants and flush refusal while busy.
        for _ in range(100):
            response = await self.client.post("/flush_cache")
            if response.status_code == 200:
                break
            await asyncio.sleep(0.1)
        self.assertEqual(response.status_code, 200, response.text)
        response = await self.client.post(
            "/generate",
            json={
                "text": "Hello",
                "rid": rid or f"probe-{uuid.uuid4().hex}",
                "sampling_params": {"max_new_tokens": 8, "ignore_eos": True},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["meta_info"]["finish_reason"]["type"], "length"
        )
        self.assertEqual(response.json()["meta_info"]["completion_tokens"], 8)

    async def run_concurrent(self, rids, payload, expected):
        ready = [asyncio.Event() for _ in rids]
        tasks = [
            asyncio.create_task(self.generate(rid, event))
            for rid, event in zip(rids, ready)
        ]
        try:
            await asyncio.wait_for(
                asyncio.gather(*(event.wait() for event in ready)), 60
            )
            self.assertTrue(
                all(not task.done() for task in tasks), "abort raced with completion"
            )
            await self.abort(**payload)
            outputs = await asyncio.wait_for(asyncio.gather(*tasks), 120)
            for rid, output in zip(rids, outputs):
                meta = output[0]
                print(rid, meta["finish_reason"], meta["completion_tokens"], flush=True)
                self.assertEqual(meta["finish_reason"]["type"], expected[rid], rid)
                if expected[rid] == "length":
                    self.assertEqual(meta["completion_tokens"], 512, rid)
                else:
                    self.assertLess(meta["completion_tokens"], 512, rid)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.assert_idle_and_reusable()

    async def test_exact_rid_and_independent_underscore_ids(self):
        rids = ["job-1", "job-10", "job-11", "job-2", "job-1_0", "job-1_0_extra"]
        await self.run_concurrent(
            rids,
            {"rid": "job-1"},
            {rid: "abort" if rid == "job-1" else "length" for rid in rids},
        )

    async def test_underscore_target(self):
        rids = ["under-1_0", "under-1_0_extra", "under-1_01"]
        await self.run_concurrent(
            rids,
            {"rid": "under-1_0"},
            {rid: "abort" if rid == "under-1_0" else "length" for rid in rids},
        )

    async def test_abort_all(self):
        rids = ["all-1", "all-10", "all-1_0", "all-2"]
        await self.run_concurrent(
            rids, {"abort_all": True}, {rid: "abort" for rid in rids}
        )

    async def test_missing_id_is_noop(self):
        rids = ["missing0", "missing_0"]
        await self.run_concurrent(
            rids, {"rid": "missing"}, {rid: "length" for rid in rids}
        )

    async def test_finished_id_is_noop(self):
        await self.assert_idle_and_reusable(rid="completed-1")
        rids = ["completed-10", "completed-1_0"]
        await self.run_concurrent(
            rids, {"rid": "completed-1"}, {rid: "length" for rid in rids}
        )

    async def test_waiting_queue_after_retract(self):
        rids = ["queued-1", "queued-10", "queued-1_0", "queued-2"]
        ready = [asyncio.Event() for _ in rids]
        tasks = [
            asyncio.create_task(self.generate(rid, event))
            for rid, event in zip(rids, ready)
        ]
        try:
            await asyncio.wait_for(
                asyncio.gather(*(event.wait() for event in ready)), 60
            )
            response = await self.client.post(
                "/pause_generation", json={"mode": "retract"}
            )
            self.assertEqual(response.status_code, 200, response.text)
            # While paused, retract moves the active requests to waiting_queue.
            # An abort response before continue_generation proves this is not
            # the running path (which needs another forward pass to finish).
            await self.abort(rid=rids[0])
            result = await asyncio.wait_for(asyncio.shield(tasks[0]), 10)
            self.assertEqual(result[0]["finish_reason"]["type"], "abort")
            self.assertTrue(all(not task.done() for task in tasks[1:]))
        finally:
            response = await self.client.post("/continue_generation", json={})
            self.assertEqual(response.status_code, 200, response.text)
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), 120
            )
        for rid, result in zip(rids[1:], results[1:]):
            self.assertIsInstance(result, dict, result)
            self.assertEqual(result[0]["finish_reason"]["type"], "length", rid)
            self.assertEqual(result[0]["completion_tokens"], 512, rid)
        await self.assert_idle_and_reusable()

    async def test_disconnect_parallel_samples(self):
        # Cancel the HTTP owner after every actual n > 1 child has started.
        # This exercises generate_request's explicit request_rids cleanup.
        ready = asyncio.Event()
        task = asyncio.create_task(
            self.generate("parallel_parent", ready, n=3, tokens=1500)
        )
        try:
            await asyncio.wait_for(ready.wait(), 60)
            self.assertFalse(task.done())
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self.assert_idle_and_reusable()


if __name__ == "__main__":
    unittest.main(verbosity=2)
