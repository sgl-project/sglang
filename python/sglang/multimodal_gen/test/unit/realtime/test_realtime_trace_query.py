# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading
import time

from sglang.multimodal_gen.runtime.realtime.trace_query import (
    CloudWatchTraceQuery,
)


def test_cloudwatch_trace_query_is_cached_and_returns_incremental_events():
    class FakeLogs:
        def __init__(self):
            self.started = []

        def start_query(self, **kwargs):
            self.started.append(kwargs)
            return {"queryId": f"query-{len(self.started)}"}

        def get_query_results(self, *, queryId):
            del queryId
            return {
                "status": "Complete",
                "results": [
                    [
                        {
                            "field": "@message",
                            "value": "prefix "
                            + json.dumps(
                                {
                                    "trace_id": "trace-a",
                                    "event": "server.chunk_complete",
                                    "trace_seq": 8,
                                    "chunk_total_ms": 490,
                                }
                            ),
                        }
                    ]
                ],
            }

    async def run():
        now = [100.0]
        logs = FakeLogs()
        query = CloudWatchTraceQuery(
            logs,
            log_group="/aws/eks/minwm/realtime",
            cache_ttl_s=15,
            clock=lambda: now[0],
        )
        first = await query.query("trace-a", after=7, limit=220)
        second = await query.query("trace-a", after=7, limit=220)
        assert first == second
        assert first["events"][0]["trace_seq"] == 8
        assert first["next_cursor"] == 8
        assert len(logs.started) == 1
        assert "filter @message like /trace-a/" in logs.started[0]["queryString"]
        assert "sort @timestamp desc" in logs.started[0]["queryString"]

        now[0] = 116.0
        await query.query("trace-a", after=7, limit=220)
        assert len(logs.started) == 2

    asyncio.run(run())


def test_cloudwatch_trace_query_unwraps_container_insights_log_events():
    event = {
        "trace_id": "trace-container",
        "event": "server.vae_decode_complete",
        "trace_seq": 42,
        "duration_ms": 27.2,
    }
    message = json.dumps(
        {
            "time": "2026-08-05T13:31:54Z",
            "stream": "stdout",
            "log": "[08-05 13:31:54] realtime_trace " + json.dumps(event),
            "kubernetes": {"pod_name": "minwm-b300-vae"},
        }
    )

    assert CloudWatchTraceQuery._parse_message(message) == event


def test_cloudwatch_trace_query_rejects_unsafe_trace_ids_without_a_query():
    class FakeLogs:
        def start_query(self, **_kwargs):
            raise AssertionError("CloudWatch must not be queried")

    async def run():
        query = CloudWatchTraceQuery(FakeLogs(), log_group="logs")
        try:
            await query.query("bad' | limit 10000", after=0, limit=220)
        except ValueError as exc:
            assert "trace_id" in str(exc)
        else:
            raise AssertionError("unsafe trace ID was accepted")

    asyncio.run(run())


def test_cloudwatch_trace_query_coalesces_identical_requests_and_bounds_concurrency():
    class FakeLogs:
        def __init__(self):
            self.started = 0
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def start_query(self, **_kwargs):
            with self.lock:
                self.started += 1
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                return {"queryId": f"query-{self.started}"}

        def get_query_results(self, *, queryId):
            time.sleep(0.03)
            with self.lock:
                self.active -= 1
            trace_id = queryId.replace("query-", "trace-")
            return {
                "status": "Complete",
                "results": [[{"field": "@message", "value": json.dumps({"trace_id": trace_id, "trace_seq": 1})}]],
            }

    async def run():
        logs = FakeLogs()
        query = CloudWatchTraceQuery(
            logs,
            log_group="logs",
            max_concurrent_queries=2,
        )
        await asyncio.gather(*(query.query("trace-same") for _ in range(5)))
        assert logs.started == 1

        await asyncio.gather(*(query.query(f"trace-{index}") for index in range(6)))
        assert logs.max_active <= 2

    asyncio.run(run())


def test_cloudwatch_trace_query_survives_waiter_cancellation_and_cleans_inflight():
    class FakeLogs:
        def __init__(self):
            self.started = 0

        def start_query(self, **_kwargs):
            self.started += 1
            return {"queryId": "query-shared"}

        def get_query_results(self, *, queryId):
            del queryId
            time.sleep(0.04)
            return {
                "status": "Complete",
                "results": [
                    [
                        {
                            "field": "@message",
                            "value": json.dumps(
                                {"trace_id": "trace-shared", "trace_seq": 1}
                            ),
                        }
                    ]
                ],
            }

    async def run():
        logs = FakeLogs()
        query = CloudWatchTraceQuery(logs, log_group="logs")
        cancelled = asyncio.create_task(query.query("trace-shared"))
        await asyncio.sleep(0.01)
        cancelled.cancel()
        try:
            await cancelled
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelled waiter unexpectedly completed")

        await asyncio.sleep(0.06)
        result = await query.query("trace-shared")
        assert result["next_cursor"] == 1
        assert logs.started == 1
        assert query._inflight == {}

    asyncio.run(run())


def test_cloudwatch_trace_query_retries_transient_api_throttling():
    class ThrottlingException(Exception):
        response = {"Error": {"Code": "ThrottlingException"}}

    class FakeLogs:
        def __init__(self):
            self.started = 0

        def start_query(self, **_kwargs):
            self.started += 1
            if self.started == 1:
                raise ThrottlingException("rate exceeded")
            return {"queryId": "query-retried"}

        def get_query_results(self, *, queryId):
            assert queryId == "query-retried"
            return {"status": "Complete", "results": []}

    async def run():
        logs = FakeLogs()
        query = CloudWatchTraceQuery(logs, log_group="logs")

        result = await query.query("trace-retry")

        assert result["events"] == []
        assert logs.started == 2

    asyncio.run(run())
