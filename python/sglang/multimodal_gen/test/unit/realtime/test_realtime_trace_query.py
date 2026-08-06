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


def test_cloudwatch_trace_query_escapes_trace_id_regex_metacharacters():
    class FakeLogs:
        def __init__(self):
            self.query_string = ""

        def start_query(self, **kwargs):
            self.query_string = kwargs["queryString"]
            return {"queryId": "query-escaped"}

        def get_query_results(self, *, queryId):
            assert queryId == "query-escaped"
            return {"status": "Complete", "results": []}

    async def run():
        logs = FakeLogs()
        query = CloudWatchTraceQuery(logs, log_group="logs")

        await query.query("trace.with:parts-1")

        assert "filter @message like /trace\\.with:parts-1/" in logs.query_string

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


def test_cloudwatch_trace_query_returns_five_minute_stage_aggregates():
    events = [
        {
            "trace_id": "trace-aggregate",
            "event": "client.generate_clicked",
            "trace_seq": 1,
            "client_epoch_ms": 1_000,
        },
        {
            "trace_id": "trace-aggregate",
            "event": "client.ws_open",
            "trace_seq": 2,
            "client_epoch_ms": 1_025,
        },
        {
            "trace_id": "trace-aggregate",
            "event": "gateway.coordinator_admit_complete",
            "trace_seq": 3,
            "coordinator_admit_ms": 8,
        },
        {
            "trace_id": "trace-aggregate",
            "event": "server.chunk_complete",
            "trace_seq": 4,
            "chunk_index": 1,
            "event_id": 7,
            "request_prepare_ms": 10,
            "scheduler_forward_ms": 20,
            "raw_payload_build_ms": 2,
            "ws_write_ms": 3,
        },
        {
            "trace_id": "trace-aggregate",
            "event": "server.model_denoise_complete",
            "trace_seq": 5,
            "chunk_index": 1,
            "event_id": 7,
            "duration_ms": 40,
        },
        {
            "trace_id": "trace-aggregate",
            "event": "server.model_denoise_complete",
            "trace_seq": 6,
            "chunk_index": 2,
            "event_id": 8,
            "duration_ms": 80,
        },
        {
            "trace_id": "trace-aggregate",
            "event": "server.vae_decode_complete",
            "trace_seq": 7,
            "chunk_index": 1,
            "event_id": 7,
            "cuda_ms": 12,
        },
        {
            "trace_id": "trace-aggregate",
            "event": "client.chunk_first_rendered",
            "trace_seq": 8,
            "chunk_index": 1,
            "event_id": 7,
            "decode_ms": 1,
            "display_lag_ms": 9,
        },
    ]

    class FakeLogs:
        def start_query(self, **kwargs):
            assert kwargs["startTime"] == 700
            assert kwargs["endTime"] == 1_000
            assert kwargs["limit"] == 10_000
            return {"queryId": "query-aggregate"}

        def get_query_results(self, *, queryId):
            assert queryId == "query-aggregate"
            return {
                "status": "Complete",
                "results": [
                    [{"field": "@message", "value": json.dumps(event)}]
                    for event in events
                ],
            }

    async def run():
        query = CloudWatchTraceQuery(
            FakeLogs(),
            log_group="logs",
            wall_clock=lambda: 1_000,
        )
        result = await query.query("trace-aggregate", window_s=300)

        assert result["stale"] is False
        assert result["window"]["seconds"] == 300
        assert result["window"]["start_epoch_ms"] == 700_000
        assert result["window"]["end_epoch_ms"] == 1_000_000
        assert result["observed_at"]
        stages = {stage["id"]: stage for stage in result["stages"]}
        assert stages["browser"]["avg_ms"] == 25
        assert stages["gateway"]["avg_ms"] == 8
        assert stages["api"]["avg_ms"] == 10
        assert stages["scheduler"]["avg_ms"] == 20
        assert stages["denoise"] == {
            "id": "denoise",
            "title": "Denoising",
            "count": 2,
            "avg_ms": 60,
            "p50_ms": 40,
            "p95_ms": 80,
            "max_ms": 80,
        }
        assert stages["vae_decode"]["avg_ms"] == 12
        assert stages["transport"]["avg_ms"] == 5
        assert stages["frontend"]["avg_ms"] == 10

    asyncio.run(run())


def test_cloudwatch_trace_query_retains_last_good_aggregate_on_empty_result():
    class FakeLogs:
        def __init__(self):
            self.started = 0

        def start_query(self, **_kwargs):
            self.started += 1
            return {"queryId": f"query-{self.started}"}

        def get_query_results(self, *, queryId):
            if queryId == "query-1":
                event = {
                    "trace_id": "trace-sticky",
                    "event": "server.model_denoise_complete",
                    "trace_seq": 1,
                    "chunk_index": 1,
                    "duration_ms": 55,
                }
                return {
                    "status": "Complete",
                    "results": [[{"field": "@message", "value": json.dumps(event)}]],
                }
            return {"status": "Complete", "results": []}

    async def run():
        now = [100.0]
        query = CloudWatchTraceQuery(
            FakeLogs(),
            log_group="logs",
            cache_ttl_s=1,
            clock=lambda: now[0],
            wall_clock=lambda: 1_000,
        )
        first = await query.query("trace-sticky")
        now[0] = 102.0
        second = await query.query("trace-sticky")

        assert first["stale"] is False
        assert second["stale"] is True
        assert second["stale_reason"] == "no_results"
        assert second["observed_at"] == first["observed_at"]
        assert second["stages"] == first["stages"]

    asyncio.run(run())


def test_cloudwatch_trace_query_retains_last_good_aggregate_on_transient_failure():
    class FakeLogs:
        def __init__(self):
            self.started = 0

        def start_query(self, **_kwargs):
            self.started += 1
            if self.started == 2:
                raise RuntimeError("temporary CloudWatch failure")
            return {"queryId": "query-good"}

        def get_query_results(self, *, queryId):
            assert queryId == "query-good"
            event = {
                "trace_id": "trace-fallback",
                "event": "server.vae_decode_complete",
                "trace_seq": 9,
                "chunk_index": 3,
                "duration_ms": 14,
            }
            return {
                "status": "Complete",
                "results": [[{"field": "@message", "value": json.dumps(event)}]],
            }

    async def run():
        now = [100.0]
        query = CloudWatchTraceQuery(
            FakeLogs(),
            log_group="logs",
            cache_ttl_s=1,
            clock=lambda: now[0],
        )
        first = await query.query("trace-fallback")
        now[0] = 102.0
        second = await query.query("trace-fallback")

        assert second["stale"] is True
        assert second["stale_reason"] == "query_failed"
        assert second["observed_at"] == first["observed_at"]
        assert second["stages"] == first["stages"]

    asyncio.run(run())
