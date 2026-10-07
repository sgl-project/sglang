import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
if "aiohttp" not in sys.modules:
    sys.modules["aiohttp"] = types.ModuleType("aiohttp")
SCRIPT = Path(__file__).parents[1] / "scripts" / "run_infercast_real_open_loop.py"
SPEC = importlib.util.spec_from_file_location("run_infercast_real_open_loop", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def measurement(index: int, latency_ms: float, ttft_ms: float, success: bool = True):
    return MODULE.RequestMeasurement(
        request_index=index,
        success=success,
        scheduled_timestamp_ms=0.0,
        actual_start_timestamp_ms=0.0,
        input_length=64,
        requested_output_length=3,
        output_length=3 if success else 0,
        cached_tokens=63,
        cached_tokens_details={"device": 63},
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        itl_ms=[2.0, 4.0] if success else [],
        metadata={"case_id": "test"},
        error="" if success else "failed",
    )


def trace_request(timestamp_ms: float, input_length: int):
    return MODULE.TraceRequest(
        timestamp_ms=timestamp_ms,
        input_length=input_length,
        output_length=3,
        token_ids=[1] * input_length,
        metadata={},
    )


def test_group_same_timestamp_requests_preserves_indices_and_time_order():
    requests = [
        trace_request(1.0, 64),
        trace_request(0.0, 128),
        trace_request(1.0, 256),
    ]

    groups = MODULE.group_same_timestamp_requests(requests)

    assert [[index for index, _ in group] for group in groups] == [[1], [0, 2]]
    assert [[request.input_length for _, request in group] for group in groups] == [
        [128],
        [64, 256],
    ]


class _StreamingContent:
    def __init__(self, events):
        self.lines = iter([f"data: {json.dumps(event)}\n".encode() for event in events])

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.lines)
        except StopIteration as error:
            raise StopAsyncIteration from error


class _StreamingResponse:
    status = 200

    def __init__(self, events):
        self.content = _StreamingContent(events)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class _BatchSession:
    def __init__(self, events):
        self.events = events
        self.payload = None

    def post(self, _url, *, json):
        self.payload = json
        return _StreamingResponse(self.events)


def test_send_request_group_uses_native_batch_and_demultiplexes_stream():
    events = [
        {"index": 0, "meta_info": {"completion_tokens": 1, "cached_tokens": 63}},
        {"index": 1, "meta_info": {"completion_tokens": 1, "cached_tokens": 127}},
        {"index": 0, "meta_info": {"completion_tokens": 3}},
        {"index": 1, "meta_info": {"completion_tokens": 3}},
    ]
    session = _BatchSession(events)
    requests = [trace_request(0.0, 64), trace_request(0.0, 128)]

    measurements = asyncio.run(
        MODULE.send_request_group(
            session,
            "http://server",
            list(enumerate(requests)),
            MODULE.time.perf_counter(),
        )
    )

    assert session.payload["input_ids"] == [[1] * 64, [1] * 128]
    assert isinstance(session.payload["sampling_params"], list)
    assert [item.output_length for item in measurements] == [3, 3]
    assert [item.cached_tokens for item in measurements] == [63, 127]
    assert all(len(item.itl_ms) == 2 for item in measurements)


def test_summarize_keeps_failures_out_of_latency_statistics():
    summary = MODULE.summarize(
        [
            measurement(0, 10.0, 4.0),
            measurement(1, 20.0, 6.0),
            measurement(2, 1.0, 0.0, False),
        ]
    )

    assert summary["request_count"] == 3
    assert summary["success_count"] == 2
    assert summary["failure_count"] == 1
    assert summary["latency_ms"] == {
        "mean": 15.0,
        "p50": 10.0,
        "p95": 20.0,
        "max": 20.0,
    }
    assert summary["ttft_ms"]["mean"] == 5.0
    assert summary["itl_ms"]["mean"] == 3.0


class _Response:
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def text(self):
        return self.body


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)

    def post(self, _url):
        return self.responses.pop(0)


def test_flush_cache_retries_transient_pending_request_rejection():
    session = _Session([_Response(400, "pending"), _Response(200, '{"ok": true}')])

    result = asyncio.run(
        MODULE.flush_cache(
            session,
            "http://server",
            timeout_seconds=1.0,
            retry_interval_seconds=0.0,
        )
    )

    assert result == {"ok": True, "_client_flush_attempts": 2}


def test_flush_cache_does_not_retry_non_transient_status():
    session = _Session([_Response(500, "broken")])

    try:
        asyncio.run(MODULE.flush_cache(session, "http://server"))
    except RuntimeError as error:
        assert "after 1 attempt(s) (500): broken" in str(error)
    else:
        raise AssertionError("flush_cache should reject a non-transient status")
