"""Minimal fake SGLang worker for kind E2E integration testing.

Responds to:
  GET  /health                   -> {"status": "ok"}
  GET  /server_info              -> {"served_model_name": MODEL_ID}
  GET  /v1/models                -> list with a single MODEL_ID model entry
  POST /v1/chat/completions      -> echoes the last user message back
  POST /v1/start_reporting       -> renews an unauthenticated gRPC load stream
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

import grpc
import grpc_tools
import uvicorn
from fastapi import FastAPI, Request
from grpc_tools import protoc


def _load_generated_proto():
    """Generate and import Python bindings for the Router-local test proto.

    Returns:
        A tuple containing the protobuf messages module and gRPC stub module.

    Raises:
        RuntimeError: If the vendored grpc-tools compiler rejects the schema.
    """
    output = tempfile.mkdtemp(prefix="load-monitor-proto-")
    include = Path(grpc_tools.__file__).parent / "_proto"
    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{Path(__file__).parent}",
            f"-I{include}",
            f"--python_out={output}",
            f"--grpc_python_out={output}",
            str(Path(__file__).parent / "load_monitor.proto"),
        ]
    )
    if result != 0:
        raise RuntimeError(f"grpc_tools.protoc failed with exit code {result}")
    sys.path.insert(0, output)
    import load_monitor_pb2  # pylint: disable=import-outside-toplevel
    import load_monitor_pb2_grpc  # pylint: disable=import-outside-toplevel

    return load_monitor_pb2, load_monitor_pb2_grpc


load_monitor_pb2, load_monitor_pb2_grpc = _load_generated_proto()

app = FastAPI()

MODEL_ID = os.environ.get("MODEL_ID", "tiny")
POD_IP = os.environ.get("POD_IP", "127.0.0.1")
_reporting_task = None
_reporting_config = None
_lease_deadline = 0.0
_sequence_id = 0


@app.get("/health")
async def health():
    """Return fake-engine health status."""
    return {"status": "ok"}


@app.get("/server_info")
async def server_info():
    """Return the model identity consumed by Router introspection."""
    # The sgl-router worker manager fetches this on every Added event and
    # uses `served_model_name` to populate the registry's model index.
    return {"served_model_name": MODEL_ID}


@app.get("/v1/models")
async def models():
    """Return one OpenAI-compatible model descriptor."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "sglang",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Echo the final user message in an OpenAI-compatible response.

    Args:
        request: Incoming FastAPI request containing a JSON chat payload.

    Returns:
        A deterministic non-streaming chat completion object.
    """
    payload = await request.json()
    messages = payload.get("messages", [])
    last_content = messages[-1]["content"] if messages else ""
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": payload.get("model", MODEL_ID),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"echo: {last_content}",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


async def _report_stream():
    """Maintain a reconnecting gRPC report stream until its lease expires.

    Returns:
        None. The coroutine exits after the last renewed lease deadline.
    """
    global _sequence_id
    backoff = 0.2
    while time.monotonic() < _lease_deadline:
        config = dict(_reporting_config)
        target = f"{config['ip']}:{config['port']}"
        try:
            async with grpc.aio.insecure_channel(target) as channel:
                stub = load_monitor_pb2_grpc.LoadMonitorServiceStub(channel)

                async def reports():
                    """Yield periodic healthy reports while the lease is live."""
                    global _sequence_id
                    while time.monotonic() < _lease_deadline:
                        _sequence_id += 1
                        yield load_monitor_pb2.LoadReport(
                            source_instance_id=f"fake-{POD_IP}",
                            sequence_id=_sequence_id,
                            report_time_unix_ms=int(time.time() * 1000),
                            worker=load_monitor_pb2.Worker(
                                worker_addr=f"{POD_IP}:30000",
                                worker_type=load_monitor_pb2.WORKER_TYPE_REGULAR,
                                model=MODEL_ID,
                            ),
                            status=load_monitor_pb2.REPORT_STATUS_HEALTHY,
                            ranks=[
                                load_monitor_pb2.RankLoad(
                                    dp_rank=0,
                                    snapshot_time_unix_ms=int(time.time() * 1000),
                                    num_running_reqs=0,
                                    num_waiting_reqs=0,
                                    num_waiting_uncached_tokens=0,
                                    num_used_tokens=1,
                                    num_total_tokens=1,
                                    max_total_num_tokens=1024,
                                    max_running_requests=32,
                                    token_usage=1.0 / 1024.0,
                                    gen_throughput=1.0,
                                    cache_hit_rate=0.0,
                                    utilization=0.0,
                                )
                            ],
                        )
                        await asyncio.sleep(config["report_interval_ms"] / 1000.0)

                await stub.Report(reports())
                backoff = 0.2
        except (grpc.aio.AioRpcError, OSError):
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 5.0)


@app.post("/v1/start_reporting")
async def start_reporting(request: Request):
    """Start or renew the fake engine's unauthenticated reporting lease.

    Args:
        request: JSON request containing callback IP/port, interval, and TTL.

    Returns:
        A small acknowledgement showing that the lease was renewed.
    """
    global _lease_deadline, _reporting_config, _reporting_task
    config = await request.json()
    _reporting_config = config
    _lease_deadline = time.monotonic() + config["lease_ttl_ms"] / 1000.0
    if _reporting_task is None or _reporting_task.done():
        _reporting_task = asyncio.create_task(_report_stream())
    return {"status": "reporting"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30000)
