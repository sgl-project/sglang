"""Minimal fake SGLang Worker for kind E2E integration testing.

The HTTP server exposes inference/discovery fixtures on port 30000. A separate
h2c gRPC server exposes the canonical Load Reporter service on port 31000.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import grpc
import grpc_tools
import uvicorn
from fastapi import FastAPI, Request
from grpc_tools import protoc


def _load_generated_proto():
    """Generate Python bindings from the repository's canonical reporter proto.

    Returns:
        A tuple containing the protobuf messages module and gRPC module.

    Raises:
        RuntimeError: If grpc-tools cannot compile the canonical schema.
    """
    output = tempfile.mkdtemp(prefix="load-monitor-proto-")
    app_root = Path(__file__).parent
    include = Path(grpc_tools.__file__).parent / "_proto"
    proto_file = app_root / "sglang/router/loadmonitor/v1/load_monitor.proto"
    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{app_root}",
            f"-I{include}",
            f"--python_out={output}",
            f"--grpc_python_out={output}",
            str(proto_file),
        ]
    )
    if result != 0:
        raise RuntimeError(f"grpc_tools.protoc failed with exit code {result}")
    sys.path.insert(0, output)
    from sglang.router.loadmonitor.v1 import (  # pylint: disable=import-outside-toplevel
        load_monitor_pb2,
        load_monitor_pb2_grpc,
    )

    return load_monitor_pb2, load_monitor_pb2_grpc


load_monitor_pb2, load_monitor_pb2_grpc = _load_generated_proto()

MODEL_ID = os.environ.get("MODEL_ID", "tiny")
POD_IP = os.environ.get("POD_IP", "127.0.0.1")
REPORTER_PORT = int(os.environ.get("LOAD_REPORTER_PORT", "31000"))
_sequence_id = 0


def _healthy_report():
    """Build the next healthy single-rank LoadReport.

    Returns:
        A canonical LoadReport with a strictly increasing sequence number.
    """
    global _sequence_id
    _sequence_id += 1
    now_ms = int(time.time() * 1000)
    return load_monitor_pb2.LoadReport(
        source_instance_id=f"fake-{POD_IP}",
        sequence_id=_sequence_id,
        report_time_unix_ms=now_ms,
        worker=load_monitor_pb2.Worker(
            worker_addr=f"{POD_IP}:30000",
            worker_type=load_monitor_pb2.WORKER_TYPE_REGULAR,
            model=MODEL_ID,
        ),
        status=load_monitor_pb2.REPORT_STATUS_HEALTHY,
        ranks=[
            load_monitor_pb2.RankLoad(
                dp_rank=0,
                snapshot_time_unix_ms=now_ms,
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


class FakeLoadMonitorService(load_monitor_pb2_grpc.LoadMonitorServiceServicer):
    """Serve the Worker side of the Router-initiated bidi protocol."""

    async def Monitor(self, request_iterator, _context):
        """Acknowledge registration and stream reports until stop or lease expiry.

        Args:
            request_iterator: Async RouterFrame stream from one Router session.
            _context: gRPC request context, unused by the fake implementation.

        Yields:
            WorkerFrame registration, report, or validation-error messages.
        """
        try:
            first = await request_iterator.__anext__()
        except StopAsyncIteration:
            return
        if first.WhichOneof("payload") != "register":
            yield load_monitor_pb2.WorkerFrame(
                error=load_monitor_pb2.StreamError(
                    code="INVALID_FIRST_FRAME",
                    message="first frame must register",
                )
            )
            return

        register = first.register
        if (
            not register.router_id
            or register.report_interval_ms <= 0
            or register.lease_ttl_ms <= 0
        ):
            yield load_monitor_pb2.WorkerFrame(
                error=load_monitor_pb2.StreamError(
                    code="INVALID_ARGUMENT",
                    message="invalid registration timing or router_id",
                )
            )
            return

        interval = register.report_interval_ms / 1000.0
        lease_ttl = register.lease_ttl_ms / 1000.0
        lease_deadline = time.monotonic() + lease_ttl
        stopped = asyncio.Event()

        async def consume_controls():
            """Apply keep-alive, reconfiguration, and stop control frames.

            Returns:
                None. The coroutine ends at EOF or after a stop frame.
            """
            nonlocal interval, lease_ttl, lease_deadline
            async for frame in request_iterator:
                payload = frame.WhichOneof("payload")
                if payload == "keep_alive":
                    lease_deadline = time.monotonic() + lease_ttl
                elif payload == "update_config":
                    update = frame.update_config
                    if update.HasField("report_interval_ms"):
                        interval = update.report_interval_ms / 1000.0
                    if update.HasField("lease_ttl_ms"):
                        lease_ttl = update.lease_ttl_ms / 1000.0
                    lease_deadline = time.monotonic() + lease_ttl
                elif payload == "stop":
                    stopped.set()
                    return
            stopped.set()

        control_task = asyncio.create_task(consume_controls())
        try:
            yield load_monitor_pb2.WorkerFrame(
                registered=load_monitor_pb2.RegisterResponse(
                    lease_ttl_ms=register.lease_ttl_ms,
                    renew_after_ms=max(1, register.lease_ttl_ms // 2),
                )
            )
            while not stopped.is_set() and time.monotonic() < lease_deadline:
                yield load_monitor_pb2.WorkerFrame(report=_healthy_report())
                timeout = min(interval, max(0.0, lease_deadline - time.monotonic()))
                try:
                    await asyncio.wait_for(stopped.wait(), timeout=timeout)
                except TimeoutError:
                    pass
        finally:
            control_task.cancel()
            await asyncio.gather(control_task, return_exceptions=True)


@asynccontextmanager
async def lifespan(_app):
    """Own the fake Worker's independent gRPC reporter server.

    Args:
        _app: FastAPI application instance, unused by this fixture.

    Yields:
        Control to FastAPI after the reporter port has started listening.
    """
    server = grpc.aio.server()
    load_monitor_pb2_grpc.add_LoadMonitorServiceServicer_to_server(
        FakeLoadMonitorService(), server
    )
    bound_port = server.add_insecure_port(f"[::]:{REPORTER_PORT}")
    if bound_port == 0:
        raise RuntimeError(f"failed to bind load reporter port {REPORTER_PORT}")
    await server.start()
    try:
        yield
    finally:
        await server.stop(grace=0)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    """Return fake-engine health status."""
    return {"status": "ok"}


@app.get("/server_info")
async def server_info():
    """Return the model identity consumed by Router introspection."""
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30000)
