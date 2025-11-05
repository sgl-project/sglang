"""
Lightweight mock worker HTTP server for router integration tests.

Implements minimal endpoints used by the router:
- GET /health, /health_generate
- POST /generate, /v1/completions, /v1/chat/completions
- POST /flush_cache
- GET /get_server_info, /get_model_info, /v1/models

Behavior knobs are controlled via CLI flags to simulate failures, latency, and load.
"""

import argparse
import asyncio
import json
import os
import random
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

# Global state (per-process)
_inflight = 0
_failures_seen = 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--worker-id", default=None)
    p.add_argument("--latency-ms", type=int, default=0)
    p.add_argument("--timeout", action="store_true")
    p.add_argument("--status-code", type=int, default=200)
    p.add_argument("--fail-first-n", type=int, default=0)
    p.add_argument("--random-fail-rate", type=float, default=0.0)
    p.add_argument("--require-api-key", action="store_true")
    p.add_argument("--api-key", default=None)
    p.add_argument("--max-payload-bytes", type=int, default=10 * 1024 * 1024)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--dp-size", type=int, default=1)
    p.add_argument("--crash-on-request", action="store_true")
    p.add_argument("--health-fail-after-ms", type=int, default=0)
    # TLS/mTLS configuration
    p.add_argument(
        "--ssl-certfile", type=str, default=None, help="Path to SSL certificate file"
    )
    p.add_argument("--ssl-keyfile", type=str, default=None, help="Path to SSL key file")
    p.add_argument(
        "--ssl-ca-certs",
        type=str,
        default=None,
        help="Path to CA certificates for client verification",
    )
    return p.parse_args()


def _extract_worker_id(args: argparse.Namespace) -> str:
    if args.worker_id:
        return str(args.worker_id)
    # default to port (unique enough for tests)
    return f"worker-{args.port}"


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI()
    worker_id = _extract_worker_id(args)
    start_ts = time.time()
    crashed = {"done": False}

    async def maybe_delay():
        if args.latency_ms > 0:
            await asyncio.sleep(args.latency_ms / 1000.0)

    def should_fail() -> Optional[int]:
        global _failures_seen
        # Fail first N requests (500)
        if args.fail_first_n > 0 and _failures_seen < args.fail_first_n:
            _failures_seen += 1
            return 500
        # Random failure probability (500)
        if args.random_fail_rate > 0.0 and random.random() < args.random_fail_rate:
            return 500
        # Forced status code override (non-200) for all responses
        if args.status_code != 200:
            return int(args.status_code)
        return None

    def check_api_key(request: Request):
        if not args.require_api_key:
            return
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized")
        key = auth.split(" ", 1)[1]
        if args.api_key and key != args.api_key:
            raise HTTPException(status_code=401, detail="Unauthorized")

    @asynccontextmanager
    async def track_inflight():
        global _inflight
        _inflight += 1
        try:
            yield
        finally:
            _inflight -= 1

    @app.get("/health")
    async def health():
        if (
            args.health_fail_after_ms
            and (time.time() - start_ts) * 1000.0 >= args.health_fail_after_ms
        ):
            return PlainTextResponse("bad", status_code=500)
        return PlainTextResponse("ok", status_code=200)

    @app.get("/health_generate")
    async def health_generate():
        return PlainTextResponse("ok", status_code=200)

    @app.post("/flush_cache")
    async def flush_cache():
        return PlainTextResponse("ok", status_code=200)

    @app.get("/get_model_info")
    async def get_model_info():
        return JSONResponse({"model": "mock", "vocab_size": 32000})

    @app.get("/v1/models")
    async def list_models():
        return JSONResponse({"data": [{"id": "mock", "object": "model"}]})

    @app.get("/get_server_info")
    async def get_server_info(request: Request):
        # Enforce API key on server info when required (used by dp_aware probing)
        check_api_key(request)
        return JSONResponse(
            {
                "worker_id": worker_id,
                "load_in_flight": _inflight,
                "cache": {"size": 0, "hit_rate": 0.0},
                "dp_size": int(args.dp_size),
            }
        )

    @app.get("/get_load")
    async def get_load(request: Request):
        check_api_key(request)
        # Return format matching real workers: array of load info per DP rank
        return JSONResponse(
            [
                {
                    "dp_rank": 0,
                    "num_reqs": _inflight,
                    "num_waiting_reqs": 0,
                    "num_tokens": _inflight,
                }
            ]
        )

    def make_json_response(obj: dict, status_code: int = 200) -> JSONResponse:
        resp = JSONResponse(obj, status_code=status_code)
        resp.headers["X-Worker-Id"] = worker_id
        return resp

    async def handle_text_request(request: Request):
        # Authorization
        check_api_key(request)

        # Payload limit
        body = await request.body()
        if len(body) > args.max_payload_bytes:
            return make_json_response({"error": "payload too large"}, status_code=413)

        # Simulate crash on first request
        if args.crash_on_request and not crashed["done"]:
            crashed["done"] = True
            os._exit(1)

        # Optional timeout (simulate hang)
        if args.timeout:
            await asyncio.sleep(3600)

        # Optional latency
        await maybe_delay()

        # Optional failures
        fail_code = should_fail()
        if fail_code is not None and fail_code != 200:
            return make_json_response(
                {"error": f"mock failure {fail_code}"}, status_code=fail_code
            )

        # Build response echoing minimal shape
        try:
            data = await request.json()
        except (json.JSONDecodeError, ValueError):
            data = {}

        now = time.time()
        ret = {
            "id": f"cmpl-{int(now*1000)}",
            "object": "text_completion",
            "created": int(now),
            "model": "mock",
            "choices": [
                {
                    "text": "ok",
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "worker_id": worker_id,
            "echo": data,
        }
        return make_json_response(ret, status_code=200)

    async def handle_stream_request(request: Request):
        check_api_key(request)

        async def gen():
            # minimal 2-chunk stream then [DONE]
            for i in range(2):
                await asyncio.sleep(0.01)
                chunk = {
                    "choices": [{"delta": {"content": "x"}}],
                    "worker_id": worker_id,
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        headers = {"X-Worker-Id": worker_id}
        return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

    @app.post("/generate")
    async def generate(request: Request):
        async with track_inflight():
            if args.stream:
                return await handle_stream_request(request)
            return await handle_text_request(request)

    @app.post("/v1/completions")
    async def completions(request: Request):
        async with track_inflight():
            if args.stream:
                return await handle_stream_request(request)
            return await handle_text_request(request)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        async with track_inflight():
            if args.stream:
                return await handle_stream_request(request)
            return await handle_text_request(request)

    return app


def main() -> None:
    args = _parse_args()
    app = create_app(args)
    # Handle SIGTERM gracefully for fast test teardown
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # Configure SSL if certificates are provided
    ssl_config = {}
    if args.ssl_certfile and args.ssl_keyfile:
        ssl_config["ssl_certfile"] = args.ssl_certfile
        ssl_config["ssl_keyfile"] = args.ssl_keyfile
        # If CA certs provided, require client certificates (mTLS)
        if args.ssl_ca_certs:
            ssl_config["ssl_ca_certs"] = args.ssl_ca_certs
            ssl_config["ssl_cert_reqs"] = 2  # ssl.CERT_REQUIRED

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", **ssl_config)


if __name__ == "__main__":
    main()
