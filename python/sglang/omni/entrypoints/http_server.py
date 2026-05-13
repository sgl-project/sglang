# SPDX-License-Identifier: Apache-2.0
"""Standalone experimental omni FastAPI app.

The production U1 path mounts `http_api` into SRT. This app is kept as a small
shell for tests and future backends that do not need SRT-owned sessions.
"""

from __future__ import annotations

import argparse
import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.omni.backends import UnsupportedARBackend, UnsupportedGenerationBackend
from sglang.omni.core.coordinator import OmniCoordinator
from sglang.omni.core.protocol import OmniRequest
from sglang.omni.entrypoints.serialization import serialize_response
from sglang.omni.entrypoints.streaming import OmniStreamSink
from sglang.version import __version__

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    factory = getattr(app.state, "orchestrator_factory", None)
    if getattr(app.state, "orchestrator", None) is None:
        app.state.orchestrator = (
            factory() if callable(factory) else _default_orchestrator()
        )
    try:
        yield
    finally:
        shutdown = getattr(app.state.orchestrator, "shutdown", None)
        if callable(shutdown):
            shutdown()


def create_app(
    orchestrator: OmniCoordinator | None = None,
    orchestrator_factory: Callable[[], OmniCoordinator] | None = None,
) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.state.orchestrator = orchestrator
    app.state.orchestrator_factory = orchestrator_factory

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/server_info")
    async def server_info():
        return {"version": __version__, "server": "sglang.omni"}

    @app.post("/v1/omni/generate")
    async def omni_generate(raw_request: Request):
        try:
            payload = await raw_request.json()
            request = OmniRequest.from_payload(payload)
            if bool(payload.get("stream", False)):
                return StreamingResponse(
                    _stream_omni_generate(raw_request.app.state.orchestrator, request),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache"},
                )
            response = raw_request.app.state.orchestrator.generate(request)
            return ORJSONResponse(serialize_response(response))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    return app


async def _stream_omni_generate(orchestrator: OmniCoordinator, request: OmniRequest):
    events: list[dict[str, Any]] = []
    stream_sink = OmniStreamSink(events.append)
    try:
        response, _ = orchestrator.generate_with_context(
            request,
            stream_sink=stream_sink,
        )
        stream_sink.done(serialize_response(response))
    except (ValueError, RuntimeError) as exc:
        stream_sink.error(message=str(exc), status_code=400)
    for event in events:
        yield _encode_sse_event(event)
    yield b"data: [DONE]\n\n"


def _encode_sse_event(event: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n\n"


def create_sensenova_u1_app(
    *,
    scheduler: "Scheduler",
    server_args: Any,
    srt_request_executor: Any | None = None,
    srt_ar_decode_max_new_tokens: int | None = None,
) -> FastAPI:
    from sglang.omni.configs.sensenova_u1 import (
        build_sensenova_u1_orchestrator_from_scheduler,
    )

    return create_app(
        orchestrator_factory=lambda: build_sensenova_u1_orchestrator_from_scheduler(
            scheduler=scheduler,
            srt_request_executor=srt_request_executor,
            srt_ar_decode_max_new_tokens=srt_ar_decode_max_new_tokens,
            server_args=server_args,
        )
    )


def launch_server(
    *,
    host: str = "127.0.0.1",
    port: int = 30001,
    orchestrator: OmniCoordinator | None = None,
) -> None:
    import uvicorn

    app = create_app(orchestrator=orchestrator)
    uvicorn.run(app, host=host, port=port)


def _default_orchestrator() -> OmniCoordinator:
    return OmniCoordinator(
        ar_backend=UnsupportedARBackend(),
        mm_generation_backend=UnsupportedGenerationBackend(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the experimental omni server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30001)
    args = parser.parse_args()
    launch_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
