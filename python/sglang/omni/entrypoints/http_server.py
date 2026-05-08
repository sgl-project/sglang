# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import base64
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse

from sglang.omni.backends.base import UnsupportedARBackend, UnsupportedGenerationBackend
from sglang.omni.coordinator import OmniCoordinator
from sglang.omni.protocol import OmniRequest, OmniResponse
from sglang.version import __version__


@asynccontextmanager
async def lifespan(app: FastAPI):
    factory = getattr(app.state, "orchestrator_factory", None)
    if getattr(app.state, "orchestrator", None) is None:
        app.state.orchestrator = factory() if callable(factory) else _default_orchestrator()
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
            response = raw_request.app.state.orchestrator.generate(request)
            return ORJSONResponse(_serialize_response(response))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

    return app


def create_sensenova_u1_app(
    *,
    scheduler: Any,
    server_args: Any,
    srt_request_executor: Any | None = None,
    srt_u_decode_max_new_tokens: int | None = None,
) -> FastAPI:
    from sglang.omni.configs.sensenova_u1 import (
        build_sensenova_u1_orchestrator_from_scheduler,
    )

    return create_app(
        orchestrator_factory=lambda: build_sensenova_u1_orchestrator_from_scheduler(
            scheduler=scheduler,
            srt_request_executor=srt_request_executor,
            srt_u_decode_max_new_tokens=srt_u_decode_max_new_tokens,
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
        generation_backend=UnsupportedGenerationBackend(),
    )


def _serialize_response(response: OmniResponse) -> dict[str, Any]:
    payload = response.to_dict()
    for segment in payload["segments"]:
        if segment["type"] == "image":
            segment["image"] = _serialize_image(segment["image"])
    return payload


def _serialize_image(image: Any) -> Any:
    if image is None:
        return None
    if isinstance(image, dict):
        return image
    if isinstance(image, bytes):
        return {
            "b64_json": base64.b64encode(image).decode("ascii"),
            "mime_type": "application/octet-stream",
        }
    save = getattr(image, "save", None)
    if callable(save):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return {
            "b64_json": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "mime_type": "image/png",
        }
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the experimental omni server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30001)
    args = parser.parse_args()
    launch_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
