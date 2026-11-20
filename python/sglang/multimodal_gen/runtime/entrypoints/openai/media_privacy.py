"""Bind media jobs to a tenant asserted by an authenticated gateway."""

import hmac
import re
from pathlib import Path

from fastapi import HTTPException, Request
from starlette._utils import get_route_path
from starlette.datastructures import Headers, UploadFile
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

from sglang.multimodal_gen.runtime.entrypoints.openai.stores import media_tenant

PRIVATE_CACHE_CONTROL = "private, no-store"
_MEDIA_PREFIXES = ("/v1/images/", "/v1/videos/", "/v1/meshes/")
_SAMPLING_FIELDS = frozenset(
    "model prompt negative_prompt seed n size width height num_frames seconds fps "
    "num_inference_steps guidance_scale true_cfg_scale quality response_format "
    "output_format background".split()
)
_UPLOAD_FIELDS = {"image", "image[]", "mask", "input_reference", "video_reference"}


async def validate_gateway_payload(request: Request):
    """Public clients supply sampling values and uploads, not paths or backend kwargs."""
    if request.method != "POST":
        return
    media_type = (
        request.headers.get("content-type", "").partition(";")[0].strip().lower()
    )
    if media_type == "application/json":
        try:
            payload = await request.json()
        except (UnicodeError, ValueError):
            raise HTTPException(400, "Invalid JSON") from None
        if not isinstance(payload, dict):
            raise HTTPException(400, "Expected a JSON object")
        fields = payload.items()
    elif media_type == "multipart/form-data":
        fields = (await request.form()).multi_items()
    else:
        raise HTTPException(415, "Expected JSON or multipart media")
    for name, value in fields:
        if isinstance(value, UploadFile) and name in _UPLOAD_FIELDS:
            continue
        if name in _SAMPLING_FIELDS and isinstance(
            value, (str, int, float, type(None))
        ):
            continue
        if (
            name == "seed"
            and isinstance(value, list)
            and all(type(v) is int for v in value)
        ):
            continue
        raise HTTPException(400, "Unsupported field or value in media gateway mode")


def read_gateway_key(key_file: str) -> bytes:
    key = Path(key_file).read_bytes().rstrip(b"\r\n")
    if not re.fullmatch(rb"[a-fA-F0-9]{64}", key):
        raise ValueError(
            "Media gateway key file must contain a 64-digit hexadecimal key"
        )

    return key


class MediaGatewayMiddleware:
    """Authenticate the gateway before parsing a body or entering the warmup queue."""

    def __init__(self, app: ASGIApp, key: bytes):
        self.app = app
        self.key = key

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = get_route_path(scope)
        if scope["method"] in {"GET", "HEAD"} and path in {"/health", "/liveness"}:
            return await self.app(scope, receive, send)
        if not (path + "/").startswith(_MEDIA_PREFIXES):
            return await Response(
                status_code=404, headers={"Cache-Control": PRIVATE_CACHE_CONTROL}
            )(scope, receive, send)

        headers = Headers(scope=scope)
        keys = headers.getlist("x-sglang-gateway-key")
        tenants = headers.getlist("x-sglang-tenant")
        if (
            len(keys) != 1
            or not hmac.compare_digest(keys[0].encode("latin-1"), self.key)
            or len(tenants) != 1
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", tenants[0])
        ):
            return await Response(
                status_code=403, headers={"Cache-Control": PRIVATE_CACHE_CONTROL}
            )(scope, receive, send)
        # Do not propagate hop credentials into handlers, traces or error reports.
        scope = dict(
            scope,
            headers=[
                (k, v)
                for k, v in scope["headers"]
                if k.lower() not in {b"x-sglang-gateway-key", b"x-sglang-tenant"}
            ],
        )
        token = media_tenant.set(tenants[0])
        try:
            await self.app(scope, receive, send)
        finally:
            media_tenant.reset(token)
