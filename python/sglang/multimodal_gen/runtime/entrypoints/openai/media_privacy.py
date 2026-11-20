"""Bind media jobs to a tenant asserted by an authenticated gateway."""

import hmac
import re
from pathlib import Path

from fastapi import HTTPException, Request

from sglang.multimodal_gen.runtime.entrypoints.openai.stores import media_tenant

PRIVATE_CACHE_CONTROL = "private, no-store"


def gateway_tenant_dependency(key_file: str):
    key = Path(key_file).read_bytes().rstrip(b"\r\n")
    if not re.fullmatch(rb"[a-fA-F0-9]{64}", key):
        raise ValueError(
            "Media gateway key file must contain a 64-digit hexadecimal key"
        )

    async def bind_tenant(request: Request):
        keys = request.headers.getlist("x-sglang-gateway-key")
        tenants = request.headers.getlist("x-sglang-tenant")
        if (
            len(keys) != 1
            or not hmac.compare_digest(keys[0].encode("latin-1"), key)
            or len(tenants) != 1
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", tenants[0])
        ):
            raise HTTPException(status_code=403, detail="Verified tenant required")
        token = media_tenant.set(tenants[0])
        try:
            yield
        finally:
            media_tenant.reset(token)

    return bind_tenant
