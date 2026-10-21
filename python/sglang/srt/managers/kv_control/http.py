# SPDX-License-Identifier: Apache-2.0
"""HTTP facade for the same typed scheduler control service used by Engine."""

from fastapi.responses import JSONResponse

from sglang.srt.utils.auth import AuthLevel, auth_level


def install_routes(app, get_manager):
    async def send(command):
        reply = await get_manager().kv_control(command)
        code = reply.get("error", {}).get("code")
        status = (
            200
            if reply["success"]
            else {
                "unsupported": 503,
                "capacity": 429,
                "missing": 404,
                "conflict": 409,
                "stale_epoch": 409,
                "wrong_engine": 409,
                "incompatible_namespace": 409,
                "busy": 409,
            }.get(code, 400)
        )
        return JSONResponse(reply, status_code=status)

    @app.get("/v1/kv/capabilities")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def capabilities():
        return await send({"kind": "capabilities"})

    @app.post("/v1/kv/control")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def control(command: dict):
        return await send(command)
