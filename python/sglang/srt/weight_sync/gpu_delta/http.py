"""HTTP integration for the GPU-delta control plane."""

from collections.abc import Callable
from http import HTTPStatus
from typing import Annotated

from fastapi import Body, FastAPI
from fastapi.responses import ORJSONResponse

from sglang.srt.utils.auth import AuthLevel, auth_level
from sglang.srt.weight_sync.gpu_delta.io import (
    AbortGpuDeltaReqInput,
    ApplyGpuDeltaReqInput,
    ClearGpuDeltaStateReqInput,
    GetGpuDeltaInfoReqInput,
    GetGpuDeltaStatusReqInput,
    PrepareGpuDeltaReqInput,
    ResumeGpuDeltaReqInput,
    UpdateWeightsFromGpuDeltaReqInput,
)


def register_gpu_delta_routes(app: FastAPI, get_global_state: Callable):
    """Use the app's route class and resolve its tokenizer manager per request."""

    async def dispatch(obj):
        content = await get_global_state().tokenizer_manager.gpu_delta.request(obj)
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.OK if content["success"] else HTTPStatus.CONFLICT,
        )

    @app.post("/update_weights_from_gpu_delta")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def update_weights_from_gpu_delta(
        obj: Annotated[UpdateWeightsFromGpuDeltaReqInput, Body()],
    ):
        return await dispatch(obj)

    @app.post("/clear_gpu_delta_state")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def clear_gpu_delta_state(
        obj: Annotated[ClearGpuDeltaStateReqInput, Body()],
    ):
        return await dispatch(obj)

    @app.post("/get_gpu_delta_info")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def get_gpu_delta_info(
        obj: Annotated[GetGpuDeltaInfoReqInput, Body()],
    ):
        return await dispatch(obj)

    @app.post("/prepare_gpu_delta")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def prepare_gpu_delta(
        obj: Annotated[PrepareGpuDeltaReqInput, Body()],
    ):
        return await dispatch(obj)

    @app.post("/get_gpu_delta_status")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def get_gpu_delta_status(
        obj: Annotated[GetGpuDeltaStatusReqInput, Body()],
    ):
        return await dispatch(obj)

    @app.post("/apply_gpu_delta")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def apply_gpu_delta(
        obj: Annotated[ApplyGpuDeltaReqInput, Body()],
    ):
        return await dispatch(obj)

    @app.post("/resume_gpu_delta")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def resume_gpu_delta(
        obj: Annotated[ResumeGpuDeltaReqInput, Body()],
    ):
        return await dispatch(obj)

    @app.post("/abort_gpu_delta")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def abort_gpu_delta(
        obj: Annotated[AbortGpuDeltaReqInput, Body()],
    ):
        return await dispatch(obj)
