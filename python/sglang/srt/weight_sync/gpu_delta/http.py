"""HTTP integration for the GPU-delta control plane."""

from collections.abc import Callable
from http import HTTPStatus
from typing import Annotated

from fastapi import Body, FastAPI
from fastapi.responses import ORJSONResponse

from sglang.srt.utils.auth import AuthLevel, auth_level
from sglang.srt.weight_sync.gpu_delta.io import (
    AbortWeightsDeltaReqInput,
    ApplyWeightsDeltaReqInput,
    ClearWeightsDeltaStateReqInput,
    GetWeightsDeltaInfoReqInput,
    GetWeightsDeltaStatusReqInput,
    PrepareWeightsDeltaReqInput,
    ResumeWeightsDeltaReqInput,
    UpdateWeightsFromDeltaReqInput,
)


def register_gpu_delta_routes(app: FastAPI, get_global_state: Callable):
    """Use the app's route class and resolve its tokenizer manager per request."""

    async def dispatch(obj):
        content = await get_global_state().tokenizer_manager.gpu_delta.request(obj)
        return ORJSONResponse(
            content,
            status_code=HTTPStatus.OK if content["success"] else HTTPStatus.CONFLICT,
        )

    @app.post("/update_weights_from_delta")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def update_weights_from_delta(
        obj: Annotated[UpdateWeightsFromDeltaReqInput, Body()],
    ):
        return await dispatch(obj)

    @app.post("/clear_weights_delta_state")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def clear_weights_delta_state(
        obj: Annotated[ClearWeightsDeltaStateReqInput, Body()],
    ):
        return await dispatch(obj)

    @app.post("/get_weights_delta_info")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def get_weights_delta_info(
        obj: Annotated[GetWeightsDeltaInfoReqInput, Body()],
    ):
        return await dispatch(obj)

    @app.post("/prepare_weights_delta")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def prepare_weights_delta(
        obj: Annotated[PrepareWeightsDeltaReqInput, Body()],
    ):
        return await dispatch(obj)

    @app.post("/get_weights_delta_status")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def get_weights_delta_status(
        obj: Annotated[GetWeightsDeltaStatusReqInput, Body()],
    ):
        return await dispatch(obj)

    @app.post("/apply_weights_delta")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def apply_weights_delta(
        obj: Annotated[ApplyWeightsDeltaReqInput, Body()],
    ):
        return await dispatch(obj)

    @app.post("/resume_weights_delta")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def resume_weights_delta(
        obj: Annotated[ResumeWeightsDeltaReqInput, Body()],
    ):
        return await dispatch(obj)

    @app.post("/abort_weights_delta")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    async def abort_weights_delta(
        obj: Annotated[AbortWeightsDeltaReqInput, Body()],
    ):
        return await dispatch(obj)
