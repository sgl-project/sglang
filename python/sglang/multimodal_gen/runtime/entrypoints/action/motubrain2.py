# SPDX-License-Identifier: Apache-2.0
"""Serve Motubrain actions with a private continuation for each robot connection."""

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request, WebSocket

from sglang.multimodal_gen.configs.sample.motubrain2 import _ConnectionState
from sglang.multimodal_gen.runtime.entrypoints.action.api import (
    action_metadata_endpoint,
    handle_action_generation,
)
from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
    action_generation_response,
)
from sglang.multimodal_gen.runtime.entrypoints.action.ws_utils import (
    run_action_msgpack_ws,
)

router = APIRouter(tags=["actions"])
router.add_api_route("/v1/actions/metadata", action_metadata_endpoint, methods=["GET"])


def _client_fields(fields: dict | None) -> dict:
    if fields is not None and not isinstance(fields, dict):
        raise ValueError("Expected an action request mapping")
    return {
        key: value
        for key, value in (fields or {}).items()
        if key not in ("connection_state", "_motubrain2_connection_state")
    }


def prepare_motubrain2_payload(
    payload: dict[str, Any],
    *,
    session_id: str | None = None,
    connection_state: dict | None = None,
) -> None:
    """Normalize both flat and enveloped observations before action inference."""
    request = _client_fields(payload)
    if "input" in request:
        inputs = _client_fields(request["input"])
        observation = _client_fields(inputs.get("observation"))
        request["input"] = {**inputs, "observation": observation}
    else:
        observation = request
    parameters = _client_fields(request.get("parameters"))
    runtime = _client_fields(request.get("runtime"))
    request.update(parameters=parameters, runtime=runtime)
    for name, default in (("cuda_graph", False), ("prefix_cache", True)):
        value = runtime.get(name)
        if value is None:
            value = observation.get(f"enable_{name}")
        runtime[name] = default if value is None else value

    native = parameters.get(
        "return_native_outputs", observation.get("return_native_outputs")
    )
    observation["return_native_outputs"] = False if native is None else native
    if session_id is not None:
        observation["session_id"] = session_id
    if connection_state is not None:
        observation["_motubrain2_connection_state"] = _ConnectionState(connection_state)
    payload.clear()
    payload.update(request)


@router.post("/v1/actions/generations")
async def create_action_generation(request: Request):
    return await handle_action_generation(request, prepare_motubrain2_payload)


@router.websocket("/v1/actions/realtime")
async def action_realtime_ws(websocket: WebSocket):
    await _serve(websocket, raw=False)


@router.websocket("/openpi/policy")
async def openpi_policy_ws(websocket: WebSocket):
    await _serve(websocket, raw=True)


async def _serve(websocket: WebSocket, *, raw: bool) -> None:
    server_args = websocket.app.state.server_args
    session_id = uuid4().hex
    connection_state = {}

    def prepare_payload(payload: dict) -> None:
        options = payload if raw else payload.setdefault("runtime", {})
        options.setdefault("output_format", "numpy")
        prepare_motubrain2_payload(
            payload, session_id=session_id, connection_state=connection_state
        )

    def build_response(output: dict) -> dict:
        nonlocal connection_state
        connection_state = output.pop("_motubrain2_connection_state")
        if raw:
            return output
        return action_generation_response(output, server_args, preserve_numpy=True)

    await run_action_msgpack_ws(
        websocket,
        server_args,
        prepare_payload=prepare_payload,
        build_response=build_response,
    )
