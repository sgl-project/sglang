# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, WebSocket

from sglang.multimodal_gen.runtime.entrypoints.vla.ws_utils import (
    run_action_msgpack_ws,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

router = APIRouter()


def _prefer_numpy_output(observation: dict[str, Any]) -> None:
    observation.setdefault("output_format", "numpy")


@router.websocket("/openpi/policy")
async def openpi_policy_ws(websocket: WebSocket):
    server_args: ServerArgs = websocket.app.state.server_args
    await run_action_msgpack_ws(
        websocket,
        server_args,
        prepare_payload=_prefer_numpy_output,
        build_response=lambda output: output,
    )
