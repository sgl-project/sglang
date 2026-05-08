# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from sglang.omni.entrypoints.http_server import _serialize_response
from sglang.omni.protocol import OmniRequest

_SENSENOVA_U1_CACHE_KEY = "sensenova-u1"


def handle_omni_generate_from_scheduler(
    *,
    scheduler: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    request = OmniRequest.from_payload(payload)
    orchestrator = _get_sensenova_u1_orchestrator(scheduler, request)
    return _serialize_response(orchestrator.generate(request))


def _get_sensenova_u1_orchestrator(scheduler: Any, request: OmniRequest) -> Any:
    model = (request.model or _SENSENOVA_U1_CACHE_KEY).lower()
    if "sensenova" not in model and "u1" not in model:
        raise ValueError(f"Unsupported omni model {request.model!r}")

    cache = getattr(scheduler, "_omni_orchestrators", None)
    if cache is None:
        cache = {}
        setattr(scheduler, "_omni_orchestrators", cache)
    if _SENSENOVA_U1_CACHE_KEY not in cache:
        from sglang.omni.configs.sensenova_u1 import (
            build_sensenova_u1_orchestrator_from_scheduler,
        )

        cache[_SENSENOVA_U1_CACHE_KEY] = build_sensenova_u1_orchestrator_from_scheduler(
            scheduler=scheduler,
            server_args=getattr(scheduler, "server_args", None),
        )
    return cache[_SENSENOVA_U1_CACHE_KEY]
