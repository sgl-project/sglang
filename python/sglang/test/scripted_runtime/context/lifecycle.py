from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

from sglang.srt.managers.io_struct import (
    AbortReq,
    ContinueGenerationReqInput,
    FlushCacheReqInput,
    PauseGenerationReqInput,
)

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext

logger = logging.getLogger(__name__)

CONTROL_ARRIVAL_TIMEOUT_S: float = 60.0


def _server_url(ctx: "ScriptedContext", path: str) -> str:
    server_args = ctx._scheduler.server_args
    return f"http://{server_args.host}:{server_args.port}{path}"


def _post_and_await_control(
    ctx: "ScriptedContext",
    *,
    path: str,
    json: Optional[Dict[str, Any]],
    expect_type: type,
    timeout_s: float = CONTROL_ARRIVAL_TIMEOUT_S,
) -> None:
    url = _server_url(ctx, path)

    async def _post() -> None:
        try:
            await ctx._http_poster.post(url, json)
        except Exception:  # noqa: BLE001 — fire-and-forget control POST
            logger.exception("scripted_runtime: control POST %s failed", path)

    ctx._http_poster.submit_coro(_post())
    ctx._tokenizer_recv_proxy.wait_until_arrived(
        lambda obj: isinstance(obj, expect_type),
        timeout_s=timeout_s,
        description=expect_type.__name__,
    )


def pause_generation(
    ctx: "ScriptedContext", *, mode: Literal["retract", "in_place"]
) -> None:
    assert ctx._is_driver, "pause_generation is only callable from the driver rank"
    _post_and_await_control(
        ctx,
        path="/pause_generation",
        json={"mode": mode},
        expect_type=PauseGenerationReqInput,
    )


def continue_generation(ctx: "ScriptedContext", *, torch_empty_cache: bool) -> None:
    assert ctx._is_driver, "continue_generation is only callable from the driver rank"
    _post_and_await_control(
        ctx,
        path="/continue_generation",
        json={"torch_empty_cache": torch_empty_cache},
        expect_type=ContinueGenerationReqInput,
    )


def abort_all(ctx: "ScriptedContext") -> None:
    assert ctx._is_driver, "abort_all is only callable from the driver rank"
    _post_and_await_control(
        ctx,
        path="/abort_request",
        json={"rid": "", "abort_all": True},
        expect_type=AbortReq,
    )


def flush_cache(ctx: "ScriptedContext") -> None:
    assert ctx._is_driver, "flush_cache is only callable from the driver rank"
    _post_and_await_control(
        ctx,
        path="/flush_cache",
        json=None,
        expect_type=FlushCacheReqInput,
    )
