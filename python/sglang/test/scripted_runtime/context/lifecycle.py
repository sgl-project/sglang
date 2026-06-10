from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from sglang.srt.managers.io_struct import (
    AbortReq,
    ContinueGenerationReqInput,
    FlushCacheReqInput,
    PauseGenerationReqInput,
)
from sglang.test.scripted_runtime.context.http_post import (
    _http_post_and_await_recv_msg,
    _http_post_fire_and_forget,
)

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext


def _await_control(
    ctx: "ScriptedContext",
    *,
    path: str,
    json,
    expect_type: type,
    await_arrival: bool = True,
) -> None:
    if not await_arrival:
        _http_post_fire_and_forget(ctx, path=path, json=json)
        return
    _http_post_and_await_recv_msg(
        ctx,
        path=path,
        json=json,
        predicate=lambda obj: isinstance(obj, expect_type),
        description=expect_type.__name__,
    )


def pause_generation(
    ctx: "ScriptedContext", *, mode: Literal["retract", "in_place"]
) -> None:
    _await_control(
        ctx,
        path="/pause_generation",
        json={"mode": mode},
        expect_type=PauseGenerationReqInput,
    )


def continue_generation(ctx: "ScriptedContext", *, torch_empty_cache: bool) -> None:
    _await_control(
        ctx,
        path="/continue_generation",
        json={"torch_empty_cache": torch_empty_cache},
        expect_type=ContinueGenerationReqInput,
    )


def abort_all(ctx: "ScriptedContext") -> None:
    _await_control(
        ctx,
        path="/abort_request",
        json={"rid": "", "abort_all": True},
        expect_type=AbortReq,
    )


def abort(ctx: "ScriptedContext", *, rid: str, await_arrival: bool = True) -> None:
    _await_control(
        ctx,
        path="/abort_request",
        json={"rid": rid, "abort_all": False},
        expect_type=AbortReq,
        await_arrival=await_arrival,
    )


def flush_cache(ctx: "ScriptedContext") -> None:
    _await_control(
        ctx,
        path="/flush_cache",
        json=None,
        expect_type=FlushCacheReqInput,
    )
