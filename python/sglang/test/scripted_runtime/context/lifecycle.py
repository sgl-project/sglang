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
    from sglang.test.scripted_runtime.req_handle import ScriptedReqHandle


def _await_control(
    ctx: ScriptedContext,
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
    ctx: ScriptedContext, *, mode: Literal["retract", "in_place"]
) -> None:
    _await_control(
        ctx,
        path="/pause_generation",
        json={"mode": mode},
        expect_type=PauseGenerationReqInput,
    )


def continue_generation(ctx: ScriptedContext, *, torch_empty_cache: bool) -> None:
    _await_control(
        ctx,
        path="/continue_generation",
        json={"torch_empty_cache": torch_empty_cache},
        expect_type=ContinueGenerationReqInput,
    )


def abort_all(ctx: ScriptedContext) -> None:
    _http_post_and_await_recv_msg(
        ctx,
        path="/abort_request",
        json={"rid": "", "abort_all": True},
        predicate=lambda obj: isinstance(obj, AbortReq) and obj.abort_all,
        description="abort-all request",
    )
    for epoch in ctx._request_epochs.values():
        epoch.abort_requested = True


def abort(
    ctx: ScriptedContext, *, handle: ScriptedReqHandle, await_arrival: bool = True
) -> None:
    epoch = handle._epoch
    if epoch is not None and ctx._request_epochs.get(handle.rid) is not epoch:
        return
    repeated = epoch is not None and epoch.abort_requested
    payload = {"rid": handle.rid, "abort_all": False}
    if not await_arrival:
        _http_post_fire_and_forget(ctx, path="/abort_request", json=payload)
    else:
        _http_post_and_await_recv_msg(
            ctx,
            path="/abort_request",
            json=payload,
            predicate=lambda obj: (
                isinstance(obj, AbortReq)
                and not obj.abort_all
                and obj.rid == handle.rid
            ),
            description=f"abort request with rid {handle.rid!r}",
            # Repeated/finished aborts are acknowledged without a scheduler message.
            allow_no_arrival=lambda: (
                epoch is None or repeated or epoch.post_future.done()
            ),
        )
    if epoch is not None:
        epoch.abort_requested = True


def flush_cache(ctx: ScriptedContext) -> None:
    _await_control(
        ctx,
        path="/flush_cache",
        json=None,
        expect_type=FlushCacheReqInput,
    )
