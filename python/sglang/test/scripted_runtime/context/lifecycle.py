from __future__ import annotations

from typing import TYPE_CHECKING, Generator, Literal

from sglang.srt.managers.io_struct import (
    AbortReq,
    ContinueGenerationReqInput,
    FlushCacheReqInput,
    PauseGenerationReqInput,
)
from sglang.test.scripted_runtime.context.http_post import (
    _http_post_and_await_recv_msg,
)

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext


def _await_control(
    ctx: "ScriptedContext", *, path: str, json, expect_type: type
) -> None:
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


def flush_cache(ctx: "ScriptedContext", *, assert_flushed: bool = True) -> Generator:
    flush_wrapper = ctx._scheduler.flush_wrapper
    flush_wrapper.last_success = None
    _await_control(
        ctx,
        path="/flush_cache",
        json=None,
        expect_type=FlushCacheReqInput,
    )
    # The scheduler consumes the just-arrived FlushCacheReqInput during this same
    # recv_requests() call, but only after the script yields control back to it
    # (the hook step runs before _pull_raw_reqs). Yield once so the flush has
    # actually run and recorded its result before we read it.
    yield
    if assert_flushed:
        assert flush_wrapper.last_success is True, (
            f"flush_cache did not flush the cache: scheduler.flush_cache() "
            f"returned {flush_wrapper.last_success!r} (the scheduler was likely "
            f"not fully idle)"
        )
