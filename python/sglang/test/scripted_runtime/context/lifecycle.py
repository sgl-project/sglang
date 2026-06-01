from __future__ import annotations

from typing import TYPE_CHECKING, Generator, Literal

from sglang.srt.managers.io_struct import (
    AbortReq,
    ContinueGenerationReqInput,
    FlushCacheReqInput,
    PauseGenerationReqInput,
)
from sglang.test.scripted_runtime.context import radix
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
    _await_control(
        ctx,
        path="/flush_cache",
        json=None,
        expect_type=FlushCacheReqInput,
    )
    # The scheduler consumes the just-arrived FlushCacheReqInput during this same
    # recv_requests() call, but only after the script yields control back to it
    # (the hook step runs before _pull_raw_reqs). Yield once so the flush is
    # actually applied before we verify it.
    yield
    if assert_flushed:
        _assert_cache_flushed(ctx)


def _assert_cache_flushed(ctx: "ScriptedContext") -> None:
    scheduler = ctx._scheduler
    kv_allocator = scheduler.token_to_kv_pool_allocator
    req_pool = scheduler.req_to_token_pool

    problems: list[str] = []

    node_count = len(radix.get_all_node_hit_counts(ctx))
    if node_count:
        problems.append(f"radix tree still holds {node_count} node(s)")

    kv_available = kv_allocator.available_size()
    if kv_available != ctx._fully_free_kv_size:
        problems.append(
            f"kv allocator not fully free: "
            f"{kv_available}/{ctx._fully_free_kv_size} tokens available"
        )

    req_available = req_pool.available_size()
    if req_available != ctx._fully_free_req_slots:
        problems.append(
            f"req_to_token pool not fully free: "
            f"{req_available}/{ctx._fully_free_req_slots} slots available"
        )

    assert not problems, "flush_cache did not fully flush the cache: " + "; ".join(
        problems
    )
