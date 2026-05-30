"""Free functions for engine-wide control verbs.

These drive ``pause_generation`` / ``continue_generation`` / ``abort_all`` /
``flush_cache``. Each takes the facade ``ctx`` first.

All of them have an HTTP API and must apply across all ranks, so they go
through the real server like ``start_req``: fire an HTTP POST on the hook's
shared async loop, then wait until the resulting control object reaches the
wrapped ``recv_from_tokenizer`` socket (so the scheduler broadcasts it to
every rank on the next ``yield``). They never touch a scheduler object
directly.
"""

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
    """Fire a control POST on the shared async loop, then await its ZMQ obj.

    Mirrors ``start_req``: the HTTP request travels the production path
    (server -> tokenizer manager -> ZMQ PUSH) and lands on the wrapped
    ``recv_from_tokenizer`` socket; we wait until an object of
    ``expect_type`` is buffered so it is delivered on the next ``yield``.
    """
    url = _server_url(ctx, path)

    async def _post() -> None:
        try:
            await ctx._http_poster.post_no_body(url, json)
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
    """POST ``/pause_generation`` and await the PauseGenerationReqInput.

    ``mode="abort"`` is intentionally not supported here: the scheduler
    has no abort branch inside ``pause_generation``; use :meth:`abort_all`.
    """
    assert ctx._is_driver, "pause_generation is only callable from the driver rank"
    _post_and_await_control(
        ctx,
        path="/pause_generation",
        json={"mode": mode},
        expect_type=PauseGenerationReqInput,
    )


def continue_generation(ctx: "ScriptedContext", *, torch_empty_cache: bool) -> None:
    """POST ``/continue_generation`` and await the ContinueGenerationReqInput.

    Resume after :meth:`pause_generation`. The facade defaults
    ``torch_empty_cache`` to False to keep scripted runs deterministic.
    """
    assert ctx._is_driver, "continue_generation is only callable from the driver rank"
    _post_and_await_control(
        ctx,
        path="/continue_generation",
        json={"torch_empty_cache": torch_empty_cache},
        expect_type=ContinueGenerationReqInput,
    )


def abort_all(ctx: "ScriptedContext") -> None:
    """POST ``/abort_request`` (abort_all) and await the AbortReq.

    ``pause_generation(mode="abort")`` does not abort in the scheduler
    today; this is the only way to reach the abort branch from a script.
    """
    assert ctx._is_driver, "abort_all is only callable from the driver rank"
    _post_and_await_control(
        ctx,
        path="/abort_request",
        json={"rid": "", "abort_all": True},
        expect_type=AbortReq,
    )


def flush_cache(ctx: "ScriptedContext") -> None:
    """POST ``/flush_cache`` and await the FlushCacheReqInput.

    Reaches every rank through the normal request-broadcast path, so it is
    safe under TP/PP. Like ``start_req``, the flush is visible on the next
    ``yield``, and the scheduler only honors it when the engine is idle
    (no in-flight reqs). The dispatch loop issues one before each script so
    a sub-script starts from a clean cache.
    """
    assert ctx._is_driver, "flush_cache is only callable from the driver rank"
    _post_and_await_control(
        ctx,
        path="/flush_cache",
        json=None,
        expect_type=FlushCacheReqInput,
    )
