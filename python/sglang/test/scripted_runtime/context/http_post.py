from __future__ import annotations

import logging
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext

logger = logging.getLogger(__name__)

RECV_MSG_ARRIVAL_TIMEOUT_S: float = 60.0
# Loopback POST to this same process; anything unsettled a second past the
# arrival deadline is stuck, not racing the arrival wait.
POST_SETTLE_TIMEOUT_S: float = 1.0


def _http_post_and_await_recv_msg(
    ctx: ScriptedContext,
    *,
    path: str,
    json: Optional[Dict[str, Any]],
    predicate: Callable[[Any], bool],
    description: str,
    timeout_s: float = RECV_MSG_ARRIVAL_TIMEOUT_S,
) -> None:
    post_future = _submit_post(ctx, path=path, json=json)
    try:
        ctx._tokenizer_recv_proxy.wait_until_arrived(
            predicate,
            timeout_s=timeout_s,
            description=description,
        )
    except TimeoutError:
        rejection = _post_rejection(post_future)
        if rejection is not None:
            raise AssertionError(
                f"POST {path} was rejected by the server, so no {description} "
                f"could arrive: {type(rejection).__name__}: {rejection}"
            ) from rejection
        raise


def _post_rejection(post_future: Future) -> Optional[BaseException]:
    try:
        return post_future.exception(timeout=POST_SETTLE_TIMEOUT_S)
    except FutureTimeoutError:
        return None


def _http_post_fire_and_forget(
    ctx: ScriptedContext,
    *,
    path: str,
    json: Optional[Dict[str, Any]],
) -> None:
    _submit_post(ctx, path=path, json=json)


def _submit_post(
    ctx: ScriptedContext,
    *,
    path: str,
    json: Optional[Dict[str, Any]],
) -> Future:
    server_args = ctx.scheduler.server_args
    url = f"http://{server_args.host}:{server_args.port}{path}"

    async def _post() -> None:
        await ctx._http_poster.post(url, json)

    return ctx._http_poster.submit_coro(_post())
