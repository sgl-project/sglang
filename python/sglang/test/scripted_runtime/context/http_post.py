from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext

logger = logging.getLogger(__name__)

RECV_MSG_ARRIVAL_TIMEOUT_S: float = 60.0


def _http_post_and_await_recv_msg(
    ctx: ScriptedContext,
    *,
    path: str,
    json: Optional[Dict[str, Any]],
    predicate: Callable[[Any], bool],
    description: str,
    timeout_s: float = RECV_MSG_ARRIVAL_TIMEOUT_S,
) -> None:
    _submit_post(ctx, path=path, json=json)
    try:
        ctx._tokenizer_recv_proxy.wait_until_arrived(
            predicate,
            timeout_s=timeout_s,
            description=description,
        )
    except TimeoutError:
        # The POST is fire-and-forget, so a rejected request (e.g. duplicate rid)
        # would otherwise present as "nothing arrived on the socket" — pointing at
        # the transport instead of the server's actual complaint.
        failures = ctx._http_poster.take_failures()
        if failures:
            raise AssertionError(
                f"POST {path} was rejected by the server, so no {description} "
                f"could arrive: " + "; ".join(failures)
            ) from None
        raise


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
) -> None:
    server_args = ctx.scheduler.server_args
    url = f"http://{server_args.host}:{server_args.port}{path}"

    async def _post() -> None:
        await ctx._http_poster.post(url, json)

    ctx._http_poster.submit_coro(_post())
