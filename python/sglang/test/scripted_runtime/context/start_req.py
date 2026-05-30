"""Free functions backing the ``start_req`` verb of :class:`ScriptedContext`.

``start_req`` is the most complex script-facing verb — it fires a real
``/generate`` HTTP call on a background thread, drains the wrapped tokenizer
socket until the request arrives, and registers a :class:`ScriptedReqHandle`.
Each function takes the facade ``ctx`` as its first argument and reads shared
state through it.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Optional

from sglang.test.scripted_runtime.req_handle import ScriptedReqHandle

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext

logger = logging.getLogger(__name__)

START_REQ_ARRIVAL_TIMEOUT_S: float = 60.0


def start_req(
    ctx: "ScriptedContext",
    *,
    prompt_len: int,
    max_new_tokens: int,
    rid: Optional[str],
) -> ScriptedReqHandle:
    """Submit a synthetic request via a real ``/generate`` HTTP call.

    Fires the request asynchronously (a background thread streams and
    discards the response, so the scheduler never blocks on it), then
    drains the wrapped ``recv_from_tokenizer`` socket until the request
    carrying this ``rid`` has been buffered by the proxy. The request
    has arrived but is not yet handed to the scheduler — it is popped on
    the next ``yield`` (next ``recv_requests`` iteration), keeping each
    step deterministic.
    """
    assert ctx._is_driver, "start_req is only callable from the driver rank"
    if rid is None:
        rid = f"scripted-{ctx._req_counter}-{uuid.uuid4().hex}"
        ctx._req_counter += 1

    _post_generate_async(
        ctx, rid=rid, prompt_len=prompt_len, max_new_tokens=max_new_tokens
    )
    ctx._tokenizer_recv_proxy.wait_until_arrived(
        lambda obj: getattr(obj, "rid", None) == rid,
        timeout_s=START_REQ_ARRIVAL_TIMEOUT_S,
        description=f"request with rid {rid!r}",
    )

    handle = ScriptedReqHandle(rid=rid, scheduler_hook=ctx._scheduler_hook)
    ctx._req_handles[rid] = handle
    return handle


def _generate_url(ctx: "ScriptedContext") -> str:
    host = ctx._scheduler.server_args.host
    port = ctx._scheduler.server_args.port
    return f"http://{host}:{port}/generate"


def _post_generate_async(
    ctx: "ScriptedContext",
    *,
    rid: str,
    prompt_len: int,
    max_new_tokens: int,
) -> None:
    """Fire a ``/generate`` request fire-and-forget on the shared async loop.

    The scheduler must never block waiting for the HTTP response, so the
    streamed POST runs as a coroutine on the hook's single background event
    loop (``ScriptedSchedulerHook.submit_coro``) with the body drained and
    discarded. The request carries the caller-supplied ``rid`` so the proxy
    can match it on the socket. Token id 1 is BOS for most tokenizers; any
    valid token works since the harness does not validate decode quality.
    """
    url = _generate_url(ctx)
    payload = {
        "input_ids": [1] * prompt_len,
        "sampling_params": {"max_new_tokens": max_new_tokens},
        "rid": rid,
        "stream": True,
    }

    async def _post() -> None:
        try:
            await ctx._http_poster.post_and_drain(url, payload)
        except Exception:  # noqa: BLE001 — fire-and-forget background coroutine
            logger.exception("scripted_runtime: /generate request rid=%s failed", rid)

    ctx._http_poster.submit_coro(_post())
