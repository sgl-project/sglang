from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Optional

from sglang.test.scripted_runtime.req_handle import ScriptedReqHandle

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext

logger = logging.getLogger(__name__)

START_REQ_ARRIVAL_TIMEOUT_S: float = 60.0


class ScriptedContextReqStarter:
    def __init__(self, ctx: "ScriptedContext") -> None:
        self._ctx = ctx
        self._req_counter = 0

    def start_req(
        self,
        *,
        prompt_len: int,
        max_new_tokens: int,
        rid: Optional[str],
    ) -> ScriptedReqHandle:
        ctx = self._ctx

        if rid is None:
            rid = f"scripted-{self._req_counter}-{uuid.uuid4().hex}"
            self._req_counter += 1

        self._post_generate_async(
            rid=rid, prompt_len=prompt_len, max_new_tokens=max_new_tokens
        )
        ctx._tokenizer_recv_proxy.wait_until_arrived(
            lambda obj: getattr(obj, "rid", None) == rid,
            timeout_s=START_REQ_ARRIVAL_TIMEOUT_S,
            description=f"request with rid {rid!r}",
        )

        ctx._started_rids.add(rid)
        return ScriptedReqHandle(rid=rid, context=ctx)

    def _generate_url(self) -> str:
        host = self._ctx._scheduler.server_args.host
        port = self._ctx._scheduler.server_args.port
        return f"http://{host}:{port}/generate"

    def _post_generate_async(
        self,
        *,
        rid: str,
        prompt_len: int,
        max_new_tokens: int,
    ) -> None:
        ctx = self._ctx

        url = self._generate_url()
        payload = {
            "input_ids": [1] * prompt_len,
            "sampling_params": {"max_new_tokens": max_new_tokens},
            "rid": rid,
            "stream": True,
        }

        async def _post() -> None:
            try:
                await ctx._http_poster.post(url, payload)
            except Exception:  # noqa: BLE001 — fire-and-forget background coroutine
                logger.exception("scripted_runtime: /generate request rid=%s failed", rid)

        ctx._http_poster.submit_coro(_post())
