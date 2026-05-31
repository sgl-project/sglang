from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sglang.test.scripted_runtime.context.http_post import (
    _http_post_and_await_recv_msg,
)
from sglang.test.scripted_runtime.req_handle import ScriptedReqHandle

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext


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
        ignore_eos: bool,
        prompt_token: int = 1,
    ) -> ScriptedReqHandle:
        ctx = self._ctx

        if rid is None:
            rid = f"scripted-{self._req_counter}-{uuid.uuid4().hex}"
            self._req_counter += 1

        payload = {
            "input_ids": [prompt_token] * prompt_len,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "ignore_eos": ignore_eos,
            },
            "rid": rid,
            "stream": True,
        }
        _http_post_and_await_recv_msg(
            ctx,
            path="/generate",
            json=payload,
            predicate=lambda obj: getattr(obj, "rid", None) == rid,
            description=f"request with rid {rid!r}",
        )

        return ScriptedReqHandle(rid=rid, context=ctx)
