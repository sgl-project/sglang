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
        ignore_eos: bool = False,
        priority: Optional[int] = None,
        dp_rank: Optional[int] = None,
    ) -> ScriptedReqHandle:
        ctx = self._ctx

        if rid is None:
            rid = f"scripted-{self._req_counter}-{uuid.uuid4().hex}"
            self._req_counter += 1

        sampling_params = {"max_new_tokens": max_new_tokens, "ignore_eos": ignore_eos}
        payload = {
            "input_ids": [1] * prompt_len,
            "sampling_params": sampling_params,
            "rid": rid,
            "stream": True,
        }
        if priority is not None:
            payload["priority"] = priority
        if dp_rank is not None:
            payload["routed_dp_rank"] = dp_rank
        _http_post_and_await_recv_msg(
            ctx,
            path="/generate",
            json=payload,
            predicate=lambda obj: getattr(obj, "rid", None) == rid,
            description=f"request with rid {rid!r}",
        )

        ctx._started_rids.add(rid)
        return ScriptedReqHandle(rid=rid, context=ctx)
