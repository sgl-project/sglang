from __future__ import annotations

import uuid
from concurrent.futures import wait
from typing import TYPE_CHECKING, Generator, List, Optional

import orjson
from aiohttp import ClientResponseError

from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.test.scripted_runtime.context.http_post import (
    _http_post_and_await_recv_msg,
)
from sglang.test.scripted_runtime.req_handle import ScriptedReqHandle

if TYPE_CHECKING:
    from sglang.test.scripted_runtime.context.api import ScriptedContext


class ScriptedContextReqStarter:
    def __init__(self, ctx: ScriptedContext) -> None:
        self._ctx = ctx
        self._req_counter = 0

    def start_req(
        self,
        *,
        prompt_len: int,
        max_new_tokens: int,
        rid: Optional[str],
        ignore_eos: bool,
        priority: Optional[int],
        dp_rank: Optional[int],
        prompt_token: int = 1,
        return_logprob: bool = False,
        logprob_start_len: Optional[int] = None,
        top_logprobs_num: Optional[int] = None,
        stop_token_ids: Optional[List[int]] = None,
        temperature: Optional[float] = None,
        lora_path: Optional[str] = None,
    ) -> ScriptedReqHandle:
        ctx = self._ctx

        if rid is None:
            rid = f"scripted-{self._req_counter}-{uuid.uuid4().hex}"
            self._req_counter += 1

        previous = ctx._request_epochs.get(rid)
        if previous is not None and not previous.post_future.done():
            raise ValueError(
                f"Request ID {rid!r} still has an open HTTP response; "
                "use yield from start_req_with_retry(...) to wait before reusing it"
            )

        sampling_params = {"max_new_tokens": max_new_tokens, "ignore_eos": ignore_eos}
        if stop_token_ids is not None:
            sampling_params["stop_token_ids"] = stop_token_ids
        if temperature is not None:
            sampling_params["temperature"] = temperature
        payload = {
            "input_ids": [prompt_token] * prompt_len,
            "sampling_params": sampling_params,
            "rid": rid,
            # SSE response cleanup can abort a later request that reuses this rid.
            "stream": False,
        }
        payload["return_logprob"] = return_logprob
        if logprob_start_len is not None:
            payload["logprob_start_len"] = logprob_start_len
        if top_logprobs_num is not None:
            payload["top_logprobs_num"] = top_logprobs_num
        if priority is not None:
            payload["priority"] = priority
        if dp_rank is not None:
            payload["routed_dp_rank"] = dp_rank
        if lora_path is not None:
            payload["lora_path"] = lora_path
        future = _http_post_and_await_recv_msg(
            ctx,
            path="/generate",
            json=payload,
            predicate=lambda obj: (
                isinstance(obj, TokenizedGenerateReqInput) and obj.rid == rid
            ),
            description=f"request with rid {rid!r}",
        )

        return ctx._register_request(rid=rid, post_future=future)

    def start_req_with_retry(
        self, *, rid: str, max_steps: int, **kwargs
    ) -> Generator[None, None, ScriptedReqHandle]:
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        previous = self._ctx._request_epochs.get(rid)
        for step in range(max_steps + 1):
            # The old HTTP handler can still clean up by rid after Req.finished().
            if previous is not None and not previous.post_future.done():
                if step == max_steps:
                    raise TimeoutError(
                        f"Request ID {rid!r} still has an open HTTP response after "
                        f"{max_steps} scheduler steps"
                    )
                yield
                wait((previous.post_future,), timeout=0.005)
                continue
            try:
                return self._ctx.start_req(rid=rid, **kwargs)
            except ClientResponseError as error:
                if not _is_duplicate_rid_error(error=error, rid=rid):
                    raise
                if step == max_steps:
                    raise TimeoutError(
                        f"Request ID {rid!r} was not released after {max_steps} "
                        f"scheduler steps: {error.message}"
                    ) from error
            # TokenizerManager cannot release the old rid until the scheduler runs.
            yield


def _is_duplicate_rid_error(*, error: ClientResponseError, rid: str) -> bool:
    if error.status != 400:
        return False
    try:
        payload = orjson.loads(error.message)
    except orjson.JSONDecodeError:
        return False
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("error"), dict)
        and payload["error"].get("message") == f"Duplicate request ID detected: {rid}"
    )
