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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

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
    # === Wishlist kwargs (additive, see 2026-05-26-round-5-de-skip-and-api-wishlist.md §5.2) ===
    prompt_tokens: Optional[List[int]],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    seed: Optional[int],
    ignore_eos: bool,
    min_new_tokens: Optional[int],
    stop: Optional[Union[str, List[str]]],
    stop_token_ids: Optional[List[int]],
    repetition_penalty: Optional[float],
    frequency_penalty: Optional[float],
    presence_penalty: Optional[float],
    return_logprob: bool,
    top_logprobs_num: Optional[int],
    logprob_start_len: Optional[int],
    priority: Optional[int],
    lora_path: Optional[str],
    session_id: Optional[str],
    dp_rank: Optional[int],
    return_hidden_states: bool,
    grammar: Optional[str],
    stream: bool,
) -> ScriptedReqHandle:
    """Submit a synthetic request via a real ``/generate`` HTTP call.

    Fires the request asynchronously (a background thread streams and
    discards the response, so the scheduler never blocks on it), then
    drains the wrapped ``recv_from_tokenizer`` socket until the request
    carrying this ``rid`` has been buffered by the proxy. The request
    has arrived but is not yet handed to the scheduler — it is popped on
    the next ``yield`` (next ``recv_requests`` iteration), keeping each
    step deterministic.

    The first three parameters (``prompt_len``, ``max_new_tokens``,
    ``rid``) are fully implemented. All other keyword-only parameters
    are wishlist additions described in
    ``2026-05-26-round-5-de-skip-and-api-wishlist.md`` §5.2 — passing
    any of them to a non-default value raises ``NotImplementedError``
    until the engine-side wiring lands.
    """
    assert ctx._is_driver, "start_req is only callable from the driver rank"
    _check_start_req_wishlist_kwargs(
        prompt_tokens=prompt_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        ignore_eos=ignore_eos,
        min_new_tokens=min_new_tokens,
        stop=stop,
        stop_token_ids=stop_token_ids,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        return_logprob=return_logprob,
        top_logprobs_num=top_logprobs_num,
        logprob_start_len=logprob_start_len,
        priority=priority,
        lora_path=lora_path,
        session_id=session_id,
        dp_rank=dp_rank,
        return_hidden_states=return_hidden_states,
        grammar=grammar,
        stream=stream,
    )
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


def _check_start_req_wishlist_kwargs(
    *,
    prompt_tokens: Optional[List[int]],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    seed: Optional[int],
    ignore_eos: bool,
    min_new_tokens: Optional[int],
    stop: Optional[Union[str, List[str]]],
    stop_token_ids: Optional[List[int]],
    repetition_penalty: Optional[float],
    frequency_penalty: Optional[float],
    presence_penalty: Optional[float],
    return_logprob: bool,
    top_logprobs_num: Optional[int],
    logprob_start_len: Optional[int],
    priority: Optional[int],
    lora_path: Optional[str],
    session_id: Optional[str],
    dp_rank: Optional[int],
    return_hidden_states: bool,
    grammar: Optional[str],
    stream: bool,
) -> None:
    """Raise NotImplementedError for any non-default wishlist kwarg.

    Centralised so each wishlist kwarg surfaces with a specific name in
    the error message, making the unimplemented-call site easy to find.
    """
    non_default_optionals: Dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "min_new_tokens": min_new_tokens,
        "stop": stop,
        "stop_token_ids": stop_token_ids,
        "repetition_penalty": repetition_penalty,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "top_logprobs_num": top_logprobs_num,
        "logprob_start_len": logprob_start_len,
        "priority": priority,
        "lora_path": lora_path,
        "session_id": session_id,
        "dp_rank": dp_rank,
        "grammar": grammar,
    }
    for name, value in non_default_optionals.items():
        if value is not None:
            raise NotImplementedError(
                f"scripted_runtime: start_req kwarg '{name}' is wishlist"
            )
    non_default_flags: Dict[str, bool] = {
        "ignore_eos": ignore_eos,
        "return_logprob": return_logprob,
        "return_hidden_states": return_hidden_states,
        "stream": stream,
    }
    for name, flag in non_default_flags.items():
        if flag:
            raise NotImplementedError(
                f"scripted_runtime: start_req kwarg '{name}' is wishlist"
            )
