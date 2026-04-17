"""Fallback B: post-inference auto-recovery for unclosed reasoning sections.

When a chat completion finishes with non-empty reasoning_content but
essentially empty content (and no tool calls), this module transparently
issues a follow-up generation that appends the think_end token to the
existing output_ids and continues, then merges the two responses before
returning to the client.

This acts as a safety net for the redirect logit processor (Path 1) and
also works on multi-token think_end models for which Path 1 cannot run.

The module is intentionally split into small pure helpers and a thin async
orchestration function so it can be unit-tested without bringing up a real
server.
"""

from __future__ import annotations

import copy
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
)
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.reasoning_parser import ReasoningParser

logger = logging.getLogger(__name__)


# --- decision helpers ----------------------------------------------------


def is_recovery_enabled(
    request: ChatCompletionRequest, server_default: bool
) -> bool:
    """Per-request override > server-side default."""
    user = request.recover_unclosed_reasoning
    return server_default if user is None else bool(user)


def is_recovery_eligible_request(
    request: ChatCompletionRequest, reasoning_parser_name: Optional[str]
) -> bool:
    """Quick gating that does not depend on the actual generation result."""
    if not reasoning_parser_name:
        return False
    if not request.separate_reasoning:
        return False
    if request.n is not None and request.n != 1:
        # n > 1 produces multiple ret items; this simplified recovery path
        # supports n == 1 only. (When n is None we treat it as 1.)
        return False
    # Only hard-exclude when the caller FORCES a tool call ("required"): the
    # model is contractually obliged to emit one, so an "empty content + only
    # reasoning" response is expected and must not be overwritten by a
    # follow-up generation.
    # For tool_choice in {"auto", "none", <specific tool>} we defer the
    # decision to the runtime check in should_recover_ret_item /
    # mark_stream_state_tool_call, which inspects whether a tool call was
    # actually produced.
    if request.tools and request.tool_choice == "required":
        return False
    if request.continue_final_message:
        # User is already explicitly continuing; do not double-tap.
        return False
    return True


def should_recover_ret_item(
    ret_item: Dict[str, Any],
    parser: ReasoningParser,
    *,
    tool_call_parser: Optional[str] = None,
    tools: Optional[List[Any]] = None,
) -> bool:
    """True iff this ret item looks like an unclosed reasoning emission.

    Conditions:
      - finish_reason.type == "stop" (length-stops are user-visible budget hits)
      - reasoning_text is non-empty after parsing
      - normal_text is empty (or whitespace only) after parsing
      - when tools were offered: the normal_text does NOT already contain a
        tool call (otherwise the "empty content" is intentional — the content
        IS the tool call).
    """
    finish = (ret_item.get("meta_info") or {}).get("finish_reason") or {}
    if finish.get("type") != "stop":
        return False

    text = ret_item.get("text") or ""
    try:
        reasoning_text, normal_text = parser.parse_non_stream(text)
    except Exception:
        return False
    if not reasoning_text:
        return False
    if (normal_text or "").strip():
        return False

    # Only when tools were offered AND the caller configured a tool call
    # parser do we bother checking for actual tool-call markers in the
    # generated text. If any is found, skip recovery (the "empty content" is
    # intentional — the model is calling a tool).
    if tool_call_parser and tools:
        try:
            from sglang.srt.function_call.function_call_parser import (
                FunctionCallParser,
            )

            fcp = FunctionCallParser(tools, tool_call_parser)
            # Check the raw generated text, not the post-reasoning-parser
            # normal_text, because some models emit the tool-call sigil
            # inside the reasoning block's tail, or the tool-call parser's
            # start tokens may themselves look like "reasoning". has_tool_call
            # inspects the full unparsed text.
            if fcp.has_tool_call(text):
                return False
        except Exception:
            # Be conservative: if we can't tell, allow recovery. The worst
            # case is the continuation generates extra text after what the
            # model considered "done".
            pass
    return True


# --- request building ----------------------------------------------------


_REDIRECT_OWNED_KEYS = (
    "think_end_token_id",
    "think_start_token_id",
    "redirect_eos_token_ids",
    "force_reasoning",
    "prob_threshold",
)

# Sentinel placed in the continuation request's `sampling_params.custom_params`
# so that, in the (forbidden) case the orchestration helpers ever recursively
# observe a follow-up generation, they immediately bail out instead of
# launching another continuation. The key lives under `custom_params` because
# the top-level `sampling_params` dict is forwarded to `SamplingParams(**…)`
# which would reject an unknown kwarg, but `custom_params` is a free-form dict.
_RECOVERY_SENTINEL_KEY = "_recover_unclosed_reasoning_attempted"


def _has_recovery_sentinel(adapted_request: GenerateReqInput) -> bool:
    sp = getattr(adapted_request, "sampling_params", None)
    if isinstance(sp, list):
        sp = sp[0] if sp else None
    if not isinstance(sp, dict):
        return False
    cp = sp.get("custom_params")
    return isinstance(cp, dict) and bool(cp.get(_RECOVERY_SENTINEL_KEY))


def _strip_redirect_owned_keys(custom_params: Optional[Dict[str, Any]]):
    """Remove the redirect processor's per-request keys from a custom_params
    dict. Returns None if nothing meaningful remains."""
    if not isinstance(custom_params, dict):
        return None
    cleaned = {k: v for k, v in custom_params.items() if k not in _REDIRECT_OWNED_KEYS}
    return cleaned or None


def build_continuation_request(
    *,
    adapted_request: GenerateReqInput,
    ret_item: Dict[str, Any],
    think_end_token_ids: List[int],
    rid: Optional[str] = None,
) -> Optional[GenerateReqInput]:
    """Build the follow-up GenerateReqInput that continues with think_end.

    Returns None if the request shape is not recoverable (text-mode prompt,
    batched input, missing input_ids, etc.).
    """
    original_input_ids = getattr(adapted_request, "input_ids", None)
    if not original_input_ids:
        return None
    if isinstance(original_input_ids[0], list):
        # Batched input → multi-prompt request, not supported by this path.
        return None

    output_ids = ret_item.get("output_ids") or []
    new_input_ids = (
        list(original_input_ids) + list(output_ids) + list(think_end_token_ids)
    )

    new_sp = copy.deepcopy(getattr(adapted_request, "sampling_params", None) or {})
    if isinstance(new_sp, list):
        new_sp = new_sp[0] if new_sp else {}
    if not isinstance(new_sp, dict):
        new_sp = {}
    cleaned_cp = _strip_redirect_owned_keys(new_sp.get("custom_params"))
    sentinel_cp = dict(cleaned_cp) if isinstance(cleaned_cp, dict) else {}
    sentinel_cp[_RECOVERY_SENTINEL_KEY] = True
    new_sp["custom_params"] = sentinel_cp

    extras: Dict[str, Any] = {}
    for k in ("lora_path", "routed_dp_rank"):
        v = getattr(adapted_request, k, None)
        if v is not None:
            extras[k] = v

    return GenerateReqInput(
        input_ids=new_input_ids,
        sampling_params=new_sp,
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        stream=False,
        return_text_in_logprobs=False,
        rid=rid or str(uuid.uuid4()),
        # NOTE: explicitly do NOT carry custom_logit_processor — the redirect
        # processor must not run on the continuation.
        **extras,
    )


# --- response merging ----------------------------------------------------


def merge_continuation_into_ret(
    *,
    original_ret_item: Dict[str, Any],
    follow_ret_item: Dict[str, Any],
    think_end_str: str,
    think_end_token_ids: List[int],
) -> Dict[str, Any]:
    """Build a new ret_item that concatenates `original + bridge + follow`."""
    merged_text = (
        (original_ret_item.get("text") or "")
        + think_end_str
        + (follow_ret_item.get("text") or "")
    )
    merged_output_ids = (
        list(original_ret_item.get("output_ids") or [])
        + list(think_end_token_ids)
        + list(follow_ret_item.get("output_ids") or [])
    )

    merged_meta = dict(original_ret_item.get("meta_info") or {})
    follow_meta = follow_ret_item.get("meta_info") or {}

    if "finish_reason" in follow_meta:
        merged_meta["finish_reason"] = follow_meta["finish_reason"]

    for k in ("completion_tokens", "cached_tokens", "reasoning_tokens"):
        a = merged_meta.get(k)
        b = follow_meta.get(k)
        if isinstance(a, int) and isinstance(b, int):
            merged_meta[k] = a + b

    # prompt_tokens: the follow-up's prefill already consumed (original prompt
    # + first-pass output_ids + think_end bridge). Summing a+b would
    # double-count the original prompt; taking max() reflects the true compute
    # cost for billing without over-reporting.
    a_pt = merged_meta.get("prompt_tokens")
    b_pt = follow_meta.get("prompt_tokens")
    if isinstance(a_pt, int) and isinstance(b_pt, int):
        merged_meta["prompt_tokens"] = max(a_pt, b_pt)
    elif isinstance(b_pt, int):
        merged_meta["prompt_tokens"] = b_pt

    new_item = dict(original_ret_item)
    new_item["text"] = merged_text
    new_item["output_ids"] = merged_output_ids
    new_item["meta_info"] = merged_meta
    return new_item


# --- async orchestration -------------------------------------------------


async def maybe_recover_non_streaming(
    *,
    handler,
    adapted_request: GenerateReqInput,
    request: ChatCompletionRequest,
    raw_request,
    ret_list: List[Dict[str, Any]],
    generate_request_fn: Optional[Callable[..., Awaitable]] = None,
) -> List[Dict[str, Any]]:
    """Apply Fallback B to a non-streaming response.

    `handler` must expose:
      - `tokenizer_manager.server_args.auto_recover_unclosed_reasoning`
      - `tokenizer_manager.tokenizer` (must implement .encode())
      - `tokenizer_manager.generate_request(...)` (async iterator)
      - `template_manager.force_reasoning`
      - `_get_reasoning_from_request(request)`
      - `reasoning_parser` (str)

    `generate_request_fn` is for testing — when None we use
    `handler.tokenizer_manager.generate_request`.
    """
    server_default = bool(
        getattr(
            handler.tokenizer_manager.server_args,
            "auto_recover_unclosed_reasoning",
            False,
        )
    )
    if not is_recovery_enabled(request, server_default):
        return ret_list
    if not is_recovery_eligible_request(request, handler.reasoning_parser):
        return ret_list
    # Anti-recursion: if this request is itself a continuation we issued
    # earlier, never trigger another round.
    if _has_recovery_sentinel(adapted_request):
        return ret_list

    template_manager = getattr(handler, "template_manager", None)
    is_force_reasoning = bool(
        getattr(template_manager, "force_reasoning", False)
    ) or bool(handler._get_reasoning_from_request(request))

    # Capture tool-call context for runtime judgement of "did the model
    # actually emit a tool call?". When either is missing we fall through to
    # the plain reasoning-only check.
    tool_call_parser = getattr(handler, "tool_call_parser", None)
    tools = getattr(request, "tools", None)

    if generate_request_fn is None:

        def generate_request_fn(req, rr):  # type: ignore[no-redef]
            return handler.tokenizer_manager.generate_request(req, rr)

    new_ret_list: List[Dict[str, Any]] = []
    for ret_item in ret_list:
        try:
            parser = ReasoningParser(
                model_type=handler.reasoning_parser,
                stream_reasoning=False,
                force_reasoning=is_force_reasoning,
                request=request,
            )
        except Exception:
            new_ret_list.append(ret_item)
            continue

        if not should_recover_ret_item(
            ret_item,
            parser,
            tool_call_parser=tool_call_parser,
            tools=tools,
        ):
            new_ret_list.append(ret_item)
            continue

        think_end_str = parser.detector.think_end_token
        try:
            think_end_token_ids = handler.tokenizer_manager.tokenizer.encode(
                think_end_str, add_special_tokens=False
            )
        except Exception:
            new_ret_list.append(ret_item)
            continue
        if not think_end_token_ids:
            new_ret_list.append(ret_item)
            continue

        follow_req = build_continuation_request(
            adapted_request=adapted_request,
            ret_item=ret_item,
            think_end_token_ids=think_end_token_ids,
        )
        if follow_req is None:
            new_ret_list.append(ret_item)
            continue

        try:
            follow_ret = await generate_request_fn(
                follow_req, raw_request
            ).__anext__()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "Fallback B continuation failed (rid=%s): %s",
                follow_req.rid,
                e,
            )
            new_ret_list.append(ret_item)
            continue

        if isinstance(follow_ret, list):
            follow_ret = follow_ret[0] if follow_ret else None
        if not isinstance(follow_ret, dict):
            new_ret_list.append(ret_item)
            continue

        merged = merge_continuation_into_ret(
            original_ret_item=ret_item,
            follow_ret_item=follow_ret,
            think_end_str=think_end_str,
            think_end_token_ids=think_end_token_ids,
        )
        new_ret_list.append(merged)

    return new_ret_list


# --- streaming Fallback B ------------------------------------------------


def new_stream_recovery_state() -> Dict[str, Any]:
    """Per-index state we track during a streaming request to decide whether
    to issue a Fallback B continuation after the first stream finishes.
    """
    return {
        "reasoning_text": "",
        "content_text": "",
        "has_tool_calls": False,
        "finish_reason": None,
        "last_output_ids": [],
        "last_meta_info": {},
    }


def update_stream_state_with_reasoning(state: Dict[str, Any], delta: str):
    if delta:
        state["reasoning_text"] += delta


def update_stream_state_with_content(state: Dict[str, Any], delta: str):
    if delta:
        state["content_text"] += delta


def mark_stream_state_tool_call(state: Dict[str, Any]):
    state["has_tool_calls"] = True


def update_stream_state_meta(
    state: Dict[str, Any],
    meta_info: Dict[str, Any],
    output_ids: List[int],
    *,
    incremental: bool = False,
):
    """Fold one stream chunk's meta / output_ids into the recovery state.

    ``incremental`` MUST match the server's ``incremental_streaming_output``
    setting:

    - ``incremental=False`` (default): each chunk carries the cumulative
      output_ids list so far; we overwrite.
    - ``incremental=True``: each chunk carries only the delta; we append so
      the continuation request's ``input_ids`` include ALL previously emitted
      tokens, not just the last delta.
    """
    state["last_meta_info"] = dict(meta_info or {})
    if output_ids:
        if incremental:
            state["last_output_ids"].extend(output_ids)
        else:
            state["last_output_ids"] = list(output_ids)
    fr = (meta_info or {}).get("finish_reason")
    if fr:
        state["finish_reason"] = fr


def should_recover_stream_state(state: Dict[str, Any]) -> bool:
    """Mirrors should_recover_ret_item but works on accumulated stream text."""
    fr = state.get("finish_reason") or {}
    if fr.get("type") != "stop":
        return False
    if state.get("has_tool_calls"):
        return False
    if not state.get("reasoning_text"):
        return False
    if (state.get("content_text") or "").strip():
        return False
    return True


async def stream_recovery_chunks(
    *,
    handler,
    request: ChatCompletionRequest,
    raw_request,
    adapted_request: GenerateReqInput,
    state: Dict[str, Any],
    request_id: str,
    request_model: str,
    continuous_usage_stats: bool,
    prompt_tokens: int,
    reasoning_tokens_acc: int,
    completion_tokens_acc: int,
    generate_request_fn: Optional[Callable[..., Any]] = None,
    on_meta_update: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> AsyncGenerator[str, None]:
    """Issue the continuation generation as a stream and yield SSE chunks.

    The generated chunks are content deltas (no role chunk, no reasoning
    chunks). After the continuation finishes a final ``finish_reason`` chunk
    is yielded with the continuation's finish_reason. Caller is responsible
    for the trailing ``[DONE]`` line and any usage chunks.

    NOTE: Every early-return path (parser build fails, tokenizer encode
    fails, continuation request shape invalid, ...) MUST still yield a
    fallback finish_reason chunk before returning — otherwise the upstream
    caller's ``continue`` will skip the normal finish chunk and the client
    stream would lose its content-termination marker.
    """

    def _fallback_finish_chunk() -> str:
        fr = state.get("finish_reason")
        fc = ChatCompletionStreamResponse(
            id=request_id,
            created=int(time.time()),
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason=fr["type"] if fr else "stop",
                    matched_stop=(
                        fr.get("matched") if isinstance(fr, dict) else None
                    ),
                )
            ],
            model=request_model,
        )
        return f"data: {fc.model_dump_json()}\n\n"

    parser_name = handler.reasoning_parser
    if not parser_name:
        yield _fallback_finish_chunk()
        return

    is_force_reasoning = bool(
        getattr(getattr(handler, "template_manager", None), "force_reasoning", False)
    ) or bool(handler._get_reasoning_from_request(request))

    try:
        parser = ReasoningParser(
            model_type=parser_name,
            stream_reasoning=False,
            force_reasoning=is_force_reasoning,
            request=request,
        )
    except Exception:
        yield _fallback_finish_chunk()
        return
    think_end_str = parser.detector.think_end_token

    try:
        think_end_token_ids = handler.tokenizer_manager.tokenizer.encode(
            think_end_str, add_special_tokens=False
        )
    except Exception:
        yield _fallback_finish_chunk()
        return
    if not think_end_token_ids:
        yield _fallback_finish_chunk()
        return

    pseudo_ret_item = {
        "output_ids": list(state.get("last_output_ids") or []),
        "text": state.get("reasoning_text", ""),
        "meta_info": state.get("last_meta_info") or {},
    }
    follow_req = build_continuation_request(
        adapted_request=adapted_request,
        ret_item=pseudo_ret_item,
        think_end_token_ids=think_end_token_ids,
    )
    if follow_req is None:
        yield _fallback_finish_chunk()
        return
    # Streaming continuation: ask for incremental text deltas if available.
    follow_req.stream = True

    if generate_request_fn is None:

        def generate_request_fn(req, rr):  # type: ignore[no-redef]
            return handler.tokenizer_manager.generate_request(req, rr)

    last_seen_text = ""
    final_finish_reason = state.get("finish_reason")
    incremental_streaming = bool(
        getattr(handler.tokenizer_manager.server_args, "incremental_streaming_output", False)
    )
    last_follow_meta: Dict[str, Any] = {}

    try:
        async for content in generate_request_fn(follow_req, raw_request):
            meta = content.get("meta_info") or {}
            if meta:
                last_follow_meta = meta
            text = content.get("text") or ""
            if incremental_streaming:
                delta = text
            else:
                delta = text[len(last_seen_text) :]
                last_seen_text = text

            if delta:
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(content=delta),
                    finish_reason=None,
                )
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=int(time.time()),
                    choices=[choice_data],
                    model=request_model,
                )
                if continuous_usage_stats:
                    # meta fields from the continuation are cumulative over
                    # the continuation (completion_tokens == len(output_ids)
                    # of the follow-up). Add onto the caller's first-pass
                    # accumulator so per-chunk usage is monotone non-decreasing.
                    cur_completion = (
                        completion_tokens_acc + int(meta.get("completion_tokens") or 0)
                    )
                    chunk.usage = UsageProcessor.calculate_token_usage(
                        prompt_tokens=prompt_tokens,
                        reasoning_tokens=reasoning_tokens_acc,
                        completion_tokens=cur_completion,
                    )
                yield f"data: {chunk.model_dump_json()}\n\n"

            fr = meta.get("finish_reason")
            if fr:
                final_finish_reason = fr
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "Streaming Fallback B continuation crashed (rid=%s): %s",
            getattr(follow_req, "rid", "?"),
            e,
        )

    # Push the final follow-up meta back to the caller so it can accumulate
    # completion/cached/prompt_tokens for the terminal usage chunk. The meta
    # fields are cumulative over the continuation, so we only need to flush
    # once after the stream ends.
    if on_meta_update is not None and last_follow_meta:
        try:
            on_meta_update(last_follow_meta)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "stream_recovery_chunks on_meta_update failed (rid=%s): %s",
                getattr(follow_req, "rid", "?"),
                e,
            )

    final_chunk = ChatCompletionStreamResponse(
        id=request_id,
        created=int(time.time()),
        choices=[
            ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason=(
                    final_finish_reason["type"] if final_finish_reason else "stop"
                ),
                matched_stop=(
                    final_finish_reason.get("matched")
                    if final_finish_reason and isinstance(final_finish_reason, dict)
                    else None
                ),
            )
        ],
        model=request_model,
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"


def is_stream_recovery_enabled_for_request(
    handler, request: ChatCompletionRequest
) -> bool:
    """Combine server default + per-request override + eligibility gating."""
    server_default = bool(
        getattr(
            handler.tokenizer_manager.server_args,
            "auto_recover_unclosed_reasoning",
            False,
        )
    )
    if not is_recovery_enabled(request, server_default):
        return False
    if not is_recovery_eligible_request(request, handler.reasoning_parser):
        return False
    # The streaming entrypoint receives the user-facing GenerateReqInput, not
    # a continuation, but check anyway in case some future caller forwards a
    # follow-up here directly.
    return True
