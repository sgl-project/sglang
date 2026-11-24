"""Text chat processor for Dynamo's HTTP frontend and token-input workers."""

import asyncio
import json
import time
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path

import msgspec
from dynamo.frontend.thinking import (
    apply_default_thinking_mode_to_template_kwargs,
    runtime_default_thinking_mode,
)
from dynamo.frontend.usage import response_usage, update_usage_metadata
from dynamo.frontend.utils import request_id_from_context
from dynamo.llm import PythonAsyncEngine
from dynamo.llm.exceptions import InvalidArgument

from . import ChatProcessor

_SAMPLING = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
}


def _validate(request):
    messages = request.get("messages")
    if not messages or any(
        msg.get("role") not in ("system", "user", "assistant")
        or not isinstance(msg.get("content"), str)
        or msg.get("tool_calls")
        for msg in messages
    ):
        raise InvalidArgument("sglang-rust requires plain-text chat messages")
    if request.get("n", 1) != 1:
        raise InvalidArgument("sglang-rust supports a single completion (n=1)")
    if request.get("response_format") not in (None, {"type": "text"}):
        raise InvalidArgument("sglang-rust supports only text response_format")
    supported = _SAMPLING | {
        "model",
        "messages",
        "n",
        "stream",
        "stream_options",
        "stop",
        "stop_token_ids",
        "max_tokens",
        "max_completion_tokens",
        "ignore_eos",
        "skip_special_tokens",
        "cache_salt",
        "nvext",
        "chat_template_args",
        "chat_template_kwargs",
        "continue_final_message",
        "add_generation_prompt",
        "thinking",
        "reasoning_effort",
        "response_format",
        "tool_choice",
        "user",
    }
    for key, value in request.items():
        if key not in supported and value is not None:
            # Some clients explicitly disable optional OpenAI features.
            if (
                key in {"tools", "logprobs", "parallel_tool_calls", "store"}
                and not value
            ):
                continue
            raise InvalidArgument(f"sglang-rust does not support {key}")
    if request.get("tool_choice") not in (None, "none", "auto"):
        raise InvalidArgument("sglang-rust does not support tool_choice")
    if request.get("add_generation_prompt") is False and not request.get(
        "continue_final_message"
    ):
        raise InvalidArgument(
            "add_generation_prompt=false requires continue_final_message"
        )
    nvext = request.get("nvext") or {}
    extra = {k for k, v in nvext.items() if v is not None} - {
        "cache_salt",
        "extra_fields",
    }
    if extra or set(nvext.get("extra_fields") or []) - {"prompt_token_ids"}:
        raise InvalidArgument("unsupported nvext option for sglang-rust")


def _worker_request(request, ids, eos, template_stops):
    nvext = request.get("nvext") or {}
    salt = nvext.get("cache_salt") or request.get("cache_salt")
    sampling = {k: request[k] for k in _SAMPLING if request.get(k) is not None}
    if sampling.get("top_k") == 0:
        sampling["top_k"] = -1
    stop = request.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    if any(not isinstance(s, str) or not s for s in stop):
        raise InvalidArgument("stop must contain non-empty strings")
    limit = request.get("max_completion_tokens")
    ignore_eos = request.get("ignore_eos", False)
    worker = {
        "model": request["model"],
        "token_ids": ids,
        "sampling_options": sampling,
        "stop_conditions": {
            "max_tokens": request.get("max_tokens") if limit is None else limit,
            "stop": stop if ignore_eos else template_stops + stop,
            "stop_token_ids": request.get("stop_token_ids") or [],
            "ignore_eos": ignore_eos,
        },
        "output_options": {
            "skip_special_tokens": request.get("skip_special_tokens", True)
        },
        "eos_token_ids": eos,
    }
    # Dynamo's SGLang worker forwards hidden stop IDs into SamplingParams.
    if eos and not ignore_eos:
        worker["stop_conditions"]["stop_token_ids_hidden"] = eos
    if salt:
        worker["routing"] = {"cache_salt": salt}
        worker["extra_args"] = {"nvext": {"cache_salt": salt}}
    return worker


def _stop_boundary(text, stops, finished):
    """Index up to which text is safe, and whether a stop is complete."""
    first = min((i for stop in stops if (i := text.find(stop)) >= 0), default=None)
    if first is not None:
        return first, True
    held = 0
    if not finished:
        for stop in stops:
            for size in range(min(len(text), len(stop) - 1), held, -1):
                if text.endswith(stop[:size]):
                    held = size
                    break
    return len(text) - held, False


def _display_prefix(raw, shown, limit):
    if not limit or not shown:
        return ""
    if raw == shown:
        return shown[:limit]
    # A match inside a token can cut a delta that also contains a buffered
    # character or a special. Map that one cut between the two decoder outputs.
    parts = []
    for tag, start, end, left, right in SequenceMatcher(
        None, raw, shown, autojunk=False
    ).get_opcodes():
        if end <= limit:
            parts.append(shown[left:right])
        elif tag == "equal" and start < limit:
            parts.append(shown[left : left + limit - start])
    return "".join(parts)


class _StringStopDecoder:
    """Match raw decoded text and commit the corresponding display text."""

    def __init__(self, processor, tail, skip_special, stops):
        self.raw = processor.decoder(tail, False)
        self.display = processor.decoder(tail, True) if skip_special else self.raw
        self.stops = stops
        self.pending = deque()

    def push(self, tokens, finished):
        output = []
        for index, token in enumerate(tokens):
            text, matched = self._step([token], False)
            output.append(text)
            if matched:
                return "".join(output), True, index + 1
        matched = False
        if finished:
            text, matched = self._step([], True)
            output.append(text)
        return "".join(output), matched, len(tokens)

    def _step(self, ids, finished):
        raw = self.raw.push(ids, finished)
        shown = raw if self.display is self.raw else self.display.push(ids, finished)
        if raw or shown:
            self.pending.append((raw, shown))
        buffered = "".join(raw for raw, _ in self.pending)
        limit, matched = _stop_boundary(buffered, self.stops, finished)
        output = []
        while self.pending:
            raw, shown = self.pending[0]
            if len(raw) > limit:
                # Retain the entire boundary token until the stop resolves. Only
                # a terminal match can require cutting a token's display text.
                if matched:
                    output.append(_display_prefix(raw, shown, limit))
                break
            self.pending.popleft()
            output.append(shown)
            limit -= len(raw)
        return "".join(output), matched


class DynamoChat:
    def __init__(self, processor, routed_engine, eos, default_thinking_mode=None):
        self.processor = processor
        self.routed_engine = routed_engine
        self.eos = eos
        self.template_stops = processor.template_stops
        self.default_thinking_mode = default_thinking_mode

    def _encode(self, request):
        # The Dynamo wire name for OpenAI chat_template_kwargs is chat_template_args.
        kwargs = (
            request.get("chat_template_args")
            or request.get("chat_template_kwargs")
            or {}
        )
        kwargs = apply_default_thinking_mode_to_template_kwargs(
            kwargs,
            self.default_thinking_mode,
            request_has_root_thinking=any(
                request.get(key) is not None for key in ("thinking", "reasoning_effort")
            ),
        )
        if kwargs:
            request = dict(request, chat_template_kwargs=kwargs)
        return self.processor.encode_chat(msgspec.json.encode(request))

    async def generate(self, request, context=None):
        _validate(request)
        try:
            ids = await asyncio.to_thread(self._encode, request)
        except ValueError as error:
            raise InvalidArgument(str(error)) from error
        worker = _worker_request(request, ids, self.eos, self.template_stops)
        stops = worker["stop_conditions"]
        skip_special = worker["output_options"]["skip_special_tokens"]
        if stops["stop"]:
            decoder = _StringStopDecoder(
                self.processor, ids[-5:], skip_special, stops["stop"]
            )
        else:
            decoder = self.processor.decoder(ids[-5:], skip_special)
        hidden = (
            set()
            if stops["ignore_eos"]
            else set(stops["stop_token_ids"]).union(self.eos)
        )
        stream = await self.routed_engine.generate(worker, context=context)
        header = {
            "id": "chatcmpl-" + request_id_from_context(context),
            "model": request["model"],
            "created": int(time.time()),
            "object": "chat.completion.chunk",
        }
        first = True
        count = 0
        metadata = {"prompt_tokens": len(ids)}
        reported_count = 0
        async for item in stream:
            if item.is_error():
                raise RuntimeError(
                    "worker stream failed: " + " ".join(item.comments() or [])
                )
            frame = item.data()
            if frame is None:
                continue
            tokens = frame["token_ids"]
            reason = frame.get("finish_reason")
            if reason in ("cancelled", "eos"):
                reason = "stop"
            if reason is not None and reason not in (
                "stop",
                "length",
                "content_filter",
            ):
                raise RuntimeError(f"worker terminated generation: {reason}")
            end = next((i for i, token in enumerate(tokens) if token in hidden), None)
            if end is not None:
                reason = "stop"
                tokens = tokens[:end]
            if stops["stop"]:
                text, matched, consumed = decoder.push(tokens, reason is not None)
                if matched:
                    reason = "stop"
            else:
                text = decoder.push(tokens, reason is not None)
                matched, consumed = False, len(tokens)
            consumed += int(end is not None and not matched)
            count += consumed
            update_usage_metadata(metadata, frame.get("completion_usage") or {})
            if not text and reason is None and not first:
                continue
            delta = {"content": text}
            if first:
                delta["role"] = "assistant"
                first = False
            data = dict(
                header, choices=[{"index": 0, "delta": delta, "finish_reason": reason}]
            )
            usage = response_usage(
                request,
                metadata,
                prompt_tokens=len(ids),
                completion_tokens=count,
                finished=reason is not None,
            )
            data["llm_metrics"] = {
                "input_tokens": len(ids),
                "output_tokens": count,
                "chunk_tokens": count - reported_count,
            }
            reported_count = count
            cached = (metadata.get("prompt_tokens_details") or {}).get("cached_tokens")
            if cached is not None:
                data["llm_metrics"]["cached_tokens"] = cached
            if usage:
                data["usage"] = usage
            if reason and "prompt_token_ids" in (
                (request.get("nvext") or {}).get("extra_fields") or []
            ):
                data["nvext"] = {"prompt_token_ids": ids}
            yield data
            if reason is not None:
                return
        raise RuntimeError("worker stream ended without a finish reason")


def _load(model_dir, template):
    processor = ChatProcessor(model_dir, template)
    eos = set()
    for name in ("config.json", "generation_config.json"):
        path = Path(model_dir) / name
        if path.exists():
            value = json.loads(path.read_bytes()).get("eos_token_id")
            if value is not None:
                eos.update(value if isinstance(value, list) else [value])
    path = Path(model_dir) / "tokenizer_config.json"
    if path.exists():
        token = json.loads(path.read_bytes()).get("eos_token")
        if isinstance(token, dict):
            token = token.get("content")
        if (
            isinstance(token, str)
            and (token_id := processor.token_id(token)) is not None
        ):
            eos.add(token_id)
    return processor, sorted(eos)


def provider(config, options):
    """Dynamo calls the returned factory for each token-input model registration."""
    if set(options) - {"chat_template"}:
        raise ValueError("sglang-rust only accepts the chat_template file option")

    async def factory(instance_id, mdc, routed_engine):
        processor, eos = await asyncio.to_thread(
            _load, str(mdc.local_dir()), options.get("chat_template")
        )
        chat = DynamoChat(
            processor,
            routed_engine,
            eos,
            runtime_default_thinking_mode(mdc.runtime_config()),
        )
        return PythonAsyncEngine(chat.generate, asyncio.get_running_loop())

    return factory
