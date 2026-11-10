"""Kimi-K3 serving glue for the ``kimi_k3`` chat encoding spec; output parsing
stays with the kimi_k3 text parsers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.entrypoints.openai.kimi_k3_wire import (
    GENERATION_STUB_TOKENS,
    KimiK3ChatWire,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageGenericParam,
    ChatCompletionRequest,
)

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.chat_encoding_adapters import ChatOutputHooks

KIMI_K3_IMAGE_PLACEHOLDER = "<|kimi_image_placeholder|>"
KIMI_K3_IMAGE_PLACEHOLDER_ESCAPED = "<| kimi_image_placeholder |>"


# OpenAI effort tiers mapped onto the nearest K3 level; K3.1 adds "medium".
_KIMI_K3_LEVEL_BY_TIER = {
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
    "xhigh": "max",
    "ultra": "max",
    "max": "max",
}
_KIMI_K31_LEVEL_BY_TIER = {**_KIMI_K3_LEVEL_BY_TIER, "medium": "medium"}


_KIMI_K3_THINKING_TYPES = (None, "enabled", "disabled", "adaptive")


def _kimi_k3_levels_text(level_by_tier: dict) -> str:
    return ", ".join(repr(level) for level in dict.fromkeys(level_by_tier.values()))


def _map_kimi_k3_effort(reasoning_effort: Any, level_by_tier: dict) -> Optional[str]:
    if reasoning_effort is None or reasoning_effort == "none":
        return None
    if isinstance(reasoning_effort, str) and reasoning_effort in level_by_tier:
        return level_by_tier[reasoning_effort]
    raise ValueError(
        f"Kimi K3 supports reasoning_effort {_kimi_k3_levels_text(level_by_tier)} "
        f"(the other OpenAI tiers map onto them); got {reasoning_effort!r}."
    )


def neutralize_kimi_k3_image_placeholder(text: str) -> str:
    return text.replace(KIMI_K3_IMAGE_PLACEHOLDER, KIMI_K3_IMAGE_PLACEHOLDER_ESCAPED)


def neutralize_kimi_k3_image_placeholder_value(value: Any) -> Any:
    if isinstance(value, str):
        return neutralize_kimi_k3_image_placeholder(value)
    if isinstance(value, list):
        return [neutralize_kimi_k3_image_placeholder_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: neutralize_kimi_k3_image_placeholder_value(item)
            for key, item in value.items()
        }
    return value


_KIMI_K3_TOOL_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,255}")


def _validate_kimi_k3_message_tools(request: ChatCompletionRequest) -> str | None:
    """Message-level tool rules of the Kimi-K3 encoding, which renders such
    tools only from a system message without content."""
    for index in request._message_tools_indices:
        message = request.messages[index]
        if message.role not in ("system", "developer"):
            return (
                f"messages[{index}]: only a system or developer message can "
                f"declare tools, got role {message.role!r}."
            )
        if message.content:
            return f"messages[{index}]: a message that declares tools must have no content."
        for tool in message.tools:
            # `type` defaults to "function", so a missing one must be caught here.
            if tool.type != "function" or "type" not in tool.model_fields_set:
                return f"messages[{index}]: tool 'type' must be 'function'."
            name = tool.function.name
            if not _KIMI_K3_TOOL_NAME_RE.fullmatch(name):
                return (
                    f"messages[{index}]: invalid tool name {name!r}: it must start "
                    "with a letter or '_', contain only letters, digits, '_' or "
                    "'-', and be at most 256 characters."
                )
    return None


class KimiK3ChatEncodingAdapter:
    collects_openai_media = True
    # The encoder parses history tool-call arguments itself and keeps the
    # original bytes of nested values; parsing them first would re-serialize
    # them and change the prompt.
    history_tool_call_arguments = "as_sent"
    renders_response_format = True
    uncounted_prompt_tokens = GENERATION_STUB_TOKENS
    output = None

    def __init__(self, *, tokenizer: Any, hf_config: Any, chat_hooks: ChatOutputHooks):
        self._tokenizer = tokenizer
        self._level_by_tier = (
            _KIMI_K31_LEVEL_BY_TIER
            if hf_config.model_type == "kimi_k31"
            else _KIMI_K3_LEVEL_BY_TIER
        )
        self._hooks = chat_hooks
        self.wire = KimiK3ChatWire(tokenizer=tokenizer)

    def validate_request(self, request: ChatCompletionRequest) -> Optional[str]:
        """Request rules of the K3 serving contract that the encoder cannot enforce."""
        error = _validate_kimi_k3_message_tools(request)
        if error:
            return error
        # `thinking` stays a plain dict on the request; `keep` is accepted and ignored.
        thinking = request.thinking
        if thinking is not None and not (
            isinstance(thinking, dict)
            and thinking.get("type") in _KIMI_K3_THINKING_TYPES
        ):
            return (
                f"invalid thinking: {thinking!r}; expected an object whose type, if "
                "set, is 'enabled', 'disabled' or 'adaptive'."
            )
        return self.wire.internal_content_error(request)

    def encode_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        request: ChatCompletionRequest,
        tools: Optional[list[dict]],
        thinking: bool,
        image_data: list,
    ) -> list[int]:
        # The encoder renders tools from the request and thinking from the
        # template kwargs; the handler collects the media.
        messages, image_count, assistant_prefix = self._prepare_messages(
            messages, request
        )
        template_kwargs = dict(request.chat_template_kwargs or {})
        template_kwargs.pop("tokenize", None)
        template_kwargs.pop("return_dict", None)
        template_kwargs.pop("image_prompts", None)
        # The encoder call below always passes add_generation_prompt=True.
        template_kwargs.pop("add_generation_prompt", None)
        for key in ("conversation", "tools"):
            if key in template_kwargs:
                raise ValueError(
                    f"chat_template_kwargs cannot set {key!r}; "
                    "use the request fields instead."
                )
        if image_count:
            template_kwargs["image_prompts"] = ["<|media_pad|>"] * image_count

        ctk_effort = template_kwargs.get("thinking_effort")
        if ctk_effort is not None and ctk_effort not in self._level_by_tier.values():
            raise ValueError(
                f"chat_template_kwargs.thinking_effort must be one of "
                f"{_kimi_k3_levels_text(self._level_by_tier)}; got {ctk_effort!r}."
            )
        if (
            "thinking_effort" not in template_kwargs
            and template_kwargs.get("thinking") is not False
        ):
            effort = _map_kimi_k3_effort(request.reasoning_effort, self._level_by_tier)
            if effort is not None:
                template_kwargs["thinking_effort"] = effort

        effective_tools = self._hooks.effective_tools(request)
        # The encoder renders tool_choice="none" with or without tools, so a
        # client's explicit "none" reaches it either way; the default "none"
        # of a request without tools does not.
        passes_tool_choice = (
            request.tool_choice == "required" and effective_tools
        ) or (
            request.tool_choice == "none"
            and (effective_tools or request._tool_choice_explicit)
        )
        if passes_tool_choice:
            template_kwargs.setdefault("tool_choice", request.tool_choice)
        if request.response_format is not None:
            template_kwargs.setdefault(
                "response_format",
                request.response_format.model_dump(exclude_unset=True, by_alias=True),
            )

        request_tools = (
            [
                tool.model_dump(exclude_unset=True, by_alias=True)
                for tool in request.tools
            ]
            if request.tools
            else None
        )
        prompt_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            tools=request_tools,
            return_dict=False,
            **template_kwargs,
        )
        if assistant_prefix:
            prompt_ids = self._hooks.append_assistant_prefix(
                prompt_ids, assistant_prefix
            )
        return prompt_ids

    def _prepare_messages(
        self,
        messages: list[dict[str, Any]],
        request: ChatCompletionRequest,
    ) -> tuple[list[dict[str, Any]], int, str | None]:
        image_count = 0
        for index, message in enumerate(messages):
            content = message.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    if part_type in ("text", "input_text"):
                        parts.append(
                            {
                                "type": "text",
                                "text": neutralize_kimi_k3_image_placeholder(
                                    part["text"]
                                ),
                            }
                        )
                    elif part_type in ("image_url", "input_image"):
                        image = part.get("image_url") or {}
                        if isinstance(image, str):
                            image = {"url": image, "detail": part.get("detail")}
                        parts.append({"type": "image_url", "image_url": image})
                        image_count += 1
                message["content"] = parts
            elif isinstance(content, str):
                message["content"] = neutralize_kimi_k3_image_placeholder(content)
            elif content is None:
                message["content"] = ""

            if message.get("role") == "assistant":
                for key in ("reasoning_content", "reasoning"):
                    if key in message:
                        message[key] = neutralize_kimi_k3_image_placeholder_value(
                            message[key]
                        )
                for tool_call in message.get("tool_calls") or []:
                    function = (
                        tool_call.get("function")
                        if isinstance(tool_call, dict)
                        else None
                    )
                    if isinstance(function, dict) and "arguments" in function:
                        function["arguments"] = (
                            neutralize_kimi_k3_image_placeholder_value(
                                function["arguments"]
                            )
                        )

            source = request.messages[index]
            if (
                isinstance(source, ChatCompletionMessageGenericParam)
                and source.role in ("system", "developer")
                and source.tools
            ):
                message["tools"] = [
                    tool.model_dump(exclude_unset=True, by_alias=True)
                    for tool in source.tools
                ]
            if message.get("role") == "developer":
                message["role"] = "system"

        assistant_prefix = None
        if request.continue_final_message:
            messages, assistant_prefix = self._hooks.split_assistant_prefix(
                messages, request
            )
        return messages, image_count, assistant_prefix
