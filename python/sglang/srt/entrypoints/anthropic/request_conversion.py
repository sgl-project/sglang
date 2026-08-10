"""Reusable Anthropic-to-OpenAI request conversion."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from typing import Any, Optional, Union

from pydantic import BaseModel

from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicMessagesRequest,
    is_server_tool,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    StreamOptions,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)

logger = logging.getLogger(__name__)


def _extract_system_text(
    content: Union[str, list[AnthropicContentBlock]],
) -> Optional[str]:
    """Flatten a system message's content to a trimmed string, or ``None``."""
    if isinstance(content, str):
        return content.strip() or None
    texts = []
    for block in content:
        if isinstance(block, BaseModel) and getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
        else:
            continue
        text = (text or "").strip()
        if text:
            texts.append(text)
    return "\n".join(texts) if texts else None


def convert_anthropic_to_openai_request(
    anthropic_request: AnthropicMessagesRequest,
    *,
    merge_inline_system: bool,
    wrap_reasoning_history: Callable[[str], str],
    apply_reasoning_enabled: Callable[[ChatCompletionRequest, bool], None],
) -> ChatCompletionRequest:
    """Convert an Anthropic Messages request to an OpenAI chat request.

    The callbacks isolate model- and chat-template-specific reasoning behavior
    from the protocol conversion so non-HTTP consumers can reuse this function.
    """
    openai_messages = []

    def _convert_anthropic_image_source_to_openai_part(
        source: Any,
    ) -> Optional[dict]:
        # Source may arrive as a Pydantic model (typed ImageBlock.source)
        # or as a raw dict when parsed from a nested tool_result payload.
        if isinstance(source, BaseModel):
            source = source.model_dump(exclude_none=True)
        if not isinstance(source, dict):
            return None

        source_type = source.get("type")
        if source_type == "base64":
            media_type = source.get("media_type", "image/png")
            data = source.get("data", "")
            if not data:
                return None
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{data}",
                },
            }

        url = source.get("url")
        if url:
            return {
                "type": "image_url",
                "image_url": {
                    "url": url,
                },
            }

        return None

    def _text_from_search_result(item: dict[str, Any]) -> str:
        search_parts = []
        title = item.get("title")
        if title:
            search_parts.append(f"Title: {title}")

        source = item.get("source")
        if isinstance(source, dict):
            source_text = source.get("url") or source.get("text")
            if source_text:
                search_parts.append(f"Source: {source_text}")
        elif source:
            search_parts.append(f"Source: {source}")

        content = item.get("content")
        content_parts = []
        if isinstance(content, str):
            content_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and part.get("text"):
                    content_parts.append(part["text"])
        if content_parts:
            search_parts.append("Content: " + "\n".join(content_parts))

        return "\n".join(search_parts)

    def _convert_tool_result_content(
        content: Any,
    ) -> tuple[list[Union[str, list[dict]]], str]:
        if isinstance(content, list):
            tool_content_parts = []
            tool_text_parts = []

            for raw_item in content:
                # Items may be typed Pydantic blocks (after request
                # validation) or raw dicts (from legacy callers). Coerce
                # to dict so the existing key-based logic still works.
                if isinstance(raw_item, BaseModel):
                    item = raw_item.model_dump(exclude_none=True)
                elif isinstance(raw_item, dict):
                    item = raw_item
                else:
                    continue

                item_type = item.get("type")
                if item_type == "text":
                    text = item.get("text", "")
                    if text:
                        tool_text_parts.append(text)
                        tool_content_parts.append({"type": "text", "text": text})
                elif item_type == "image":
                    image_part = _convert_anthropic_image_source_to_openai_part(
                        item.get("source")
                    )
                    if image_part is not None:
                        tool_content_parts.append(image_part)
                elif item_type == "tool_reference":
                    # Anthropic uses `tool_name`; the SGLang chat template
                    # matches on `name`. Translate at the boundary.
                    ref_name = item.get("tool_name") or item.get("name")
                    if ref_name:
                        tool_content_parts.append(
                            {"type": "tool_reference", "name": ref_name}
                        )
                elif item_type == "search_result":
                    search_text = _text_from_search_result(item)
                    if search_text:
                        tool_text_parts.append(search_text)
                        tool_content_parts.append({"type": "text", "text": search_text})

            tool_text = "\n".join(tool_text_parts)
            # GLM templates expand references only at the start of a tool
            # message, so isolate reference runs without changing part order.
            tool_content_groups: list[list[dict]] = []
            for part in tool_content_parts:
                is_reference = part["type"] == "tool_reference"
                if (
                    not tool_content_groups
                    or (tool_content_groups[-1][0]["type"] == "tool_reference")
                    != is_reference
                ):
                    tool_content_groups.append([])
                tool_content_groups[-1].append(part)

            tool_contents: list[Union[str, list[dict]]] = []
            for group in tool_content_groups:
                if len(group) == 1 and group[0]["type"] == "text":
                    tool_contents.append(group[0]["text"])
                else:
                    tool_contents.append(group)
            return tool_contents or [""], tool_text

        tool_text = str(content) if content else ""
        return [tool_text], tool_text

    def _convert_assistant_thinking_blocks(
        blocks: list[AnthropicContentBlock],
    ) -> Optional[str]:
        """Re-wrap prior-turn thinking blocks in the parser's own tokens.

        ``redacted_thinking`` carries encrypted bytes that no local parser can
        interpret, so we raise rather than silently drop it. On non-reasoning
        models the rewrap is best-effort: log a warning and drop the thinking
        text so a history echo does not reject the whole request.
        """
        if any(block.type == "redacted_thinking" for block in blocks):
            raise ValueError("Anthropic redacted_thinking history is not supported")

        thinking_parts = [
            block.thinking
            for block in blocks
            if block.type == "thinking" and block.thinking
        ]
        if not thinking_parts:
            return None

        try:
            return wrap_reasoning_history("\n".join(thinking_parts))
        except ValueError as e:
            logger.warning(
                "Dropping prior-turn thinking history (%d blocks): %s",
                len(thinking_parts),
                e,
            )
            return None

    system_parts: list[str] = []
    if anthropic_request.system:
        if isinstance(anthropic_request.system, str):
            if anthropic_request.system.strip():
                system_parts.append(anthropic_request.system)
        else:
            for block in anthropic_request.system:
                if block.type == "text" and block.text:
                    system_parts.append(block.text)

    if merge_inline_system:
        for msg in anthropic_request.messages:
            if msg.role != "system":
                continue
            text = _extract_system_text(msg.content)
            if text:
                system_parts.append(text)

    if system_parts:
        openai_messages.append({"role": "system", "content": "\n".join(system_parts)})

    def _emit_user_message(parts: list[dict]) -> None:
        """Append accumulated parts as a user message, then clear them."""
        if not parts:
            return
        if len(parts) == 1 and parts[0]["type"] == "text":
            openai_messages.append({"role": "user", "content": parts[0]["text"]})
        else:
            openai_messages.append({"role": "user", "content": list(parts)})
        parts.clear()

    # Convert messages
    for msg in anthropic_request.messages:
        if msg.role == "system" and merge_inline_system:
            continue
        if isinstance(msg.content, str):
            openai_messages.append({"role": msg.role, "content": msg.content})
            continue

        # Complex content with blocks
        openai_msg = {"role": msg.role}
        content_parts: list[dict] = []
        tool_calls: list[dict] = []

        if msg.role == "assistant":
            reasoning_history = _convert_assistant_thinking_blocks(msg.content)
            if reasoning_history is not None:
                content_parts.append({"type": "text", "text": reasoning_history})

        for block in msg.content:
            # ``thinking``/``redacted_thinking`` blocks are surfaced via
            # the reasoning-history reconstruction above; skip them here
            # to avoid double-injecting their text into the prompt.
            if block.type in ("thinking", "redacted_thinking"):
                continue

            # ``is not None`` (not truthy) so an empty-string text block
            # still produces a placeholder text part — without it, an
            # assistant turn whose only content is "" vanishes and
            # subsequent user→user pairs trip strict chat templates.
            if block.type == "text" and block.text is not None:
                content_parts.append({"type": "text", "text": block.text})

            elif block.type == "image" and block.source:
                image_part = _convert_anthropic_image_source_to_openai_part(
                    block.source
                )
                if image_part is not None:
                    content_parts.append(image_part)

            elif block.type == "search_result":
                search_text = _text_from_search_result(block.model_dump())
                if search_text:
                    content_parts.append({"type": "text", "text": search_text})

            elif block.type == "tool_use":
                tool_call = {
                    "id": block.id or f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": block.name or "",
                        "arguments": json.dumps(block.input or {}),
                    },
                }
                tool_calls.append(tool_call)

            elif block.type == "tool_result":
                tool_contents, tool_text = _convert_tool_result_content(block.content)

                # Use tool_use_id (per spec) with fallback to id
                tool_call_id = block.tool_use_id or block.id or ""

                # Tool results from user become separate tool messages.
                # Flush any pending text/image first so the wire order is
                # preserved.
                if msg.role == "user":
                    _emit_user_message(content_parts)
                    for tool_content in tool_contents:
                        openai_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": tool_content,
                            }
                        )
                else:
                    content_parts.append(
                        {
                            "type": "text",
                            "text": f"Tool result: {tool_text}",
                        }
                    )

        # Attach tool calls to assistant messages
        if tool_calls:
            openai_msg["tool_calls"] = tool_calls

        # Attach content
        if content_parts:
            if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                openai_msg["content"] = content_parts[0]["text"]
            else:
                openai_msg["content"] = content_parts
            openai_messages.append(openai_msg)
        elif tool_calls:
            openai_messages.append(openai_msg)
        elif msg.role == "user":
            # User turn that was entirely tool_results — the tool
            # messages were already emitted above, nothing left.
            continue
        else:
            # Assistant turn with no content and no tool_calls: emit
            # an empty-string placeholder so strict templates still
            # see a valid role-alternation sequence.
            openai_msg["content"] = ""
            openai_messages.append(openai_msg)

    # Build ChatCompletionRequest
    request_data = {
        "messages": openai_messages,
        "model": anthropic_request.model,
        "max_tokens": anthropic_request.max_tokens,
        "stream": anthropic_request.stream or False,
    }

    if anthropic_request.temperature is not None:
        request_data["temperature"] = anthropic_request.temperature
    if anthropic_request.top_p is not None:
        request_data["top_p"] = anthropic_request.top_p
    if anthropic_request.top_k is not None:
        request_data["top_k"] = anthropic_request.top_k
    if anthropic_request.stop_sequences is not None:
        request_data["stop"] = anthropic_request.stop_sequences

    # Enable usage in stream so we can report it
    if anthropic_request.stream:
        request_data["stream_options"] = StreamOptions(
            include_usage=True,
            continuous_usage_stats=True,
        )

    chat_request = ChatCompletionRequest(**request_data)

    if anthropic_request.thinking is not None:
        # The protocol layer already enforces the SDK shape. The local backend
        # has no equivalent hard budget, so accept it and make the limitation
        # visible to operators.
        if anthropic_request.thinking.budget_tokens is not None:
            logger.warning(
                "Anthropic thinking.budget_tokens=%d is accepted for "
                "SDK compatibility but the local backend has no "
                "equivalent hard-cap knob — the budget is not enforced",
                anthropic_request.thinking.budget_tokens,
            )
        enabled = anthropic_request.thinking.type != "disabled"
        if anthropic_request.thinking.display == "omitted":
            logger.warning(
                "Anthropic thinking.display='omitted' is accepted for "
                "SDK compatibility but reasoning text will still be "
                "emitted to the client"
            )
        apply_reasoning_enabled(chat_request, enabled)

    # Claude 4.7 ``output_config``: map ``effort`` onto the OpenAI
    # ``reasoning_effort`` knob. ``xhigh`` collapses to ``max`` because
    # the OpenAI Literal does not include the Anthropic-only ``xhigh``.
    if anthropic_request.output_config is not None:
        output_config = anthropic_request.output_config
        if output_config.effort is not None:
            chat_request.reasoning_effort = (
                "max" if output_config.effort == "xhigh" else output_config.effort
            )
        if output_config.task_budget is not None:
            logger.info(
                "Anthropic output_config.task_budget hint: %d %s",
                output_config.task_budget.total,
                output_config.task_budget.type,
            )

    # The local backend has no equivalent beta system; accept-and-log so
    # requests do not fail validation.
    if anthropic_request.betas:
        logger.info(
            "Anthropic request opted into betas %s — no-op locally",
            anthropic_request.betas,
        )

    # Deferred tools remain in the list with defer_loading=True; the chat
    # template decides when to render them.
    if anthropic_request.tools:
        converted_tools = []
        for tool in anthropic_request.tools:
            if is_server_tool(tool):
                logger.info(
                    "Skipping built-in Anthropic server tool %r (type=%r): "
                    "no native support in the OpenAI-compatible backend",
                    tool.name,
                    tool.type,
                )
                continue

            converted_tools.append(
                Tool(
                    type="function",
                    defer_loading=tool.defer_loading,
                    function={
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                )
            )

        if converted_tools:
            chat_request.tools = converted_tools

    if anthropic_request.tool_choice is not None:
        tool_choice_type = anthropic_request.tool_choice.type
        if tool_choice_type == "none":
            chat_request.tool_choice = "none"
        elif chat_request.tools:
            if tool_choice_type == "auto":
                chat_request.tool_choice = "auto"
            elif tool_choice_type == "any":
                chat_request.tool_choice = "required"
            elif tool_choice_type == "tool":
                tool_name = anthropic_request.tool_choice.name
                if not any(
                    tool.function.name == tool_name for tool in chat_request.tools
                ):
                    raise ValueError(
                        f"tool_choice references tool {tool_name!r} but it "
                        f"is not in the forwarded tools list "
                        f"(server-side Anthropic tools cannot be selected)"
                    )
                chat_request.tool_choice = ToolChoice(
                    type="function",
                    function=ToolChoiceFuncName(name=tool_name),
                )
        elif tool_choice_type in ("any", "tool"):
            raise ValueError(
                f"tool_choice={tool_choice_type!r} requires at least one custom "
                f"tool; all supplied tools were server-side Anthropic "
                f"built-ins which the OpenAI-compatible backend cannot invoke"
            )
    elif chat_request.tools:
        chat_request.tool_choice = "auto"

    return chat_request
