"""Reusable OpenAI-to-Anthropic response conversion."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Optional

from sglang.srt.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicMessagesResponse,
    AnthropicUsage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    UsageInfo,
)

logger = logging.getLogger(__name__)

# Only values allowed by AnthropicMessagesResponse.stop_reason appear here.
# OpenAI's content_filter and abort reasons have no exact Anthropic equivalent,
# so callers log them and fall back to end_turn.
OPENAI_TO_ANTHROPIC_STOP_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def _cached_prompt_tokens(usage: UsageInfo) -> int:
    prompt_tokens_details = usage.prompt_tokens_details
    return getattr(prompt_tokens_details, "cached_tokens", 0) or 0


def _anthropic_input_tokens(usage: UsageInfo) -> int:
    prompt = usage.prompt_tokens or 0
    cached = _cached_prompt_tokens(usage)
    if cached > prompt:
        # Upstream telemetry bug: cached cannot exceed the prompt it caches.
        # Clamping silently here would hide the discrepancy from billing
        # dashboards, so make it visible at WARNING level.
        logger.warning(
            "Cached tokens (%d) exceed prompt tokens (%d); clamping "
            "input_tokens to 0. This usually indicates an upstream "
            "telemetry bug.",
            cached,
            prompt,
        )
    return max(prompt - cached, 0)


def anthropic_usage_from_openai(
    usage: Optional[UsageInfo],
    *,
    include_input: bool,
    include_output: bool,
    force_zero_output: bool = False,
) -> AnthropicUsage:
    """Convert OpenAI usage counters to Anthropic usage counters."""
    if usage is None:
        return AnthropicUsage(
            input_tokens=0 if include_input else None,
            output_tokens=0 if include_output else None,
        )

    usage_fields: dict[str, int] = {}
    cached_tokens = _cached_prompt_tokens(usage)
    if include_input:
        usage_fields["input_tokens"] = _anthropic_input_tokens(usage)
        if cached_tokens:
            usage_fields["cache_read_input_tokens"] = cached_tokens
    if include_output:
        usage_fields["output_tokens"] = (
            0 if force_zero_output else (usage.completion_tokens or 0)
        )
    return AnthropicUsage(**usage_fields)


def convert_openai_to_anthropic_response(
    response: ChatCompletionResponse,
) -> AnthropicMessagesResponse:
    """Convert a complete OpenAI chat response to an Anthropic response."""
    if not response.choices:
        return AnthropicMessagesResponse(
            content=[TextBlock(text="")],
            model=response.model,
            stop_reason="end_turn",
            usage=AnthropicUsage(input_tokens=0, output_tokens=0),
        )

    choice = response.choices[0]
    content: list[AnthropicContentBlock] = []

    # A signature is omitted when the backend does not provide one. An empty
    # signature could be mistaken for a real value by downstream verifiers.
    if choice.message.reasoning_content:
        content.append(ThinkingBlock(thinking=choice.message.reasoning_content))

    if choice.message.content:
        content.append(TextBlock(text=choice.message.content))

    if choice.message.tool_calls:
        for tool_call in choice.message.tool_calls:
            raw_args = tool_call.function.arguments
            try:
                tool_input = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Tool %r emitted invalid JSON arguments: %r — "
                    "defaulting to empty input",
                    tool_call.function.name,
                    (raw_args or "")[:200],
                )
                tool_input = {}

            content.append(
                ToolUseBlock(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=tool_input,
                )
            )

    finish_reason = choice.finish_reason or "stop"
    if finish_reason not in OPENAI_TO_ANTHROPIC_STOP_REASON:
        logger.warning(
            "Unmapped OpenAI finish_reason %r; defaulting to end_turn",
            finish_reason,
        )
    stop_reason = OPENAI_TO_ANTHROPIC_STOP_REASON.get(finish_reason, "end_turn")

    # Anthropic requires at least one content block, including for an empty
    # completion or a response stopped by a content filter.
    if not content:
        content.append(TextBlock(text=""))

    return AnthropicMessagesResponse(
        id=f"msg_{uuid.uuid4().hex}",
        content=content,
        model=response.model,
        stop_reason=stop_reason,
        usage=anthropic_usage_from_openai(
            response.usage,
            include_input=True,
            include_output=True,
        ),
    )
