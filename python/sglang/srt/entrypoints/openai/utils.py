import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from sglang.srt.entrypoints.openai.protocol import (
    CachedTokensDetails,
    ChatCompletionRequest,
    CompletionRequest,
    LogProbs,
)

logger = logging.getLogger(__name__)


def to_openai_style_logprobs(
    input_token_logprobs=None,
    output_token_logprobs=None,
    input_top_logprobs=None,
    output_top_logprobs=None,
):
    ret_logprobs = LogProbs()

    def append_token_logprobs(token_logprobs):
        for logprob, _, token_text in token_logprobs:
            ret_logprobs.tokens.append(token_text)
            ret_logprobs.token_logprobs.append(logprob)

            # Not supported yet
            ret_logprobs.text_offset.append(-1)

    def append_top_logprobs(top_logprobs):
        for tokens in top_logprobs:
            if tokens is not None:
                ret_logprobs.top_logprobs.append(
                    {token[2]: token[0] for token in tokens}
                )
            else:
                ret_logprobs.top_logprobs.append(None)

    if input_token_logprobs is not None:
        append_token_logprobs(input_token_logprobs)
    if output_token_logprobs is not None:
        append_token_logprobs(output_token_logprobs)
    if input_top_logprobs is not None:
        append_top_logprobs(input_top_logprobs)
    if output_top_logprobs is not None:
        append_top_logprobs(output_top_logprobs)

    return ret_logprobs


def process_hidden_states_from_ret(
    ret_item: Dict[str, Any],
    request: Union[
        ChatCompletionRequest,
        CompletionRequest,
    ],
) -> Optional[List]:
    """Process hidden states from a ret item in non-streaming response.

    Args:
        ret_item: Response item containing meta_info
        request: The original request object

    Returns:
        Processed hidden states for the last token, or None
    """
    if not request.return_hidden_states:
        return None

    hidden_states = ret_item["meta_info"].get("hidden_states", None)
    if hidden_states is not None:
        hidden_states = hidden_states[-1] if len(hidden_states) > 1 else []
    return hidden_states


def process_routed_experts_from_ret(
    ret_item: Dict[str, Any],
    request: Union[
        ChatCompletionRequest,
        CompletionRequest,
    ],
) -> Optional[str]:
    """Process routed experts from a ret item in non-streaming response."""
    if not getattr(request, "return_routed_experts", False):
        return None
    return ret_item["meta_info"].get("routed_experts", None)


def process_cached_tokens_details_from_ret(
    ret_item: Dict[str, Any],
    request: Union[
        ChatCompletionRequest,
        CompletionRequest,
    ],
) -> Optional[CachedTokensDetails]:
    """Process cached tokens details from a ret item in non-streaming response."""
    if not getattr(request, "return_cached_tokens_details", False):
        return None

    details = ret_item["meta_info"].get("cached_tokens_details", None)
    if details is None:
        return None

    # Check if L3 storage fields are present
    if "storage" in details:
        return CachedTokensDetails(
            device=details.get("device", 0),
            host=details.get("host", 0),
            storage=details.get("storage", 0),
            storage_backend=details.get("storage_backend"),
        )
    else:
        return CachedTokensDetails(
            device=details.get("device", 0),
            host=details.get("host", 0),
        )

@dataclass
class ChatTemplateTokenizationResult:
    token_ids: list[int]
    meta_info: Dict[str, Any] = field(default_factory=dict)


_DUMMY_USER = {"role": "user", "content": "dummy"}

def apply_chat_template_to_additional_message(
    new_messages: list[dict[str, Any]],
    tokenizer,
    tools=None,
    reasoning_effort=None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    pretokenized_last_token_id: Optional[int] = None,
) -> ChatTemplateTokenizationResult:
    if chat_template_kwargs is None:
        chat_template_kwargs = {}
    dummy_assistant = _build_dummy_assistant(new_messages)
    # Add a dummy tool response here to avoid templates like glm-4.7 use
    # <|observation|> as the start of consecutive tool responses in chat
    # template, but also use <|observation|> as the stop token in assistant
    # message.
    base_messages = [_DUMMY_USER, dummy_assistant, _build_dummy_tool_response()]

    messages_without = base_messages
    messages_with = base_messages + new_messages

    tokens_with = tokenizer.apply_chat_template(
        messages_with,
        tokenize=True,
        add_generation_prompt=True,
        tools=tools,
        reasoning_effort=reasoning_effort,
        return_dict=False,
        **chat_template_kwargs,
    )
    tokens_without = tokenizer.apply_chat_template(
        messages_without,
        tokenize=True,
        add_generation_prompt=False,
        tools=tools,
        reasoning_effort=reasoning_effort,
        return_dict=False,
        **chat_template_kwargs,
    )

    # Validate prefix match: only raise in debug mode, otherwise warn
    prefix_mismatch = tokens_with[: len(tokens_without)] != tokens_without
    if prefix_mismatch:
        msg = (
            "Token prefix mismatch when tokenizing additional messages. "
            "This can happen for thinking models or models with special "
            "chat templates that do not produce append-only token id lists. "
            f"tokens_without_len={len(tokens_without)}, "
            f"tokens_with_len={len(tokens_with)}, "
            f"decoded_with={tokenizer.decode(tokens_with)!r}, "
            f"decoded_without={tokenizer.decode(tokens_without)!r}"
        )
        if logger.isEnabledFor(logging.DEBUG):
            raise ValueError(msg)
        logger.warning(msg)

    incremental_ids = tokens_with[len(tokens_without) :]

    # Fault tolerance: templates like qwen3 insert a trailing `\n` after
    # the last assistant message.  Its loss-mask is 0, so we only prepend
    # it for alignment when it really is a whitespace-only token.
    trailing_token_prepended = False
    if (
        pretokenized_last_token_id is not None
        and tokens_without[-1] != pretokenized_last_token_id
        and tokenizer.decode([tokens_without[-1]]).strip() == ""
    ):
        incremental_ids = [tokens_without[-1]] + incremental_ids
        trailing_token_prepended = True

    return ChatTemplateTokenizationResult(
        token_ids=incremental_ids,
        meta_info={
            "prefix_mismatch": prefix_mismatch,
            "trailing_token_prepended": trailing_token_prepended,
        },
    )


def _build_dummy_assistant(tool_responses: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": " ",
        "tool_calls": [
            {
                "id": resp.get("tool_call_id", f"call0000{i}"),
                "type": "function",
                "function": {
                    "name": resp.get("name", "dummy_func"),
                    "arguments": {},
                },
            }
            for i, resp in enumerate(tool_responses)
        ],
    }

def _build_dummy_tool_response() -> dict[str, Any]:
    return {
        "role": "tool",
        "content": "",
        "tool_call_id": "call0000",
        "type": "function",
        "function": {
            "name": "dummy_func",
            "arguments": {},
        },
    }
