import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall

from sglang.srt.entrypoints.openai.protocol import (
    CachedTokensDetails,
    ChatCompletionRequest,
    CompletionRequest,
    LogProbs,
    StreamOptions,
    Tool,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.utils import random_uuid

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


def should_include_usage(
    stream_options: StreamOptions | None, stream_response_default_include_usage: bool
) -> tuple[bool, bool]:
    # When stream_options are specified in the request
    if stream_options:
        include_usage = (
            stream_options.include_usage or stream_response_default_include_usage
        )
        continuous_usage_stats = bool(stream_options.continuous_usage_stats)
    else:
        include_usage, continuous_usage_stats = (
            stream_response_default_include_usage,
            False,
        )
    return include_usage, continuous_usage_stats


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


def cached_tokens_details_from_dict(
    details: Dict[str, Any],
) -> CachedTokensDetails:
    """Convert a raw cached_tokens_details dict to a CachedTokensDetails object."""
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


def process_cached_tokens_details_from_ret(
    ret_item: Dict[str, Any],
    request: Union[
        ChatCompletionRequest,
        CompletionRequest,
    ],
) -> Optional[CachedTokensDetails]:
    """Process cached tokens details from a ret item in non-streaming response."""
    if not request.return_cached_tokens_details:
        return None

    details = ret_item["meta_info"].get("cached_tokens_details", None)
    if details is None:
        return None

    return cached_tokens_details_from_dict(details)


def convert_embeds_to_tensors(
    embeds: Optional[Union[List[Optional[List[List[float]]]], List[List[float]]]],
) -> Optional[List[Optional[List[torch.Tensor]]]]:
    """Convert nested float lists from the HTTP API to lists of tensors.

    Accepts either:
      - None -> returns None
      - List[List[float]] (single input) -> [[tensor, ...]]
      - List[Optional[List[List[float]]]] (batch) -> [Optional[List[tensor]], ...]
    Each innermost List[float] becomes a 1-D torch.Tensor.
    Per-input None entries are preserved (no overrides for that input).
    """
    if embeds is None:
        return None
    if len(embeds) == 0:
        return []
    # Find first non-None entry to detect nesting depth
    first_non_none = next((e for e in embeds if e is not None), None)
    if first_non_none is None:
        # All entries are None
        return [None] * len(embeds)
    # Detect nesting depth by checking the first non-None entry:
    # - Single input [num_replacements][hidden_size]: first element is List[float]
    # - Batch [num_inputs][num_replacements][hidden_size]: first element is List[List[float]]
    if not first_non_none or not isinstance(first_non_none[0], list):
        # Single input: each entry is a float vector
        return [[torch.tensor(vec, dtype=torch.float32) for vec in embeds]]
    # Otherwise it's batch: [num_inputs][num_replacements][hidden_size]
    return [
        (
            [torch.tensor(vec, dtype=torch.float32) for vec in per_input]
            if per_input is not None
            else None
        )
        for per_input in embeds
    ]


def parse_tool_calls_from_content(
    content: str,
    tools: List[Tool],
    tool_call_parser: str,
    generate_tool_call_id: Callable[[Any, int], str],
    history_tool_calls_cnt: int = 0,
) -> Tuple[str, List[ResponseFunctionToolCall]]:
    """Parse tool calls from model output content.

    Args:
        content: The model output text to parse
        tools: List of Tool objects (Chat API format)
        tool_call_parser: The parser type to use (e.g., "llama3", "qwen25")
        generate_tool_call_id: Function to generate tool call IDs,
            takes (call_info, history_count) and returns tool_call_id string

    Returns:
        Tuple of (remaining_text, list of ResponseFunctionToolCall objects)
    """
    tool_calls: List[ResponseFunctionToolCall] = []
    remaining_text = content

    parser = FunctionCallParser(tools, tool_call_parser)
    if parser.has_tool_call(content):
        try:
            text, call_info_list = parser.parse_non_stream(content)
            for call_info in call_info_list:
                tool_id = generate_tool_call_id(call_info, history_tool_calls_cnt)
                function_tool_call = ResponseFunctionToolCall(
                    id=f"fc_{random_uuid()[:32]}",
                    type="function_call",
                    call_id=tool_id,
                    name=call_info.name,
                    arguments=call_info.parameters or "",
                    status="completed",
                )
                tool_calls.append(function_tool_call)

            remaining_text = text
        except Exception as e:
            logger.error(f"Tool call parsing error: {e}")
            # Fall back to returning original content if parsing fails
            remaining_text = content

    return remaining_text, tool_calls
