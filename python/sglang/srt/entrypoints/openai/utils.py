# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions for OpenAI API server"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

import jinja2.nodes
import transformers.utils.chat_template_utils as hf_chat_utils

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageParam,
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
    LogProbs,
    OpenAIServingRequest,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.validation import ValidationRule

logger = logging.getLogger(__name__)


# ============================================================================
# JINJA TEMPLATE CONTENT FORMAT DETECTION
# ============================================================================
#
# This adapts vLLM's approach for detecting chat template content format:
# https://github.com/vllm-project/vllm/blob/02f0c7b220422792f5e53de2a7d51d2d3ff2df28/vllm/entrypoints/chat_utils.py#L296-L313
# - Analyzes Jinja template AST to detect content iteration patterns
# - 'openai' format: templates with {%- for content in message['content'] -%} loops
# - 'string' format: templates that expect simple string content
# - Processes content accordingly to match template expectations


def _is_var_access(node: jinja2.nodes.Node, varname: str) -> bool:
    """Check if node is a variable access like {{ varname }}"""
    if isinstance(node, jinja2.nodes.Name):
        return node.ctx == "load" and node.name == varname
    return False


def _is_attr_access(node: jinja2.nodes.Node, varname: str, key: str) -> bool:
    """Check if node is an attribute access like {{ varname['key'] }} or {{ varname.key }}"""
    if isinstance(node, jinja2.nodes.Getitem):
        return (
            _is_var_access(node.node, varname)
            and isinstance(node.arg, jinja2.nodes.Const)
            and node.arg.value == key
        )

    if isinstance(node, jinja2.nodes.Getattr):
        return _is_var_access(node.node, varname) and node.attr == key

    return False


def _is_var_or_elems_access(
    node: jinja2.nodes.Node,
    varname: str,
    key: str = None,
) -> bool:
    """Check if node accesses varname or varname[key] with filters/tests"""
    if isinstance(node, jinja2.nodes.Filter):
        return node.node is not None and _is_var_or_elems_access(
            node.node, varname, key
        )
    if isinstance(node, jinja2.nodes.Test):
        return _is_var_or_elems_access(node.node, varname, key)

    if isinstance(node, jinja2.nodes.Getitem) and isinstance(
        node.arg, jinja2.nodes.Slice
    ):
        return _is_var_or_elems_access(node.node, varname, key)

    return _is_attr_access(node, varname, key) if key else _is_var_access(node, varname)


def _try_extract_ast(chat_template: str):
    """Try to parse the Jinja template into an AST"""
    try:
        jinja_compiled = hf_chat_utils._compile_jinja_template(chat_template)
        return jinja_compiled.environment.parse(chat_template)
    except Exception as e:
        logger.debug(f"Error when compiling Jinja template: {e}")
        return None


def detect_template_content_format(chat_template: str) -> str:
    """
    Detect whether a chat template expects 'string' or 'openai' content format.

    - 'string': content is a simple string (like DeepSeek templates)
    - 'openai': content is a list of structured dicts (like Llama4 templates)

    Detection logic:
    - If template has loops like {%- for content in message['content'] -%} → 'openai'
    - Otherwise → 'string'
    """
    jinja_ast = _try_extract_ast(chat_template)
    if jinja_ast is None:
        return "string"

    try:
        # Look for patterns like: {%- for content in message['content'] -%}
        for loop_ast in jinja_ast.find_all(jinja2.nodes.For):
            loop_iter = loop_ast.iter

            # Check if iterating over message['content'] or similar
            if _is_var_or_elems_access(loop_iter, "message", "content"):
                return "openai"  # Found content iteration → openai format

        return "string"  # No content loops found → string format
    except Exception as e:
        logger.debug(f"Error when parsing AST of Jinja template: {e}")
        return "string"


def process_content_for_template_format(
    msg_dict: dict,
    content_format: str,
    image_data: list,
    audio_data: list,
    modalities: list,
) -> dict:
    """
    Process message content based on detected template format.

    Args:
        msg_dict: Message dictionary with content
        content_format: 'string' or 'openai' (detected via AST analysis)
        image_data: List to append extracted image URLs
        audio_data: List to append extracted audio URLs
        modalities: List to append modalities

    Returns:
        Processed message dictionary
    """
    if not isinstance(msg_dict.get("content"), list):
        # Already a string or None, no processing needed
        return {k: v for k, v in msg_dict.items() if v is not None}

    if content_format == "openai":
        # OpenAI format: preserve structured content list, normalize types
        processed_content_parts = []
        for chunk in msg_dict["content"]:
            if isinstance(chunk, dict):
                chunk_type = chunk.get("type")

                if chunk_type == "image_url":
                    image_data.append(chunk["image_url"]["url"])
                    if chunk.get("modalities"):
                        modalities.append(chunk.get("modalities"))
                    # Normalize to simple 'image' type for template compatibility
                    processed_content_parts.append({"type": "image"})
                elif chunk_type == "audio_url":
                    audio_data.append(chunk["audio_url"]["url"])
                    # Normalize to simple 'audio' type
                    processed_content_parts.append({"type": "audio"})
                else:
                    # Keep other content as-is (text, etc.)
                    processed_content_parts.append(chunk)

        new_msg = {
            k: v for k, v in msg_dict.items() if v is not None and k != "content"
        }
        new_msg["content"] = processed_content_parts
        return new_msg

    else:  # content_format == "string"
        # String format: flatten to text only (for templates like DeepSeek)
        text_parts = []
        for chunk in msg_dict["content"]:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                text_parts.append(chunk["text"])
            # Note: For string format, we ignore images/audio since the template
            # doesn't expect structured content - multimodal placeholders would
            # need to be inserted differently

        new_msg = msg_dict.copy()
        new_msg["content"] = " ".join(text_parts) if text_parts else ""
        new_msg = {k: v for k, v in new_msg.items() if v is not None}
        return new_msg


def calculate_token_usage(
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: Optional[Dict[str, int]] = None,
) -> UsageInfo:
    """Calculate token usage information"""
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=cached_tokens,
    )


def aggregate_token_usage(
    responses: List[Dict[str, Any]],
    n_choices: int = 1,
    enable_cache_report: bool = False,
) -> UsageInfo:
    """Aggregate token usage from multiple responses

    Args:
        responses: List of response dictionaries with meta_info
        n_choices: Number of choices per request (for prompt token counting)
        enable_cache_report: Whether to include cached token details

    Returns:
        Aggregated UsageInfo
    """
    # Sum completion tokens from all responses
    completion_tokens = sum(
        response["meta_info"]["completion_tokens"] for response in responses
    )

    # For prompt tokens, only count every n_choices-th response to avoid double counting
    prompt_tokens = sum(
        responses[i]["meta_info"]["prompt_tokens"]
        for i in range(0, len(responses), n_choices)
    )

    # Handle cached tokens if cache reporting is enabled
    cached_tokens_details = None
    if enable_cache_report:
        cached_tokens_sum = sum(
            response["meta_info"].get("cached_tokens", 0) for response in responses
        )
        if cached_tokens_sum > 0:
            cached_tokens_details = {"cached_tokens": cached_tokens_sum}

    return calculate_token_usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens_details,
    )


def create_error_response(
    message: str,
    err_type: str = "BadRequestError",
    status_code: int = 400,
    param: Optional[str] = None,
) -> ErrorResponse:
    """Create an error response"""
    return ErrorResponse(
        object="error",
        message=message,
        type=err_type,
        param=param,
        code=status_code,
    )


def create_streaming_error_response(
    message: str,
    err_type: str = "BadRequestError",
    status_code: int = 400,
) -> str:
    """Create a streaming error response"""
    error = create_error_response(message, err_type, status_code)
    return json.dumps({"error": error.model_dump()})


def build_base_sampling_params(request: OpenAIServingRequest) -> Dict[str, Any]:
    """Build common sampling parameters shared by both chat and completion requests"""
    params = {}

    # Define parameter mappings (request_attr -> param_name)
    direct_mappings = {
        "temperature": "temperature",
        "max_tokens": "max_new_tokens",
        "min_tokens": "min_new_tokens",
        "stop": "stop",
        "stop_token_ids": "stop_token_ids",
        "top_p": "top_p",
        "top_k": "top_k",
        "min_p": "min_p",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "repetition_penalty": "repetition_penalty",
        "regex": "regex",
        "ebnf": "ebnf",
        "n": "n",
        "no_stop_trim": "no_stop_trim",
        "ignore_eos": "ignore_eos",
        "logit_bias": "logit_bias",
        "skip_special_tokens": "skip_special_tokens",
        "json_schema": "json_schema",
    }

    # Apply direct mappings
    for request_attr, param_name in direct_mappings.items():
        if hasattr(request, request_attr):
            params[param_name] = getattr(request, request_attr)

    # Handle special cases
    # max_completion_tokens overrides max_tokens for chat requests
    if isinstance(request, ChatCompletionRequest) and request.max_completion_tokens:
        params["max_new_tokens"] = request.max_completion_tokens

    return params


def sanitize_model_name(model: str) -> str:
    """Sanitize model name for safe usage

    Args:
        model: Model name to sanitize

    Returns:
        Sanitized model name
    """
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>:"|?*]', "", model)

    # Limit length
    if len(sanitized) > 256:
        sanitized = sanitized[:256]

    return sanitized.strip()


def extract_error_message(exception: Exception) -> str:
    """Extract a clean error message from an exception

    Args:
        exception: Exception to extract message from

    Returns:
        Clean error message string
    """
    error_msg = str(exception)

    # Remove common prefixes that aren't user-friendly
    prefixes_to_remove = [
        "ValidationError: ",
        "ValueError: ",
        "TypeError: ",
        "KeyError: ",
    ]

    for prefix in prefixes_to_remove:
        if error_msg.startswith(prefix):
            error_msg = error_msg[len(prefix) :]
            break

    # Limit length for safety
    if len(error_msg) > 500:
        error_msg = error_msg[:500] + "..."

    return error_msg


def format_validation_errors(errors: List[Dict[str, Any]]) -> str:
    """Format Pydantic validation errors into a user-friendly message

    Args:
        errors: List of validation error dictionaries

    Returns:
        Formatted error message
    """
    if not errors:
        return "Unknown validation error"

    messages = []
    for error in errors[:5]:  # Limit to first 5 errors
        loc = " -> ".join(str(x) for x in error.get("loc", []))
        msg = error.get("msg", "Unknown error")
        if loc:
            messages.append(f"{loc}: {msg}")
        else:
            messages.append(msg)

    result = "; ".join(messages)

    if len(errors) > 5:
        result += f" (and {len(errors) - 5} more errors)"

    return result


def is_multimodal_content(content: Any) -> bool:
    """Check if content contains multimodal elements

    Args:
        content: Content to check

    Returns:
        True if content is multimodal, False otherwise
    """
    if isinstance(content, list):
        return any(
            isinstance(item, dict) and item.get("type") in ["image_url", "audio_url"]
            for item in content
        )
    return False


def count_message_tokens_estimate(messages: List[ChatCompletionMessageParam]) -> int:
    """Rough estimate of token count for messages (for validation purposes)

    Args:
        messages: List of chat messages

    Returns:
        Estimated token count
    """
    total_chars = 0

    for msg in messages:
        if isinstance(msg.content, str):
            total_chars += len(msg.content)
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    total_chars += len(item.get("text", ""))

        # Add some tokens for role and structure
        total_chars += 10

    # Rough estimate: 1 token ≈ 4 characters for English text
    return total_chars // 4


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


def _get_enable_thinking_from_request(request_obj):
    """Extracts the 'enable_thinking' flag from request chat_template_kwargs.

    Args:
        request_obj: The request object (or an item from a list of requests).

    Returns:
        The boolean value of 'enable_thinking' if found and not True, otherwise True.
    """
    if (
        hasattr(request_obj, "chat_template_kwargs")
        and request_obj.chat_template_kwargs
        and request_obj.chat_template_kwargs.get("enable_thinking") is not None
    ):
        return request_obj.chat_template_kwargs.get("enable_thinking")
    return True


def create_streaming_chunk_data(chunk_data: str) -> str:
    """Create a streaming response chunk in the proper format"""
    return f"data: {chunk_data}\n\n"


def create_stream_done() -> str:
    """Create the final [DONE] message for streaming responses"""
    return "data: [DONE]\n\n"
