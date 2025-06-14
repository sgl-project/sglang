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
"""
Pre-built validation rules for OpenAI API parameters.

This module provides comprehensive validation for all OpenAI API request parameters,
ensuring requests meet both OpenAI standards and SGLang-specific requirements.

Key Components:
- ValidationRule: Encapsulates parameter validation logic
- Parameter Validators: Specific validation functions for different parameter types
- Request Type Handlers: Validation rule sets for different endpoint types
- Common Validators: Shared validation logic across endpoints

Validation Categories:
- Basic Types: String, number, boolean validation
- Ranges: Min/max validation for numeric parameters
- Formats: Pattern matching for structured data
- Content: Message and prompt content validation
- Constraints: Cross-parameter dependency validation

The validation system is designed to provide clear, actionable error messages
that help users understand and fix request issues quickly.

Usage:
Validation rules are automatically applied based on request type. Each rule
specifies the parameter name, validation function, and parameter accessor,
allowing for flexible and comprehensive validation coverage.
"""

import logging
import re
from typing import Any, Callable, List, Optional, Union

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageParam,
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingInput,
    EmbeddingRequest,
    OpenAIServingRequest,
)

logger = logging.getLogger(__name__)


class ValidationRule:
    """Represents a validation rule for request parameters"""

    def __init__(
        self,
        param_name: str,
        validator_func: Callable[[Any], Optional[str]],
        param_getter: Callable[[OpenAIServingRequest], Any],
    ):
        self.param_name = param_name
        self.validator_func = validator_func
        self.param_getter = param_getter


def validate_chat_messages(messages: List[ChatCompletionMessageParam]) -> Optional[str]:
    """Validate chat messages format and content

    Args:
        messages: List of chat messages

    Returns:
        Error message if validation fails, None if valid
    """
    if not messages:
        return "Messages cannot be empty"

    # Check for alternating user/assistant pattern (optional validation)
    roles = [msg.role for msg in messages]

    # First message should typically be from user or system
    if roles[0] not in ["user", "system"]:
        return "First message should be from 'user' or 'system'"

    # Check for consecutive assistant messages (which might indicate an error)
    for i in range(1, len(roles)):
        if roles[i] == "assistant" and roles[i - 1] == "assistant":
            # This is actually allowed in some cases, so just warn
            pass

    # Validate message content
    for i, msg in enumerate(messages):
        if msg.role == "user":
            if not msg.content:
                return f"User message at index {i} has no content"
        elif msg.role == "assistant":
            # Assistant messages can have no content if they have tool_calls
            if not msg.content and not getattr(msg, "tool_calls", None):
                return f"Assistant message at index {i} has no content or tool calls"

    return None


def validate_completion_prompt(
    prompt: Union[str, List[str], List[int], List[List[int]]]
) -> Optional[str]:
    """Validate completion prompt format and content

    Args:
        prompt: The prompt to validate

    Returns:
        Error message if validation fails, None if valid
    """
    if prompt is None:
        return "Prompt cannot be None"

    if isinstance(prompt, str):
        if not prompt.strip():
            return "Prompt cannot be empty or whitespace only"
    elif isinstance(prompt, list):
        if not prompt:
            return "Prompt list cannot be empty"

        # Check if it's a list of strings
        if all(isinstance(item, str) for item in prompt):
            for i, item in enumerate(prompt):
                if not item.strip():
                    return f"Prompt at index {i} cannot be empty or whitespace only"

        # Check if it's a list of token IDs (integers)
        elif all(isinstance(item, int) for item in prompt):
            if any(item < 0 for item in prompt):
                return "Token IDs must be non-negative"

        # Check if it's a list of lists (multiple token sequences)
        elif all(isinstance(item, list) for item in prompt):
            for i, item in enumerate(prompt):
                if not item:
                    return f"Token sequence at index {i} cannot be empty"
                if not all(isinstance(token, int) for token in item):
                    return f"Token sequence at index {i} must contain only integers"
                if any(token < 0 for token in item):
                    return f"Token sequence at index {i} contains negative token IDs"
        else:
            return "Prompt must be string, list of strings, list of integers, or list of integer lists"
    else:
        return "Prompt must be string or list"

    return None


def validate_model_name(model: str) -> Optional[str]:
    """Validate model name format

    Args:
        model: Model name to validate

    Returns:
        Error message if validation fails, None if valid
    """
    if not model:
        return "Model name cannot be empty"

    if not isinstance(model, str):
        return "Model name must be a string"

    # Basic validation - model names should be reasonable
    if len(model) > 256:
        return "Model name too long (maximum 256 characters)"

    # Check for invalid characters (basic validation)
    if re.search(r'[<>:"|?*]', model):
        return "Model name contains invalid characters"

    return None


def validate_temperature(temperature: float) -> Optional[str]:
    """Validate temperature parameter

    Args:
        temperature: Temperature value to validate

    Returns:
        Error message if validation fails, None if valid
    """
    if not isinstance(temperature, (int, float)):
        return "Temperature must be a number"

    if temperature < 0:
        return "Temperature must be non-negative"

    # OpenAI allows up to 2.0, but some models may support higher
    if temperature > 2.0:
        return "Temperature should typically be between 0 and 2"

    return None


def validate_max_tokens(max_tokens: Optional[int]) -> Optional[str]:
    """Validate max_tokens parameter

    Args:
        max_tokens: Maximum tokens value to validate

    Returns:
        Error message if validation fails, None if valid
    """
    if max_tokens is None:
        return None

    if not isinstance(max_tokens, int):
        return "max_tokens must be an integer"

    if max_tokens <= 0:
        return "max_tokens must be positive"

    # Reasonable upper limit (can be adjusted based on model capabilities)
    if max_tokens > 100000:
        return "max_tokens is too large (maximum 100000)"

    return None


def validate_stop_sequences(stop: Optional[Union[str, List[str]]]) -> Optional[str]:
    """Validate stop sequences

    Args:
        stop: Stop sequences to validate

    Returns:
        Error message if validation fails, None if valid
    """
    if stop is None:
        return None

    if isinstance(stop, str):
        if len(stop) > 100:
            return "Stop sequence too long (maximum 100 characters)"
        return None

    if isinstance(stop, list):
        if len(stop) > 4:  # OpenAI limit
            return "Too many stop sequences (maximum 4)"

        for i, seq in enumerate(stop):
            if not isinstance(seq, str):
                return f"Stop sequence at index {i} must be a string"
            if len(seq) > 100:
                return f"Stop sequence at index {i} too long (maximum 100 characters)"

        return None

    return "Stop sequences must be string or list of strings"


def validate_top_p(top_p: float) -> Optional[str]:
    """Validate top_p parameter

    Args:
        top_p: Top-p value to validate

    Returns:
        Error message if validation fails, None if valid
    """
    if not isinstance(top_p, (int, float)):
        return "top_p must be a number"

    if top_p <= 0 or top_p > 1:
        return "top_p must be between 0 and 1"

    return None


def validate_frequency_penalty(frequency_penalty: float) -> Optional[str]:
    """Validate frequency_penalty parameter

    Args:
        frequency_penalty: Frequency penalty value to validate

    Returns:
        Error message if validation fails, None if valid
    """
    if not isinstance(frequency_penalty, (int, float)):
        return "frequency_penalty must be a number"

    if frequency_penalty < -2.0 or frequency_penalty > 2.0:
        return "frequency_penalty must be between -2.0 and 2.0"

    return None


def validate_presence_penalty(presence_penalty: float) -> Optional[str]:
    """Validate presence_penalty parameter

    Args:
        presence_penalty: Presence penalty value to validate

    Returns:
        Error message if validation fails, None if valid
    """
    if not isinstance(presence_penalty, (int, float)):
        return "presence_penalty must be a number"

    if presence_penalty < -2.0 or presence_penalty > 2.0:
        return "presence_penalty must be between -2.0 and 2.0"

    return None


def validate_embedding_input(input: EmbeddingInput) -> Optional[str]:
    """Validate that the input is not empty or whitespace only."""
    if not input:
        return "Input cannot be empty"

    # Handle single string
    if isinstance(input, str):
        if not input.strip():
            return "Input cannot be empty or whitespace only"
        return None

    # Handle list inputs
    if isinstance(input, list):
        if len(input) == 0:
            return "Input cannot be empty"

        # Check first element to determine type
        first_item = input[0]

        if isinstance(first_item, str):
            # List of strings
            for i, item in enumerate(input):
                if not isinstance(item, str):
                    return f"All items in input list must be strings"
                if not item.strip():
                    return f"Input at index {i} cannot be empty or whitespace only"
        elif isinstance(first_item, int):
            # List of integers (token IDs)
            for i, item in enumerate(input):
                if not isinstance(item, int):
                    return f"All items in input list must be integers"
                if item < 0:
                    return f"Token ID at index {i} must be non-negative"
        elif isinstance(first_item, list):
            # List of lists (multiple token sequences)
            for i, item in enumerate(input):
                if not isinstance(item, list):
                    return f"Input at index {i} must be a list"
                if not item:
                    return f"Input at index {i} cannot be empty"
                if not all(isinstance(token, int) for token in item):
                    return f"Input at index {i} must contain only integers"
                if any(token < 0 for token in item):
                    return f"Input at index {i} contains negative token IDs"
        # Note: MultimodalEmbeddingInput validation would be handled by Pydantic

    return None


def get_common_validation_rules() -> List[ValidationRule]:
    """Get validation rules common to both chat and completion requests"""
    return [
        ValidationRule(
            param_name="model",
            validator_func=validate_model_name,
            param_getter=lambda request: request.model,
        ),
        ValidationRule(
            param_name="temperature",
            validator_func=validate_temperature,
            param_getter=lambda request: request.temperature,
        ),
        ValidationRule(
            param_name="max_tokens",
            validator_func=validate_max_tokens,
            param_getter=lambda request: request.max_tokens,
        ),
        ValidationRule(
            param_name="stop",
            validator_func=validate_stop_sequences,
            param_getter=lambda request: request.stop,
        ),
    ]


def get_chat_specific_validation_rules() -> List[ValidationRule]:
    """Get validation rules specific to chat completion requests"""
    return [
        ValidationRule(
            param_name="messages",
            validator_func=validate_chat_messages,
            param_getter=lambda request: request.messages,
        ),
    ]


def get_completion_specific_validation_rules() -> List[ValidationRule]:
    """Get validation rules specific to completion requests"""
    return [
        ValidationRule(
            param_name="prompt",
            validator_func=validate_completion_prompt,
            param_getter=lambda request: request.prompt,
        ),
    ]


def get_embedding_specific_validation_rules() -> List[ValidationRule]:
    """Get validation rules specific to embedding requests"""
    return [
        ValidationRule(
            param_name="input",
            validator_func=validate_embedding_input,
            param_getter=lambda request: request.input,
        ),
    ]


def get_validation_rules(request: OpenAIServingRequest) -> List[ValidationRule]:
    """Get all validation rules for the request"""
    if isinstance(request, ChatCompletionRequest):
        return get_common_validation_rules() + get_chat_specific_validation_rules()
    elif isinstance(request, CompletionRequest):
        # Echo + logprobs warning
        if request.echo and request.logprobs:
            logger.warning(
                "Echo is not compatible with logprobs. "
                "To compute logprobs of input prompt, please use the native /generate API."
            )
        return (
            get_common_validation_rules() + get_completion_specific_validation_rules()
        )
    elif isinstance(request, EmbeddingRequest):
        return get_embedding_specific_validation_rules()
    else:
        raise ValueError(f"Unsupported request type: {type(request)}")
