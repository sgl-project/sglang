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
"""Utility functions for server argument validation and processing."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def validate_parser_model_compatibility(
    tool_call_parser: Optional[str],
    reasoning_parser: Optional[str],
    model_path: str,
    get_hf_config_func,
):
    """Validate that parser selections are compatible with the loaded model.

    Args:
        tool_call_parser: The selected tool call parser name, or None
        reasoning_parser: The selected reasoning parser name, or None
        model_path: Path to the model
        get_hf_config_func: Function to retrieve HuggingFace config
    """
    # Skip validation if both parsers are None
    if tool_call_parser is None and reasoning_parser is None:
        return

    # Define parser-to-model mappings
    # Each entry maps parser name patterns to expected model name patterns
    parser_model_map = {
        "qwen": ["qwen"],
        "qwen25": ["qwen"],
        "qwen3": ["qwen"],
        "qwen3_coder": ["qwen"],
        "gpt-oss": ["gpt-oss", "gpt_oss"],
        "deepseek": ["deepseek"],
        "deepseekv3": ["deepseek"],
        "deepseekv31": ["deepseek"],
        "glm": ["glm", "chatglm"],
        "glm45": ["glm", "chatglm"],
        "kimi": ["kimi", "moonshot"],
        "kimi_k2": ["kimi", "moonshot"],
        "mistral": ["mistral"],
        "llama3": ["llama"],
        "pythonic": [],  # Generic parser, no specific model
        "step3": ["step"],
        "minimax": ["minimax"],
    }

    # Reverse mapping: suggest parsers based on detected model type
    model_parser_suggestions = {
        "qwen": "qwen",
        "gpt-oss": "gpt-oss",
        "gpt_oss": "gpt-oss",
        "deepseek": "deepseek",
        "glm": "glm",
        "chatglm": "glm",
        "kimi": "kimi",
        "moonshot": "kimi",
        "mistral": "mistral",
        "llama": "llama3",
        "step": "step3",
        "minimax": "minimax",
    }

    # Try to get model information
    # Handle edge cases where model_path might be None or empty
    if not model_path:
        logger.debug(
            "validate_parser_model_compatibility: model_path is empty, skipping validation"
        )
        return

    model_path_lower = model_path.lower()
    model_type = None

    try:
        hf_config = get_hf_config_func()
        if hasattr(hf_config, "model_type"):
            model_type = hf_config.model_type.lower()
    except (OSError, ValueError, RuntimeError) as e:
        # OSError: File not found, network issues, permission errors
        # ValueError: Invalid config format
        # RuntimeError: Model loading failures
        logger.debug(
            f"Could not load HuggingFace config for parser validation: {e}. "
            "Validation will continue using model path only."
        )
    except Exception as e:
        # Catch any other unexpected exceptions but log them
        logger.debug(
            f"Unexpected error loading config for parser validation: {type(e).__name__}: {e}. "
            "Validation will continue using model path only."
        )

    def check_parser_compatibility(parser_name: str, parser_type: str):
        """Check if a parser is compatible with the model."""
        if parser_name is None or not parser_name:
            return

        try:
            parser_lower = parser_name.lower()
        except (AttributeError, TypeError):
            logger.warning(f"Invalid {parser_type} value: {parser_name!r}")
            return

        expected_model_patterns = parser_model_map.get(parser_lower, [])

        # Skip validation for generic parsers
        if not expected_model_patterns:
            return

        # Check if model matches any expected pattern
        is_compatible = False
        for pattern in expected_model_patterns:
            if pattern in model_path_lower or (model_type and pattern in model_type):
                is_compatible = True
                break

        if not is_compatible:
            # Try to suggest a better parser
            suggested_parser = None
            for model_pattern, suggested in model_parser_suggestions.items():
                if model_pattern in model_path_lower or (
                    model_type and model_pattern in model_type
                ):
                    suggested_parser = suggested
                    break

            suggestion_text = (
                f" Consider using '{suggested_parser}' parser instead."
                if suggested_parser
                else ""
            )

            logger.warning(
                f"{parser_type} '{parser_name}' may not be compatible with "
                f"model '{model_path}'.{suggestion_text}"
            )

    # Validate both parsers
    check_parser_compatibility(tool_call_parser, "tool_call_parser")
    check_parser_compatibility(reasoning_parser, "reasoning_parser")
