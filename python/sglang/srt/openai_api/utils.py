"""
Utility functions for OpenAI API adapter.
"""

import logging
from typing import Dict, List

import jinja2.nodes
import transformers.utils.chat_template_utils as hf_chat_utils

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
