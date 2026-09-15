"""Tool-reference compatibility decisions for Anthropic requests.

CALLING SPEC:
    native = template_supports_deferred_tool_loading(chat_template)
    part = make_tool_reference_part(name, native_support=native)
    visible = should_forward_tool(
        name=name,
        defer_loading=defer_loading,
        referenced_names=referenced_names,
        native_support=native,
    )

Inputs and outputs are plain values. Functions are deterministic and have no
side effects, so template capability and deferred-tool routing can be tested
without constructing a serving stack.
"""

from collections.abc import Mapping
from typing import Any

import jinja2
import transformers.utils.chat_template_utils as hf_chat_utils


def _template_sources(chat_template: Any) -> list[str]:
    """Return string template sources from tokenizer template configuration."""
    if isinstance(chat_template, str):
        return [chat_template]
    if isinstance(chat_template, Mapping):
        return [value for value in chat_template.values() if isinstance(value, str)]
    return []


def template_supports_deferred_tool_loading(chat_template: Any) -> bool:
    """Return whether a template implements native deferred-tool expansion.

    Jinja comments are absent from the parsed AST, avoiding the false positive
    caused by a raw substring check. Requiring both protocol fields also keeps
    templates that merely render a reference as text on the generic path.
    """
    for source in _template_sources(chat_template):
        try:
            compiled = hf_chat_utils._compile_jinja_template(source)
            template_ast = compiled.environment.parse(source)
        except (jinja2.TemplateError, TypeError, ValueError):
            continue
        constants = {
            node.value
            for node in template_ast.find_all(jinja2.nodes.Const)
            if isinstance(node.value, str)
        }
        attributes = {node.attr for node in template_ast.find_all(jinja2.nodes.Getattr)}
        identifiers = constants | attributes
        if {"tool_reference", "defer_loading"} <= identifiers:
            return True
    return False


def make_tool_reference_part(name: str, *, native_support: bool) -> dict[str, str]:
    """Build a native reference or a text marker for a generic template."""
    if native_support:
        return {"type": "tool_reference", "name": name}
    return {"type": "text", "text": f"[tool reference: {name}]"}


def should_forward_tool(
    *,
    name: str,
    defer_loading: bool | None,
    referenced_names: set[str],
    native_support: bool,
) -> bool:
    """Return whether a tool belongs in the converted request.

    Native templates receive the complete catalog and expand referenced tools
    inline. Generic templates receive only immediately available tools plus
    deferred tools already discovered in the conversation history.
    """
    return native_support or defer_loading is not True or name in referenced_names
