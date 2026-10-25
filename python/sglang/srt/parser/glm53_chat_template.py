# Adapted from the GLM-5.3 / GLM-5.3-Flash reference chat_template.py by Z.ai.
"""Python version of the GLM-5.3 and GLM-5.3-Flash chat templates.

The two templates are the same apart from the multimodal block, so a single
renderer serves both. The output is byte-identical to the Jinja templates.
Where a template fails on malformed input, the renderer fails at the same
point, raising ``ValueError`` for Jinja's ``UndefinedError`` and the very
``TypeError`` Jinja would raise, so the serving layer answers both with 400
exactly as it did for the template. Jinja semantics that are easy to get wrong
when reading the template source are called out inline; ``_UNDEFINED`` stands
for Jinja's ``Undefined``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Optional

_UNDEFINED = object()

_TOOLS_PREFIX = (
    "\n# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
)
_TOOLS_SUFFIX = (
    "</tools>\n\n"
    "For each function call, output the function name and arguments within the "
    "following XML format:\n"
    "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>"
    "<arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key>"
    "<arg_value>{arg-value-2}</arg_value>...</tool_call>"
)

# Tuples rather than sets: Jinja's `in` compares with ==, so an unhashable
# part type must not raise here.
_IMAGE_PART_TYPES = ("image", "image_url")
_VIDEO_PART_TYPES = ("video", "video_url")
_AUDIO_PART_TYPES = ("audio", "audio_url", "input_audio")
_MEDIA_PART_TYPES = _IMAGE_PART_TYPES + _VIDEO_PART_TYPES + _AUDIO_PART_TYPES

_MEDIA_REMINDER = (
    "<reminder>You are unable to process this {} because you don't have "
    "multi-modal input ability. Try different methods.</reminder>"
)
_IMAGE_TOKENS = "<|begin_of_image|><|image|><|end_of_image|>"
_VIDEO_TOKENS = "<|begin_of_video|><|video|><|end_of_video|>"
_AUDIO_TOKENS = "<|begin_of_audio|><|end_of_audio|>"


def _get(obj: Any, key: str, default: Any = _UNDEFINED) -> Any:
    """``obj.key`` in Jinja: an item of a mapping, an attribute otherwise."""
    if isinstance(obj, Mapping):
        return obj[key] if key in obj else default
    return getattr(obj, key, default)


def _first(value: Any) -> Any:
    """``value.0`` in Jinja, which is Undefined when there is no item 0."""
    try:
        return value[0]
    except (AttributeError, TypeError, LookupError):
        return _UNDEFINED


def _string(value: Any) -> str:
    """``{{ value }}`` in Jinja: Undefined renders empty, None renders "None"."""
    return "" if value is _UNDEFINED else str(value)


def _is_iterable(value: Any) -> bool:
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _render_media(part_type: str, multimodal: bool) -> str:
    if not multimodal:
        media_type = part_type.replace("_url", "").replace("input_", "")
        return _MEDIA_REMINDER.format(media_type)
    if part_type in _IMAGE_PART_TYPES:
        return _IMAGE_TOKENS
    if part_type in _VIDEO_PART_TYPES:
        return _VIDEO_TOKENS
    return _AUDIO_TOKENS


def _visible_text(content: Any, multimodal: bool) -> str:
    if isinstance(content, str):
        return content
    if _is_iterable(content) and not isinstance(content, Mapping):
        parts = []
        for item in content:
            if isinstance(item, Mapping):
                part_type = _get(item, "type")
                if part_type == "text":
                    parts.append(_string(_get(item, "text")))
                elif part_type in _MEDIA_PART_TYPES:
                    parts.append(_render_media(part_type, multimodal))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    # The template guards this branch with `content is not none`, so None
    # renders nothing instead of "None".
    return "" if content is None else _string(content)


def _unwrap_tool(tool: Any) -> Any:
    # `'function' in tool` is a membership test, so a present but empty
    # `function` still replaces the tool.
    if isinstance(tool, Mapping) and "function" in tool:
        return tool["function"]
    return tool


def _tool_to_json(tool: Any) -> str:
    # The tool_to_json macro calls `tool.items()`, which fails on a non-mapping.
    if not isinstance(tool, Mapping):
        raise ValueError(f"Each tool must be a JSON object, got {type(tool).__name__}.")
    # The macro writes `"{{ k }}": {{ v | tojson }}` by hand: values go through
    # tojson (json.dumps under transformers), but keys are interpolated raw and
    # are not JSON-escaped, so json.dumps on the whole dict would diverge.
    fields = [
        f'"{key!s}": {json.dumps(value, ensure_ascii=False)}'
        for key, value in tool.items()
        if key != "defer_loading" and key != "strict"
    ]
    return "{" + ", ".join(fields) + "}"


def _tool_references_to_response(refs: Any, tools: Any) -> str:
    # The template defines this macro under `{%- if tools -%}`, so rendering a
    # tool_reference without tools called an undefined macro.
    if not tools:
        raise ValueError(
            "Rendering a tool_reference requires `tools`; the GLM-5.3 chat "
            "template only defines its tool macros when tools are given."
        )
    parts = ["<tool_response><tools>\n"]
    for ref in refs:
        ref_name = _get(ref, "name")
        for tool in tools:
            tool = _unwrap_tool(tool)
            # Undefined equals Undefined in Jinja, so a reference without a
            # name matches every tool without one, as it does here.
            if _get(tool, "name") == ref_name:
                parts.append(_tool_to_json(tool) + "\n")
    parts.append("</tools></tool_response>")
    return "".join(parts)


def _tool_response(text: str) -> str:
    return "<tool_response>" + text + "</tool_response>"


def _id_of(obj: Any) -> str:
    tool_call_id = _get(obj, "tool_call_id", None)
    if tool_call_id:
        return str(tool_call_id)
    obj_id = _get(obj, "id", None)
    if obj_id:
        return str(obj_id)
    return ""


def _is_list_of_outputs(message: Any) -> bool:
    # `m.content and m.content.0.output is defined`: falsy content
    # short-circuits, while reading `.output` from a missing item 0 (a mapping
    # or a scalar content) raises UndefinedError.
    content = _get(message, "content", None)
    if not content:
        return False
    first = _first(content)
    if first is _UNDEFINED:
        raise ValueError(
            "A tool message `content` must be a string or a list, got "
            f"{type(content).__name__}."
        )
    return _get(first, "output") is not _UNDEFINED


def _is_tool_reference_list(value: Any) -> bool:
    # `v is iterable and v is not string and v is not mapping and v and
    # v.0.type == "tool_reference"`; reading `.type` from a missing item 0
    # raises UndefinedError like `_is_list_of_outputs`.
    if isinstance(value, (str, Mapping)) or not _is_iterable(value) or not value:
        return False
    first = _first(value)
    if first is _UNDEFINED:
        raise ValueError(
            f"A tool_reference list must be a list, got {type(value).__name__}."
        )
    return _get(first, "type") == "tool_reference"


def _render_output(entry: Any, tools: Any, multimodal: bool) -> str:
    output = _get(entry, "output")
    if _is_tool_reference_list(output):
        return _tool_references_to_response(output, tools)
    return _tool_response(_visible_text(output, multimodal))


def _render_tool_response(message: Any, tools: Any, multimodal: bool) -> str:
    content = _get(message, "content", None)
    if isinstance(content, str):
        return _tool_response(content)
    if not isinstance(content, Mapping) and _is_tool_reference_list(content):
        return _tool_references_to_response(content, tools)
    if _is_list_of_outputs(message):
        return "".join(_render_output(entry, tools, multimodal) for entry in content)
    return _tool_response(_visible_text(content, multimodal))


def _render_tool_block(
    messages: list[Any], start: int, end: int, tools: Any, multimodal: bool
) -> str:
    """Render the consecutive tool messages ``messages[start:end + 1]``.

    Results are reordered to follow the driving assistant's ``tool_calls`` when
    every id is non-empty and unique on both sides and every result id names
    one of the calls; otherwise they render in message order.
    """
    # One (message index, entry index or -1, id) per tool result. The template
    # collects these for the whole block before it looks at the tool calls, so
    # a malformed tool message fails here whether or not results get sorted.
    results = []
    for index in range(start, end + 1):
        message = messages[index]
        if _is_list_of_outputs(message):
            for entry_index, entry in enumerate(_get(message, "content")):
                results.append((index, entry_index, _id_of(entry)))
        else:
            results.append((index, -1, _id_of(message)))

    assistant = messages[start - 1] if start > 0 else None
    tool_calls = None
    if assistant is not None and _get(assistant, "role") == "assistant":
        tool_calls = _get(assistant, "tool_calls", None)

    can_sort = bool(tool_calls)
    if can_sort:
        call_ids = [_id_of(tool_call) for tool_call in tool_calls]
        result_ids = [result_id for _, _, result_id in results]
        can_sort = (
            "" not in result_ids
            and "" not in call_ids
            and len(set(result_ids)) == len(result_ids)
            and len(set(call_ids)) == len(call_ids)
            and all(result_id in call_ids for result_id in result_ids)
        )

    if not can_sort:
        return "".join(
            _render_tool_response(messages[index], tools, multimodal)
            for index in range(start, end + 1)
        )

    call_position = {call_id: position for position, call_id in enumerate(call_ids)}
    parts = []
    for index, entry_index, _ in sorted(
        results, key=lambda result: call_position[result[2]]
    ):
        message = messages[index]
        if entry_index >= 0:
            entry = _get(message, "content")[entry_index]
            parts.append(_render_output(entry, tools, multimodal))
        else:
            parts.append(_render_tool_response(message, tools, multimodal))
    return "".join(parts)


def _render_tool_call(tool_call: Any, multimodal: bool) -> str:
    # `{%- if tc.function %}{%- set tc = tc.function %}`: a truthiness test.
    function = _get(tool_call, "function", None)
    if function:
        tool_call = function
    name = _get(tool_call, "name")
    if multimodal:
        # GLM-5.3-Flash: `'<tool_call>' ~ tc.name` stringifies the name.
        parts = ["<tool_call>" + _string(name)]
    else:
        # GLM-5.3: `'<tool_call>' + tc.name` is Python's +, so a non-string
        # name raises the very same TypeError; do not coerce it.
        if name is _UNDEFINED:
            raise ValueError(
                "Each tool_call needs a `name`; the GLM-5.3 chat template "
                "reads `tc.name` without a guard."
            )
        parts = ["<tool_call>" + name]
    # The name is concatenated first, so it fails before bad arguments do.
    arguments = _get(tool_call, "arguments", None)
    if not isinstance(arguments, Mapping):
        raise ValueError(
            "tool_call `arguments` must be a JSON object, got "
            f"{type(arguments).__name__}."
        )
    for key, value in arguments.items():
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        parts.append(f"<arg_key>{key!s}</arg_key><arg_value>{value}</arg_value>")
    parts.append("</tool_call>")
    return "".join(parts)


def _render(
    messages: list[Any],
    tools: Optional[list[Any]],
    add_generation_prompt: bool,
    reasoning_effort: Any,
    clear_thinking: Any,
    reasoning_content: Any,
    multimodal: bool,
) -> list[str]:
    effective_reasoning_effort = (
        reasoning_effort if reasoning_effort in ("low", "high") else "max"
    )
    parts = [
        "[gMASK]",
        "<sop>",
        "<|system|>",
        f"Reasoning Effort: {effective_reasoning_effort.capitalize()}",
    ]

    if tools:
        parts.append("<|system|>")
        parts.append(_TOOLS_PREFIX)
        for tool in tools:
            tool = _unwrap_tool(tool)
            defer_loading = _get(tool, "defer_loading")
            if defer_loading is _UNDEFINED or not defer_loading:
                parts.append(_tool_to_json(tool) + "\n")
        parts.append(_TOOLS_SUFFIX)

    last_user_index = -1
    for index, message in enumerate(messages):
        if _get(message, "role") == "user":
            last_user_index = index

    index = 0
    while index < len(messages):
        message = messages[index]
        role = _get(message, "role")
        if role == "user":
            parts.append("<|user|>")
            parts.append(_visible_text(_get(message, "content"), multimodal))
        elif role == "assistant":
            parts.append("<|assistant|>")
            content = _visible_text(_get(message, "content"), multimodal)
            reasoning = _get(message, "reasoning_content")
            if not isinstance(reasoning, str):
                if "</think>" in content:
                    reasoning = content.split("</think>")[0].split("<think>")[-1]
                    content = content.split("</think>")[-1]
                else:
                    # The template only assigns reasoning_content in the two
                    # branches above, so otherwise the name resolves to the
                    # template variable of that name, which is Undefined unless
                    # the caller passed `reasoning_content` as a template kwarg.
                    reasoning = reasoning_content
            if (not clear_thinking or index > last_user_index) and (
                reasoning is not _UNDEFINED
            ):
                parts.append("<think>" + reasoning + "</think>")
            else:
                parts.append("<think></think>")
            if content.strip():
                parts.append(content.strip())
            tool_calls = _get(message, "tool_calls", None)
            if tool_calls:
                for tool_call in tool_calls:
                    parts.append(_render_tool_call(tool_call, multimodal))
        elif role == "tool":
            if index == 0 or _get(messages[index - 1], "role") != "tool":
                end = index
                while (
                    end + 1 < len(messages)
                    and _get(messages[end + 1], "role") == "tool"
                ):
                    end += 1
                parts.append("<|observation|>")
                parts.append(
                    _render_tool_block(messages, index, end, tools, multimodal)
                )
                index = end
        elif role == "system":
            parts.append("<|system|>")
            parts.append(_visible_text(_get(message, "content"), multimodal))
        index += 1

    if add_generation_prompt:
        parts.append("<|assistant|>")
        parts.append("<think>")
    return parts


def render_glm53_segments(
    messages: list[Any],
    tools: Optional[list[Any]] = None,
    add_generation_prompt: bool = False,
    reasoning_effort: Any = None,
    clear_thinking: Any = False,
    reasoning_content: Any = _UNDEFINED,
    **_: Any,
) -> list[str]:
    """Render a GLM-5.3 (text-only) chat prompt as the strings it concatenates.

    The special tokens that open the prompt and each turn are strings of their
    own, so the prompt can be encoded one segment at a time.
    """
    return _render(
        messages,
        tools,
        add_generation_prompt,
        reasoning_effort,
        clear_thinking,
        reasoning_content,
        multimodal=False,
    )


def render_glm53_flash_segments(
    messages: list[Any],
    tools: Optional[list[Any]] = None,
    add_generation_prompt: bool = False,
    reasoning_effort: Any = None,
    clear_thinking: Any = False,
    reasoning_content: Any = _UNDEFINED,
    **_: Any,
) -> list[str]:
    """``render_glm53_segments`` for GLM-5.3-Flash (multimodal)."""
    return _render(
        messages,
        tools,
        add_generation_prompt,
        reasoning_effort,
        clear_thinking,
        reasoning_content,
        multimodal=True,
    )


def render_glm53(*args: Any, **kwargs: Any) -> str:
    """Render a GLM-5.3 (text-only) chat prompt."""
    return "".join(render_glm53_segments(*args, **kwargs))


def render_glm53_flash(*args: Any, **kwargs: Any) -> str:
    """Render a GLM-5.3-Flash (multimodal) chat prompt."""
    return "".join(render_glm53_flash_segments(*args, **kwargs))
