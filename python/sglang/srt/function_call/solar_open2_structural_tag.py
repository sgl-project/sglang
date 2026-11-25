# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

from collections.abc import Mapping
from typing import Any, Literal

from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    AnyTextFormat,
    AnyTokensFormat,
    ConstStringFormat,
    Format,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    SequenceFormat,
    StarFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TokenFormat,
    TokenTriggeredTagsFormat,
    TriggeredTagsFormat,
)

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice

_CALL_START = "<|tool_call:start|>"
_CALL_END = "<|tool_call:end|>"
_ARG_START = "<|tool_arg:start|>"
_ARG_VALUE = "<|tool_arg:value|>"
_ARG_END = "<|tool_arg:end|>"
_THINK_END = "<|think:end|>"
_IM_END = "<|im:end|>"
_SENTINELS = (
    _CALL_START,
    _CALL_END,
    _ARG_START,
    _ARG_VALUE,
    _ARG_END,
    _THINK_END,
    _IM_END,
)
_RAW_TYPES = frozenset({"string", "str", "text", "varchar", "char", "enum"})


class _Markers:
    """Bind sentinel boundaries to token IDs when the vocabulary is available."""

    def __init__(self, vocab: Mapping[str, int] | None):
        self.ids = (
            {marker: vocab[marker] for marker in _SENTINELS}
            if vocab is not None and all(marker in vocab for marker in _SENTINELS)
            else None
        )

    def boundary(self, marker: str) -> str | TokenFormat:
        return marker if self.ids is None else TokenFormat(token=self.ids[marker])

    def element(self, marker: str) -> Format:
        if self.ids is None:
            return ConstStringFormat(value=marker)
        return TokenFormat(token=self.ids[marker])

    def free_text(self, *excludes: str) -> Format:
        if self.ids is None:
            return AnyTextFormat(excludes=list(excludes))
        return AnyTokensFormat(exclude_tokens=[self.ids[marker] for marker in excludes])

    def triggered_tags(
        self, tags: list[TagFormat], stop_after_first: bool = False
    ) -> Format:
        if self.ids is None:
            return TriggeredTagsFormat(
                triggers=[_CALL_START], tags=tags, stop_after_first=stop_after_first
            )
        return TokenTriggeredTagsFormat(
            trigger_tokens=[self.ids[_CALL_START]],
            tags=tags,
            stop_after_first=stop_after_first,
        )


def _value_format(spec: Any, parameters: dict, markers: _Markers) -> Format:
    raw_value = markers.free_text(_ARG_END)
    if not isinstance(spec, dict):
        return raw_value
    spec = {
        **spec,
        **{
            key: parameters[key]
            for key in ("$defs", "definitions")
            if key in parameters and key not in spec
        },
    }
    # Match the parser: missing types are strings; unions use the first
    # non-null type. Strings are emitted raw, other values as JSON.
    value_type = spec.get("type")
    if isinstance(value_type, list):
        value_type = next((item for item in value_type if item != "null"), None)
    if value_type is None or str(value_type).strip().lower() in _RAW_TYPES:
        enum = spec.get("enum")
        if (
            isinstance(enum, list)
            and enum
            and all(isinstance(item, str) for item in enum)
        ):
            return OrFormat(elements=[ConstStringFormat(value=item) for item in enum])
        return raw_value
    return JSONSchemaFormat(json_schema=spec)


def _argument(key: Format, value: Format, markers: _Markers) -> Format:
    return SequenceFormat(
        elements=[
            markers.element(_ARG_START),
            key,
            markers.element(_ARG_VALUE),
            value,
            markers.element(_ARG_END),
            ConstStringFormat(value="\n"),
        ]
    )


def _arguments(parameters: Any, markers: _Markers) -> Format:
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    if not isinstance(properties, dict):
        return StarFormat(
            content=_argument(
                markers.free_text(_ARG_VALUE, _ARG_END),
                markers.free_text(_ARG_END),
                markers,
            )
        )
    required = parameters.get("required")
    required = set(required) if isinstance(required, list) else set()
    elements = []
    for name, spec in properties.items():
        argument = _argument(
            ConstStringFormat(value=name),
            _value_format(spec, parameters, markers),
            markers,
        )
        elements.append(
            argument if name in required else OptionalFormat(content=argument)
        )
    return (
        SequenceFormat(elements=elements) if elements else ConstStringFormat(value="")
    )


def get_solar_open2_structural_tag(
    tools: list[Tool],
    tool_choice: ToolChoice | Literal["auto", "required", "none"] = "auto",
    thinking_mode: bool = False,
    vocab: Mapping[str, int] | None = None,
    parallel_tool_calls: bool = True,
) -> StructuralTag | None:
    if not tools or tool_choice == "none":
        return None
    named_choice = isinstance(tool_choice, ToolChoice)
    if named_choice:
        tools = [
            tool for tool in tools if tool.function.name == tool_choice.function.name
        ]
        if not tools:
            raise ValueError(f"Unknown tool: {tool_choice.function.name}")
    markers = _Markers(vocab)
    tags = [
        TagFormat(
            begin=markers.boundary(_CALL_START),
            content=SequenceFormat(
                elements=[
                    ConstStringFormat(value=f"{tool.function.name}\n"),
                    _arguments(
                        True
                        if tool.function.strict is False
                        else tool.function.parameters,
                        markers,
                    ),
                ]
            ),
            end=markers.boundary(_CALL_END),
        )
        for tool in tools
    ]
    if tool_choice == "auto":
        auto_format = markers.triggered_tags(
            tags, stop_after_first=parallel_tool_calls is False
        )
        if parallel_tool_calls is False:
            auto_format = SequenceFormat(
                elements=[
                    auto_format,
                    OptionalFormat(content=ConstStringFormat(value="\n")),
                    OptionalFormat(content=markers.element(_IM_END)),
                ]
            )
        return StructuralTag(format=auto_format)

    elements = []
    if thinking_mode:
        # With --reasoning-parser, SGLang's ReasonerGrammarBackend owns this
        # prefix and passes thinking_mode=False for the suffix grammar.
        elements.append(
            OptionalFormat(
                content=SequenceFormat(
                    elements=[
                        markers.free_text(_CALL_START, _THINK_END),
                        markers.element(_THINK_END),
                    ]
                )
            )
        )
    elements.extend(
        [
            TagsWithSeparatorFormat(
                tags=tags,
                separator="\n",
                at_least_one=True,
                stop_after_first=named_choice or parallel_tool_calls is False,
            ),
            OptionalFormat(content=ConstStringFormat(value="\n")),
            # Solar ends turns with im:end rather than the tokenizer's EOS.
            OptionalFormat(content=markers.element(_IM_END)),
        ]
    )
    return StructuralTag(format=SequenceFormat(elements=elements))
