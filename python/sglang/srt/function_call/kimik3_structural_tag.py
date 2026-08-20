from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    AnyTextFormat,
    AnyTokensFormat,
    ConstStringFormat,
    ExcludeTokenFormat,
    Format,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    RegexFormat,
    SequenceFormat,
    StarFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TokenFormat,
    TriggeredTagsFormat,
)

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.kimik3_format import (
    ARGUMENT_CLOSE,
    CALL_CLOSE,
    CALL_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)

_JSON_TYPES = (
    "string",
    "number",
    "integer",
    "boolean",
    "array",
    "object",
    "null",
)
_CLOSE_TOKEN = "<|close|>"
_ARGUMENT_CLOSE_SUFFIX = ARGUMENT_CLOSE.removeprefix(_CLOSE_TOKEN)
_JSON_TO_XTML_TYPE = {
    "string": "string",
    "number": "number",
    "integer": "number",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
    "null": "null",
}
_STRING_KEYWORDS = {"format", "maxLength", "minLength", "pattern"}
_NUMBER_KEYWORDS = {
    "exclusiveMaximum",
    "exclusiveMinimum",
    "maximum",
    "minimum",
    "multipleOf",
}
_ARRAY_KEYWORDS = {
    "contains",
    "items",
    "maxContains",
    "maxItems",
    "minContains",
    "minItems",
    "prefixItems",
    "uniqueItems",
}
_OBJECT_KEYWORDS = {
    "additionalProperties",
    "dependentRequired",
    "dependentSchemas",
    "maxProperties",
    "minProperties",
    "patternProperties",
    "properties",
    "propertyNames",
    "required",
}


def _escape_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;")


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _matches_json_type(value: Any, json_type: str) -> bool:
    value_type = _json_type(value)
    return value_type == json_type or (
        json_type == "number" and value_type == "integer"
    )


def _resolve_local_ref(
    ref: str, root_schema: Dict[str, Any]
) -> Optional[Union[bool, Dict[str, Any]]]:
    if not ref.startswith("#/"):
        return None
    value: Any = root_schema
    for part in ref[2:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, (bool, dict)):
        return value
    return None


def _schema_types(
    schema: Union[bool, Dict[str, Any]],
    root_schema: Dict[str, Any],
    seen_refs: Optional[Set[str]] = None,
) -> List[str]:
    if schema is False:
        return []
    if schema is True:
        return list(_JSON_TYPES)

    ref = schema.get("$ref")
    if isinstance(ref, str):
        seen_refs = set() if seen_refs is None else set(seen_refs)
        if ref not in seen_refs:
            target = _resolve_local_ref(ref, root_schema)
            if target is not None:
                seen_refs.add(ref)
                return _schema_types(target, root_schema, seen_refs)

    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return [schema_type] if schema_type in _JSON_TYPES else list(_JSON_TYPES)
    if isinstance(schema_type, list):
        return [item for item in _JSON_TYPES if item in schema_type]

    for keyword in ("anyOf", "oneOf"):
        options = schema.get(keyword)
        if isinstance(options, list):
            option_types = {
                item
                for option in options
                if isinstance(option, (bool, dict))
                for item in _schema_types(option, root_schema, seen_refs)
            }
            return [item for item in _JSON_TYPES if item in option_types]

    options = schema.get("allOf")
    if isinstance(options, list):
        type_sets = [
            set(_schema_types(option, root_schema, seen_refs))
            for option in options
            if isinstance(option, (bool, dict))
        ]
        constrained = [types for types in type_sets if types != set(_JSON_TYPES)]
        if constrained:
            result = constrained[0] | (
                {"integer"} if "number" in constrained[0] else set()
            )
            for types in constrained[1:]:
                result &= types | ({"integer"} if "number" in types else set())
            # number survives the intersection only if every branch allows it,
            # making the widened integer redundant rather than the other way.
            if "number" in result and "integer" in result:
                result.remove("integer")
            return [item for item in _JSON_TYPES if item in result]

    if "const" in schema:
        return [_json_type(schema["const"])]
    enum = schema.get("enum")
    if isinstance(enum, list):
        enum_types = {_json_type(value) for value in enum}
        return [item for item in _JSON_TYPES if item in enum_types]
    if _OBJECT_KEYWORDS.intersection(schema):
        return ["object"]
    if _ARRAY_KEYWORDS.intersection(schema):
        return ["array"]
    if _STRING_KEYWORDS.intersection(schema):
        return ["string"]
    if _NUMBER_KEYWORDS.intersection(schema):
        return ["number"]
    return list(_JSON_TYPES)


def _with_root_definitions(
    schema: Union[bool, Dict[str, Any]], root_schema: Dict[str, Any]
) -> Union[bool, Dict[str, Any]]:
    if not isinstance(schema, dict):
        return schema
    result = dict(schema)
    for key in ("$defs", "definitions"):
        if key in root_schema and key not in result:
            result[key] = root_schema[key]
    return result


def _restrict_schema_type(
    schema: Union[bool, Dict[str, Any]],
    json_type: str,
    root_schema: Dict[str, Any],
) -> Union[bool, Dict[str, Any]]:
    if not isinstance(schema, dict):
        return {"type": json_type} if schema else False

    ref = schema.get("$ref")
    if isinstance(ref, str):
        target = _resolve_local_ref(ref, root_schema)
        if target is not None:
            return _with_root_definitions(
                _restrict_schema_type(target, json_type, root_schema), root_schema
            )

    result = dict(schema)
    schema_type = result.get("type")
    if isinstance(schema_type, list):
        if json_type not in schema_type:
            return False
        result["type"] = json_type
    elif isinstance(schema_type, str):
        if schema_type != json_type:
            return False
    else:
        result["type"] = json_type

    for keyword in ("anyOf", "oneOf"):
        options = result.get(keyword)
        if not isinstance(options, list):
            continue
        restricted = [
            _restrict_schema_type(option, json_type, root_schema)
            for option in options
            if isinstance(option, (bool, dict))
            and json_type in _schema_types(option, root_schema)
        ]
        if not restricted:
            return False
        if len(restricted) == 1 and isinstance(restricted[0], dict):
            result.pop(keyword)
            result.update(restricted[0])
        else:
            result[keyword] = restricted

    enum = result.get("enum")
    if isinstance(enum, list):
        result["enum"] = [
            value for value in enum if _matches_json_type(value, json_type)
        ]
        if not result["enum"]:
            return False
    if "const" in result and not _matches_json_type(result["const"], json_type):
        return False
    return _with_root_definitions(result, root_schema)


def _value_format(
    schema: Union[bool, Dict[str, Any]],
    json_type: str,
    loose_string: bool = False,
) -> Format:
    if loose_string and json_type == "string":
        return AnyTextFormat()
    # XGrammar 0.2.1 miscompiles a one-sided negative integer lower bound:
    # {"type": "integer", "minimum": -N} accepts the incomplete value "-"
    # and rejects every valid negative integer. Splitting the range at zero
    # avoids that converter bug without weakening the schema.
    if (
        json_type == "integer"
        and isinstance(schema, dict)
        and "maximum" not in schema
        and "exclusiveMaximum" not in schema
        and "multipleOf" not in schema
    ):
        lower_bounds = []
        minimum = schema.get("minimum")
        if isinstance(minimum, int) and not isinstance(minimum, bool):
            lower_bounds.append(minimum)
        exclusive_minimum = schema.get("exclusiveMinimum")
        if isinstance(exclusive_minimum, int) and not isinstance(
            exclusive_minimum, bool
        ):
            lower_bounds.append(exclusive_minimum + 1)
        if lower_bounds and max(lower_bounds) < 0:
            negative = dict(schema)
            negative["maximum"] = -1
            nonnegative = dict(schema)
            nonnegative.pop("exclusiveMinimum", None)
            nonnegative["minimum"] = 0
            schema = {"anyOf": [negative, nonnegative]}
    return JSONSchemaFormat(
        json_schema=schema,
        style="qwen_xml" if json_type == "string" else "json",
    )


def _argument_value_variants(
    schema: Union[bool, Dict[str, Any]],
    root_schema: Dict[str, Any],
    loose_strings: bool = False,
) -> List[Tuple[str, Format]]:
    return [
        (
            json_type,
            _value_format(restricted, json_type, loose_string=loose_strings),
        )
        for json_type in _schema_types(schema, root_schema)
        if (restricted := _restrict_schema_type(schema, json_type, root_schema))
        is not False
    ]


def _known_argument_format(
    key: str,
    schema: Union[bool, Dict[str, Any]],
    root_schema: Dict[str, Any],
) -> Optional[Format]:
    escaped_key = _escape_attr(key)
    variants = [
        TagFormat(
            begin=(
                f'<|open|>argument key="{escaped_key}" '
                f'type="{_JSON_TO_XTML_TYPE[json_type]}"<|sep|>'
            ),
            content=value_format,
            end=ARGUMENT_CLOSE,
        )
        for json_type, value_format in _argument_value_variants(schema, root_schema)
    ]
    if not variants:
        return None
    if len(variants) == 1:
        return variants[0]
    return OrFormat(elements=variants)


def _dynamic_argument_format(
    schema: Union[bool, Dict[str, Any]],
    root_schema: Dict[str, Any],
    loose_strings: bool = False,
) -> Format:
    variants = [
        SequenceFormat(
            elements=[
                RegexFormat(pattern=r'[^"& \t\r\n\f\v=<>]+'),
                ConstStringFormat(
                    value=(f'" type="{_JSON_TO_XTML_TYPE[json_type]}"<|sep|>')
                ),
                value_format,
            ]
        )
        for json_type, value_format in _argument_value_variants(
            schema, root_schema, loose_strings=loose_strings
        )
    ]
    if not variants:
        raise ValueError("Kimi K3 additional parameter schema accepts no values")
    content = variants[0] if len(variants) == 1 else OrFormat(elements=variants)
    return TagFormat(
        begin='<|open|>argument key="',
        content=content,
        end=ARGUMENT_CLOSE,
    )


def _strict_arguments_format(parameters: Dict[str, Any]) -> Format:
    properties = parameters.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("Kimi K3 tool parameters 'properties' must be an object")
    required = parameters.get("required", [])
    if not isinstance(required, list) or not all(
        isinstance(item, str) for item in required
    ):
        raise ValueError("Kimi K3 tool parameters 'required' must be a string list")

    required_set = set(required)
    missing = required_set.difference(properties)
    if missing:
        raise ValueError(
            f"Kimi K3 required parameters are missing schemas: {sorted(missing)!r}"
        )

    elements: List[Format] = []
    for key, schema in properties.items():
        if not isinstance(key, str) or not isinstance(schema, (bool, dict)):
            raise ValueError("Kimi K3 tool property schemas must be JSON schemas")
        argument = _known_argument_format(key, schema, parameters)
        if argument is None:
            if key in required_set:
                raise ValueError(
                    f"Kimi K3 required parameter {key!r} accepts no values"
                )
            continue
        elements.append(
            argument if key in required_set else OptionalFormat(content=argument)
        )

    additional = parameters.get("additionalProperties", True)
    if additional is True:
        elements.append(StarFormat(content=_dynamic_argument_format(True, parameters)))
    elif isinstance(additional, dict):
        elements.append(
            StarFormat(content=_dynamic_argument_format(additional, parameters))
        )
    elif additional is not False:
        raise ValueError(
            "Kimi K3 tool parameters 'additionalProperties' must be a schema"
        )
    if not elements:
        return ConstStringFormat(value="")
    return SequenceFormat(elements=elements)


def _tool_arguments_format(tool: Tool) -> Format:
    parameters = tool.function.parameters
    if not tool.function.strict:
        root_schema = parameters if isinstance(parameters, dict) else {}
        return StarFormat(
            content=_dynamic_argument_format(True, root_schema, loose_strings=True)
        )
    if parameters is None:
        # Server-side strict levels mark tools without parameters strict too;
        # they take no arguments rather than failing the whole constraint.
        return ConstStringFormat(value="")
    if not isinstance(parameters, dict):
        raise ValueError(
            f"Kimi K3 strict tool {tool.function.name!r} must define parameters"
        )
    schema_types = _schema_types(parameters, parameters)
    if "object" not in schema_types:
        raise ValueError(
            f"Kimi K3 tool {tool.function.name!r} parameters must be an object schema"
        )
    return _strict_arguments_format(parameters)


def _tool_call_tag(tool: Tool, arguments_format: Optional[Format] = None) -> TagFormat:
    name = _escape_attr(tool.function.name)
    if arguments_format is None:
        arguments_format = _tool_arguments_format(tool)
    return TagFormat(
        begin=f'{CALL_OPEN} tool="{name}" index="',
        content=SequenceFormat(
            elements=[
                RegexFormat(pattern=r"[1-9][0-9]*"),
                ConstStringFormat(value='"<|sep|>'),
                arguments_format,
            ]
        ),
        end=CALL_CLOSE,
    )


def _tool_calls_tag(call_tags: List[TagFormat], parallel_tool_calls: bool) -> TagFormat:
    if parallel_tool_calls:
        content: Format = TagsWithSeparatorFormat(
            tags=call_tags, separator="", at_least_one=True
        )
    elif len(call_tags) == 1:
        content = call_tags[0]
    else:
        content = OrFormat(elements=call_tags)
    return TagFormat(begin=TOOLS_OPEN, content=content, end=TOOLS_CLOSE)


def _auto_suffix(
    tools_tag: TagFormat, parallel_tool_calls: bool
) -> TriggeredTagsFormat:
    # A retriggered second tools section would evade the single-call limit.
    return TriggeredTagsFormat(
        triggers=[TOOLS_OPEN],
        tags=[tools_tag],
        excludes=[THINK_OPEN, THINK_CLOSE, CALL_OPEN],
        stop_after_first=not parallel_tool_calls,
    )


def _with_reasoning(suffix: Format, thinking_mode: bool) -> Format:
    if not thinking_mode:
        return suffix
    return SequenceFormat(
        elements=[
            TagFormat(begin="", content=AnyTextFormat(), end=THINK_CLOSE),
            suffix,
        ]
    )


def _single_xtml_type(
    schema: Union[bool, Dict[str, Any]], root_schema: Dict[str, Any]
) -> Optional[str]:
    schema_types = _schema_types(schema, root_schema)
    if len(schema_types) != 1:
        return None
    return _JSON_TO_XTML_TYPE[schema_types[0]]


def _nonempty_argument_format(key: str, xtml_type: str) -> Format:
    # A token-based end keeps the first close token out of both content formats.
    argument = TagFormat(
        begin=(
            f'<|open|>argument key="{_escape_attr(key)}" ' f'type="{xtml_type}"<|sep|>'
        ),
        content=SequenceFormat(elements=[ExcludeTokenFormat(), AnyTokensFormat()]),
        end=TokenFormat(token=_CLOSE_TOKEN),
    )
    return SequenceFormat(
        elements=[
            argument,
            ConstStringFormat(value=_ARGUMENT_CLOSE_SUFFIX),
        ]
    )


def _auto_tool_arguments_format(tool: Tool) -> Format:
    parameters = tool.function.parameters
    if not isinstance(parameters, dict):
        return AnyTextFormat()
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        return AnyTextFormat()

    required_tags: List[Format] = []
    for key in required:
        schema = properties.get(key)
        if not isinstance(key, str) or not isinstance(schema, (bool, dict)):
            return AnyTextFormat()
        xtml_type = _single_xtml_type(schema, parameters)
        if xtml_type is None:
            return AnyTextFormat()
        required_tags.append(_nonempty_argument_format(key, xtml_type))

    if required_tags:
        elements = required_tags
    else:
        alternatives = [
            _nonempty_argument_format(key, xtml_type)
            for key, schema in properties.items()
            if isinstance(key, str)
            and isinstance(schema, (bool, dict))
            and (xtml_type := _single_xtml_type(schema, parameters)) is not None
        ]
        if not alternatives:
            return AnyTextFormat()
        elements = [
            (
                alternatives[0]
                if len(alternatives) == 1
                else OrFormat(elements=alternatives)
            )
        ]

    return SequenceFormat(
        elements=[*elements, AnyTextFormat(excludes=[CALL_OPEN, CALL_CLOSE])]
    )


def get_kimik3_auto_tool_call_structural_tag(
    tools: List[Tool],
    thinking_mode: bool = False,
    parallel_tool_calls: bool = True,
) -> Optional[StructuralTag]:
    if not tools:
        return None

    call_tags = [
        _tool_call_tag(tool, _auto_tool_arguments_format(tool)) for tool in tools
    ]
    suffix = _auto_suffix(
        _tool_calls_tag(call_tags, parallel_tool_calls), parallel_tool_calls
    )
    return StructuralTag(format=_with_reasoning(suffix, thinking_mode))


def _select_tools(
    tools: List[Tool],
    tool_choice: Union[ToolChoice, Literal["auto", "required"]],
) -> Tuple[List[Tool], bool]:
    if not isinstance(tool_choice, ToolChoice):
        return tools, tool_choice == "required"
    name = tool_choice.function.name
    selected = [tool for tool in tools if tool.function.name == name]
    if not selected:
        raise ValueError(f"Kimi K3 tool choice {name!r} is not in the tools list")
    return selected, True


def get_kimik3_structural_tag(
    tools: List[Tool],
    tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
    thinking_mode: bool = False,
    parallel_tool_calls: bool = True,
) -> StructuralTag:
    selected_tools, at_least_one = _select_tools(tools, tool_choice)
    if not selected_tools:
        raise ValueError("Kimi K3 structural tags require at least one tool")

    call_tags = [_tool_call_tag(tool) for tool in selected_tools]
    tools_tag = _tool_calls_tag(call_tags, parallel_tool_calls)
    if at_least_one:
        suffix: Format = SequenceFormat(
            elements=[
                AnyTextFormat(
                    excludes=[TOOLS_OPEN, THINK_OPEN, THINK_CLOSE, CALL_OPEN]
                ),
                tools_tag,
            ]
        )
    else:
        suffix = _auto_suffix(tools_tag, parallel_tool_calls)
    return StructuralTag(format=_with_reasoning(suffix, thinking_mode))
