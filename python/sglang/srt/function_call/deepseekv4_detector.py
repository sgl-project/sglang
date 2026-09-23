import json
import logging
import re
from copy import deepcopy

from jsonschema import Draft202012Validator, SchemaError
from referencing import Registry
from referencing.exceptions import NoSuchResource, Unresolvable

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.core_types import StreamingParseResult, ToolCallItem
from sglang.srt.function_call.deepseekv4_format import mask_literals as _mask_literals
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.utils import normalize_json_schema_types

logger = logging.getLogger(__name__)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate DSML JSON argument key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid DSML JSON constant {value}")


def _strict_json_loads(text: str):
    return json.loads(
        text,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )


def _reject_schema_retrieval(uri: str):
    raise NoSuchResource(ref=uri)


def _validate_string_parameter(name: str, value: str, *, complete: bool) -> None:
    prefix = "</｜DSML｜parameter"
    if prefix not in value:
        return
    visible = _mask_literals(value, heredocs=True)
    for match in re.finditer(re.escape(prefix), visible):
        previous = match.start() - 1
        while previous >= 0 and value[previous] == "\\":
            previous -= 1
        if (match.start() - previous - 1) % 2:
            continue
        if match.end() == len(value):
            if not complete:
                continue
        elif (
            value[match.end()] == ">"
            or value[match.end()].isalnum()
            or value[match.end()] == "_"
        ):
            continue
        raise ValueError(
            f"Malformed DSML parameter terminator in string parameter {name!r}"
        )


class DeepSeekV4Detector(DeepSeekV32Detector):
    """
    Detector for DeepSeek V4 model function call format.

    The DeepSeek V4 format uses XML-like DSML tags to delimit function calls.
    Supports two parameter formats:

    Format 1 - XML Parameter Tags:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>
        ...
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Format 2 - Direct JSON:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="function_name">
        {
            "param_name": "value"
        }
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Examples:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        { "city": "San Francisco" }
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜DSML｜tool_calls>` and `</｜DSML｜tool_calls>`
    - Individual Tool Call: Wrapped between `<｜DSML｜invoke name="...">` and `</｜DSML｜invoke>`
    - Parameters: Either XML tags or direct JSON format
    - Supports multiple tool calls

    Protocol structure, completed JSON arguments, and the provided tool schema
    are validated by default. Validation neither coerces values nor supplies
    missing fields, and schema references are resolved locally only.
    Malformed calls fail explicitly instead of falling back to assistant text.
    Protocol-like literals in ordinary content and valid arguments are not
    subject to a blanket marker-content filter. Confirmed reasoning-boundary
    normalization is handled separately by the reasoning parser; arbitrary
    reasoning text is not rewritten.

    Reference: DeepSeek V4 format specification
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜DSML｜tool_calls>"
        self.eot_token = "</｜DSML｜tool_calls>"
        self.function_calls_regex = r"<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>"
        self._schema_validators: dict[str, Draft202012Validator] = {}
        self._quote_history: list[str] = []
        self._active_invoke_start = 0

    def _find_invoke(self, text: str) -> re.Match | None:
        pattern = re.compile(self.invoke_regex, re.DOTALL)
        if self.current_tool_name_sent:
            return pattern.match(text, self._active_invoke_start)
        history = "".join(self._quote_history)
        masked = _mask_literals(history + text)[len(history) :]
        for marker in re.finditer(r"<｜DSML｜invoke\b", masked):
            match = pattern.match(text, marker.start())
            if match is not None:
                self._active_invoke_start = match.start()
                return match
        return None

    def _extract_preamble(self, text: str, invoke_start: int) -> str:
        history = "".join(self._quote_history)
        prefix = _mask_literals(history + text[:invoke_start])[len(history) :]
        marker = re.search(r"</?｜DSML｜", prefix)
        # A parsed invoke establishes the protocol boundary even when its outer
        # wrapper is damaged. Such wrapper bytes are not assistant prose.
        start = marker.start() if marker else invoke_start
        self._quote_history.clear()
        return text[:start].removesuffix("\n\n")

    def _raise_parse_error(self, error: Exception) -> None:
        raise ValueError(f"Failed to parse DeepSeek V4 tool output: {error}") from error

    def _parse_parameters_from_xml(
        self, invoke_content: str, allow_partial: bool = False
    ) -> str:
        if invoke_content.lstrip().startswith("{"):
            # Keep direct-JSON arguments private until their strings can be
            # validated; do not repair or reserialize partial JSON.
            if allow_partial:
                return ""
            parameters = _strict_json_loads(invoke_content)
            for name, value in parameters.items():
                if isinstance(value, str):
                    _validate_string_parameter(name, value, complete=True)
            return super()._parse_parameters_from_xml(invoke_content, False)
        last_match_end = 0
        names: set[str] = set()
        for match in re.finditer(self.parameter_regex, invoke_content, re.DOTALL):
            if invoke_content[last_match_end : match.start()].strip():
                raise ValueError("Malformed DSML parameter boundary")
            last_match_end = match.end()
            self._validate_xml_parameter(match, names, complete=True)
        remaining = invoke_content[last_match_end:]
        if allow_partial:
            partial = re.search(self.partial_parameter_regex, remaining, re.DOTALL)
            if partial:
                if remaining[: partial.start()].strip():
                    raise ValueError("Malformed DSML parameter boundary")
                self._validate_xml_parameter(partial, names, complete=False)
            elif tail := remaining.lstrip():
                parameter_start = "<｜DSML｜parameter"
                if not (
                    parameter_start.startswith(tail)
                    or tail.startswith(parameter_start)
                    or self.invoke_end_token.startswith(tail)
                ):
                    raise ValueError("Malformed DSML parameter boundary")
        elif remaining.strip():
            raise ValueError("Incomplete DSML parameter at end of invoke")
        return super()._parse_parameters_from_xml(invoke_content, allow_partial)

    def _validate_xml_parameter(
        self, match: re.Match, names: set[str], *, complete: bool
    ) -> None:
        name, string_flag, value = match.groups()
        if name in names:
            raise ValueError(f"Duplicate DSML parameter {name!r}")
        names.add(name)
        if string_flag not in {"true", "false"}:
            raise ValueError(f"Invalid DSML string flag for parameter {name!r}")
        if string_flag == "true":
            _validate_string_parameter(name, value, complete=complete)
        elif complete:
            try:
                _strict_json_loads(value.strip())
            except ValueError as error:
                raise ValueError(
                    f"Invalid JSON in DSML non-string parameter {name!r}"
                ) from error

    def _validate_arguments(
        self,
        name: str,
        arguments: str,
        tools: list[Tool],
        *,
        complete: bool,
    ) -> None:
        sent = self.streamed_args_for_tool[self.current_tool_id]
        if not arguments.startswith(sent):
            raise ValueError(f"Non-monotonic DSML argument stream for tool {name!r}")
        if complete:
            try:
                parameters = _strict_json_loads(arguments)
            except ValueError as error:
                raise ValueError(
                    f"Invalid JSON in completed DSML arguments for tool {name!r}"
                ) from error
            if not isinstance(parameters, dict):
                raise ValueError(f"DSML arguments for tool {name!r} must be an object")
            self._validate_against_tool_schema(name, parameters, tools)

    def _validate_against_tool_schema(
        self, name: str, parameters: dict, tools: list[Tool]
    ) -> None:
        validator = self._schema_validators.get(name)
        if validator is None:
            tool = next(
                (tool for tool in reversed(tools) if tool.function.name == name), None
            )
            if tool is None:
                if envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                    logger.warning("No schema for forwarded DeepSeek V4 tool: %s", name)
                    return
                raise ValueError(f"Undefined DSML tool {name!r}")
            schema = deepcopy(tool.function.parameters or {})
            normalize_json_schema_types(schema)
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as error:
                raise ValueError(f"Invalid schema for DSML tool {name!r}") from error
            validator = Draft202012Validator(
                schema, registry=Registry(retrieve=_reject_schema_retrieval)
            )
            self._schema_validators[name] = validator
        try:
            error = next(validator.iter_errors(parameters), None)
        except Unresolvable as error:
            raise ValueError(
                f"Unresolvable schema reference for DSML tool {name!r}; "
                "external retrieval is disabled"
            ) from error
        if error is not None:
            path = "/".join(
                str(part).replace("~", "~0").replace("/", "~1")
                for part in error.absolute_path
            )
            raise ValueError(
                f"DSML tool arguments violate schema for {name!r}: "
                f"{error.validator} at /{path}"
            )

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        result = super().parse_streaming_increment(new_text, tools)
        if result.calls:
            self._quote_history.clear()
        elif result.normal_text:
            self._quote_history.append(result.normal_text)
        return result

    def finish(self, tools: list[Tool]) -> StreamingParseResult:
        super().finish(tools)
        if self.current_tool_name_sent:
            raise ValueError("Incomplete DSML invoke at end of stream")
        remaining = self._buffer
        if self.current_tool_id >= 0:
            while remaining.lstrip().startswith(self.eot_token):
                remaining = remaining.lstrip()[len(self.eot_token) :]
            if not remaining.strip():
                remaining = ""
        history = "".join(self._quote_history)
        masked = _mask_literals(history + remaining)[len(history) :]
        if re.search(r"</?｜DSML｜", masked):
            raise ValueError("Incomplete or malformed DSML tool block at end of stream")
        self._buffer = ""
        self._quote_history.clear()
        return StreamingParseResult(normal_text=remaining)

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        detector = type(self)()
        parsed = detector.parse_streaming_increment(text, tools)
        tail = detector.finish(tools)
        accumulated: dict[int, ToolCallItem] = {}
        for item in parsed.calls + tail.calls:
            call = accumulated.setdefault(
                item.tool_index,
                ToolCallItem(tool_index=item.tool_index, parameters=""),
            )
            if item.name:
                call.name = item.name
            call.parameters += item.parameters
        calls = []
        for call in accumulated.values():
            calls.extend(
                self.parse_base_json(
                    {"name": call.name, "parameters": json.loads(call.parameters)},
                    tools,
                )
            )
        return StreamingParseResult(
            normal_text=parsed.normal_text + tail.normal_text, calls=calls
        )

    def get_structural_tag_name(self) -> str:
        return "deepseek_v4"
