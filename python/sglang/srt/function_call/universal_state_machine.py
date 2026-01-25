import json
import logging
from typing import Any, Dict, List, Optional

from sglang.srt.function_call.state_machine import (
    JsonConfig,
    ParseResult,
    ParserEvent,
    ToolParserStateMachine,
    UniversalToolParserState,
    XmlConfig,
)

logger = logging.getLogger(__name__)


class UniversalXmlStateMachine(ToolParserStateMachine):
    def __init__(self, config: XmlConfig):
        self.config = config
        self.state = UniversalToolParserState.IDLE
        self.buffer = ""
        self.current_tool_name = ""
        self.current_parameter_name = ""
        self.current_parameter_value = ""
        self.current_parameters: Dict[str, Any] = {}
        self._param_tag_complete = False
        self.normal_text = ""

        self.tool_start_marker = f"<{self.config.tool_tag}"
        self.tool_end_marker = f"</{self.config.tool_tag}>"

        if self.config.root_tag:
            self.root_start_marker: Optional[str] = f"<{self.config.root_tag}"
            self.root_end_marker: Optional[str] = f"</{self.config.root_tag}>"
        else:
            self.root_start_marker = None
            self.root_end_marker = None

        if self.config.tool_name_tag:
            self.name_start_marker: Optional[str] = f"<{self.config.tool_name_tag}"
            if self.config.tool_name_tag_attr:
                self.name_start_marker += (
                    f" {self.config.tool_name_tag_attr}{self.config.attr_sep}"
                )
                self.name_end_marker = '"' if self.config.attr_sep == '="' else ">"
            elif self.config.attr_sep == "=":
                self.name_start_marker += "="
                self.name_end_marker = ">"
            else:
                self.name_start_marker += ">"
                self.name_end_marker = f"</{self.config.tool_name_tag}>"
        elif self.config.tool_name_attr:
            self.name_start_marker = (
                f" {self.config.tool_name_attr}{self.config.attr_sep}"
            )
            self.name_end_marker = '"' if self.config.attr_sep == '="' else ">"
        else:
            self.name_start_marker = None
            self.name_end_marker = ""

        if self.config.param_tag:
            self.param_start_marker: Optional[str] = f"<{self.config.param_tag}"
            self.param_end_marker: Optional[str] = f"</{self.config.param_tag}>"
            if self.config.param_name_attr:
                self.param_name_marker: Optional[str] = (
                    f" {self.config.param_name_attr}{self.config.attr_sep}"
                )
            elif self.config.attr_sep == "=":
                self.param_name_marker = "="
            else:
                self.param_name_marker = None
        else:
            self.param_start_marker = None
            self.param_end_marker = None
            self.param_name_marker = None

        if self.config.param_key_tag:
            self.key_start_marker: Optional[str] = f"<{self.config.param_key_tag}>"
            self.key_end_marker: Optional[str] = f"</{self.config.param_key_tag}>"
            self.val_start_marker: Optional[str] = f"<{self.config.param_value_tag}>"
            self.val_end_marker: Optional[str] = f"</{self.config.param_value_tag}>"
        else:
            self.key_start_marker = None
            self.key_end_marker = None
            self.val_start_marker = None
            self.val_end_marker = None

    def _ends_with_partial(self, buffer: str, tokens: List[Optional[str]]) -> int:
        max_partial = 0
        for token in tokens:
            if not token:
                continue
            for i in range(1, min(len(buffer), len(token)) + 1):
                if token.startswith(buffer[-i:]):
                    max_partial = max(max_partial, i)
        return max_partial

    def reset(self):
        self.state = UniversalToolParserState.IDLE
        self.buffer = ""
        self.current_tool_name = ""
        self.current_parameter_name = ""
        self.current_parameter_value = ""
        self.current_parameters = {}
        self._param_tag_complete = False
        self.normal_text = ""

    def parse(self, data: str) -> ParseResult:
        self.buffer += data
        completed_tools_to_return: List[Dict[str, Any]] = []
        events: List[ParserEvent] = []
        self.normal_text = ""
        error = None

        while True:
            match self.state:
                case UniversalToolParserState.IDLE:
                    markers = [
                        m
                        for m in [
                            self.tool_start_marker,
                            self.root_start_marker,
                            self.root_end_marker,
                        ]
                        if m
                    ]
                    positions = [
                        (self.buffer.find(m), m)
                        for m in markers
                        if self.buffer.find(m) != -1
                    ]

                    if not positions:
                        max_p = self._ends_with_partial(self.buffer, markers)
                        if max_p > 0:
                            self.normal_text += self.buffer[:-max_p]
                            self.buffer = self.buffer[-max_p:]
                            break
                        else:
                            self.normal_text += self.buffer
                            self.buffer = ""
                            break

                    pos, marker = min(positions, key=lambda x: x[0])
                    self.normal_text += self.buffer[:pos]
                    self.buffer = self.buffer[pos:]

                    if self.buffer.startswith(self.tool_start_marker):
                        self.buffer = self.buffer[len(self.tool_start_marker) :]
                        self.state = UniversalToolParserState.TOOL_START
                        # Nameless tool start to initialize detector tracking
                        events.append(ParserEvent(event_type="tool_start"))
                        continue
                    elif self.root_start_marker and self.buffer.startswith(
                        self.root_start_marker
                    ):
                        end_idx = self.buffer.find(">")
                        if end_idx != -1:
                            self.buffer = self.buffer[end_idx + 1 :]
                            continue
                        else:
                            break
                    elif self.root_end_marker and self.buffer.startswith(
                        self.root_end_marker
                    ):
                        self.buffer = self.buffer[len(self.root_end_marker) :]
                        continue
                    else:
                        break

                case UniversalToolParserState.TOOL_START:
                    if self.name_start_marker and self.name_start_marker in self.buffer:
                        pos = self.buffer.find(self.name_start_marker)
                        # Ensure we consume junk before the name start marker
                        self.buffer = self.buffer[pos + len(self.name_start_marker) :]
                        self.state = UniversalToolParserState.IN_TOOL_NAME
                        continue

                    # Skip leading whitespace or junk if we are not looking for an attribute name
                    self.buffer = self.buffer.lstrip()
                    if self.buffer.startswith(">"):
                        self.buffer = self.buffer[1:]
                        if (
                            not self.config.tool_name_tag
                            and not self.config.tool_name_attr
                        ):
                            self.state = UniversalToolParserState.IN_TOOL_NAME
                            self.name_end_marker = "<"
                        else:
                            self.state = UniversalToolParserState.TOOL_NAME_END
                        continue
                    elif ">" in self.buffer:
                        pos = self.buffer.find(">")
                        self.buffer = self.buffer[pos + 1 :]
                        if (
                            not self.config.tool_name_tag
                            and not self.config.tool_name_attr
                        ):
                            self.state = UniversalToolParserState.IN_TOOL_NAME
                            self.name_end_marker = "<"
                        else:
                            self.state = UniversalToolParserState.TOOL_NAME_END
                        continue
                    elif (
                        not self.config.tool_name_tag and not self.config.tool_name_attr
                    ):
                        self.state = UniversalToolParserState.IN_TOOL_NAME
                        self.name_end_marker = "<"
                        continue
                    else:
                        break

                case UniversalToolParserState.IN_TOOL_NAME:
                    if self.name_end_marker in self.buffer:
                        pos = self.buffer.find(self.name_end_marker)
                        name_chunk = self.buffer[:pos]
                        self.current_tool_name += name_chunk
                        self.current_tool_name = self.current_tool_name.strip()
                        self.buffer = self.buffer[
                            pos
                            + (
                                len(self.name_end_marker)
                                if self.name_end_marker != "<"
                                else 0
                            ) :
                        ]
                        self.state = UniversalToolParserState.TOOL_NAME_END
                        # Named tool start to update detector tracking
                        events.append(
                            ParserEvent(
                                event_type="tool_start", name=self.current_tool_name
                            )
                        )
                        continue
                    else:
                        self.current_tool_name += self.buffer
                        self.current_tool_name = self.current_tool_name.lstrip()
                        self.buffer = ""
                        break

                case (
                    UniversalToolParserState.TOOL_NAME_END
                    | UniversalToolParserState.PARAMETER_END
                ):
                    # Skip leading whitespace or junk
                    self.buffer = self.buffer.lstrip()
                    if self.buffer.startswith(">"):
                        self.buffer = self.buffer[1:].lstrip()

                    markers = []
                    if self.name_start_marker and not self.current_tool_name:
                        markers.append(
                            (
                                self.name_start_marker,
                                UniversalToolParserState.IN_TOOL_NAME,
                            )
                        )
                    if self.param_start_marker:
                        markers.append(
                            (
                                self.param_start_marker,
                                UniversalToolParserState.PARAMETER_START,
                            )
                        )
                    if self.key_start_marker:
                        markers.append(
                            (
                                self.key_start_marker,
                                UniversalToolParserState.IN_PARAMETER_NAME,
                            )
                        )
                    if self.tool_end_marker:
                        markers.append(
                            (self.tool_end_marker, UniversalToolParserState.TOOL_END)
                        )
                    if self.root_end_marker:
                        markers.append(
                            (self.root_end_marker, UniversalToolParserState.IDLE)
                        )

                    positions = [
                        (self.buffer.find(m), m, s)
                        for m, s in markers
                        if self.buffer.find(m) != -1
                    ]

                    if not positions:
                        # Check for partial markers
                        max_p = self._ends_with_partial(
                            self.buffer, [m for m, _ in markers]
                        )
                        if max_p > 0:
                            break
                        break

                    pos, marker, next_state = min(positions, key=lambda x: x[0])
                    self.buffer = self.buffer[pos + len(marker) :]

                    self.state = next_state
                    if next_state in [
                        UniversalToolParserState.PARAMETER_START,
                        UniversalToolParserState.IN_PARAMETER_NAME,
                    ]:
                        self._param_tag_complete = False
                        self.current_parameter_name = ""
                        self.current_parameter_value = ""
                    continue

                case UniversalToolParserState.PARAMETER_START:
                    if self.param_name_marker and self.param_name_marker in self.buffer:
                        pos = self.buffer.find(self.param_name_marker)
                        self.buffer = self.buffer[pos + len(self.param_name_marker) :]
                        self.state = UniversalToolParserState.IN_PARAMETER_NAME
                        continue
                    else:
                        break

                case UniversalToolParserState.IN_PARAMETER_NAME:
                    e_marker = self.key_end_marker or (
                        ">" if self.config.attr_sep == "=" else '"'
                    )
                    if e_marker in self.buffer:
                        pos = self.buffer.find(e_marker)
                        p_name_chunk = self.buffer[:pos].strip()
                        self.current_parameter_name += p_name_chunk
                        self.buffer = self.buffer[pos + len(e_marker) :]
                        self.state = UniversalToolParserState.IN_PARAMETER_VALUE
                        if e_marker == ">":
                            self._param_tag_complete = True
                        events.append(
                            ParserEvent(
                                event_type="parameter_start",
                                name=self.current_parameter_name,
                            )
                        )
                        continue
                    else:
                        self.current_parameter_name += self.buffer
                        self.buffer = ""
                        break

                case UniversalToolParserState.IN_PARAMETER_VALUE:
                    if not self._param_tag_complete:
                        if self.val_start_marker:
                            if self.val_start_marker in self.buffer:
                                pos = self.buffer.find(self.val_start_marker)
                                self.buffer = self.buffer[
                                    pos + len(self.val_start_marker) :
                                ]
                                self._param_tag_complete = True
                                continue
                            else:
                                if (
                                    self._ends_with_partial(
                                        self.buffer, [self.val_start_marker]
                                    )
                                    > 0
                                ):
                                    break
                                pass
                        else:
                            if self.buffer.startswith(">"):
                                self.buffer = self.buffer[1:]
                                self._param_tag_complete = True
                                continue
                            else:
                                self._param_tag_complete = True
                                continue

                    end_m = (
                        self.val_end_marker
                        if self.val_end_marker
                        else self.param_end_marker
                    )
                    if end_m and end_m in self.buffer:
                        pos = self.buffer.find(end_m)
                        content = self.buffer[:pos]
                        rest = self.buffer[pos + len(end_m) :]
                        trimmed_rest = rest.lstrip()

                        next_markers = [
                            self.param_start_marker,
                            self.key_start_marker,
                            self.val_start_marker,
                            self.tool_start_marker,
                            self.root_start_marker,
                            self.tool_end_marker,
                            self.root_end_marker,
                            self.param_end_marker,
                            self.val_end_marker,
                        ]

                        is_genuine = (not trimmed_rest) or any(
                            m and trimmed_rest.startswith(m) for m in next_markers
                        )

                        if not is_genuine:
                            # Ambiguity check
                            is_ambiguous = any(
                                m and m.startswith(trimmed_rest) for m in next_markers
                            )
                            if is_ambiguous and trimmed_rest:
                                break

                            if not trimmed_rest:
                                is_genuine = True
                            else:
                                is_genuine = False

                        if is_genuine:
                            self.current_parameter_value += content
                            if content:
                                events.append(
                                    ParserEvent(
                                        event_type="text_delta", text_delta=content
                                    )
                                )
                            self.current_parameters[self.current_parameter_name] = (
                                self.current_parameter_value
                            )
                            events.append(
                                ParserEvent(
                                    event_type="parameter",
                                    name=self.current_parameter_name,
                                    value=self.current_parameter_value,
                                )
                            )
                            self.buffer = rest
                            self.state = UniversalToolParserState.PARAMETER_END
                            continue
                        else:
                            # Not genuine, consume the marker as text delta
                            self.current_parameter_value += self.buffer[
                                : pos + len(end_m)
                            ]
                            events.append(
                                ParserEvent(
                                    event_type="text_delta",
                                    text_delta=self.buffer[: pos + len(end_m)],
                                )
                            )
                            self.buffer = rest
                            continue
                    else:
                        markers = [
                            self.param_start_marker,
                            self.key_start_marker,
                            self.val_start_marker,
                            self.tool_start_marker,
                            self.root_start_marker,
                            self.tool_end_marker,
                            self.root_end_marker,
                            self.val_end_marker,
                            self.param_end_marker,
                        ]
                        max_p = self._ends_with_partial(self.buffer, markers)
                        if len(self.buffer) > max_p:
                            to_consume = len(self.buffer) - max_p
                            delta = self.buffer[:to_consume]
                            self.current_parameter_value += delta
                            events.append(
                                ParserEvent(event_type="text_delta", text_delta=delta)
                            )
                            self.buffer = self.buffer[to_consume:]
                        break

                case UniversalToolParserState.TOOL_END:
                    completed_tools_to_return.append(
                        {
                            "name": self.current_tool_name,
                            "arguments": dict(self.current_parameters),
                        }
                    )
                    events.append(ParserEvent(event_type="tool_end"))
                    self.current_tool_name = ""
                    self.current_parameters = {}
                    self.state = UniversalToolParserState.IDLE
                    continue
                case _:
                    break

        return ParseResult(
            state=self.state,
            completed_tools=completed_tools_to_return,
            remaining=self.buffer,
            events=events,
            normal_text=self.normal_text,
            error=error,
        )


class UniversalJsonStateMachine(ToolParserStateMachine):
    def __init__(self, config: JsonConfig):
        self.config = config
        self.state = UniversalToolParserState.IDLE
        self.buffer = ""
        self.brace_count = 0
        self.bracket_count = 0
        self.in_string = False
        self.escape = False
        self.normal_text = ""
        self.current_tool_name = ""

    def reset(self):
        self.state = UniversalToolParserState.IDLE
        self.buffer = ""
        self.brace_count = 0
        self.bracket_count = 0
        self.in_string = False
        self.escape = False
        self.normal_text = ""
        self.current_tool_name = ""

    def _ends_with_partial(self, buffer: str, tokens: List[Optional[str]]) -> int:
        max_partial = 0
        for token in tokens:
            if not token:
                continue
            for i in range(1, min(len(buffer), len(token)) + 1):
                if token.startswith(buffer[-i:]):
                    max_partial = max(max_partial, i)
        return max_partial

    def parse(self, data: str) -> ParseResult:
        self.buffer += data
        completed_tools_to_return: List[Dict[str, Any]] = []
        events: List[ParserEvent] = []
        self.normal_text = ""
        error = None

        while True:
            match self.state:
                case UniversalToolParserState.IDLE:
                    prefix = self.config.prefix or self.config.name_prefix or "{"
                    pos = self.buffer.find(prefix)

                    if (
                        pos == -1
                        and self.config.prefix
                        and self.config.prefix != "{"
                        and "{" in self.buffer
                    ):
                        brace_pos = self.buffer.find("{")
                        max_p = self._ends_with_partial(
                            self.buffer[: brace_pos + 1], [prefix]
                        )
                        if max_p == 0:
                            prefix = "{"
                            pos = brace_pos

                    if pos != -1:
                        self.normal_text += self.buffer[:pos]
                        self.buffer = self.buffer[pos + len(prefix) :]
                        if (
                            self.config.name_prefix
                            and prefix == self.config.name_prefix
                        ):
                            self.state = UniversalToolParserState.IN_TOOL_NAME
                            self.current_tool_name = ""
                        else:
                            if self.buffer.startswith("\n"):
                                self.buffer = self.buffer[1:]
                            self.state = UniversalToolParserState.IN_PARAMETER_VALUE
                            self.brace_count = 1 if prefix.endswith("{") else 0
                            self.bracket_count = 1 if prefix.endswith("[") else 0
                            self.in_string = False
                            self.escape = False
                            events.append(ParserEvent(event_type="tool_start"))
                        continue
                    else:
                        max_p = self._ends_with_partial(self.buffer, [prefix])
                        if max_p > 0:
                            self.normal_text += self.buffer[:-max_p]
                            self.buffer = self.buffer[-max_p:]
                        else:
                            self.normal_text += self.buffer
                            self.buffer = ""
                        break

                case UniversalToolParserState.IN_TOOL_NAME:
                    suffix = self.config.name_suffix or "{"
                    pos = self.buffer.find(suffix)
                    if pos != -1:
                        self.current_tool_name += self.buffer[:pos].strip()
                        self.buffer = self.buffer[pos + len(suffix) :]
                        if self.buffer.startswith("\n"):
                            self.buffer = self.buffer[1:]
                        self.state = UniversalToolParserState.IN_PARAMETER_VALUE
                        self.brace_count = 1 if suffix.endswith("{") else 0
                        self.bracket_count = 1 if suffix.endswith("[") else 0
                        self.in_string = False
                        self.escape = False
                        events.append(
                            ParserEvent(
                                event_type="tool_start", name=self.current_tool_name
                            )
                        )
                        continue
                    else:
                        self.current_tool_name += self.buffer
                        self.buffer = ""
                        break

                case UniversalToolParserState.IN_PARAMETER_VALUE:
                    consumed_idx = -1
                    for i, char in enumerate(self.buffer):
                        if char == '"' and not self.escape:
                            self.in_string = not self.in_string
                        elif char == "\\" and self.in_string:
                            self.escape = not self.escape
                        elif not self.in_string:
                            if char == "{":
                                self.brace_count += 1
                            elif char == "}":
                                self.brace_count -= 1
                            elif char == "[":
                                self.bracket_count += 1
                            elif char == "]":
                                self.bracket_count -= 1

                            is_complete = (
                                self.bracket_count == 0
                                if self.config.is_array
                                else self.brace_count == 0
                            )

                            if is_complete:
                                json_str = self.buffer[: i + 1]
                                if not self.config.is_array:
                                    if (
                                        self.config.prefix
                                        and self.config.prefix.endswith("{")
                                    ) and not json_str.lstrip().startswith("{"):
                                        json_str = "{" + json_str
                                    elif (
                                        self.config.name_suffix
                                        and self.config.name_suffix.endswith("{")
                                    ) and not json_str.lstrip().startswith("{"):
                                        json_str = "{" + json_str
                                else:
                                    if (
                                        self.config.prefix
                                        and self.config.prefix.endswith("[")
                                    ) and not json_str.lstrip().startswith("["):
                                        json_str = "[" + json_str

                                try:
                                    parsed = json.loads(json_str)
                                    tools_to_add = (
                                        [parsed]
                                        if isinstance(parsed, dict)
                                        else (
                                            parsed if isinstance(parsed, list) else []
                                        )
                                    )
                                    # Emit text_delta BEFORE tool_end
                                    events.append(
                                        ParserEvent(
                                            event_type="text_delta",
                                            text_delta=self.buffer[: i + 1],
                                        )
                                    )
                                    for tool in tools_to_add:
                                        if (
                                            self.current_tool_name
                                            and "name" not in tool
                                        ):
                                            if (
                                                "arguments" not in tool
                                                and "parameters" not in tool
                                            ):
                                                tool = {
                                                    "name": self.current_tool_name,
                                                    "arguments": {
                                                        k: v for k, v in tool.items()
                                                    },
                                                }
                                            else:
                                                tool["name"] = self.current_tool_name
                                        completed_tools_to_return.append(tool)
                                    self.state = UniversalToolParserState.IDLE
                                    consumed_idx = i
                                    events.append(ParserEvent(event_type="tool_end"))
                                    break
                                except json.JSONDecodeError:
                                    pass
                        if char != "\\":
                            self.escape = False

                    if consumed_idx != -1:
                        self.buffer = self.buffer[consumed_idx + 1 :]
                        if self.config.suffix:
                            stripped_buffer = self.buffer.lstrip()
                            if stripped_buffer.startswith(self.config.suffix):
                                skip_len = self.buffer.find(self.config.suffix) + len(
                                    self.config.suffix
                                )
                                self.buffer = self.buffer[skip_len:]
                        continue
                    else:
                        # Eagerly emit text_delta for partial buffer
                        markers = [self.config.suffix] if self.config.suffix else []
                        max_p = self._ends_with_partial(self.buffer, markers)
                        if len(self.buffer) > max_p:
                            to_consume = len(self.buffer) - max_p
                            delta = self.buffer[:to_consume]
                            events.append(
                                ParserEvent(event_type="text_delta", text_delta=delta)
                            )
                            self.buffer = self.buffer[to_consume:]
                        break
                case _:
                    break

        return ParseResult(
            state=self.state,
            completed_tools=completed_tools_to_return,
            remaining=self.buffer,
            events=events,
            normal_text=self.normal_text,
            error=error,
        )
