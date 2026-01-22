import json
import sys
from typing import Any, Dict, List, Optional

from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.state_machine import (
    JsonConfig,
    ParseResult,
    ToolParserStateMachine,
    UniversalToolParserState,
    XmlConfig,
)


class UniversalXmlStateMachine(ToolParserStateMachine):
    def __init__(self, config: XmlConfig):
        self.config = config
        self.state = UniversalToolParserState.IDLE
        self.buffer = ""
        self.current_tool_name = ""
        self.current_parameter_name = ""
        self.current_parameter_value = ""
        self.current_parameters: Dict[str, Any] = {}
        self.completed_tools: List[Dict[str, Any]] = []
        self._param_tag_complete = False
        self.new_calls: List[ToolCallItem] = []
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

    def parse(self, data: str) -> ParseResult:
        sys.stderr.write(
            f"DEBUG SM: parse data='{data}', state={self.state}, buffer='{self.buffer}'\n"
        )
        self.buffer += data
        self.completed_tools = []
        self.new_calls = []
        self.normal_text = ""
        error = None

        while True:
            sys.stderr.write(
                f"DEBUG SM LOOP: state={self.state}, buffer='{self.buffer}'\n"
            )
            if self.state == UniversalToolParserState.IDLE:
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

            elif self.state == UniversalToolParserState.TOOL_START:
                m_start = self.name_start_marker
                if m_start and m_start in self.buffer:
                    pos = self.buffer.find(m_start)
                    self.buffer = self.buffer[pos + len(m_start) :]
                    self.state = UniversalToolParserState.IN_TOOL_NAME
                    continue
                elif ">" in self.buffer:
                    pos = self.buffer.find(">")
                    self.buffer = self.buffer[pos + 1 :]
                    if not self.config.tool_name_tag and not self.config.tool_name_attr:
                        self.state = UniversalToolParserState.IN_TOOL_NAME
                        self.name_end_marker = "<"
                    else:
                        self.state = UniversalToolParserState.TOOL_NAME_END
                    continue
                else:
                    break

            elif self.state == UniversalToolParserState.IN_TOOL_NAME:
                if self.name_end_marker in self.buffer:
                    pos = self.buffer.find(self.name_end_marker)
                    self.current_tool_name += self.buffer[:pos].strip()
                    self.buffer = self.buffer[
                        pos
                        + (
                            len(self.name_end_marker)
                            if self.name_end_marker != "<"
                            else 0
                        ) :
                    ]
                    self.state = UniversalToolParserState.TOOL_NAME_END
                    self.new_calls.append(
                        ToolCallItem(
                            tool_index=0, name=self.current_tool_name, parameters=""
                        )
                    )
                    sys.stderr.write(
                        f"DEBUG SM: emitted new call {self.current_tool_name}\n"
                    )
                    continue
                else:
                    self.current_tool_name += self.buffer
                    self.buffer = ""
                    break

            elif (
                self.state == UniversalToolParserState.TOOL_NAME_END
                or self.state == UniversalToolParserState.PARAMETER_END
            ):
                if (
                    self.name_start_marker
                    and self.name_start_marker in self.buffer
                    and not self.current_tool_name
                ):
                    pos = self.buffer.find(self.name_start_marker)
                    self.buffer = self.buffer[pos + len(self.name_start_marker) :]
                    self.state = UniversalToolParserState.IN_TOOL_NAME
                    continue
                elif self.param_start_marker and self.param_start_marker in self.buffer:
                    pos = self.buffer.find(self.param_start_marker)
                    self.buffer = self.buffer[pos + len(self.param_start_marker) :]
                    self.state = UniversalToolParserState.PARAMETER_START
                    self._param_tag_complete = False
                    self.current_parameter_name = ""
                    self.current_parameter_value = ""
                    continue
                elif self.key_start_marker and self.key_start_marker in self.buffer:
                    pos = self.buffer.find(self.key_start_marker)
                    self.buffer = self.buffer[pos + len(self.key_start_marker) :]
                    self.state = UniversalToolParserState.IN_PARAMETER_NAME
                    continue
                elif self.tool_end_marker in self.buffer:
                    pos = self.buffer.find(self.tool_end_marker)
                    self.buffer = self.buffer[pos + len(self.tool_end_marker) :]
                    self.state = UniversalToolParserState.TOOL_END
                    continue
                elif (
                    ">" in self.buffer
                    and self.state == UniversalToolParserState.TOOL_NAME_END
                ):
                    pos = self.buffer.find(">")
                    self.buffer = self.buffer[pos + 1 :]
                    continue
                else:
                    break

            elif self.state == UniversalToolParserState.PARAMETER_START:
                p_marker = self.param_name_marker
                if p_marker and p_marker in self.buffer:
                    pos = self.buffer.find(p_marker)
                    self.buffer = self.buffer[pos + len(p_marker) :]
                    self.state = UniversalToolParserState.IN_PARAMETER_NAME
                    continue
                else:
                    break

            elif self.state == UniversalToolParserState.IN_PARAMETER_NAME:
                if self.key_end_marker:
                    e_marker = self.key_end_marker
                elif self.config.attr_sep == "=":
                    e_marker = ">"
                else:
                    e_marker = '"'

                if e_marker in self.buffer:
                    pos = self.buffer.find(e_marker)
                    self.current_parameter_name += self.buffer[:pos].strip()
                    self.buffer = self.buffer[pos + len(e_marker) :]
                    self.state = UniversalToolParserState.IN_PARAMETER_VALUE
                    if e_marker == ">":
                        self._param_tag_complete = True
                    continue
                else:
                    self.current_parameter_name += self.buffer
                    self.buffer = ""
                    break

            elif self.state == UniversalToolParserState.IN_PARAMETER_VALUE:
                if not self._param_tag_complete and not self.key_start_marker:
                    if ">" in self.buffer:
                        pos = self.buffer.find(">")
                        self.buffer = self.buffer[pos + 1 :]
                        self._param_tag_complete = True
                        continue
                    else:
                        break
                elif self.val_start_marker and not self._param_tag_complete:
                    v_start = self.val_start_marker
                    if v_start in self.buffer:
                        pos = self.buffer.find(v_start)
                        self.buffer = self.buffer[pos + len(v_start) :]
                        self._param_tag_complete = True
                        continue
                    else:
                        break
                else:
                    end_m = (
                        self.val_end_marker
                        if self.val_end_marker
                        else self.param_end_marker
                    )
                    if end_m and end_m in self.buffer:
                        pos = self.buffer.find(end_m)
                        rest = self.buffer[pos + len(end_m) :]
                        trimmed_rest = rest.lstrip()

                        next_markers = [
                            self.param_start_marker,
                            self.key_start_marker,
                            self.tool_end_marker,
                            self.root_end_marker,
                        ]
                        is_genuine = (
                            any(m and trimmed_rest.startswith(m) for m in next_markers)
                            or not trimmed_rest
                        )

                        if is_genuine:
                            self.current_parameter_value += self.buffer[:pos]
                            self.current_parameters[self.current_parameter_name] = (
                                self.current_parameter_value
                            )
                            self.buffer = rest
                            self.state = UniversalToolParserState.PARAMETER_END
                            continue
                        else:
                            self.current_parameter_value += self.buffer[
                                : pos + len(end_m)
                            ]
                            self.buffer = rest
                            continue
                    else:
                        if len(self.buffer) > 20:
                            to_consume = len(self.buffer) - 20
                            self.current_parameter_value += self.buffer[:to_consume]
                            self.buffer = self.buffer[to_consume:]
                        break

            elif self.state == UniversalToolParserState.TOOL_END:
                self.completed_tools.append(
                    {
                        "name": self.current_tool_name,
                        "arguments": self.current_parameters,
                    }
                )
                self.current_tool_name = ""
                self.current_parameters = {}
                self.state = UniversalToolParserState.IDLE
                continue
            else:
                break

        return ParseResult(
            state=self.state,
            completed_tools=self.completed_tools,
            remaining=self.buffer,
            streaming_calls=self.new_calls,
            normal_text=self.normal_text,
            error=error,
        )


class UniversalJsonStateMachine(ToolParserStateMachine):
    def __init__(self, config: JsonConfig):
        self.config = config
        self.state = UniversalToolParserState.IDLE
        self.buffer = ""
        self.brace_count = 0
        self.in_string = False
        self.escape = False
        self.completed_tools: List[Dict[str, Any]] = []
        self.new_calls: List[ToolCallItem] = []
        self.normal_text = ""

    def parse(self, data: str) -> ParseResult:
        self.buffer += data
        self.completed_tools = []
        self.new_calls = []
        self.normal_text = ""
        error = None

        while True:
            if self.state == UniversalToolParserState.IDLE:
                prefix = self.config.prefix or "{"
                pos = self.buffer.find(prefix)
                if pos != -1:
                    self.normal_text += self.buffer[:pos]
                    rest = self.buffer[pos + len(prefix) :]
                    if rest.startswith("\n"):
                        rest = rest[1:]
                    self.buffer = rest
                    self.state = UniversalToolParserState.IN_PARAMETER_VALUE
                    self.brace_count = 1 if prefix.endswith("{") else 0
                    self.in_string = False
                    self.escape = False
                    continue
                else:
                    max_p = 0
                    for i in range(1, min(len(self.buffer), len(prefix)) + 1):
                        if prefix.startswith(self.buffer[-i:]):
                            max_p = i
                    if max_p > 0:
                        self.normal_text += self.buffer[:-max_p]
                        self.buffer = self.buffer[-max_p:]
                    else:
                        self.normal_text += self.buffer
                        self.buffer = ""
                    break

            elif self.state == UniversalToolParserState.IN_PARAMETER_VALUE:
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
                            if self.brace_count == 0:
                                json_str = self.buffer[: i + 1]
                                if (
                                    not self.config.prefix
                                    or not self.config.prefix.endswith("{")
                                ) and not json_str.lstrip().startswith("{"):
                                    json_str = "{" + json_str

                                try:
                                    parsed = json.loads(json_str)
                                    if isinstance(parsed, dict):
                                        self.completed_tools.append(parsed)
                                    elif isinstance(parsed, list):
                                        self.completed_tools.extend(parsed)
                                    self.state = UniversalToolParserState.IDLE
                                    consumed_idx = i
                                    break
                                except json.JSONDecodeError:
                                    pass
                    if char != "\\":
                        self.escape = False

                if consumed_idx != -1:
                    self.buffer = self.buffer[consumed_idx + 1 :]
                    if self.config.suffix:
                        trimmed = self.buffer.lstrip()
                        if trimmed.startswith(self.config.suffix):
                            skip = self.buffer.find(self.config.suffix) + len(
                                self.config.suffix
                            )
                            self.buffer = self.buffer[skip:]
                    continue
                else:
                    break
            else:
                break

        return ParseResult(
            state=self.state,
            completed_tools=self.completed_tools,
            remaining=self.buffer,
            streaming_calls=self.new_calls,
            normal_text=self.normal_text,
            error=error,
        )
