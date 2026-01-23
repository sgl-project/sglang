import html
import json
import logging
from types import SimpleNamespace
from typing import Any, List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.glm_state_machine import GlmStateMachine
from sglang.srt.function_call.state_machine import UniversalToolParserState
from sglang.srt.function_call.utils import get_argument_type

logger = logging.getLogger(__name__)


def _convert_to_number(value: str) -> Any:
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        else:
            return int(value)
    except (ValueError, AttributeError):
        return value


class Glm4MoeDetector(BaseFormatDetector):
    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self.sm = GlmStateMachine()
        self._tool_indices = {}
        self._reset_streaming_state()

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def _reset_streaming_state(self):
        self._buffer = ""
        self.current_tool_id = -1
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []
        self.tool_name_sent = []
        self._streamed_raw_length = 0
        self.current_tool_name_sent = False
        self.sm.reset()

    def _convert_param_value(self, value: str, param_type: str) -> Any:
        value = html.unescape(value)
        if '\\"' in value:
            try:
                val_unescaped = value.encode("utf-8").decode("unicode_escape")
                if (val_unescaped.startswith("{") and val_unescaped.endswith("}")) or (
                    val_unescaped.startswith("[") and val_unescaped.endswith("]")
                ):
                    return json.loads(val_unescaped)
            except Exception:
                try:
                    val_unescaped = value.replace('\\"', '"').replace("\\\\", "\\")
                    return json.loads(val_unescaped)
                except Exception:
                    pass

        if param_type == "integer":
            try:
                return int(value.strip())
            except (ValueError, TypeError, AttributeError):
                return value
        elif param_type == "number":
            return _convert_to_number(value)
        elif param_type == "boolean":
            lower_val = value.lower().strip()
            if lower_val in ["true", "1", "yes", "on"]:
                return True
            elif lower_val in ["false", "0", "no", "off"]:
                return False
            return value

        if (value.startswith("{") and value.endswith("}")) or (
            value.startswith("[") and value.endswith("]")
        ):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        return value

    def _parse_parameter(self, fname: str, pname: str, pval: str, tools: Any) -> Any:
        inferred_type = get_argument_type(fname, pname, tools) or "string"
        return self._convert_param_value(pval, inferred_type)

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        self.sm.reset()
        result = self.sm.parse(text)
        calls = []
        for tool in result.completed_tools:
            name = tool["name"]
            raw_args = tool["arguments"]
            converted_args = {}
            for k, v in raw_args.items():
                converted_args[k] = self._parse_parameter(name, k, v, tools)
            raw = {"name": name, "arguments": converted_args}
            parsed_calls = self.parse_base_json(raw, tools)
            calls.extend(parsed_calls)

            for _ in range(len(parsed_calls)):
                self.current_tool_id += 1
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({"name": None, "arguments": {}})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")
                while len(self.tool_name_sent) <= self.current_tool_id:
                    self.tool_name_sent.append(True)

        return StreamingParseResult(normal_text=result.normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: Any
    ) -> StreamingParseResult:
        if not self._tool_indices:
            self._tool_indices = (
                self._get_tool_indices(tools) if isinstance(tools, list) else {}
            )

        result = self.sm.parse(new_text)
        calls: List[ToolCallItem] = []

        for event in result.events:
            if event.event_type == "tool_start":
                while len(self.prev_tool_call_arr) <= self.current_tool_id + 1:
                    self.prev_tool_call_arr.append({"name": None, "arguments": {}})
                while len(self.streamed_args_for_tool) <= self.current_tool_id + 1:
                    self.streamed_args_for_tool.append("")
                while len(self.tool_name_sent) <= self.current_tool_id + 1:
                    self.tool_name_sent.append(False)

                if (
                    self.current_tool_id == -1
                    or self.tool_name_sent[self.current_tool_id]
                ):
                    self.current_tool_id += 1
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({"name": None, "arguments": {}})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")
                    while len(self.tool_name_sent) <= self.current_tool_id:
                        self.tool_name_sent.append(False)

                    tool_name = event.name or self.sm.current_tool_name or None
                    self.prev_tool_call_arr[self.current_tool_id]["name"] = tool_name
                    if tool_name:
                        self.tool_name_sent[self.current_tool_id] = True
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=tool_name,
                            parameters="",
                        )
                    )
                else:
                    tool_name = event.name or self.sm.current_tool_name or None
                    if tool_name and not self.tool_name_sent[self.current_tool_id]:
                        self.prev_tool_call_arr[self.current_tool_id][
                            "name"
                        ] = tool_name
                        self.tool_name_sent[self.current_tool_id] = True
                        updated = False
                        for call in calls:
                            if (
                                call.tool_index == self.current_tool_id
                                and call.name is None
                            ):
                                call.name = tool_name
                                updated = True
                        if not updated:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=tool_name,
                                    parameters="",
                                )
                            )

            elif event.event_type == "parameter_start" and self.current_tool_id != -1:
                if not self.tool_name_sent[self.current_tool_id]:
                    tool_name = self.sm.current_tool_name
                    self.prev_tool_call_arr[self.current_tool_id]["name"] = tool_name
                    self.tool_name_sent[self.current_tool_id] = True
                    for call in calls:
                        if (
                            call.tool_index == self.current_tool_id
                            and call.name is None
                        ):
                            call.name = tool_name

                param_name = event.name or ""
                sent = self.streamed_args_for_tool[self.current_tool_id]
                is_first = not sent

                tool_name = str(
                    self.prev_tool_call_arr[self.current_tool_id].get("name")
                    or self.sm.current_tool_name
                    or ""
                )
                arg_type = get_argument_type(tool_name, param_name, tools) or "string"
                fragment = (
                    ("{" if is_first else ", ")
                    + json.dumps(param_name, ensure_ascii=False)
                    + ": "
                )
                if arg_type == "string":
                    fragment += '"'

                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, parameters=fragment)
                )
                self.streamed_args_for_tool[self.current_tool_id] += fragment

            elif event.event_type == "text_delta" and self.current_tool_id != -1:
                delta = event.text_delta or ""
                delta = html.unescape(delta)

                tool_name = str(
                    self.prev_tool_call_arr[self.current_tool_id].get("name")
                    or self.sm.current_tool_name
                    or ""
                )
                param_name = self.sm.current_parameter_name
                arg_type = get_argument_type(tool_name, param_name, tools) or "string"

                if arg_type == "string":
                    json_delta = json.dumps(delta, ensure_ascii=False)[1:-1]
                else:
                    json_delta = delta

                if json_delta:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id, parameters=json_delta
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += json_delta

            elif event.event_type == "parameter" and self.current_tool_id != -1:
                param_name = event.name or ""
                param_value = event.value or ""
                tool_name = str(
                    self.prev_tool_call_arr[self.current_tool_id].get("name")
                    or self.sm.current_tool_name
                    or ""
                )

                converted = self._parse_parameter(
                    tool_name, param_name, param_value, tools
                )
                self.prev_tool_call_arr[self.current_tool_id]["arguments"][
                    param_name
                ] = converted

                arg_type = get_argument_type(tool_name, param_name, tools) or "string"
                if arg_type == "string":
                    sent = self.streamed_args_for_tool[self.current_tool_id]
                    if not sent.endswith('"'):
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id, parameters='"'
                            )
                        )
                        self.streamed_args_for_tool[self.current_tool_id] += '"'
                else:
                    val_str = str(param_value)
                    sent = self.streamed_args_for_tool[self.current_tool_id]
                    prefix = f"{json.dumps(param_name, ensure_ascii=False)}: "
                    if prefix in sent:
                        already_sent = sent.split(prefix)[-1]
                        if val_str.startswith(already_sent):
                            remaining = val_str[len(already_sent) :]
                            if remaining:
                                calls.append(
                                    ToolCallItem(
                                        tool_index=self.current_tool_id,
                                        parameters=remaining,
                                    )
                                )
                                self.streamed_args_for_tool[
                                    self.current_tool_id
                                ] += remaining

            elif event.event_type == "tool_end" and self.current_tool_id != -1:
                if not self.tool_name_sent[self.current_tool_id]:
                    name = self.sm.current_tool_name
                    self.prev_tool_call_arr[self.current_tool_id]["name"] = name
                    self.tool_name_sent[self.current_tool_id] = True
                    for call in calls:
                        if (
                            call.tool_index == self.current_tool_id
                            and call.name is None
                        ):
                            call.name = name

                sent = self.streamed_args_for_tool[self.current_tool_id]
                if not sent:
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, parameters="{}")
                    )
                    self.streamed_args_for_tool[self.current_tool_id] = "{}"
                elif sent.startswith("{") and not sent.endswith("}"):
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, parameters="}")
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += "}"

                self.current_tool_id += 1

        return StreamingParseResult(normal_text=result.normal_text, calls=calls)

    def _process_arguments_streaming(
        self, func_name: str, new_text: str, tools: Any, **kwargs
    ) -> Any:
        if not hasattr(self, "_streamed_raw_length"):
            self._streamed_raw_length = 0
        increment = new_text[self._streamed_raw_length :]

        if self.current_tool_name_sent and (
            self.sm.state == UniversalToolParserState.IDLE
            or self.sm.state == UniversalToolParserState.TOOL_START
        ):
            self.sm.state = UniversalToolParserState.TOOL_NAME_END
            self.sm.current_tool_name = func_name
            if self.current_tool_id == -1 or self.tool_name_sent[self.current_tool_id]:
                self.current_tool_id += 1
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({"name": func_name, "arguments": {}})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")
                while len(self.tool_name_sent) <= self.current_tool_id:
                    self.tool_name_sent.append(True)

        if (
            self.current_tool_id != -1
            and not increment
            and not self.streamed_args_for_tool[self.current_tool_id]
        ):
            self.streamed_args_for_tool[self.current_tool_id] = "{"
            return SimpleNamespace(
                normal_text="",
                calls=[ToolCallItem(tool_index=self.current_tool_id, parameters="{")],
                parameters="{",
            )

        res = self.parse_streaming_increment(increment, tools)
        self._streamed_raw_length = len(new_text)
        all_params = "".join([c.parameters for c in res.calls if c.parameters])
        return SimpleNamespace(
            normal_text=res.normal_text, calls=res.calls, parameters=all_params
        )

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError
