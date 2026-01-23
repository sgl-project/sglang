import json
import logging
from typing import Any, List, cast

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.minimax_state_machine import MinimaxStateMachine
from sglang.srt.function_call.utils import infer_type_from_json_schema

logger = logging.getLogger(__name__)


class MinimaxM2Detector(BaseFormatDetector):
    def __init__(self):
        super().__init__()
        self.tool_call_start_token: str = "<minimax:tool_call>"
        self.sm = MinimaxStateMachine()
        self._tool_indices = {}
        self._reset_streaming_state()

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def _reset_streaming_state(self):
        self._buffer = ""
        self.current_tool_id = -1
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []
        self.sm.reset()

    def _convert_param_value(self, value: str, param_type: str) -> Any:
        if param_type == "string":
            if value == "null":
                return None
            return value
        elif param_type == "integer":
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        elif param_type == "number":
            try:
                val = float(value)
                return val if val != int(val) else int(val)
            except (ValueError, TypeError):
                return value
        elif param_type == "boolean":
            lower_val = value.lower().strip()
            if lower_val in ["true", "1", "yes", "on"]:
                return True
            elif lower_val in ["false", "0", "no", "off"]:
                return False
            return value
        elif param_type in ("object", "array"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _parse_parameter(
        self, fname: str, pname: str, pval: str, tools: List[Tool]
    ) -> Any:
        param_schema = None
        for tool in tools:
            if tool.function.name == fname and tool.function.parameters is not None:
                params = cast(Any, tool.function.parameters)
                if isinstance(params, dict):
                    properties = params.get("properties")
                    if isinstance(properties, dict):
                        param_schema = properties.get(pname)
                        break

        inferred_type = (
            infer_type_from_json_schema(param_schema) if param_schema else "string"
        )
        return self._convert_param_value(pval, inferred_type or "string")

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
            calls.extend(self.parse_base_json(raw, tools))

        return StreamingParseResult(normal_text=result.normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        if not self._tool_indices:
            self._tool_indices = self._get_tool_indices(tools)

        result = self.sm.parse(new_text)
        calls: List[ToolCallItem] = []

        for event in result.events:
            if event.event_type == "tool_start":
                if event.name in self._tool_indices:
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                    else:
                        self.current_tool_id += 1

                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")

                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": event.name,
                        "arguments": {},
                    }
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=event.name,
                            parameters="",
                        )
                    )

            elif event.event_type == "parameter_start" and self.current_tool_id != -1:
                param_name = event.name or ""
                sent = self.streamed_args_for_tool[self.current_tool_id]
                is_first = not sent

                fragment = (
                    ("{" if is_first else ", ")
                    + json.dumps(param_name, ensure_ascii=False)
                    + ": "
                )
                # Minimax parameters are strings
                fragment += '"'

                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, parameters=fragment)
                )
                self.streamed_args_for_tool[self.current_tool_id] += fragment

            elif event.event_type == "text_delta" and self.current_tool_id != -1:
                delta = event.text_delta or ""
                # Escape for JSON string
                json_delta = json.dumps(delta, ensure_ascii=False)[1:-1]
                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, parameters=json_delta)
                )
                self.streamed_args_for_tool[self.current_tool_id] += json_delta

            elif event.event_type == "parameter" and self.current_tool_id != -1:
                param_name = event.name or ""
                param_value = event.value or ""
                converted_value = self._parse_parameter(
                    self.sm.current_tool_name, param_name, param_value, tools
                )

                self.prev_tool_call_arr[self.current_tool_id]["arguments"][
                    param_name
                ] = converted_value

                sent = self.streamed_args_for_tool[self.current_tool_id]
                if not sent.endswith('"'):
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, parameters='"')
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += '"'

            elif event.event_type == "tool_end" and self.current_tool_id != -1:
                sent = self.streamed_args_for_tool[self.current_tool_id]
                if sent and sent.startswith("{") and not sent.endswith("}"):
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, parameters="}")
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += "}"

        return StreamingParseResult(normal_text=result.normal_text, calls=calls)

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError
