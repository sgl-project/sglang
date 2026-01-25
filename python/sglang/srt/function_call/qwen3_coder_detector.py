import json
import logging
from typing import Any, List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.qwen3_coder_state_machine import Qwen3CoderStateMachine

logger = logging.getLogger(__name__)


class Qwen3CoderDetector(BaseFormatDetector):
    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self.sm = Qwen3CoderStateMachine()
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
        self.sm.reset()

    def _parse_parameter(self, fname: str, pname: str, pval: str, tools: Any) -> Any:
        return pval

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
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        if not self._tool_indices:
            self._tool_indices = self._get_tool_indices(tools)

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

                fragment = (
                    ("{" if is_first else ", ")
                    + json.dumps(param_name, ensure_ascii=False)
                    + ": "
                )
                fragment += '"'

                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, parameters=fragment)
                )
                self.streamed_args_for_tool[self.current_tool_id] += fragment

            elif event.event_type == "text_delta" and self.current_tool_id != -1:
                delta = event.text_delta or ""
                json_delta = json.dumps(delta, ensure_ascii=False)[1:-1]
                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, parameters=json_delta)
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

                converted_value = self._parse_parameter(
                    tool_name, param_name, param_value, tools
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
                if sent and sent.startswith("{") and not sent.endswith("}"):
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, parameters="}")
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += "}"

                self.current_tool_id += 1

        return StreamingParseResult(normal_text=result.normal_text, calls=calls)

    def supports_structural_tag(self) -> bool:
        return True

    def structure_info(self) -> _GetInfoFunc:
        # TODO: Implement this based on the format
        raise NotImplementedError()
