import json
import logging
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.mistral_state_machine import MistralStateMachine

logger = logging.getLogger(__name__)


class MistralDetector(BaseFormatDetector):
    def __init__(self):
        super().__init__()
        self.sm_array = MistralStateMachine(is_compact=False)
        self.sm_compact = MistralStateMachine(is_compact=True)
        self.sm = None
        self._tool_indices = {}
        self._reset_streaming_state()

    def _reset_streaming_state(self):
        self._buffer = ""
        self.current_tool_id = -1
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []
        self.tool_name_sent = []
        self.sm_array.reset()
        self.sm_compact.reset()

    def has_tool_call(self, text: str) -> bool:
        return "[TOOL_CALLS]" in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if "[TOOL_CALLS] [" in text:
            self.sm = self.sm_array
        else:
            self.sm = self.sm_compact

        self.sm.reset()
        result = self.sm.parse(text)
        calls = []
        for tool in result.completed_tools:
            if isinstance(tool, dict):
                parsed_calls = self.parse_base_json(tool, tools)
                calls.extend(parsed_calls)
                for _ in range(len(parsed_calls)):
                    self.current_tool_id += 1
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({"name": None, "arguments": {}})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")
                    while len(self.tool_name_sent) <= self.current_tool_id:
                        self.tool_name_sent.append(True)

        return StreamingParseResult(normal_text=result.normal_text.strip(), calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        if not self._tool_indices:
            self._tool_indices = self._get_tool_indices(tools)

        self._buffer += new_text

        # Determine format if not yet known
        if self.sm != self.sm_array and self.sm != self.sm_compact:
            if "[TOOL_CALLS] [" in self._buffer:
                self.sm = self.sm_array
                self.sm.reset()
                result = self.sm.parse(self._buffer)
            elif "[TOOL_CALLS]" in self._buffer:
                idx = self._buffer.find("[TOOL_CALLS]")
                rest = self._buffer[idx + len("[TOOL_CALLS]") :].lstrip()
                if rest:
                    if rest.startswith("["):
                        return StreamingParseResult()
                    else:
                        self.sm = self.sm_compact
                        self.sm.reset()
                        result = self.sm.parse(self._buffer)
                else:
                    return StreamingParseResult()
            else:
                return StreamingParseResult()
        else:
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
                    # Start a new tool
                    self.current_tool_id += 1
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({"name": None, "arguments": {}})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")
                    while len(self.tool_name_sent) <= self.current_tool_id:
                        self.tool_name_sent.append(False)

                    self.prev_tool_call_arr[self.current_tool_id]["name"] = event.name
                    if event.name:
                        self.tool_name_sent[self.current_tool_id] = True
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=event.name,
                            parameters="",
                        )
                    )
                else:
                    # Update name of current nameless tool call
                    if event.name and not self.tool_name_sent[self.current_tool_id]:
                        self.prev_tool_call_arr[self.current_tool_id][
                            "name"
                        ] = event.name
                        self.tool_name_sent[self.current_tool_id] = True
                        found = False
                        for call in calls:
                            if (
                                call.tool_index == self.current_tool_id
                                and call.name is None
                            ):
                                call.name = event.name
                                found = True
                        if not found:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=event.name,
                                    parameters="",
                                )
                            )

            elif event.event_type == "text_delta" and self.current_tool_id != -1:
                delta = event.text_delta or ""
                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, parameters=delta)
                )
                self.streamed_args_for_tool[self.current_tool_id] += delta

            elif event.event_type == "tool_end" and self.current_tool_id != -1:
                if not self.tool_name_sent[self.current_tool_id]:
                    if result.completed_tools:
                        tool = result.completed_tools[-1]
                        name = tool.get("name")
                        self.prev_tool_call_arr[self.current_tool_id]["name"] = name
                        self.tool_name_sent[self.current_tool_id] = True
                        for call in calls:
                            if (
                                call.tool_index == self.current_tool_id
                                and call.name is None
                            ):
                                call.name = name

                if self.sm == self.sm_array:
                    if result.completed_tools:
                        tool = result.completed_tools[-1]
                        name = tool.get("name")
                        self.prev_tool_call_arr[self.current_tool_id]["arguments"] = (
                            tool.get("arguments", {})
                        )

                        sent = self.streamed_args_for_tool[self.current_tool_id]
                        if not sent:
                            params = json.dumps(
                                tool.get("arguments", {}), ensure_ascii=False
                            )
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=name,
                                    parameters=params,
                                )
                            )
                            self.streamed_args_for_tool[self.current_tool_id] = params
                        else:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=name,
                                    parameters="",
                                )
                            )
                    # Advance for potential next tool in same string
                    self.current_tool_id += 1
                else:  # Compact format
                    if result.completed_tools:
                        tool = result.completed_tools[-1]
                        self.prev_tool_call_arr[self.current_tool_id]["arguments"] = (
                            tool.get("arguments", {})
                        )

                        sent = self.streamed_args_for_tool[self.current_tool_id]
                        full_params = json.dumps(
                            tool.get("arguments", {}), ensure_ascii=False
                        )
                        if not sent:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    parameters=full_params,
                                )
                            )
                            self.streamed_args_for_tool[self.current_tool_id] = (
                                full_params
                            )
                        elif full_params.startswith(sent) and full_params != sent:
                            diff = full_params[len(sent) :]
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id, parameters=diff
                                )
                            )
                            self.streamed_args_for_tool[self.current_tool_id] = (
                                full_params
                            )

                    self.current_tool_id += 1

        return StreamingParseResult(normal_text=result.normal_text, calls=calls)

    def _start_new_tool(self, name: str, calls: List[ToolCallItem]):
        if self.current_tool_id == -1:
            self.current_tool_id = 0
        else:
            self.current_tool_id += 1

        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")

        self.prev_tool_call_arr[self.current_tool_id] = {"name": name, "arguments": {}}
        calls.append(
            ToolCallItem(tool_index=self.current_tool_id, name=name, parameters="")
        )

    def supports_structural_tag(self) -> bool:
        return True

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()
