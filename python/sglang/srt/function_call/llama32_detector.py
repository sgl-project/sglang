import logging
import re
from types import SimpleNamespace
from typing import Any, List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.state_machine import JsonConfig
from sglang.srt.function_call.universal_state_machine import UniversalJsonStateMachine

logger = logging.getLogger(__name__)


class Llama32Detector(BaseFormatDetector):
    """
    Detector for Llama 3.2 models with json tool call format.

    Format Structure:
    ```
    <|python_tag|>{"name":"xxx", "arguments":{...}}
    ```
    """

    def __init__(self):
        super().__init__()
        self.sm = UniversalJsonStateMachine(JsonConfig(prefix="<|python_tag|>"))
        self._tool_indices = {}
        self._reset_streaming_state()

    def _reset_streaming_state(self):
        self._buffer = ""
        self.current_tool_id = -1
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []
        self._streamed_raw_length = 0
        self.sm.reset()

    def has_tool_call(self, text: str) -> bool:
        return "<|python_tag|>" in text or text.startswith("{")

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        self.sm.reset()
        # Fallback to base class regex if it looks like a Python dict but not valid JSON
        if "'" in text and ":" in text and "{" in text:
            processed_text = re.sub(r"'([^']*)':", r'"\1":', text)
            processed_text = re.sub(r":\s*'([^']*)'", r': "\1"', processed_text)
            text = processed_text

        result = self.sm.parse(text)
        calls = []
        for tool in result.completed_tools:
            if isinstance(tool, dict):
                parsed_calls = self.parse_base_json(tool, tools)
                calls.extend(parsed_calls)
                for _ in range(len(parsed_calls)):
                    self.current_tool_id += 1
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")
        return StreamingParseResult(normal_text=result.normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        if not self._tool_indices:
            self._tool_indices = self._get_tool_indices(tools)

        processed_text = re.sub(r"'([^']*)':", r'"\1":', new_text)
        processed_text = re.sub(r":\s*'([^']*)'", r': "\1"', processed_text)

        result = self.sm.parse(processed_text)
        calls: List[ToolCallItem] = []

        for event in result.events:
            if event.event_type == "tool_start":
                while len(self.prev_tool_call_arr) <= self.current_tool_id + 1:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id + 1:
                    self.streamed_args_for_tool.append("")

                if self.current_tool_id == -1 or self.prev_tool_call_arr[
                    self.current_tool_id
                ].get("name"):
                    self.current_tool_id += 1
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id, name=None, parameters=""
                        )
                    )
                else:
                    pass

            elif event.event_type == "text_delta" and self.current_tool_id != -1:
                delta = event.text_delta or ""
                self.streamed_args_for_tool[self.current_tool_id] += delta

                if not self.prev_tool_call_arr[self.current_tool_id].get("name"):
                    try:
                        from partial_json_parser.core.options import Allow

                        from sglang.srt.function_call.utils import _partial_json_loads

                        parsed, _ = _partial_json_loads(
                            self.streamed_args_for_tool[self.current_tool_id], Allow.ALL
                        )
                        if isinstance(parsed, dict) and "name" in parsed:
                            name = parsed["name"]
                            self.prev_tool_call_arr[self.current_tool_id]["name"] = name
                            for call in calls:
                                if (
                                    call.tool_index == self.current_tool_id
                                    and call.name is None
                                ):
                                    call.name = name
                    except:
                        pass

                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, parameters=delta)
                )

            elif event.event_type == "tool_end" and self.current_tool_id != -1:
                if result.completed_tools:
                    tool = result.completed_tools[-1]
                    name = tool.get("name")
                    self.prev_tool_call_arr[self.current_tool_id]["name"] = name
                    self.prev_tool_call_arr[self.current_tool_id]["arguments"] = (
                        tool.get("arguments", {})
                    )
                    for call in calls:
                        if (
                            call.tool_index == self.current_tool_id
                            and call.name is None
                        ):
                            call.name = name

        return StreamingParseResult(normal_text=result.normal_text, calls=calls)

    def _process_arguments_streaming(
        self, func_name: str, new_text: str, tools: Any, **kwargs
    ) -> Any:
        if not hasattr(self, "_streamed_raw_length"):
            self._streamed_raw_length = 0
        increment = new_text[self._streamed_raw_length :]
        res = self.parse_streaming_increment(increment, tools)
        self._streamed_raw_length = len(new_text)
        all_params = "".join([c.parameters for c in res.calls if c.parameters])
        return SimpleNamespace(
            normal_text=res.normal_text, calls=res.calls, parameters=all_params
        )

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<|python_tag|>{"name":"' + name + '", "arguments":',
            end="}",
            trigger="<|python_tag|>",
        )

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<|python_tag|>{"name":"' + name + '", "arguments":',
            end="}",
            trigger="<|python_tag|>",
        )
