import logging
from typing import List

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


class Qwen25Detector(BaseFormatDetector):
    """
    Detector for Qwen 2.5 and Qwen 3 model function call format.

    Format Structure:
    ```
    <tool_call>\n{"name":"func1", "arguments":{...}}\n</tool_call>\n<tool_call>\n{"name":"func2", "arguments":{...}}\n</tool_call>
    ```

    Key Components:
    - Tool Call Tags: `<tool_call>` and `</tool_call>` wrap each individual call
    - Function Call Object: JSON object with "name" and "arguments" fields

    Reference: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct?chat_template=default
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.sm = UniversalJsonStateMachine(
            JsonConfig(prefix="<tool_call>\n", suffix="\n</tool_call>")
        )
        self._tool_indices = {}
        self._reset_streaming_state()

    def _reset_streaming_state(self):
        self._buffer = ""
        self.current_tool_id = -1
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []
        self.sm.reset()

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Qwen 2.5 format tool call."""
        return "<tool_call>" in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        self.sm.reset()
        result = self.sm.parse(text)
        calls = []
        for tool in result.completed_tools:
            if isinstance(tool, dict):
                parsed_calls = self.parse_base_json(tool, tools)
                calls.extend(parsed_calls)
                self.current_tool_id += len(parsed_calls)
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
                    # Nameless tool call already started, nothing to update until JSON starts
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

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<tool_call>\n{"name":"' + name + '", "arguments":',
            end="}\n</tool_call>",
            trigger="<tool_call>",
        )
