import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from json import JSONDecodeError, JSONDecoder
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

import partial_json_parser
from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow
from pydantic import BaseModel

from sglang.srt.openai_api.protocol import (
    StructuralTagResponseFormat,
    StructuresResponseFormat,
    Tool,
)

logger = logging.getLogger(__name__)

TOOLS_TAG_LIST = [
    "<|plugin|>",
    "<function=",
    "<tool_call>",
    "<|python_tag|>",
    "[TOOL_CALLS]",
]


class ToolCallItem(BaseModel):
    """Simple encapsulation of the parsed ToolCall result for easier usage in streaming contexts."""

    tool_index: int
    name: Optional[str] = None
    parameters: str  # JSON string


def _find_common_prefix(s1: str, s2: str) -> str:
    prefix = ""
    min_length = min(len(s1), len(s2))
    for i in range(0, min_length):
        if s1[i] == s2[i]:
            prefix += s1[i]
        else:
            break
    return prefix


def _partial_json_loads(input_str: str, flags: Allow) -> Tuple[Any, int]:
    try:
        return (partial_json_parser.loads(input_str, flags), len(input_str))
    except JSONDecodeError as e:
        if "Extra data" in e.msg:
            dec = JSONDecoder()
            return dec.raw_decode(input_str)
        raise


def _is_complete_json(input_str: str) -> bool:
    try:
        json.loads(input_str)
        return True
    except JSONDecodeError:
        return False


class StreamingParseResult:
    """Result of streaming incremental parsing."""

    def __init__(
        self, normal_text: str = "", calls: Optional[List[ToolCallItem]] = None
    ):
        self.normal_text = normal_text
        self.calls = calls or []


@dataclass
class StructureInfo:
    begin: str
    end: str
    trigger: str


_GetInfoFunc = Callable[[str], StructureInfo]
"""
helper alias of function
ususally it is a function that takes a name string and returns a StructureInfo object,
which can be used to construct a structural_tag object
"""


class BaseFormatDetector(ABC):
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    def __init__(self):
        # initialize properties used for state when parsing tool calls in
        self._buffer = ""
        # streaming mode
        self.prev_tool_call_arr: List[Dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: List[str] = (
            []
        )  # map what has been streamed for each tool so far to a list
        self.bot_token = ""
        self.eot_token = ""

    def parse_base_json(self, action: Any, tools: List[Tool]) -> List[ToolCallItem]:
        tool_indices = {
            tool.function.name: i for i, tool in enumerate(tools) if tool.function.name
        }
        if not isinstance(action, list):
            action = [action]

        results = []
        for act in action:
            name = act.get("name")
            if name and name in tool_indices:
                results.append(
                    ToolCallItem(
                        tool_index=tool_indices[name],
                        name=name,
                        parameters=json.dumps(
                            act.get("parameters") or act.get("arguments", {}),
                            ensure_ascii=False,
                        ),
                    )
                )
            else:
                logger.warning(f"Model attempted to call undefined function: {name}")

        return results

    @abstractmethod
    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Parses the text in one go. Returns success=True if the format matches, otherwise False.
        Note that leftover_text here represents "content that this parser will not consume further".
        """
        action = json.loads(text)
        return StreamingParseResult(calls=self.parse_base_json(action, tools))

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing with tool validation.
        """
        # Append new text to buffer
        self._buffer += new_text
        current_text = self._buffer
        if not (self.bot_token in current_text or current_text.startswith("{")):
            self._buffer = ""
            if self.eot_token in new_text:
                new_text = new_text.replace(self.eot_token, "")
            return StreamingParseResult(normal_text=new_text)

        # Build tool indices if not already built
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = {
                tool.function.name: i
                for i, tool in enumerate(tools)
                if tool.function and tool.function.name
            }

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            tool_call_arr = []
            is_complete = []
            try:
                start_idx = (
                    len(self.bot_token)
                    if current_text.startswith(self.bot_token)
                    else 0
                )
                while start_idx < len(current_text):
                    (obj, end_idx) = _partial_json_loads(
                        current_text[start_idx:], flags
                    )
                    is_complete.append(
                        _is_complete_json(current_text[start_idx : start_idx + end_idx])
                    )
                    start_idx += end_idx + len("; ")

                    # Validate tool name if present
                    if "name" in obj and obj["name"] not in self._tool_indices:
                        # Invalid tool name - reset state
                        self._buffer = ""
                        self.current_tool_id = -1
                        self.current_tool_name_sent = False
                        if self.streamed_args_for_tool:
                            self.streamed_args_for_tool.pop()
                        return StreamingParseResult()

                    # Handle parameters/arguments consistency
                    if "parameters" in obj:
                        assert (
                            "arguments" not in obj
                        ), "model generated both parameters and arguments"
                        obj["arguments"] = obj["parameters"]
                    tool_call_arr.append(obj)

            except MalformedJSON:
                return StreamingParseResult()

            if len(tool_call_arr) == 0:
                return StreamingParseResult()

            current_tool_call: Dict = (
                tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}
            )

            # Handle new tool in array
            if len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1:
                if self.current_tool_id >= 0:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments:
                        cur_args_json = json.dumps(cur_arguments)
                        sent = len(self.streamed_args_for_tool[self.current_tool_id])
                        argument_diff = cur_args_json[sent:]

                        res = StreamingParseResult(
                            calls=[
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name="",
                                    parameters=argument_diff,
                                )
                            ],
                        )
                        self.streamed_args_for_tool[
                            self.current_tool_id
                        ] += argument_diff
                    else:
                        res = StreamingParseResult()
                else:
                    res = StreamingParseResult()

                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                return res

            # Handle tool name
            elif not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name and function_name in self._tool_indices:
                    res = StreamingParseResult(
                        calls=[
                            ToolCallItem(
                                tool_index=self._tool_indices[function_name],
                                name=function_name,
                                parameters="",
                            )
                        ],
                    )
                    self.current_tool_name_sent = True
                else:
                    res = StreamingParseResult()

            # Handle streaming arguments
            else:
                cur_arguments = current_tool_call.get("arguments")
                res = StreamingParseResult()

                if cur_arguments:
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments)
                    prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                        "arguments"
                    )

                    argument_diff = None
                    if is_complete[self.current_tool_id]:
                        argument_diff = cur_args_json[sent:]
                        self._buffer = ""
                        self.prev_tool_call_arr[self.current_tool_id].clear()
                        self.current_tool_name_sent = False
                        self.streamed_args_for_tool[self.current_tool_id] = ""

                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments)
                        if cur_args_json != prev_args_json:
                            prefix = _find_common_prefix(prev_args_json, cur_args_json)
                            argument_diff = prefix[sent:]

                    if argument_diff is not None:
                        res = StreamingParseResult(
                            calls=[
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    parameters=argument_diff,
                                )
                            ],
                        )
                        if not is_complete[self.current_tool_id]:
                            self.streamed_args_for_tool[
                                self.current_tool_id
                            ] += argument_diff

            self.prev_tool_call_arr = tool_call_arr
            return res

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult()

    @abstractmethod
    def has_tool_call(self, text: str) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()


class Qwen25Detector(BaseFormatDetector):
    """
    Detector for Qwen 2.5 models.
    Assumes function call format:
      <tool_call>{"name":"xxx", "arguments":{...}}</tool_call>
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Qwen 2.5 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        pattern = rf"{self.bot_token}(.*?){self.eot_token}"
        match_result_list = re.findall(pattern, text, re.DOTALL)
        calls = []
        for match_result in match_result_list:
            match_result = json.loads(match_result)
            calls.extend(self.parse_base_json(match_result, tools))
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<tool_call>{"name":"' + name + '", "arguments":',
            end="}</tool_call>",
            trigger="<tool_call>",
        )


class MistralDetector(BaseFormatDetector):
    """
    Detector for Mistral models.
    Assumes function call format:
      <|action_start|><|plugin|>{"name":"xxx", "arguments":{...}}<|action_end|>
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.bot_token = "[TOOL_CALLS] ["
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Mistral format tool call."""
        return self.bot_token in text

    def _clean_text(self, text: str) -> str:
        """
        clean text to only leave ''[TOOL_CALLS] [{"name": xxx, "arguments": {xxx}}]'
        for example,
            text = '[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}]\n\nToday\'s weather in Boston is :{function call result} (in Fahrenheit)\n\nIf you prefer Celsius, please let me know.'
            return '[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}]'
        The key pattern is [TOOL_CALLS] [...]
        """
        find_results = re.findall(r"\[TOOL_CALLS\] \[.*?\]", text, re.DOTALL)
        if len(find_results) > 0:
            return find_results[0]
        else:
            return ""

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        text = self._clean_text(text)
        tool_content = text.replace("[TOOL_CALLS]", "").strip()
        raw_tool_calls = self.tool_call_regex.findall(tool_content)
        calls = []
        if len(raw_tool_calls) > 0:
            raw_tool_call = raw_tool_calls[0]
            function_call_arr = json.loads(raw_tool_call)
            for match_result in function_call_arr:
                calls.extend(self.parse_base_json(match_result, tools))
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='[TOOL_CALLS] [{"name":"' + name + '", "arguments":',
            end="}]",
            trigger="[TOOL_CALLS]",
        )


class Llama32Detector(BaseFormatDetector):
    """
    Detector for Llama 3.2 models.
    Assumes function call format:
      <|python_tag|>{"name":"xxx", "arguments":{...}}
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<|python_tag|>"

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Llama 3.2 format tool call."""
        # depending on the prompt format the Llama model may or may not
        # prefix the output with the <|python_tag|> token
        return "<|python_tag|>" in text or text.startswith("{")

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse function calls from text, handling multiple JSON objects."""
        if "<|python_tag|>" not in text and not text.startswith("{"):
            return StreamingParseResult(normal_text=text, calls=[])

        if "<|python_tag|>" in text:
            normal_text, action_text = text.split("<|python_tag|>")
        else:
            normal_text, action_text = "", text

        # Split by semicolon and process each part
        json_parts = [part.strip() for part in action_text.split(";") if part.strip()]
        all_actions = []
        for part in json_parts:
            try:
                # Parse each individual JSON object
                action = json.loads(part)
                all_actions.append(action)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON part: {part}")
                logger.warning(f"JSON parse error: {str(e)}")
                continue
        calls = []
        # Only process if we found valid JSON objects
        if all_actions:
            calls = self.parse_base_json(all_actions, tools)
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<|python_tag|>{"name":"' + name + '", "arguments":',
            end="}",
            trigger="<|python_tag|>",
        )


class MultiFormatParser:
    def __init__(self, detectors: List[BaseFormatDetector]):
        """
        :param detectors: A series of available Detector instances passed in
        """
        self.detectors = detectors

    def parse_once(
        self, text: str, tools: List[Tool]
    ) -> Tuple[str, list[ToolCallItem]]:
        """
        One-time parsing: Loop through detectors until there are no new matches or text is exhausted
        Return: (final_text, all_calls)
        - final_text: The remaining text after parsing that was not consumed by any Detector (can be treated as normal text)
        - all_calls: All calls parsed by the Detectors
        """
        final_calls = []
        final_normal_text = text
        for detector in self.detectors:
            parsed_result = detector.detect_and_parse(text, tools)
            tool_call_list = parsed_result.calls
            if len(tool_call_list) > 0:  # parsed successfully
                final_calls = tool_call_list
                final_normal_text = parsed_result.normal_text
                break

        # leftover_text is the normal text not consumed by any Detector
        return final_normal_text, final_calls

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> Tuple[str, list[ToolCallItem]]:
        """
        Streaming incremental parsing: Feed new_text to each detector's parse_streaming_increment
        and merge their produced normal_text/calls to return.
        (The logic here can be "priority-based" or "parallel parsing" based on your needs)
        """
        final_normal_text = ""
        final_calls = []

        for detector in self.detectors:
            sp_result = detector.parse_streaming_increment(new_text, tools)
            # Merge normal_text and calls
            # If one sp_result contains result call, this should be a successful parse
            # If one sp_result only contains normal_text, this can either be a successful
            # parse or it is not using the desired parsing tool.
            if sp_result.normal_text:
                final_normal_text = sp_result.normal_text
            if sp_result.calls:
                final_calls.extend(sp_result.calls)
                final_normal_text = sp_result.normal_text
                break

        return final_normal_text, final_calls


class FunctionCallParser:
    """
    In streaming scenarios, each time new_text is received, it calls multi_format_parser.parse_streaming_increment
    and returns the resulting normal_text and calls to the upper layer (or SSE).
    """

    ToolCallParserEnum: Dict[str, Type[BaseFormatDetector]] = {
        "llama3": Llama32Detector,
        "qwen25": Qwen25Detector,
        "mistral": MistralDetector,
    }

    def __init__(self, tools: List[Tool], tool_call_parser: str):
        detectors = []
        if tool_call_parser:
            detector_class = self.ToolCallParserEnum.get(tool_call_parser)
            if detector_class:
                detectors.append(detector_class())
            else:
                raise ValueError(f"Unsupported tool_call_parser: {tool_call_parser}")
        else:
            raise ValueError("Tool Call Parser Not Given!")

        self.multi_format_parser = MultiFormatParser(detectors)
        self.tools = tools

    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains a tool call in the format supported by this parser.
        This delegates to the detector's implementation.

        :param text: The text to check for tool calls
        :return: True if the text contains a tool call, False otherwise
        """
        # Check all detectors in the multi_format_parser
        for detector in self.multi_format_parser.detectors:
            if detector.has_tool_call(text):
                return True
        return False

    def parse_non_stream(self, full_text: str) -> Tuple[str, list[ToolCallItem]]:
        """
        Non-streaming call: one-time parsing
        """
        full_normal_text, calls = self.multi_format_parser.parse_once(
            full_text, self.tools
        )
        return full_normal_text, calls

    def parse_stream_chunk(self, chunk_text: str) -> Tuple[str, list[ToolCallItem]]:
        """
        Streaming call: incremental parsing
        """
        normal_text, calls = self.multi_format_parser.parse_streaming_increment(
            chunk_text, self.tools
        )
        return normal_text, calls

    def structure_infos(self) -> List[_GetInfoFunc]:
        """
        Returns a list of structure_info functions for each detector
        """
        return [
            detector.structure_info() for detector in self.multi_format_parser.detectors
        ]

    def get_structure_tag(self) -> StructuralTagResponseFormat:
        tool_structures: List[StructuresResponseFormat] = list()
        tool_trigger_set: Set[str] = set()

        for wrapper in self.structure_infos():
            for tool in self.tools:
                function = tool.function
                name = function.name
                assert name is not None
                info = wrapper(name)

                # accept all if not strict, otherwise only accept the schema
                schema = function.parameters if function.strict else {}

                tool_structures.append(
                    StructuresResponseFormat(
                        begin=info.begin,
                        schema=schema,  # type: ignore
                        end=info.end,
                    )
                )
                tool_trigger_set.add(info.trigger)

        return StructuralTagResponseFormat(
            type="structural_tag",
            structures=tool_structures,
            triggers=list(tool_trigger_set),
        )
