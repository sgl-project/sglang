import json
import re
from abc import ABC, abstractmethod
from json import JSONDecodeError, JSONDecoder
from typing import Any, Dict, List, Optional, Tuple

import partial_json_parser
from partial_json_parser.core.options import Allow
from pydantic import BaseModel, Field

TOOLS_TAG_LIST = [
    "<|plugin|>",
    "<function=",
    "<tool_call>",
    "<|python_tag|>",
    "[TOOL_CALLS]",
]


class Function(BaseModel):
    """Function Tool Template."""

    description: Optional[str] = Field(default=None, examples=[None])
    name: Optional[str] = None
    parameters: Optional[object] = None


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


class BaseFormatDetector:
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

    def parse_base_json(self, action: Dict, tools: List[Function]):
        name, parameters = action["name"], json.dumps(
            action.get("parameters", action.get("arguments", {})),
            ensure_ascii=False,
        )
        tool_index = [tool.function.name for tool in tools].index(name)
        tool_call_item = ToolCallItem(
            tool_index=tool_index, name=name, parameters=parameters
        )
        calls = [tool_call_item]
        return calls

    def detect_and_parse(self, text: str, tools: List[Function]) -> List[ToolCallItem]:
        """
        Parses the text in one go. Returns success=True if the format matches, otherwise False.
        Note that leftover_text here represents "content that this parser will not consume further".
        """
        action = json.loads(text)
        return self.parse_base_json(action, tools)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Function]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing, referencing the logic of Llama32Detector.
        We partially parse JSON within <tool_call>...</tool_call>, and handle
        incremental argument output.
        """
        # Append new text to buffer
        self._buffer += new_text
        current_text = self._buffer
        if not (self.bot_token in current_text or current_text.startswith("{")):
            self._buffer = ""
            if self.eot_token in new_text:
                new_text = new_text.replace(self.eot_token, "")
            return StreamingParseResult(normal_text=new_text)

        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending
        # an incomplete string since OpenAI only ever (as far as I have
        # seen) allows sending the entire tool/ function name at once.
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            tool_call_arr = []
            is_complete = []
            try:
                # depending on the prompt format the Llama model may or may not
                # prefix the output with the <|python_tag|> token
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
                    # depending on the prompt Llama can use
                    # either arguments or parameters
                    if "parameters" in obj:
                        assert (
                            "arguments" not in obj
                        ), "model generated both parameters and arguments"
                        obj["arguments"] = obj["parameters"]
                    tool_call_arr.append(obj)

            except partial_json_parser.core.exceptions.MalformedJSON:
                # not enough tokens to parse into JSON yet
                return StreamingParseResult()

            # select as the current tool call the one we're on the state at
            current_tool_call: Dict = (
                tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}
            )

            # case -- if no tokens have been streamed for the tool, e.g.
            #   only the array brackets, stream nothing
            if len(tool_call_arr) == 0:
                return StreamingParseResult()

            # case: we are starting a new tool in the array
            #   -> array has > 0 length AND length has moved past cursor
            elif (
                len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1
            ):

                # if we're moving on to a new call, first make sure we
                # haven't missed anything in the previous one that was
                # auto-generated due to JSON completions, but wasn't
                # streamed to the client yet.
                if self.current_tool_id >= 0:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments:
                        cur_args_json = json.dumps(cur_arguments)
                        sent = len(self.streamed_args_for_tool[self.current_tool_id])
                        argument_diff = cur_args_json[sent:]

                        res = StreamingParseResult(
                            normal_text=None,
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
                # re-set stuff pertaining to progress in the current tool
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                print("starting on new tool %d", self.current_tool_id)
                return res

            # if the current tool name hasn't been sent, send if available
            # - otherwise send nothing
            elif not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:
                    res = StreamingParseResult(
                        normal_text=None,
                        calls=[
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=function_name,
                                parameters="",
                            )
                        ],
                    )
                    self.current_tool_name_sent = True
                else:
                    res = StreamingParseResult()

            # now we know we're on the same tool call and we're streaming
            # arguments
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
                        self.current_tool_name_sent: bool = False
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
                                    name="",
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
            print(e)
            # Skipping chunk as a result of tool streaming extraction error
            return StreamingParseResult()


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

    def detect_and_parse(self, text: str, tools: List[Function]) -> List[ToolCallItem]:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        if "<tool_call>" not in text:
            return []
        pattern = r"<tool_call>(.*?)</tool_call>"
        match_result_list = re.findall(pattern, text, re.DOTALL)
        calls = []
        for match_result in match_result_list:
            match_result = json.loads(match_result)
            calls.extend(self.parse_base_json(match_result, tools))
        return calls


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

    def detect_and_parse(self, text: str, tools: List[Function]) -> List[ToolCallItem]:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        text = self._clean_text(text)
        tool_content = text.replace("[TOOL_CALLS]", "").strip()
        raw_tool_calls = self.tool_call_regex.findall(tool_content)
        calls = []
        if len(raw_tool_calls) > 0:
            raw_tool_call = raw_tool_calls[0]
            function_call_arr = json.loads(raw_tool_call)
            for match_result in function_call_arr:
                calls.extend(self.parse_base_json(match_result, tools))
        return calls


class Llama32Detector(BaseFormatDetector):
    """
    Detector for Llama 3.2 models.
    Assumes function call format:
      <|python_tag|>{"name":"xxx", "arguments":{...}}
    Does not require a closing tag "</python_tag|>",
    relies on json.loads(...) success to determine if JSON is complete.
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.bot_token = "<|python_tag|>"

    def detect_and_parse(self, text: str, tools: List[Function]) -> List[ToolCallItem]:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """

        if "<|python_tag|>" not in text:
            return []
        _, action = text.split("<|python_tag|>")
        action = json.loads(action)
        return self.parse_base_json(action, tools)


class MultiFormatParser:
    def __init__(self, detectors: List[BaseFormatDetector]):
        """
        :param detectors: A series of available Detector instances passed in
        """
        self.detectors = detectors

    def parse_once(self, text: str, tools: List[Function]):
        """
        One-time parsing: Loop through detectors until there are no new matches or text is exhausted
        Return: (final_text, all_calls)
        - final_text: The remaining text after parsing that was not consumed by any Detector (can be treated as normal text)
        - all_calls: All calls parsed by the Detectors
        """
        final_calls = []
        final_normal_text = text
        for detector in self.detectors:
            tool_call_list = detector.detect_and_parse(text, tools)
            if len(tool_call_list) > 0:  # parsed successfully
                final_calls = tool_call_list
                break

        # leftover_text is the normal text not consumed by any Detector
        return final_normal_text, final_calls

    def parse_streaming_increment(self, new_text: str, tools: List[Function]):
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

    ToolCallParserEnum: Dict[str, BaseFormatDetector] = {
        "llama3": Llama32Detector,
        "qwen25": Qwen25Detector,
        "mistral": MistralDetector,
    }

    def __init__(self, tools: List[Function], tool_call_parser: str = None):
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

    def parse_non_stream(self, full_text: str):
        """
        Non-streaming call: one-time parsing
        """
        full_normal_text, calls = self.multi_format_parser.parse_once(
            full_text, self.tools
        )
        return full_normal_text, calls

    def parse_stream_chunk(self, chunk_text: str):
        """
        Streaming call: incremental parsing
        """
        normal_text, calls = self.multi_format_parser.parse_streaming_increment(
            chunk_text, self.tools
        )
        return normal_text, calls
