import json, re
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional, Any

from json import JSONDecodeError, JSONDecoder
import partial_json_parser
from partial_json_parser.core.options import Allow


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


class ToolCallItem:
    """Used to store information about a single recognized function/tool call."""

    def __init__(
        self, tool_index: Optional[int], name: Optional[str], arguments: Optional[str]
    ):
        self.tool_index = tool_index
        self.name = name
        self.arguments = arguments


class StreamingParseResult:
    """Result of streaming incremental parsing."""

    def __init__(
        self, normal_text: str = "", calls: Optional[List[ToolCallItem]] = None
    ):
        self.normal_text = normal_text
        self.calls = calls or []


class BaseFormatDetector(ABC):
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    @abstractmethod
    def detect_and_parse(self, text: str, tools: List["Tool"]) -> List[ToolCallItem]:
        """
        Parses the text in one go. Returns success=True if the format matches, otherwise False.
        Note that leftover_text here represents "content that this parser will not consume further".
        """
        pass

    @abstractmethod
    def parse_streaming_increment(
        self, new_text: str, tools: List["Tool"]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing, internally maintains a buffer or state.
        Each time new_text is received, attempts to parse it. If a function call is matched, returns calls.
        It may also return some normal_text.
        """
        pass


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
        self._buffer = ""
        self.bot_token = "<|python_tag|>"

        # Indicates the index of the tool call currently being processed; -1 means not started or already finished
        self.current_tool_id: int = -1
        # Indicates whether the name of the current tool has already been output (it will only be output once for the same function call)
        self.current_tool_name_sent: bool = False
        # Stores the arguments (strings) already sent for each tool, for incremental sending
        self.streamed_args_for_tool: List[str] = []
        # Stores the list of all tool calls (JSON objects) parsed in the "previous" iteration
        self.prev_tool_call_arr: List[Dict] = []

        self.tool_call_regex = re.compile(r"\[{.*?}\]", re.DOTALL)

    def detect_and_parse(self, text: str, tools: List["Tool"]) -> List[ToolCallItem]:
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
        name, parameters = action["name"], json.dumps(
            action.get("parameters", action.get("arguments", {})),
            ensure_ascii=False,
        )
        tool_index = [tool.function.name for tool in tools].index(name)
        tool_call_item = ToolCallItem(
            tool_index=tool_index, name=name, arguments=parameters
        )
        calls = [tool_call_item]
        return calls

    def parse_streaming_increment(
        self, new_text: str, tools: List["Tool"]
    ) -> StreamingParseResult:
        self._buffer += new_text
        current_text = self._buffer
        if not (
            current_text.startswith(self.bot_token) or current_text.startswith("{")
        ):
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
                print("not enough tokens to parse into JSON yet")
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

                        print("got arguments diff: %s", argument_diff)
                        res = StreamingParseResult(
                            normal_text=None,
                            calls=[
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name="",
                                    arguments=argument_diff,
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
                                arguments="",
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
                                    arguments=argument_diff,
                                )
                            ],
                        )
                        self.streamed_args_for_tool[
                            self.current_tool_id
                        ] += argument_diff

            self.prev_tool_call_arr = tool_call_arr
            return res

        except Exception:
            print("Error trying to handle streaming tool call.")
            print("Skipping chunk as a result of tool streaming extraction " "error")
            return StreamingParseResult()


class MultiFormatParser:
    def __init__(self, detectors: List[BaseFormatDetector]):
        """
        :param detectors: A series of available Detector instances passed in
        """
        self.detectors = detectors

    def parse_once(self, text: str, tools: List["Tool"]):
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
            final_calls.extend(tool_call_list)

        # leftover_text is the normal text not consumed by any Detector
        return final_normal_text, final_calls

    def parse_streaming_increment(self, new_text: str, tools: List["Tool"]):
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


class StreamingJSONParser:
    """
    Maintains a buffer and attempts to parse complete JSON strings from it.
    Each time parse_increment() is called, new text is appended to the buffer and an attempt is made to parse.
    If one or more complete JSONs are present in the buffer, they are removed from the buffer and returned.
    Remaining (incomplete or non-JSON) text is retained in the buffer or returned as normal text.
    """

    def __init__(self):
        self.buffer = ""

    def parse_increment(self, new_text: str):
        """
        :param new_text: New text increment
        :return: (normal_text, list_of_parsed_json)

         normal_text:  The part confirmed this time that can be output as "normal text"
         list_of_parsed_json: Array of JSON objects successfully parsed from the buffer
        """
        self.buffer += new_text

        # First separate the 'successfully extracted JSON' from the buffer
        # The buffer may also contain some normal strings not inside JSON, so need to distinguish
        normal_text_segments = []
        parsed_json_objs = []

        # A simple idea:
        #   - Scan the entire buffer, continuously finding { ... } to extract JSON objects
        #   - If braces do not match => indicates incomplete JSON => need to wait for next increment
        #   - If JSON deserialization fails => either incomplete or parsing error (incorrect format)
        # Below is just a minimal implementation; for complex scenarios, refer to the more complete handling logic in vLLM

        idx = 0
        while idx < len(self.buffer):
            # Find the first '{' in the buffer
            start_brace = self.buffer.find("{", idx)
            if start_brace == -1:
                # No more '{' found, the rest is all normal text
                normal_text_segments.append(self.buffer[idx:])
                idx = len(self.buffer)
                break

            # First, add the normal text before start_brace to normal_text_segments
            if start_brace > idx:
                normal_text_segments.append(self.buffer[idx:start_brace])

            # Now try to find the matching '}' for this '{'
            # Using a simple brace count here. A more robust JSON decoding + exception catching can also be used.
            brace_count = 0
            end_pos = start_brace
            while end_pos < len(self.buffer):
                if self.buffer[end_pos] == "{":
                    brace_count += 1
                elif self.buffer[end_pos] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        # Found matching '}', indicating a complete JSON
                        break
                end_pos += 1

            if brace_count != 0:
                # Indicates that no matching '}' was found up to the end of the buffer -> incomplete -> wait for next increment
                # Then the normal text ends before start_brace
                idx = start_brace
                break

            # Extract this complete JSON string
            json_str = self.buffer[start_brace : end_pos + 1]
            try:
                json_obj = json.loads(json_str)
                # Successfully parsed -> record it
                parsed_json_objs.append(json_obj)
            except json.JSONDecodeError:
                # In some cases braces match, but parsing fails -> consider it incomplete or incorrectly formatted
                # If incomplete, still set idx back to start_brace, allowing subsequent increments to retry
                idx = start_brace
                break

            # After successfully parsing JSON, move the pointer to after end_pos to continue
            idx = end_pos + 1

        # After processing, content after idx remains in the buffer (not yet recognized)
        # normal_text is the concatenation of normal_text_segments
        # parsed_json_objs are the successfully recognized JSON objects

        normal_text = "".join(normal_text_segments)
        leftover = self.buffer[idx:]
        # Update self.buffer to the leftover that wasn't consumed this time
        self.buffer = leftover

        return normal_text, parsed_json_objs


class FunctionCallParser:
    """
    In streaming scenarios, each time new_text is received, it calls multi_format_parser.parse_streaming_increment
    and returns the resulting normal_text and calls to the upper layer (or SSE).
    """

    def __init__(self, tools: List["Tool"]):
        # Inject a set of Detectors here. To support Qwen25, InternLM2 in the future,
        # simply instantiate the corresponding Detector and add it to the list:
        self.multi_format_parser = MultiFormatParser(
            [
                Llama32Detector(),
                # Qwen25Detector(),
                # InternLM2Detector(),
                # ...
            ]
        )
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
