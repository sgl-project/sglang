import json, re
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional, Any


class ToolCallItem:
    """Used to store information about a single recognized function/tool call."""

    def __init__(self, tool_index: int, name: str, parameters: str):
        self.tool_index = tool_index
        self.name = name
        self.parameters = parameters


class ParseResult:
    """Result of a single parse."""

    def __init__(
        self,
        success: bool,
        consumed_text: str = "",
        leftover_text: str = "",
        calls: Optional[List[ToolCallItem]] = None,
    ):
        """
        :param success: Whether parsing was successful
        :param consumed_text: Text consumed during parsing
        :param leftover_text: Text not yet consumed, left for the next parser or subsequent steps
        :param calls: List of ToolCallItem parsed
        """
        self.success = success
        self.consumed_text = consumed_text
        self.leftover_text = leftover_text
        self.calls = calls or []


class StreamingParseResult:
    """Result of streaming incremental parsing."""

    def __init__(self, normal_text: str, calls: Optional[List[ToolCallItem]] = None):
        """
        :param normal_text: Content that can be output as normal text this time (or an empty string)
        :param calls: Newly parsed ToolCallItems (or an empty list)
        """
        self.normal_text = normal_text
        self.calls = calls or []


class BaseFormatDetector(ABC):
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    @abstractmethod
    def detect_and_parse(self, text: str, tools: List["Tool"]) -> ParseResult:
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


class InternLM2Detector(BaseFormatDetector):
    """
    Detects formats like <|plugin|>...<|action_start|><|plugin|> ... <|action_end|>
    and parses them into ToolCallItem
    """

    def detect_and_parse(self, text: str, tools: List["Tool"]) -> ParseResult:
        if "<|plugin|>" not in text:
            return ParseResult(False)
        if "<|action_start|><|plugin|>" not in text or "<|action_end|>" not in text:
            return ParseResult(False)

        # Simplified approach: assuming it contains 'action' JSON
        try:
            left_part, action_part = text.split("<|action_start|><|plugin|>")
            action_part = action_part.split("<|action_end|>")[0]
            # action_part contains `{...}` JSON
            # Some implementations may need to first extract action_part[action_part.find("{") : ]
            action_part = action_part[action_part.find("{") :]
            action_json = json.loads(action_part)
            name = action_json["name"]
            # Could be "arguments" or "parameters"
            params = action_json.get("parameters", action_json.get("arguments", {}))
            # Assemble call
            # First find tool_index
            tool_index = -1
            tool_names = [t.function.name for t in tools]  # Depending on the situation
            if name in tool_names:
                tool_index = tool_names.index(name)
            calls = [
                ToolCallItem(tool_index, name, json.dumps(params, ensure_ascii=False))
            ]

            # Separate text_before and text_after
            text_after = ""
            # In this example, treat left_part as text_before
            text_before = left_part

            return ParseResult(
                success=True,
                consumed_text=text_before,
                leftover_text=text_after,
                calls=calls,
            )
        except Exception as e:
            return ParseResult(False)


class Llama31Detector(BaseFormatDetector):
    """For formats like <function=...>...</function>."""

    def detect_and_parse(self, text: str, tools: List["Tool"]) -> ParseResult:
        if "<function=" not in text or "</function>" not in text:
            return ParseResult(False)
        try:
            action, after = text.split("</function>", 1)
            # action part is like: "...<function=name>{...}"
            # First separate the name
            name_part = action.split("<function=")[1]
            name = name_part.split(">{")[0]  # Get the part before the curly brace

            # Then parse JSON
            json_part = action[action.find("{") :]
            params = json_part.strip()
            # May need to load first here
            # Depending on whether you want it structured or preserved as a string
            # For example:
            try:
                dict_params = json.loads(params)
            except:
                dict_params = {}

            tool_index = -1
            tool_names = [t.function.name for t in tools]
            if name in tool_names:
                tool_index = tool_names.index(name)

            calls = [
                ToolCallItem(
                    tool_index, name, json.dumps(dict_params, ensure_ascii=False)
                )
            ]

            # Assume that before <function=xxx> ... </function> is text_before, and after is text_after
            before_part, _ = text.split("<function=")
            text_before = before_part
            text_after = after

            return ParseResult(
                success=True,
                consumed_text=text_before,
                leftover_text=text_after,
                calls=calls,
            )

        except Exception as e:
            return ParseResult(False)


class Qwen25Detector(BaseFormatDetector):
    """Detects the format <tool_call>...</tool_call>."""

    def detect_and_parse(self, text: str, tools: List["Tool"]) -> ParseResult:
        if "<tool_call>" not in text or "</tool_call>" not in text:
            return ParseResult(False)
        pattern = r"<tool_call>(.*?)</tool_call>"
        match_result_list = re.findall(pattern, text, re.DOTALL)
        if not match_result_list:
            return ParseResult(False)

        calls = []
        try:
            for match_result in match_result_list:
                action = json.loads(match_result)
                name = action["name"]
                params = action["arguments"]
                tool_index = -1
                tool_names = [t.function.name for t in tools]
                if name in tool_names:
                    tool_index = tool_names.index(name)
                calls.append(
                    ToolCallItem(
                        tool_index=tool_index,
                        name=name,
                        parameters=json.dumps(params, ensure_ascii=False),
                    )
                )
            # Split out text_before and text_after
            start_idx = text.find("<tool_call>")
            text_before = text[:start_idx]
            # Assume only the first <tool_call>... is parsed, and the rest is put into text_after (can also loop to handle)
            end_idx = text.rfind("</tool_call>") + len("</tool_call>")
            text_after = text[end_idx:]
            return ParseResult(
                success=True,
                consumed_text=text_before,
                leftover_text=text_after,
                calls=calls,
            )
        except:
            return ParseResult(False)


class Llama32Detector(BaseFormatDetector):
    """
    For Llama3.2: Assumes its function call format is:
      <|python_tag|>{"name":"xxx", "arguments":{...}}
    No longer requires a closing tag "</python_tag|>",
    but instead relies on whether `json.loads(...)` succeeds to determine if a complete JSON is obtained.
    """

    def __init__(self):
        self._buffer = ""
        self._tag_open = "<|python_tag|>"

    def detect_and_parse(self, text: str, tools: List["Tool"]) -> ParseResult:
        """
        One-time parsing: In the given text, if <|python_tag|> is detected, immediately attempts to parse JSON after it.
        Returns calls if successful; otherwise returns success=False.
        """
        open_idx = text.find(self._tag_open)
        if open_idx == -1:
            return ParseResult(success=False)

        # Extract the part before <|python_tag|> as normal content (not necessarily required here)
        text_before = text[:open_idx]

        # Take all content after <|python_tag|>
        start_json = open_idx + len(self._tag_open)
        json_part = text[start_json:]

        try:
            json_obj = json.loads(json_part)
            # Parsing succeeded => construct call
            name = json_obj.get("name", "unknown_func")
            params_dict = json_obj.get("arguments") or json_obj.get("parameters") or {}
            params_str = json.dumps(params_dict, ensure_ascii=False)

            # Find tool_index
            tool_index = -1
            tool_names = [t.function.name for t in tools]
            if name in tool_names:
                tool_index = tool_names.index(name)

            call = ToolCallItem(tool_index, name, params_str)

            # consumed_text = entire <|python_tag|> + JSON
            consumed_text = text[: start_json + len(json_part)]
            leftover_text = ""  # Assume one-time parsing only supports parsing once
            return ParseResult(
                success=True,
                consumed_text=consumed_text,
                leftover_text=leftover_text,
                calls=[call],
            )
        except Exception:
            return ParseResult(success=False)

    def parse_streaming_increment(
        self, new_text: str, tools: List["Tool"]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing: Whenever <|python_tag|> is seen, attempt to json.loads the content after it.
        If it fails, assume the increment is insufficient (leave for next time); if successful, produce a ToolCallItem.
        Do not look for any "closing" tags.
        """
        self._buffer += new_text

        normal_text = ""
        calls = []
        idx = 0
        while True:
            # Find <|python_tag|>
            open_pos = self._buffer.find(self._tag_open, idx)
            if open_pos == -1:
                # No more <|python_tag|> => treat the rest as normal text
                normal_text += self._buffer[idx:]
                idx = len(self._buffer)
                break

            # First, treat the content before <|python_tag|> as normal text
            normal_text += self._buffer[idx:open_pos]
            # Next search starts after <|python_tag|>
            json_start = open_pos + len(self._tag_open)

            # Attempt to parse JSON from json_start to the end
            # If not complete, will throw an exception => leave for next increment
            try:
                json_obj = json.loads(self._buffer[json_start:])
                # Parsing succeeded => construct call
                name = json_obj.get("name", "unknown_func")
                params_dict = (
                    json_obj.get("arguments") or json_obj.get("parameters") or {}
                )
                params_str = json.dumps(params_dict, ensure_ascii=False)

                # tool_index
                tool_index = -1
                tool_names = [t.function.name for t in tools]
                if name in tool_names:
                    tool_index = tool_names.index(name)

                calls.append(ToolCallItem(tool_index, name, params_str))

                # This indicates the entire remainder has been read as JSON
                # Move idx to the end => no further processing
                idx = len(self._buffer)
                break
            except json.JSONDecodeError:
                # Indicates that JSON from json_start to the end cannot be parsed => increment is insufficient
                # Then do not move, set idx to open_pos (or json_start),
                # thus leftover is preserved for the next time
                idx = open_pos
                break

        # Remove the part up to idx from the buffer
        leftover = self._buffer[idx:]
        self._buffer = leftover

        return StreamingParseResult(normal_text, calls)


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
        all_calls = []
        leftover_text = text
        while True:
            matched = False
            for detector in self.detectors:
                result = detector.detect_and_parse(leftover_text, tools)
                if result.success:
                    # Merge calls
                    all_calls.extend(result.calls)
                    # Replace leftover_text with the leftover from earlier
                    leftover_text = result.leftover_text
                    matched = True
                    break  # Break out of for loop to start again with the first detector
            if not matched:
                # No detector matched -> end
                break

        # leftover_text is the normal text not consumed by any Detector
        return leftover_text, all_calls

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
            if sp_result.normal_text:
                final_normal_text += sp_result.normal_text
            if sp_result.calls:
                final_calls.extend(sp_result.calls)

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
        leftover, calls = self.multi_format_parser.parse_once(full_text, self.tools)
        return leftover, calls

    def parse_stream_chunk(self, chunk_text: str):
        """
        Streaming call: incremental parsing
        """
        normal_text, calls = self.multi_format_parser.parse_streaming_increment(
            chunk_text, self.tools
        )
        return normal_text, calls
