import json
import logging
import re
from typing import Any, Dict, List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import safe_literal_eval

logger = logging.getLogger(__name__)


def get_argument_type(func_name: str, arg_key: str, defined_tools: List[Tool]) -> str:
    """Get the expected type for a function argument from tool schema."""
    name2tool = {tool.function.name: tool for tool in defined_tools}
    if func_name not in name2tool:
        return None
    tool = name2tool[func_name]
    parameters = tool.function.parameters or {}
    properties = parameters.get("properties", {})
    if arg_key not in properties:
        return None
    return properties[arg_key].get("type", None)


def parse_arguments(value: str) -> tuple[Any, bool]:
    """Parse a string value to appropriate type. Returns (parsed_value, success)."""
    try:
        try:
            parsed_value = json.loads(value)
        except:
            parsed_value = safe_literal_eval(value)
        return parsed_value, True
    except:
        return value, False


class Step3Detector(BaseFormatDetector):
    """
    Detector for Step3 model function call format.

    The Step3 format uses special Unicode tokens to delimit function calls
    with steptml XML format for invocations.

    Format Structure:
    ```
    <｜tool_calls_begin｜>
    <｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="function_name">
    <steptml:parameter name="param1">value1</steptml:parameter>
    <steptml:parameter name="param2">value2</steptml:parameter>
    </steptml:invoke><｜tool_call_end｜>
    <｜tool_calls_end｜>
    ```
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜tool_calls_begin｜>"
        self.eot_token = "<｜tool_calls_end｜>"
        self.tool_call_begin = "<｜tool_call_begin｜>"
        self.tool_call_end = "<｜tool_call_end｜>"
        self.tool_sep = "<｜tool_sep｜>"

        # Regex for parsing steptml invocations. The body is allowed to be empty
        # so that a parameter-less call (`<steptml:invoke name="x"></steptml:invoke>`)
        # is still recognised instead of being silently dropped.
        self.invoke_end = "</steptml:invoke>"
        self.invoke_regex = re.compile(
            r'<steptml:invoke name="([^"]+)">(.*?)</steptml:invoke>', re.DOTALL
        )
        self.param_regex = re.compile(
            r'<steptml:parameter name="([^"]+)">([^<]*)</steptml:parameter>', re.DOTALL
        )

        # Streaming state variables
        self._in_tool_block: bool = False
        self._tool_block_finished: bool = False
        self._current_function_name: str = ""
        self._current_parameters: Dict[str, Any] = {}
        self._in_tool_call: bool = False
        self._function_name_sent: bool = False

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Step3 format tool call."""
        return self.bot_token in text

    def _parse_steptml_invoke(
        self, text: str, tools: List[Tool] = None
    ) -> tuple[str, dict]:
        """Parse steptml invoke format to extract function name and parameters."""
        invoke_match = self.invoke_regex.search(text)
        if not invoke_match:
            return None, {}

        func_name = invoke_match.group(1)
        params_text = invoke_match.group(2)

        params = {}
        for param_match in self.param_regex.finditer(params_text):
            param_name = param_match.group(1)
            param_value = param_match.group(2).strip()

            # If tools provided, use schema-aware parsing
            if tools:
                arg_type = get_argument_type(func_name, param_name, tools)
                if arg_type and arg_type != "string":
                    parsed_value, _ = parse_arguments(param_value)
                    params[param_name] = parsed_value
                else:
                    params[param_name] = param_value
            else:
                # Fallback to generic parsing if no tools provided
                parsed_value, _ = parse_arguments(param_value)
                params[param_name] = parsed_value

        return func_name, params

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.
        """
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])

        try:
            pre_text, rest = text.split(self.bot_token, 1)

            # A block without its end token was cut short by the token limit.
            # The unfinished markup is not content, so only the text that
            # precedes the block is returned; the calls that did complete
            # before the cut are still parsed below, as in streaming.
            if self.eot_token in rest:
                tool_section, post_text = rest.split(self.eot_token, 1)
            else:
                logger.warning("Tool call block is not terminated, treating as markup")
                tool_section, post_text = rest, ""

            # Find all individual tool calls using regex
            calls = []
            tool_call_pattern = (
                f"{re.escape(self.tool_call_begin)}(.*?){re.escape(self.tool_call_end)}"
            )

            for match in re.finditer(tool_call_pattern, tool_section, re.DOTALL):
                call_content = match.group(1)

                # Check if it's a function call
                if self.tool_sep not in call_content:
                    continue

                type_part, invoke_part = call_content.split(self.tool_sep, 1)
                if type_part.strip() != "function":
                    continue

                func_name, params = self._parse_steptml_invoke(invoke_part, tools)
                if func_name:
                    # Use parse_base_json to create the ToolCallItem
                    action = {"name": func_name, "arguments": params}
                    calls.extend(self.parse_base_json(action, tools))

            # Combine pre and post text
            normal_text = pre_text + post_text

            return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # Return the original text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for Step3 format.

        The buffer is drained in a loop: a single increment may carry the
        preamble, several complete tool calls and the trailing text all at once
        (``stream_interval > 1``, speculative decoding, or simply the last
        increment of the stream, which has no successor to defer work to).
        """
        self._buffer += new_text

        # Build tool indices for validation
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []

        while self._buffer:
            # Everything after the tool block is ordinary text
            if self._tool_block_finished:
                normal_parts.append(self._buffer)
                self._buffer = ""
                break

            # Before the tool block: emit text, hold back a partial bot_token
            if not self._in_tool_block:
                idx = self._buffer.find(self.bot_token)
                if idx == -1:
                    partial_len = self._ends_with_partial_token(
                        self._buffer, self.bot_token
                    )
                    if partial_len:
                        if len(self._buffer) > partial_len:
                            normal_parts.append(self._buffer[:-partial_len])
                            self._buffer = self._buffer[-partial_len:]
                        break  # wait for more text
                    normal_parts.append(self._buffer)
                    self._buffer = ""
                    break
                normal_parts.append(self._buffer[:idx])
                self._buffer = self._buffer[idx + len(self.bot_token) :]
                self._in_tool_block = True
                continue

            # Inside the tool block, between calls
            if not self._in_tool_call:
                begin_idx = self._buffer.find(self.tool_call_begin)
                eot_idx = self._buffer.find(self.eot_token)
                if eot_idx != -1 and (begin_idx == -1 or eot_idx < begin_idx):
                    self._buffer = self._buffer[eot_idx + len(self.eot_token) :]
                    self._tool_block_finished = True
                    self._reset_streaming_state()
                    continue
                if begin_idx == -1:
                    break  # wait for the next call or the end of the block
                # Anything between two calls is protocol filler, not content
                self._buffer = self._buffer[begin_idx + len(self.tool_call_begin) :]
                self._in_tool_call = True
                self._function_name_sent = False
                self._current_function_name = ""
                self._current_parameters = {}
                continue

            # Inside a tool call
            result, consumed = self._parse_partial_tool_call(tools)
            calls.extend(result.calls)
            if consumed:
                continue
            if self.eot_token in self._buffer:
                # The block ends while this call is still unfinished: it was
                # truncated (token limit), so it is markup, not content.
                logger.warning("Tool block ended with incomplete tool call")
                eot_idx = self._buffer.find(self.eot_token)
                self._buffer = self._buffer[eot_idx + len(self.eot_token) :]
                self._tool_block_finished = True
                self._reset_streaming_state()
                continue
            break  # wait for more text

        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Flush at end of stream.

        Text held back because it looked like the start of ``bot_token`` is real
        content once no more text can arrive. A buffer still inside the tool
        block is an unfinished tool call, i.e. markup, and is dropped -- but a
        call whose name has already been streamed cannot be taken back, so its
        JSON is closed here to keep ``arguments`` parsable.
        """
        held, self._buffer = self._buffer, ""
        if held and not self._in_tool_block:
            return StreamingParseResult(normal_text=held)

        calls: List[ToolCallItem] = []
        if self._in_tool_call and self._function_name_sent:
            logger.warning("Stream ended inside a tool call, closing its arguments")
            closing = "}" if self.streamed_args_for_tool[self.current_tool_id] else "{}"
            calls.append(
                ToolCallItem(tool_index=self.current_tool_id, parameters=closing)
            )
            self.streamed_args_for_tool[self.current_tool_id] += closing
            self._reset_streaming_state()
            self.current_tool_id += 1
        return StreamingParseResult(calls=calls)

    def _current_call_segment(self) -> str:
        """The slice of the buffer that belongs to the call being parsed.

        Parameters must never be read past this call's own end: when a second
        complete call arrives in the same increment, an unbounded search grafts
        its parameters onto the current call and drops it.
        """
        end = self._buffer.find(self.tool_call_end)
        segment = self._buffer if end == -1 else self._buffer[:end]
        close = segment.find(self.invoke_end)
        if close != -1:
            segment = segment[: close + len(self.invoke_end)]
        return segment

    def _skip_current_call(self) -> bool:
        """Drop the malformed call in the buffer. True if the buffer advanced."""
        self._reset_streaming_state()
        end = self._buffer.find(self.tool_call_end)
        if end == -1:
            return False
        self._buffer = self._buffer[end + len(self.tool_call_end) :]
        return True

    def _parse_partial_tool_call(
        self, tools: List[Tool]
    ) -> tuple[StreamingParseResult, bool]:
        """Parse the tool call at the head of the buffer.

        Returns the deltas to stream plus whether the buffer was advanced past
        this call (so the caller may look at what follows it).
        """
        calls = []
        segment = self._current_call_segment()

        # Check if we have tool_sep (means we're past the type declaration)
        if self.tool_sep not in segment:
            if self.tool_call_end in self._buffer:
                logger.warning("Tool call without a type separator, skipping")
                return StreamingParseResult(calls=calls), self._skip_current_call()
            return StreamingParseResult(calls=calls), False  # Wait for more text

        type_part, invoke_part = segment.split(self.tool_sep, 1)
        if type_part.strip() != "function":
            # Invalid tool type, skip this tool call
            logger.warning(f"Unsupported tool call type: {type_part.strip()!r}")
            return StreamingParseResult(calls=calls), self._skip_current_call()

        # Try to extract function name if not sent yet
        if not self._function_name_sent:
            name_match = re.search(r'<steptml:invoke name="([^"]+)">', invoke_part)
            if name_match:
                func_name = name_match.group(1)

                # Validate function name
                if func_name in self._tool_indices:
                    self._current_function_name = func_name
                    self._function_name_sent = True

                    # Initialize tool tracking
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0

                    # Ensure tracking arrays are large enough
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")

                    # Store tool call info
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": {},
                    }

                    # Send tool name with empty parameters
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                else:
                    # Invalid function name
                    logger.warning(f"Invalid function name: {func_name}")
                    return StreamingParseResult(calls=calls), self._skip_current_call()
            else:
                # Function name not complete yet
                return StreamingParseResult(calls=calls), False

        # Parse parameters incrementally
        if self._function_name_sent:
            # Extract all complete parameters
            new_params = {}
            for param_match in self.param_regex.finditer(invoke_part):
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()

                # Use schema-aware parsing
                arg_type = get_argument_type(
                    self._current_function_name, param_name, tools
                )
                if arg_type and arg_type != "string":
                    parsed_value, _ = parse_arguments(param_value)
                    new_params[param_name] = parsed_value
                else:
                    new_params[param_name] = param_value

            # Check if we have new parameters to stream
            if new_params != self._current_parameters:
                # Build the JSON content without the closing brace for streaming
                if not self._current_parameters:
                    # First parameters - send opening brace and content
                    params_content = json.dumps(new_params, ensure_ascii=False)
                    if len(params_content) > 2:  # More than just "{}"
                        # Send everything except the closing brace
                        diff = params_content[:-1]
                    else:
                        diff = "{"
                else:
                    # Subsequent parameters - calculate the incremental diff
                    old_json = json.dumps(self._current_parameters, ensure_ascii=False)
                    new_json = json.dumps(new_params, ensure_ascii=False)

                    # Remove closing braces for comparison
                    old_without_brace = old_json[:-1]
                    new_without_brace = new_json[:-1]

                    # The new content should extend the old content
                    if new_without_brace.startswith(old_without_brace):
                        diff = new_without_brace[len(old_without_brace) :]
                    else:
                        # Parameters changed in unexpected way - shouldn't happen in normal streaming
                        diff = ""

                if diff:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            parameters=diff,
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += diff

                # Update current state
                self._current_parameters = new_params
                self.prev_tool_call_arr[self.current_tool_id]["arguments"] = new_params

            # Check if tool call is complete
            if self.tool_call_end in self._buffer:
                # Close the JSON object. A call with no parameters has streamed
                # nothing so far, so it needs a whole "{}" -- otherwise its
                # arguments stay "", which is not valid JSON.
                closing = (
                    "}" if self.streamed_args_for_tool[self.current_tool_id] else "{}"
                )
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        parameters=closing,
                    )
                )
                self.streamed_args_for_tool[self.current_tool_id] += closing

                # Find the end position
                end_idx = self._buffer.find(self.tool_call_end)
                # Remove the processed tool call from buffer
                self._buffer = self._buffer[end_idx + len(self.tool_call_end) :]

                # Reset state for next tool call
                self._reset_streaming_state()
                self.current_tool_id += 1
                return StreamingParseResult(calls=calls), True

        return StreamingParseResult(calls=calls), False

    def _reset_streaming_state(self):
        """Reset streaming state for the next tool call"""
        self._in_tool_call = False
        self._function_name_sent = False
        self._current_function_name = ""
        self._current_parameters = {}

    def supports_structural_tag(self) -> bool:
        """Return True if this detector supports structural tag format."""
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()
