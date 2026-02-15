import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import _is_complete_json

logger = logging.getLogger(__name__)


class KimiK2Detector(BaseFormatDetector):
    """
    Detector for Kimi K2 model function call format.

    Format Structure:
    ```
    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.{func_name}:{index}<|tool_call_argument_begin|>{json_args}<|tool_call_end|>
    <|tool_calls_section_end|>
    ```

    Reference: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/tool_call_guidance.md
    """

    def __init__(self):
        super().__init__()

        self.bot_token: str = "<|tool_calls_section_begin|>"
        self.eot_token: str = "<|tool_calls_section_end|>"

        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"
        self.tool_call_argument_begin_token: str = "<|tool_call_argument_begin|>"

        # Support both standard format (functions.{func_name}:{index}) and UUID format (call_{uuid})
        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>(?:[\w\.\-]+:\d+|call_[\w]+))\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*?\})\s*<\|tool_call_end\|>"
        )

        # Use a pattern that stops at <|tool_call_end|> or end of string
        # (?:(?!<\|tool_call_end\|>).)* is a tempered greedy token that matches any char
        # except when it would start <|tool_call_end|>
        # Support both standard format and UUID format
        self.stream_tool_call_portion_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>(?:[\w\.\-]+:\d+|call_[\w]+))\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{(?:(?!<\|tool_call_end\|>).)*)",
            re.DOTALL,
        )

        # Initialize streaming state
        self._buffer = ""
        self._last_arguments = ""
        # Track the length of arguments already streamed (similar to GLM47's _streamed_raw_length)
        # This ensures accurate incremental diff calculation in streaming scenarios
        self._streamed_args_length = 0
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []

        # Robust parser for ids like "functions.search:0", "search:0", or "call_{uuid}"
        # Note: function names may contain hyphens (e.g., mcp__tencent-cloud-portal__search-documents)
        # For UUID format, we'll extract function name from JSON arguments
        self.tool_call_id_regex = re.compile(
            r"^(?:(?:functions\.)?(?P<name>[\w\.\-]+):(?P<index>\d+)|(?P<call_id>call_[\w]+))$"
        )

    def reset(self):
        """
        Reset all streaming state for a new conversation/request.

        This should be called when starting to parse a new request to ensure
        no state from previous requests leaks through.
        """
        self._buffer = ""
        self._last_arguments = ""
        self._streamed_args_length = 0
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []

    def _reset_tool_call_state(self):
        """
        Reset state for the next tool call within the same request.

        This is called when a tool call is completed and we need to prepare
        for parsing the next one. Similar to GLM47's _reset_streaming_state.
        """
        self._last_arguments = ""
        self._streamed_args_length = 0
        self.current_tool_name_sent = False

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a KimiK2 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])
        try:
            # there are two possible captures - between tags, or between a
            # tag and end-of-string so the result of
            # findall is an array of tuples where one is a function call and
            # the other is None
            function_call_tuples = self.tool_call_regex.findall(text)

            logger.debug("function_call_tuples: %s", function_call_tuples)

            tool_calls = []
            for match in function_call_tuples:
                function_id, function_args = match
                m = self.tool_call_id_regex.match(function_id)
                if not m:
                    logger.warning("Unexpected tool_call_id format: %s", function_id)
                    continue

                # Handle both standard format and UUID format
                if m.group("call_id"):
                    # UUID format: The call_id (e.g., call_6b0e7c24f0c44f0d996b59e5) does NOT contain function name
                    # We need to infer it from the JSON arguments by matching against available tools
                    try:
                        args_dict = json.loads(function_args)
                        function_name = self._infer_function_name_from_args(
                            args_dict, tools
                        )
                        if not function_name:
                            # Fallback: use call_id as function name (will be updated later if possible)
                            logger.warning(
                                "Cannot infer function name from UUID format call: %s, args: %s. Using call_id as fallback.",
                                function_id,
                                list(args_dict.keys()),
                            )
                            function_name = function_id
                        function_idx = 0  # Default index for UUID format
                    except json.JSONDecodeError as e:
                        logger.warning(
                            "Failed to parse JSON arguments for UUID format: %s, error: %s",
                            function_id,
                            e,
                        )
                        continue
                else:
                    # Standard format: functions.{func_name}:{index}
                    function_name = m.group("name")
                    function_idx = int(m.group("index"))

                logger.info(f"function_name {function_name}")

                tool_calls.append(
                    ToolCallItem(
                        tool_index=function_idx,
                        name=function_name,
                        parameters=function_args,
                    )
                )

            content = text[: text.find(self.bot_token)]
            return StreamingParseResult(normal_text=content, calls=tool_calls)

        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for KimiK2 format.

        State Machine:
        1. NORMAL_TEXT: Accumulating normal text, watching for tool call markers
        2. TOOL_CALL_SECTION: Inside <|tool_calls_section_begin|>...<|tool_calls_section_end|>
        3. TOOL_CALL_PARSING: Parsing individual tool call
        4. TOOL_CALL_COMPLETE: Tool call finished, ready for next one

        Critical Requirements:
        - Never leak special tokens (<|tool_call_*|>, <|tool_calls_section_*|>) to normal_text
        - Handle partial tokens at buffer boundaries
        - Support both standard format (functions.name:0) and UUID format (call_xxx)
        - Correctly handle multiple consecutive tool calls
        - Preserve buffer state across increments for retry on error
        """
        self._buffer += new_text
        current_text = self._buffer

        # Define all special tokens that should never leak
        special_tokens = [
            self.bot_token,  # <|tool_calls_section_begin|>
            self.eot_token,  # <|tool_calls_section_end|>
            self.tool_call_start_token,  # <|tool_call_begin|>
            self.tool_call_end_token,  # <|tool_call_end|>
            self.tool_call_argument_begin_token,  # <|tool_call_argument_begin|>
        ]

        # Check if we have any tool call markers
        has_tool_call = any(token in current_text for token in special_tokens)

        if has_tool_call:
            # Find the earliest tool call marker position
            earliest_pos = len(current_text)
            for token in special_tokens:
                pos = current_text.find(token)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos

            if earliest_pos > 0:
                # There's normal text before the tool call markers
                # Return it first, keep the rest in buffer
                normal_text = current_text[:earliest_pos]
                self._buffer = current_text[earliest_pos:]
                return StreamingParseResult(normal_text=normal_text)

        if not has_tool_call:
            # No tool call markers found, but check for partial tokens at buffer end
            # This prevents leaking partial special tokens like "<|tool_ca" or "<|tool_calls_sec"
            max_partial_len = 0
            for token in special_tokens:
                partial_len = self._ends_with_partial_token(current_text, token)
                max_partial_len = max(max_partial_len, partial_len)

            if max_partial_len > 0:
                # Keep the potential partial token in buffer
                safe_text = current_text[:-max_partial_len]
                self._buffer = current_text[-max_partial_len:]
                return StreamingParseResult(normal_text=safe_text)
            else:
                # No partial token, safe to return all content
                self._buffer = ""
                return StreamingParseResult(normal_text=current_text)

        # At this point, we have tool call markers in the buffer
        calls: list[ToolCallItem] = []

        try:
            # Try to match a tool call (complete or partial)
            match = self.stream_tool_call_portion_regex.search(current_text)

            if not match:
                # No match yet, but we have tool call markers
                # This could be:
                # 1. Incomplete tool call (waiting for more data)
                # 2. Just the section markers without tool calls

                # Check if we have the section end marker
                if self.eot_token in current_text:
                    # Tool calls section is complete, remove all markers from buffer
                    eot_pos = current_text.find(self.eot_token)
                    self._buffer = current_text[eot_pos + len(self.eot_token) :]
                    # Reset tool call state for next section
                    self.current_tool_id = -1
                    self.current_tool_name_sent = False
                    return StreamingParseResult(normal_text="", calls=calls)

                # Still waiting for more data, keep everything in buffer
                return StreamingParseResult(normal_text="", calls=calls)

            # We have a match (complete or partial tool call)
            function_id = match.group("tool_call_id")
            function_args = match.group("function_arguments")

            # Parse the tool call ID
            m = self.tool_call_id_regex.match(function_id)
            if not m:
                logger.warning("Unexpected tool_call_id format: %s", function_id)
                # Skip this malformed tool call, remove it from buffer
                self._buffer = current_text[match.end() :]
                return StreamingParseResult(normal_text="", calls=calls)

            # Extract function name based on format
            function_name = None
            if m.group("call_id"):
                # UUID format: infer function name from arguments
                if _is_complete_json(function_args):
                    try:
                        args_dict = json.loads(function_args)
                        function_name = self._infer_function_name_from_args(
                            args_dict, tools
                        )
                    except json.JSONDecodeError:
                        pass

                # If inference failed, use call_id as placeholder
                if not function_name:
                    function_name = function_id
            else:
                # Standard format: extract from ID
                function_name = m.group("name")

            # Initialize state for first tool call
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = [""]

            # Ensure tracking arrays are large enough
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

            # Send tool name on first encounter
            if not self.current_tool_name_sent:
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=function_name,
                        parameters="",
                    )
                )
                self.current_tool_name_sent = True
                self._streamed_args_length = 0
                self._last_arguments = ""
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": function_name,
                    "arguments": {},
                }

            # Stream argument increments
            current_args_length = len(function_args)
            if current_args_length > self._streamed_args_length:
                # Calculate the new increment
                argument_diff = function_args[self._streamed_args_length :]

                # CRITICAL: Filter out ALL special end markers from argument_diff
                # This prevents streaming any special tokens as part of arguments
                # Order matters: filter tool_call_end first, then section_end
                for end_marker in [self.tool_call_end_token, self.eot_token]:
                    if end_marker in argument_diff:
                        argument_diff = argument_diff.split(end_marker, 1)[0]

                # Update streamed length
                self._streamed_args_length = current_args_length

                if argument_diff:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=argument_diff,
                        )
                    )
                    self._last_arguments += argument_diff
                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff

            # Check if tool call is complete (has end marker)
            if self.tool_call_end_token in current_text[match.start() :]:
                # Extract complete arguments (without end marker)
                complete_args = function_args.split(self.tool_call_end_token, 1)[0]

                # Validate and parse JSON
                if _is_complete_json(complete_args):
                    try:
                        parsed_args = json.loads(complete_args)
                        self.prev_tool_call_arr[self.current_tool_id][
                            "arguments"
                        ] = parsed_args

                        # For UUID format, update function name if we now have complete args
                        if m.group("call_id") and function_name == function_id:
                            inferred_name = self._infer_function_name_from_args(
                                parsed_args, tools
                            )
                            if inferred_name:
                                self.prev_tool_call_arr[self.current_tool_id][
                                    "name"
                                ] = inferred_name
                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse complete JSON arguments: %s", e)

                # Find and remove the completed tool call from buffer
                tool_call_end_pattern = r"<\|tool_call_begin\|>.*?<\|tool_call_end\|>"
                end_match = re.search(tool_call_end_pattern, current_text, re.DOTALL)
                if end_match:
                    remaining_text = current_text[end_match.end() :]

                    # CRITICAL: Check for section end marker and remove it
                    # Strip leading whitespace first to handle cases like " <|tool_calls_section_end|>"
                    remaining_text = remaining_text.lstrip()
                    if remaining_text.startswith(self.eot_token):
                        # Remove section end marker
                        self._buffer = remaining_text[len(self.eot_token) :]
                        # Reset state for next tool calls section
                        self.current_tool_id = -1
                        self.current_tool_name_sent = False
                    else:
                        # More tool calls may follow
                        self._buffer = remaining_text
                        self.current_tool_id += 1
                        self._reset_tool_call_state()
                else:
                    # Fallback: find tool_call_end_token position and clean up from there
                    # This handles cases where the full pattern doesn't match but we know
                    # the tool call is complete (we detected tool_call_end_token above)
                    end_token_pos = current_text.find(self.tool_call_end_token)
                    if end_token_pos != -1:
                        remaining_text = current_text[
                            end_token_pos + len(self.tool_call_end_token) :
                        ]
                        remaining_text = remaining_text.lstrip()
                        if remaining_text.startswith(self.eot_token):
                            self._buffer = remaining_text[len(self.eot_token) :]
                        else:
                            self._buffer = remaining_text
                    else:
                        self._buffer = ""
                    self.current_tool_id = -1
                    self.current_tool_name_sent = False

                return StreamingParseResult(normal_text="", calls=calls)

            # Tool call not yet complete, keep buffering
            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}", exc_info=True)
            # On error, preserve buffer for retry but don't leak tokens
            return StreamingParseResult(normal_text="", calls=calls)

    def _infer_function_name_from_args(self, args_dict: dict, tools: List[Tool]) -> str:
        """
        Infer function name from JSON arguments by matching against available tools.

        This is needed for UUID format (e.g., call_6b0e7c24f0c44f0d996b59e5) which doesn't
        include the function name in the call_id.

        Strategy:
        1. Match argument keys against tool parameter schemas
        2. Calculate a similarity score based on:
           - Number of matching required parameters
           - Number of matching optional parameters
           - Penalty for extra keys not in schema
        3. Return the tool with the highest score

        Args:
            args_dict: Parsed JSON arguments
            tools: List of available tools

        Returns:
            Inferred function name, or None if no good match found
        """
        if not tools or not args_dict:
            return None

        arg_keys = set(args_dict.keys())
        best_match = None
        best_score = -1

        for tool in tools:
            if not tool.function or not tool.function.parameters:
                continue

            # Extract parameter info from tool schema
            properties = tool.function.parameters.get("properties", {})
            required = set(tool.function.parameters.get("required", []))
            all_params = set(properties.keys())

            # Calculate matching score
            matching_required = len(arg_keys & required)
            matching_optional = len(arg_keys & (all_params - required))
            extra_keys = len(arg_keys - all_params)
            missing_required = len(required - arg_keys)

            # Scoring formula:
            # - CRITICAL: If any required parameter is missing, heavily penalize
            # - High weight for matching required parameters
            # - Medium weight for matching optional parameters
            # - Penalty for extra keys (might be wrong tool)
            # - Bonus for exact match (all required params present, no missing)
            if missing_required > 0:
                # Missing required parameters - very bad match
                score = -1000 * missing_required
            else:
                # All required parameters present - good candidate
                # Prefer tools with fewer required parameters (more specific match)
                score = (
                    matching_required * 10
                    + matching_optional * 5
                    - extra_keys * 3
                    - len(required) * 2
                )  # Penalty for having more required params

            logger.debug(
                f"Tool {tool.function.name}: score={score}, "
                f"matching_required={matching_required}/{len(required)}, "
                f"matching_optional={matching_optional}, "
                f"extra_keys={extra_keys}, missing_required={missing_required}"
            )

            if score > best_score:
                best_score = score
                best_match = tool.function.name

        if best_score > 0:
            logger.info(f"Inferred function name: {best_match} (score={best_score})")
            return best_match

        logger.warning(f"Could not infer function name from args: {list(arg_keys)}")
        return None

    def structure_info(self) -> _GetInfoFunc:
        """Return function that creates StructureInfo for guided generation."""

        def get_info(name: str) -> StructureInfo:
            return StructureInfo(
                begin=f"<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:0<|tool_call_argument_begin|>",
                end="<|tool_call_end|><|tool_calls_section_end|>",
                trigger="<|tool_calls_section_begin|>",
            )

        return get_info
