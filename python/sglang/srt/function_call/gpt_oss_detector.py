import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


class GptOssDetector(BaseFormatDetector):
    """
    Detector for T4-style function calls with channel format.

    Supports two formats:
    1. Direct function call: <|channel|>commentary to={namespace.function}<|constrain|>json<|message|>{args}<|call|>
    2. Commentary with action plan: <|channel|>commentary<|message|>{content}<|end|>

    For parallel function calls, each call is self-contained and starts with its own channel:
    <|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"location":"SF"}<|call|>
    <|channel|>commentary to=functions.search<|constrain|>json<|message|>{"query":"SF attractions"}<|call|>

    Examples:
    Single: <|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"location":"San Francisco"}<|call|>commentary
    Multiple: <|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"location":"Paris"}<|call|>commentary<|channel|>commentary to=functions.search<|constrain|>json<|message|>{"query":"Paris tourism"}<|call|>
    With Action Plan: <|channel|>commentary<|message|>**Action plan**: 1. Do X 2. Do Y<|end|><|start|>assistant<|channel|>commentary to=functions.x<|constrain|>json<|message|>{"template": "basic_html", "path": "index.html"}<|call|>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<|start|>assistant<|channel|>commentary"
        self.eot_token = "<|call|>"
        # TODO: no clear indication how parallel tool call response format is
        self.tool_call_separator = ""

        # Pattern for complete function calls with to= parameter
        # Handles both <|call|> and <|call|>commentary endings
        # Also handles optional <|start|>assistant prefix and whitespace after function name
        self.function_call_pattern = re.compile(
            r"(?:<\|start\|>assistant)?<\|channel\|>commentary to=([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*"
            r"<\|constrain\|>json<\|message\|>(.*?)<\|call\|>(?:commentary)?",
            re.DOTALL,
        )

        # Pattern for streaming function calls (incomplete)
        # Also handles optional whitespace after function name
        self.streaming_pattern = re.compile(
            r"(?:<\|start\|>assistant)?<\|channel\|>commentary to=([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*"
            r"<\|constrain\|>json<\|message\|>(.*)",
            re.DOTALL,
        )

        # Pattern for commentary with action plan (no to= parameter)
        self.commentary_pattern = re.compile(
            r"<\|channel\|>commentary<\|message\|>(.*?)<\|end\|>",
            re.DOTALL,
        )

        self._last_arguments = ""

    def has_tool_call(self, text: str) -> bool:
        """Check if text contains TypeScript-style function call markers."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse TypeScript-style function calls from complete text."""
        if not self.has_tool_call(text):
            return StreamingParseResult(normal_text=text, calls=[])

        tool_indices = self._get_tool_indices(tools)

        calls = []
        tool_index = 0

        # Process the entire text to handle mixed commentary and tool calls
        normal_text_parts = []

        # Find all commentary sections (both with and without to=)
        all_commentary_pattern = re.compile(
            r"<\|channel\|>commentary(?:\s+to=[^<]*)?<\|message\|>(.*?)(?:<\|end\|>|<\|call\|>)",
            re.DOTALL,
        )

        # Track processed positions to avoid double-processing
        processed_ranges = []

        # First, extract all tool calls
        for match in self.function_call_pattern.finditer(text):
            full_function_name = match.group(1)
            args_content = match.group(2)
            processed_ranges.append((match.start(), match.end()))

            function_name = (
                full_function_name.split(".")[-1]
                if "." in full_function_name
                else full_function_name
            )

            try:
                arguments = json.loads(args_content) if args_content.strip() else {}
            except json.JSONDecodeError:
                continue

            if function_name in tool_indices:
                calls.append(
                    ToolCallItem(
                        tool_index=tool_index,
                        name=function_name,
                        parameters=json.dumps(arguments, ensure_ascii=False),
                    )
                )
                tool_index += 1

        # Then, find non-tool-call commentary sections for normal text
        for match in all_commentary_pattern.finditer(text):
            # Check if this match overlaps with any processed tool call
            match_start, match_end = match.start(), match.end()
            is_tool_call = any(
                start <= match_start < end or start < match_end <= end
                for start, end in processed_ranges
            )

            # If this commentary is not part of a tool call, include it in normal text
            if not is_tool_call:
                content = match.group(1).strip()
                if content:
                    normal_text_parts.append(content)

        # Handle remaining text after all matches
        if processed_ranges:
            last_match_end = max(end for _, end in processed_ranges)
            if last_match_end < len(text):
                remaining_text = text[last_match_end:]

            # Clean up <|start|>assistant prefixes and extract final content
            # Remove standalone <|start|>assistant prefixes
            remaining_text = re.sub(r"<\|start\|>assistant(?!\w)", "", remaining_text)

            # Extract content from final channel if present
            final_pattern = re.compile(
                r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|$)", re.DOTALL
            )
            final_match = final_pattern.search(remaining_text)

            if final_match:
                # Get everything before final channel + final channel content
                before_final = remaining_text[: final_match.start()].strip()
                final_content = final_match.group(1).strip()

                parts = []
                if before_final:
                    parts.append(before_final)
                if final_content:
                    parts.append(final_content)
                remaining_text = " ".join(parts) if parts else ""

            remaining_text = remaining_text.strip()

            if remaining_text:
                normal_text_parts.append(remaining_text)

        # Combine all normal text parts
        final_normal_text = " ".join(part for part in normal_text_parts if part).strip()
        return StreamingParseResult(normal_text=final_normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Parse incremental streaming text for TypeScript-style function calls."""
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have a tool call
        has_tool_call = "<|channel|>commentary to=" in current_text

        if not has_tool_call and current_text:
            # Check for commentary without function calls
            commentary_match = self.commentary_pattern.search(current_text)
            if commentary_match:
                commentary_content = commentary_match.group(1)
                self._buffer = current_text[commentary_match.end() :]
                return StreamingParseResult(normal_text=commentary_content, calls=[])

            # Check for final channel content
            final_pattern = re.compile(
                r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|$)",
                re.DOTALL,
            )
            final_match = final_pattern.search(current_text)
            if final_match:
                final_content = final_match.group(1).strip()
                self._buffer = ""
                return StreamingParseResult(normal_text=final_content, calls=[])

            self._buffer = ""
            return StreamingParseResult(normal_text=new_text, calls=[])

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls = []
        try:
            # Check for streaming function call
            match = self.streaming_pattern.search(current_text)
            if match:
                full_function_name = match.group(1)
                args_content = match.group(2)

                function_name = (
                    full_function_name.split(".")[-1]
                    if "." in full_function_name
                    else full_function_name
                )

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                # Ensure we have enough entries in tracking arrays
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                if not self.current_tool_name_sent:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=function_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                    # Store the tool call info
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": function_name,
                        "arguments": {},
                    }
                    self.streamed_args_for_tool[self.current_tool_id] = ""

                # Check if we have a complete function call
                complete_match = self.function_call_pattern.search(current_text)
                if complete_match:
                    args_content = complete_match.group(2)

                    try:
                        parsed_args = json.loads(args_content)
                        self.prev_tool_call_arr[self.current_tool_id][
                            "arguments"
                        ] = parsed_args

                        # Send complete arguments if we haven't sent them yet
                        if not self.streamed_args_for_tool[self.current_tool_id]:
                            # Send the complete arguments as JSON string
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters=json.dumps(
                                        parsed_args, ensure_ascii=False
                                    ),
                                )
                            )
                            self.streamed_args_for_tool[self.current_tool_id] = (
                                json.dumps(parsed_args, ensure_ascii=False)
                            )
                    except json.JSONDecodeError:
                        pass

                    # Remove the completed function call from buffer
                    remaining_after_call = current_text[complete_match.end() :]

                    # Clean up <|start|>assistant prefixes and extract final content
                    remaining_after_call = re.sub(
                        r"<\|start\|>assistant(?!\w)", "", remaining_after_call
                    )

                    # Extract content from final channel if present
                    final_pattern = re.compile(
                        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|$)",
                        re.DOTALL,
                    )
                    final_match = final_pattern.search(remaining_after_call)

                    if final_match:
                        before_final = remaining_after_call[
                            : final_match.start()
                        ].strip()
                        final_content = final_match.group(1).strip()

                        parts = []
                        if before_final:
                            parts.append(before_final)
                        if final_content:
                            parts.append(final_content)
                        remaining_after_call = " ".join(parts) if parts else ""

                    self._buffer = remaining_after_call.strip()

                    # Reset state for next tool call
                    self.current_tool_name_sent = False
                    self.current_tool_id += 1

                    # Return final content if available
                    final_text = ""
                    if final_match and final_content:
                        final_text = final_content
                    elif remaining_after_call:
                        final_text = remaining_after_call

                    return StreamingParseResult(normal_text=final_text, calls=calls)

            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text, calls=[])

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()

    def build_ebnf(self, tools: List[Tool]) -> str:
        raise NotImplementedError()
