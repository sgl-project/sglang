import json
import logging
import re
from typing import List, Optional, Tuple

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.parser.harmony_parser import HarmonyParser

logger = logging.getLogger(__name__)


class GptOssDetector(BaseFormatDetector):
    """
    Detector for T4-style function calls using HarmonyParser.

    Handles tool calls in the format:
    <|channel|>commentary to={namespace.function}<|constrain|>json<|message|>{args}<|call|>
    """

    # Structural markers of the Harmony protocol. Text is only withheld from the
    # client while it could still turn out to be the start of one of these.
    _HARMONY_MARKERS = (
        "<|start|>",
        "<|channel|>",
        "<|message|>",
        "<|constrain|>",
        "<|end|>",
        "<|call|>",
        "<|return|>",
        "assistantfinal",
    )

    def __init__(self):
        super().__init__()
        self.harmony_parser = HarmonyParser()
        self.bot_token = "<|start|>assistant<|channel|>commentary"
        self.eot_token = "<|call|>"

        # Pattern to extract function name and JSON from tool_call event content
        self.tool_extract_pattern = re.compile(
            r"to=([a-zA-Z_][a-zA-Z0-9_.-]*)\s*<\|constrain\|>json<\|message\|>(.*?)(?:<\|call\|>|$)",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        """Check if text contains TypeScript-style function call markers."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse TypeScript-style function calls from complete text."""
        if not self.has_tool_call(text):
            return StreamingParseResult(normal_text=text, calls=[])

        # Parse with HarmonyParser
        events = self.harmony_parser.parse(text)
        # Flush buffer for complete parsing
        events += self.harmony_parser.parse("")

        tool_indices = self._get_tool_indices(tools)
        calls = []
        normal_parts = []
        tool_index = 0

        for event in events:
            if event.event_type == "tool_call":
                # Extract tool call from event content
                tool_call = self._extract_tool_call_from_event(
                    event.raw_text if event.raw_text else event.content,
                    tool_indices,
                    tool_index,
                )
                if tool_call:
                    calls.append(tool_call)
                    tool_index += 1
            elif event.event_type == "normal":
                normal_parts.append(event.content)
            # Ignore reasoning events in function call context

        normal_text = " ".join(normal_parts).strip()
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _has_harmony_marker(self, text: str) -> bool:
        return any(marker in text for marker in self._HARMONY_MARKERS)

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Release text held back when the stream ended.

        ``parse_streaming_increment`` withholds a trailing fragment only while it
        can still complete into a structural marker. Once the stream is over it
        cannot, so what is left is ordinary content and is emitted rather than
        dropped. A fragment that did complete into a marker is left alone: it
        means the protocol block was truncated, and that is not user-visible text.
        """
        pending, self._buffer = self._buffer, ""
        if pending and not self._has_harmony_marker(pending):
            return StreamingParseResult(normal_text=pending, calls=[])
        return StreamingParseResult()

    def _split_at_partial_marker(self, text: str) -> Tuple[str, str]:
        """Split ``text`` into content that is safe to stream and a tail that may
        still complete into a structural marker."""
        hold = max(
            (
                self._ends_with_partial_token(text, marker)
                for marker in self._HARMONY_MARKERS
            ),
            default=0,
        )
        if hold:
            return text[:-hold], text[-hold:]
        return text, ""

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Parse incremental streaming text for TypeScript-style function calls."""
        self._buffer += new_text

        if self.harmony_parser.strategy is None and not self._has_harmony_marker(
            self._buffer
        ):
            # HarmonyParser has not seen a structural marker yet, so it holds
            # everything it is given and will keep holding until one arrives. Plain
            # content has to stream anyway, so emit everything except a tail that
            # could still complete into a marker.
            #
            # Only that tail stays in the buffer, and only the buffer is handed to
            # HarmonyParser below. Giving it text that has already been streamed out
            # is what previously made it replay that text as a second normal event
            # once a marker finally arrived.
            emit, self._buffer = self._split_at_partial_marker(self._buffer)
            return StreamingParseResult(normal_text=emit, calls=[])

        # A marker is in play: hand over everything not interpreted yet.
        pending, self._buffer = self._buffer, ""
        events = self.harmony_parser.parse(pending)

        if not any(event.event_type == "tool_call" for event in events):
            # No tool call in this batch. Emit the normal content HarmonyParser
            # produced, with its own filtering already applied.
            normal_text = "".join(
                event.content for event in events if event.event_type == "normal"
            )
            return StreamingParseResult(normal_text=normal_text, calls=[])

        # Initialize state if needed
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls = []
        normal_text = ""

        for event in events:
            if event.event_type == "tool_call":
                # We got a complete tool call from HarmonyParser
                tool_call_info = self._extract_tool_call_from_event(
                    event.raw_text if event.raw_text else event.content,
                    self._tool_indices,
                    self.current_tool_id if self.current_tool_id >= 0 else 0,
                )

                if tool_call_info:
                    # Initialize state if first tool
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.prev_tool_call_arr = []
                        self.streamed_args_for_tool = [""]

                    # Ensure arrays are large enough
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")

                    # Store tool call info
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": tool_call_info.name,
                        "arguments": json.loads(tool_call_info.parameters),
                    }

                    # Emit the complete tool call at once
                    # (Could be modified to emit name first, then args, if needed)
                    calls.append(tool_call_info)

                    # Mark as streamed
                    self.streamed_args_for_tool[self.current_tool_id] = (
                        tool_call_info.parameters
                    )

                    # Move to next tool
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False

            elif event.event_type == "normal":
                normal_text += event.content

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _extract_tool_call_from_event(
        self, content: str, tool_indices: dict, tool_index: int
    ) -> Optional[ToolCallItem]:
        """
        Extract tool call information from HarmonyParser event content.

        Content format: "commentary to=functions.get_weather<|constrain|>json<|message|>{...}"
        """
        match = self.tool_extract_pattern.search(content)

        if not match:
            logger.debug(f"Could not extract tool call from: {content[:100]}")
            return None

        full_function_name = match.group(1)
        json_content = match.group(2)

        # Extract function name (last part after .)
        function_name = (
            full_function_name.split(".")[-1]
            if "." in full_function_name
            else full_function_name
        )

        # Check if tool exists
        if function_name not in tool_indices:
            logger.debug(f"Function {function_name} not in available tools")
            if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                return None  # Skip unknown tools (default legacy behavior)

        # Parse JSON arguments
        try:
            arguments = json.loads(json_content) if json_content.strip() else {}
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON arguments: {e}")
            return None

        return ToolCallItem(
            tool_index=tool_index,
            name=function_name,
            parameters=json.dumps(arguments, ensure_ascii=False),
        )

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError("structure_info not used with HarmonyParser")

    def get_structural_tag_name(self) -> str:
        return "harmony"
