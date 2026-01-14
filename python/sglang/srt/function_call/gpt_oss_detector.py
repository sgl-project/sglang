import json
import logging
import re
from typing import List, Optional

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
    <|channel|>commentary to={namespace.function}<|message|>{args}<|call|>
    (Note: <|constrain|>json is optional and may be omitted)
    """

    def __init__(self):
        super().__init__()
        self.harmony_parser = HarmonyParser()
        self.bot_token = "<|start|>assistant<|channel|>commentary"
        self.eot_token = "<|call|>"

        # Pattern to extract function name and JSON from tool call header
        # Format to extract from: "to=func_name<|constrain|>json<|message|>{...}" or "to=func_name<|message|>{...}"
        # We extract: to=func_name for name, and after <|message|> for JSON
        # <|constrain|>json is optional
        self.tool_extract_pattern = re.compile(
            r"to=([a-zA-Z_][a-zA-Z0-9_.-]*)(?:\s*<\|constrain\|>\w+)?\s*<\|message\|>(.*?)(?:<\|call\|>|$)",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        """Check if text contains TypeScript-style function call markers.

        Supports multiple formats:
        1. Full format: <|start|>assistant<|channel|>commentary to=...
        2. Stream format: <|channel|>commentary to=... (without start token)
        """
        # Check for full bot token format
        if self.bot_token in text:
            return True

        # Check for stream format without <|start|>assistant prefix
        # Pattern: <|channel|>commentary to=...
        if "<|channel|>commentary to=" in text or "<|channel|>commentary  to=" in text:
            return True

        return False

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

    def _get_tool_indices(self, tools: List[Tool]) -> dict:
        """Get mapping of tool names to indices."""
        tool_indices = {}
        if tools:
            for tool in tools:
                tool_name = tool.function.name
                tool_indices[tool_name] = True
        return tool_indices

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Parse incremental streaming text for TypeScript-style function calls.

        Design principle:
        - HarmonyParser maintains its internal buffer state across chunks via internal SemanticBuffer
        - GptOssDetector extracts tool calls from HarmonyParser events
        - Normal content is accumulated from 'normal' events that appear BEFORE tool calls
        - We simply parse and return whatever HarmonyParser gives us

        HarmonyParser already tracks events that have been emitted via its SemanticBuffer.
        When we call HarmonyParser.parse(new_text), it:
        1. Appends new_text to its internal SemanticBuffer
        2. Parses and returns NEW events from accumulated buffer
        3. Tracks returned events in its internal _emitted_events list
        4. Sets its internal _buffer to remaining unparsed text (for future parsing)
        """
        # Initialize tool indices once (reuse across streaming calls)
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        # Parse with HarmonyParser - it manages all buffer state
        events = self.harmony_parser.parse(new_text)

        # Accumulate normal text from events
        calls = []
        normal_text = ""
        # Process events in order
        for event in events:
            if event.event_type == "tool_call":
                # Extract tool call from event
                # Use raw_text for regex matching as it has to header format
                tool_call_info = self._extract_tool_call_from_event(
                    event.raw_text if event.raw_text else event.content,
                    self._tool_indices,
                    len(calls),  # Use current calls length as tool_index
                )
                if tool_call_info:
                    calls.append(tool_call_info)

            elif event.event_type == "normal":
                # Accumulate normal text from events
                normal_text += event.content
            # Ignore reasoning content (handled by ReasoningParser)

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _extract_tool_call_from_event(
        self, content: str, tool_indices: dict, tool_index: int
    ) -> Optional[ToolCallItem]:
        """
        Extract tool call information from HarmonyParser event content.

        Content format (with raw_text from HarmonyParser):
        "to=func_name<|message|>{...}"
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

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError("structure_info not used with HarmonyParser")
