import json
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult


class JsonArrayParser(BaseFormatDetector):
    """
    Parser for JSON array tool calls when JSON schema constraints are active.

    This parser is used when tool_choice="required" or a specific tool is named,
    bypassing model-specific parsers in favor of direct JSON array parsing.
    """

    def __init__(self):
        super().__init__()
        # Configure for JSON array parsing
        self.bot_token = "["
        self.eot_token = "]"
        self.tool_call_separator = ","
        self._pending_separator = False

    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains a JSON tool call (array or single object).
        """
        return "[" in text or "{" in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Parse JSON tool calls using the base class implementation.
        """
        raise NotImplementedError(
            "Detect and parse not supported for JSON schema constraints."
        )

    def build_ebnf(self, tools: List[Tool]) -> str:
        """
        Build an EBNF grammar for constrained generation.
        This is not used for JSON schema constraints as they are handled
        by the constraint backends directly.
        """
        raise NotImplementedError(
            "EBNF generation is not supported for JSON schema constraints."
        )

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Parse JSON array tool calls using json.loads() validation.
        Checks the last comma to find valid JSON boundaries.
        """
        # If we receive the whole array at once, remove the array brackets.
        # Otherwise the base class only removes '[' at the start and crashes because of the trailing ']'.
        if new_text.strip().startswith(self.bot_token) and new_text.strip().endswith(self.eot_token):
            new_text = new_text.lstrip(self.bot_token).rstrip(self.eot_token)
            return super().parse_streaming_increment(new_text, tools)
        # Handle pending separator from previous chunk
        if self._pending_separator:
            if new_text.strip().startswith("{"):
                new_text = self.tool_call_separator + new_text
                self._pending_separator = False
        
        last_comma_pos = new_text.rfind(self.tool_call_separator)
        if last_comma_pos > 0:
            # Found a potential separator, check if content before it is valid JSON
            combined = self._buffer + new_text[:last_comma_pos]
            start_pos = combined.find(self.bot_token) + 1
            prev_tool_call = combined[start_pos:]
            
            # Remove leading comma from prev_tool_call if present
            if prev_tool_call.lstrip().startswith(self.tool_call_separator):
                prev_tool_call = prev_tool_call.lstrip()[len(self.tool_call_separator):]
            
            remainder = new_text[last_comma_pos + 1:].strip()
            
            try:
                # Try to parse the JSON part
                json.loads(prev_tool_call)
                # Valid JSON found before separator
                
                if remainder:
                    # There's more content after the separator
                    return super().parse_streaming_increment(new_text, tools)
                else:
                    # Separator at end of chunk, set pending separator and process text without separator
                    self._pending_separator = True
                    # Remove the trailing comma from new_text before passing to base class
                    return super().parse_streaming_increment(new_text[:last_comma_pos], tools)
            except json.JSONDecodeError:
                # Not valid JSON yet, delegate to base implementation
                pass
        
        # No valid separator found, delegate to base implementation
        return super().parse_streaming_increment(new_text, tools)

    def structure_info(self) -> callable:
        """
        Return a function that creates StructureInfo for constrained generation.
        This is not used for JSON schema constraints as they are handled
        by the constraint backends directly.
        """
        raise NotImplementedError("structure_info not used for JSON schema constraints")
