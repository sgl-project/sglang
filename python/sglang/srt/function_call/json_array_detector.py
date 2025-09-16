from typing import List
import json

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult


class JsonArrayDetector(BaseFormatDetector):
    """
    Detector for JSON array tool calls when JSON schema constraints are active.
    
    This detector is used when tool_choice="required" or a specific tool is named,
    bypassing model-specific parsers in favor of direct JSON array parsing.
    """

    def __init__(self):
        super().__init__()
        # Configure for JSON array parsing
        self.bot_token = "["
        self.eot_token = "]"
        self.tool_call_separator = ","
        # Whether we detected a valid tool-call separator that was withheld
        # from the previous chunk and should be re-inserted before the next
        # tool call JSON object begins.
        self._pending_separator = False
        # Track brace count across streaming chunks
        self.brace_count = 0

    def _find_valid_separator(self, text: str) -> int:
        """
        Find the first valid separator by tracking brace count character by character.
        Updates self.brace_count as it processes the text.
        Returns the index of the separator or -1 if not found.
        """
        if not self.tool_call_separator:
            return -1
            
        for i in range(len(text)):
            if text[i] == '{':
                self.brace_count += 1
            elif text[i] == '}':
                self.brace_count -= 1
            elif text[i] == self.tool_call_separator and self.brace_count == 0:
                return i

        return -1


    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains a JSON tool call (array or single object).
        """        
        stripped = text.strip()
        return stripped.startswith("[") 

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Parse JSON tool calls using the base class implementation.
        """
        raise NotImplementedError("Detect and parse not supported for JSON schema constraints.")

    def build_ebnf(self, tools: List[Tool]) -> str:
        """
        Build an EBNF grammar for constrained generation.
        This is not used for JSON schema constraints as they are handled
        by the constraint backends directly.
        """
        raise NotImplementedError("EBNF generation is not supported for JSON schema constraints.")

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Parse JSON array tool calls with improved separator detection.
        Uses brace counting to find valid separators that are outside JSON objects.
        """
        # If we previously withheld a valid tool-call separator, prepend it
        # when we see the next JSON object start
        if self._pending_separator:
            # Check if the text (after removing leading whitespace) starts with '{'
            if new_text.lstrip().startswith('{'):
                # Re-insert the withheld separator before the new object.
                new_text = self.tool_call_separator + new_text
                self._pending_separator = False
                # Process the original text (without the prepended separator) for separator detection
                sep_index = self._find_valid_separator(new_text[1:])
                if sep_index != -1:
                    sep_index += 1  # Adjust for the prepended separator
            else:
                # No new object starting, just process normally
                sep_index = self._find_valid_separator(new_text)
        else:
            # Detect the configured tool call separator within this chunk.
            sep_index = self._find_valid_separator(new_text)
            
        if sep_index > 0:  # Ensure separator is not at position 0
            # Candidate separator found. Validate that the content before it
            # closes a complete JSON object, using the accumulated buffer.
            json_part = new_text[:sep_index]

            # Build full JSON so far and exclude outer tokens the base strips.
            full_json = self._buffer + json_part
            if full_json.startswith(self.bot_token):
                full_json = full_json[len(self.bot_token):]
            elif full_json.startswith(self.tool_call_separator):
                full_json = full_json[len(self.tool_call_separator):]
            if full_json.endswith(self.eot_token):
                full_json = full_json[:-len(self.eot_token)]

            try:
                json.loads(full_json)
                # It's a valid object followed by a separator.
                remainder = new_text[sep_index + len(self.tool_call_separator):]
                if remainder:
                    # There's more text after the separator - process normally
                    return super().parse_streaming_increment(new_text, tools)
                else:
                    # No remainder - separator was at end of chunk
                    # Withhold the separator for the next chunk
                    self._pending_separator = True
                    return super().parse_streaming_increment(json_part, tools)
            except json.JSONDecodeError:
                # Not a complete object yet; fall through to base handling.
                pass

        # Default: delegate to base implementation with unmodified text.
        return super().parse_streaming_increment(new_text, tools)

    def structure_info(self) -> callable:
        """
        Return a function that creates StructureInfo for constrained generation.
        This is not used for JSON schema constraints as they are handled
        by the constraint backends directly.
        """
        raise NotImplementedError("structure_info not used for JSON schema constraints")