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
        Wrapper for base format detector to handle JSON array specifics. To prevent partial json loader crash,
        we withhold the tool call separator until we see the next JSON object start.
        """
        # If we previously withheld a valid tool-call separator, prepend it
        # when we see the next JSON object start
        if self._pending_separator:
            # Check if the text (after removing leading whitespace) starts with '{'
            if new_text.lstrip().startswith('{'):
                # Re-insert the withheld separator before the new object.
                new_text = self.tool_call_separator + new_text
                self._pending_separator = False

        # Detect the configured tool call separator within this chunk.
        sep_index = self._find_valid_separator(new_text)
        if sep_index != -1 and sep_index > 0:  # Ensure separator is not at position 0
            # Candidate separator found. Validate that the content before it
            # closes a complete JSON object, using the accumulated buffer.
            json_part = new_text[:sep_index]

            # Build full JSON so far and exclude outer tokens the base strips.
            full_json = self._buffer + json_part
            if full_json.startswith(self.bot_token):
                full_json = full_json[len(self.bot_token):]
            if full_json.endswith(self.eot_token):
                full_json = full_json[:-len(self.eot_token)]

            try:
                json.loads(full_json)
                # It's a valid object followed by a separator.
                remainder = new_text[sep_index + len(self.tool_call_separator):]
                if remainder:
                    # Strip leading whitespace from remainder before prepending separator
                    remainder_stripped = remainder.lstrip()
                    # Process the complete chunk with separator reinserted
                    return super().parse_streaming_increment(
                        json_part + self.tool_call_separator + remainder_stripped, tools
                    )
                else:
                    # No remainder. Withhold the separator for the next chunk.
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