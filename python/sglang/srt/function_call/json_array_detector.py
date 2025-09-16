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
        # If we previously withheld a valid tool-call separator, prepend it
        # when we see the next JSON object start (typically '{' inside array).
        if self._pending_separator:
            # Find first non-space to decide if an object starts now.
            i = 0
            while i < len(new_text) and new_text[i].isspace():
                i += 1
            if i < len(new_text) and new_text[i] == '{':
                # Re-insert the withheld separator before the new object.
                new_text = self.tool_call_separator + new_text
                self._pending_separator = False

        # Detect the configured tool call separator within this chunk.
        sep = self.tool_call_separator
        sep_index = new_text.find(sep)
        if sep_index != -1:
            # Candidate separator found. Validate that the content before it
            # closes a complete JSON object, using the accumulated buffer.
            json_part_original = new_text[:sep_index]

            # Build full JSON so far and exclude outer tokens the base strips.
            full_json = self._buffer + json_part_original
            if full_json.startswith(self.bot_token):
                full_json = full_json[len(self.bot_token):]
            if full_json.endswith(self.eot_token):
                full_json = full_json[:-len(self.eot_token)]

            try:
                json.loads(full_json)
                # It's a valid object followed by a separator.
                remainder = new_text[sep_index + len(sep):]
                if remainder:
                    # Strip leading whitespace from remainder before prepending separator
                    remainder_stripped = remainder.lstrip()
                    # Process the complete chunk with separator reinserted
                    return super().parse_streaming_increment(
                        json_part_original + sep + remainder_stripped, tools
                    )
                else:
                    # No remainder. Withhold the separator for the next chunk.
                    self._pending_separator = True
                    return super().parse_streaming_increment(json_part_original, tools)
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