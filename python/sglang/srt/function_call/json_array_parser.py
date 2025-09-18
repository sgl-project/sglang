import json
import re
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
        Streaming logic:
        - If a pending separator exists and new_text isn't just whitespace, prepend a comma and clear the flag.
        - If new_text ends with "}," allowing whitespace between '}' and ',' and at end, remove the comma and set pending for next chunk.
        - Delegate to base implementation with the modified text.
        """
        # Apply pending separator if there is non-whitespace content in this chunk
        if self._pending_separator and new_text.strip():
            new_text = "," + new_text
            self._pending_separator = False

        # Detect a trailing "}," with any whitespace between '}' and ',' and at the end
        if re.search(r"}\s*,\s*$", new_text):
            # Remove the trailing comma but preserve surrounding whitespace
            new_text = re.sub(r"}(\s*),(\s*)$", r"}\1\2", new_text, count=1)
            # Mark pending separator for the next non-whitespace chunk
            self._pending_separator = True

        return super().parse_streaming_increment(new_text, tools)

    def structure_info(self) -> callable:
        """
        Return a function that creates StructureInfo for constrained generation.
        This is not used for JSON schema constraints as they are handled
        by the constraint backends directly.
        """
        raise NotImplementedError("structure_info not used for JSON schema constraints")
