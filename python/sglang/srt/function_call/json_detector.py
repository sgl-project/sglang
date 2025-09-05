import json
import logging
from typing import List

from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult, ToolCallItem
from sglang.srt.function_call.utils import _partial_json_loads, _is_complete_json

logger = logging.getLogger(__name__)


class JsonDetector(BaseFormatDetector):
    """
    Detector for JSON array tool calls when JSON schema constraints are active.
    
    This detector is used when tool_choice="required" or a specific tool is named,
    bypassing model-specific parsers in favor of direct JSON array parsing.
    """

    def __init__(self):
        super().__init__()
        # Configure for JSON array parsing
        self.bot_token = ""
        self.eot_token = ""
        self.tool_call_separator = ", "

    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains a JSON tool call (array or single object).
        
        Args:
            text: The text to check for tool calls
            
        Returns:
            True if the text starts with a JSON array or object, False otherwise
        """
        stripped = text.strip()
        return stripped.startswith("[") or stripped.startswith("{")

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Parse JSON tool calls (array or single object) in one go.
        
        Args:
            text: The text to parse
            tools: List of available tools
            
        Returns:
            StreamingParseResult with parsed tool calls
        """
        try:
            # Parse JSON
            tool_call_data = json.loads(text)
            
            # Handle both arrays and single objects
            if isinstance(tool_call_data, list):
                # Array format: [{"name": "func", "parameters": {...}}, ...]
                call_list = tool_call_data
            elif isinstance(tool_call_data, dict) and "name" in tool_call_data:
                # Single object format: {"name": "func", "parameters": {...}}
                call_list = [tool_call_data]
            else:
                return StreamingParseResult()
            
            calls = []
            for i, call_info in enumerate(call_list):
                if isinstance(call_info, dict) and "name" in call_info:
                    # Convert parameters to arguments for consistency
                    arguments = call_info.get("parameters", call_info.get("arguments", {}))
                    calls.append(ToolCallItem(
                        name=call_info["name"],
                        parameters=json.dumps(arguments, ensure_ascii=False),
                        tool_index=i
                    ))
            
            return StreamingParseResult(calls=calls)
        except (json.JSONDecodeError, KeyError, TypeError):
            return StreamingParseResult()

    def build_ebnf(self, tools: List[Tool]) -> str:
        """
        Build an EBNF grammar for constrained generation.
        This is not used for JSON schema constraints as they are handled
        by the constraint backends directly.
        """
        return ""

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Simple streaming parser for JSON tool calls.
        
        Handles both single JSON objects and JSON arrays directly.
        
        Args:
            new_text: The new chunk of text to parse
            tools: List of available tools
            
        Returns:
            StreamingParseResult with parsed tool calls
        """
        # Append new text to buffer
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have a tool call (JSON array or object)
        if not self.has_tool_call(current_text):
            # No tool call detected, return as normal text
            normal_text = self._buffer
            self._buffer = ""
            return StreamingParseResult(normal_text=normal_text) if normal_text else StreamingParseResult()

        # Build tool indices if not already built
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        # Try to parse as complete JSON first
        try:
            if current_text.strip().endswith('}') or current_text.strip().endswith(']'):
                tool_call_data = json.loads(current_text)
                
                # Handle both arrays and single objects
                if isinstance(tool_call_data, list):
                    # Array format: [{"name": "func", "parameters": {...}}, ...]
                    call_list = tool_call_data
                elif isinstance(tool_call_data, dict) and "name" in tool_call_data:
                    # Single object format: {"name": "func", "parameters": {...}}
                    call_list = [tool_call_data]
                else:
                    return StreamingParseResult()
                
                calls = []
                for i, call_info in enumerate(call_list):
                    if isinstance(call_info, dict) and "name" in call_info:
                        # Validate tool name
                        if call_info["name"] not in self._tool_indices:
                            return StreamingParseResult()
                        
                        # Convert parameters to arguments
                        arguments = call_info.get("parameters", call_info.get("arguments", {}))
                        calls.append(ToolCallItem(
                            name=call_info["name"],
                            parameters=json.dumps(arguments, ensure_ascii=False),
                            tool_index=i
                        ))
                
                # Clear buffer and return calls
                self._buffer = ""
                return StreamingParseResult(calls=calls)
        except json.JSONDecodeError:
            pass

        # For partial JSON, buffer until complete
        return StreamingParseResult()

    def structure_info(self) -> callable:
        """
        Return a function that creates StructureInfo for constrained generation.
        This is not used for JSON schema constraints as they are handled
        by the constraint backends directly.
        """
        def _get_info(tool_name: str):
            return None
        return _get_info