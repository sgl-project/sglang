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
        Check if the given text contains a JSON array tool call.
        
        Args:
            text: The text to check for tool calls
            
        Returns:
            True if the text starts with a JSON array, False otherwise
        """
        return text.strip().startswith("[")

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Parse JSON array tool calls in one go.
        
        Args:
            text: The text to parse
            tools: List of available tools
            
        Returns:
            StreamingParseResult with parsed tool calls
        """
        try:
            # Parse JSON array
            tool_call_data = json.loads(text)
            if not isinstance(tool_call_data, list):
                return StreamingParseResult()
            
            calls = []
            for i, call_info in enumerate(tool_call_data):
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
        Custom streaming parser for JSON arrays.
        
        This method handles streaming JSON arrays like [{"name": "func1", "parameters": {...}}, ...]
        by parsing them incrementally as they arrive.
        
        Args:
            new_text: The new chunk of text to parse
            tools: List of available tools
            
        Returns:
            StreamingParseResult with parsed tool calls
        """
        # Append new text to buffer
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have a tool call (JSON array)
        if not self.has_tool_call(current_text):
            # No tool call detected, return as normal text
            normal_text = self._buffer
            self._buffer = ""
            return StreamingParseResult(normal_text=normal_text) if normal_text else StreamingParseResult()

        # Build tool indices if not already built
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        try:
            # Try to parse as complete JSON array first
            if current_text.strip().startswith('[') and current_text.strip().endswith(']'):
                try:
                    tool_call_data = json.loads(current_text)
                    if isinstance(tool_call_data, list):
                        # Complete array - parse all elements
                        calls = []
                        for i, call_info in enumerate(tool_call_data):
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
                        
                        # Clear buffer and return all calls
                        self._buffer = ""
                        return StreamingParseResult(calls=calls)
                except json.JSONDecodeError:
                    pass

            # Handle partial array streaming
            # Parse incrementally to get partial results
            try:
                (obj, end_idx) = _partial_json_loads(current_text, Allow.ALL)
                
                # Check if we got a complete array
                if isinstance(obj, list) and _is_complete_json(current_text[:end_idx]):
                    calls = []
                    for i, call_info in enumerate(obj):
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
                    
                    # Update buffer to remove parsed content
                    self._buffer = current_text[end_idx:]
                    return StreamingParseResult(calls=calls)
                
                # Partial array - return empty result, keep buffering
                return StreamingParseResult()
                
            except MalformedJSON:
                # Invalid JSON - return as normal text
                normal_text = self._buffer
                self._buffer = ""
                return StreamingParseResult(normal_text=normal_text)

        except Exception as e:
            logger.error(f"Error in JSON array streaming parser: {e}")
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