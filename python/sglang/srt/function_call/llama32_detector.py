import json
import logging
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    _GetInfoFunc,
)
from sglang.srt.function_call.json_schema_composer import JSONSchemaComposer
from sglang.srt.function_call.ebnf_composer import EBNFComposer

logger = logging.getLogger(__name__)


class Llama32Detector(BaseFormatDetector):
    """
    Detector for Llama 3.2 models with json tool call format.

    Format Structure:
    ```
    <python_tag>{"name":"xxx", "arguments":{...}}
    ```
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<|python_tag|>"
        # NOTE: technically Llama3.2 doesn't support well with parallel tool calls
        # They need specific prompt engineering to support parallel tool calls
        # Here we use ';' as the separator, which might have compatibility issues
        # if users define to use a different separator in their prompt
        self.tool_call_separator = ";"

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Llama 3.2 format tool call."""
        # depending on the prompt format the Llama model may or may not
        # prefix the output with the <|python_tag|> token
        # Also check for JSON array format when using JSON schema constraints
        has_call = "<|python_tag|>" in text or text.startswith("{") or text.startswith("[")
        return has_call

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse function calls from text, handling multiple JSON objects."""
        if "<|python_tag|>" not in text and not text.startswith("{") and not text.startswith("["):
            return StreamingParseResult(normal_text=text, calls=[])

        if "<|python_tag|>" in text:
            normal_text, action_text = text.split("<|python_tag|>", maxsplit=1)
        else:
            normal_text, action_text = "", text

        decoder = json.JSONDecoder()
        idx = 0
        safe_idx = idx  # the index of the last valid JSON object
        all_actions = []
        action_text_len = len(action_text)
        
        # Handle case where action_text is a complete JSON array or object
        if action_text.strip().startswith("[") or action_text.strip().startswith("{"):
            try:
                parsed_content = json.loads(action_text)
                if isinstance(parsed_content, list):
                    all_actions = parsed_content
                    safe_idx = len(action_text)
                else:
                    all_actions = [parsed_content]
                    safe_idx = len(action_text)
            except json.JSONDecodeError:
                # Fall back to incremental parsing
                pass
        
        # If we didn't get actions from array parsing, try incremental parsing
        if not all_actions:
            while idx < action_text_len:
                try:
                    obj, end = decoder.raw_decode(action_text[idx:])
                    all_actions.append(obj)
                    idx += end + len(self.tool_call_separator)
                    safe_idx = idx
                except json.JSONDecodeError as e:
                    # Find where next `{"name"` appears and try again
                    logger.warning(
                        f"Failed to parse JSON part: {action_text[idx:]}, JSON parse error: {str(e)}"
                    )
                    next_obj_start = action_text.find('{"name":', idx + 1)
                    if next_obj_start == -1:
                        break
                    idx = next_obj_start
                    continue

        # Validate and flatten all_actions to ensure it contains only objects, not arrays
        if all_actions:
            flattened_actions = []
            for action in all_actions:
                if isinstance(action, list):
                    # If we have a nested array, flatten it
                    flattened_actions.extend(action)
                else:
                    flattened_actions.append(action)
            all_actions = flattened_actions

        # Only process if we found valid JSON objects
        calls = self.parse_base_json(all_actions, tools) if all_actions else []
        
        # Use safe_idx to avoid idx containing the last part of an invalid JSON object
        trailing_text = (
            action_text[safe_idx:].strip() if safe_idx < action_text_len else ""
        )
        return StreamingParseResult(
            normal_text=normal_text + trailing_text, calls=calls
        )

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<|python_tag|>{"name":"' + name + '", "arguments":',
            end="}",
            trigger="<|python_tag|>",
        )

    def build_json_schema(self, tools: List[Tool]):
        return JSONSchemaComposer.build_json_schema(
            tools,
            tool_choice="required",
            tool_call_separator=self.tool_call_separator,
        )
    
    def build_ebnf(self, tools: List[Tool]):
        # Keep for backward compatibility
        return EBNFComposer.build_ebnf(
            tools,
            function_format="json",
            tool_call_separator=self.tool_call_separator,
        )
