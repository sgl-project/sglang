import json
import logging
from typing import List

from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import (
    _find_common_prefix,
    _partial_json_loads,
)

logger = logging.getLogger(__name__)


class InternlmDetector(BaseFormatDetector):
    """
    Detector for InternLM2/Intern-S1 model function call format.

    The InternLM format uses special tokens to delimit function calls
    with JSON for arguments.

    Format Structure:
    ```
    text<|action_start|><|plugin|>
    {json}<|action_end|>
    ```

    Examples:
    ```
    What's the weather like?<|action_start|><|plugin|>
    {"name": "get_weather", "parameters": {"location": "Tokyo"}}<|action_end|>
    ```

    Key Components:
    - Tool Call Start: `<|action_start|><|plugin|>\n`
    - Tool Call End: `<|action_end|>`
    - Arguments: JSON object with `name` and `parameters`/`arguments`
    - Only supports single tool call at a time (no parallel tool calls)

    Reference: https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/serve/openai/tool_parsers/internlm.py
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<|action_start|><|plugin|>\n"
        self.eot_token = "<|action_end|>"
        self.position = 0

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains an InternLM format tool call."""
        return "<|action_start|>" in text

    def get_arguments(self, obj):
        """Extract arguments from object, supporting both 'parameters' and 'arguments' keys."""
        if "parameters" in obj:
            return obj.get("parameters")
        elif "arguments" in obj:
            return obj.get("arguments")
        return None

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: StreamingParseResult with normal text and parsed tool calls.
        """
        if "<|action_start|><|plugin|>" not in text:
            return StreamingParseResult(normal_text=text, calls=[])

        try:
            # Split text into content and action
            parts = text.split("<|action_start|><|plugin|>")
            normal_text = parts[0].strip()
            
            if len(parts) < 2:
                return StreamingParseResult(normal_text=text, calls=[])
            
            action_part = parts[1]
            
            # Extract action JSON (remove end token if present)
            action = action_part.split("<|action_end|>")[0].strip()
            
            # Parse the JSON
            action_dict = json.loads(action)
            name = action_dict.get("name")
            parameters = self.get_arguments(action_dict)
            
            if not parameters:
                parameters = {}
            
            # Build tool indices for validation
            tool_indices = self._get_tool_indices(tools)
            
            # Validate tool name
            if not (name and name in tool_indices):
                logger.warning(f"Model attempted to call undefined function: {name}")
                if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                    return StreamingParseResult(normal_text=text, calls=[])
            
            # Create tool call item
            tool_call = ToolCallItem(
                tool_index=0,
                name=name,
                parameters=json.dumps(parameters, ensure_ascii=False),
            )
            
            return StreamingParseResult(normal_text=normal_text, calls=[tool_call])
            
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            return StreamingParseResult(normal_text=text, calls=[])

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for InternLM format.
        
        Handles the special case where InternLM only generates one tool call at a time.
        """
        self._buffer += new_text
        current_text = self._buffer
        
        # If no tool call marker, return as normal text
        if "<|action_start|>" not in current_text:
            # Only clear buffer if we're sure no tool call is starting
            if not self._ends_with_partial_token(self._buffer, "<|action_start|>"):
                normal_text = self._buffer
                self._buffer = ""
                if self.eot_token in normal_text:
                    normal_text = normal_text.replace(self.eot_token, "")
                return StreamingParseResult(normal_text=normal_text)
            else:
                # Might be partial start token, keep buffering
                return StreamingParseResult()
        
        # If the tool call has already been sent, return empty delta
        # to make sure the finish_reason will be sent correctly
        if self.current_tool_id > 0:
            return StreamingParseResult(content="")
        
        # Check if we have the full plugin marker
        if "<|action_start|><|plugin|>\n" not in current_text:
            return StreamingParseResult()
        
        # Split into text and action
        parts = current_text.split("<|action_start|><|plugin|>\n")
        text_part = parts[0]
        
        # If we have text before the action, send it
        if text_part and len(text_part) > self.position:
            sent_text = text_part[self.position:]
            self.position = len(text_part)
            return StreamingParseResult(normal_text=sent_text)
        
        # Now parse the action part
        if len(parts) < 2:
            return StreamingParseResult()
        
        action = parts[1].strip()
        action = action.split("<|action_end|>")[0]
        
        # Build tool indices if not already built
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)
        
        # Bit mask flags for partial JSON parsing
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        
        try:
            # Try to parse as JSON
            try:
                tool_call_obj = _partial_json_loads(action, flags)[0]
            except MalformedJSON:
                logger.debug("Not enough tokens to parse into JSON yet")
                return StreamingParseResult()
            
            if not tool_call_obj:
                return StreamingParseResult()
            
            # Case 1: Send tool name if not sent yet
            if not self.current_tool_name_sent:
                function_name = tool_call_obj.get("name")
                
                if function_name:
                    # Validate tool name
                    if function_name not in self._tool_indices:
                        logger.warning(f"Model attempted to call undefined function: {function_name}")
                        if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                            return StreamingParseResult()
                    
                    # Initialize tool tracking
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.streamed_args_for_tool.append("")
                    
                    # Send tool name with empty parameters
                    self.current_tool_name_sent = True
                    return StreamingParseResult(
                        calls=[
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=function_name,
                                parameters="",
                            )
                        ]
                    )
                else:
                    return StreamingParseResult()
            
            # Case 2: Stream arguments
            else:
                cur_arguments = self.get_arguments(tool_call_obj)
                
                # No arguments generated yet
                if not cur_arguments:
                    if not self.prev_tool_call_arr:
                        return StreamingParseResult()
                    prev_arguments = self.get_arguments(self.prev_tool_call_arr[0]) if self.prev_tool_call_arr else None
                    if not prev_arguments:
                        return StreamingParseResult()
                
                # First time getting parameters
                if cur_arguments and (not self.prev_tool_call_arr or not self.get_arguments(self.prev_tool_call_arr[0])):
                    cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)
                    
                    # Send all arguments accumulated so far
                    arguments_delta = cur_arguments_json
                    self.streamed_args_for_tool[self.current_tool_id] += arguments_delta
                    
                    # Update prev_tool_call_arr
                    tool_call_obj["arguments"] = cur_arguments
                    self.prev_tool_call_arr = [tool_call_obj]
                    
                    return StreamingParseResult(
                        calls=[
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                parameters=arguments_delta,
                            )
                        ]
                    )
                
                # Both prev and cur parameters exist, send the increase
                elif cur_arguments:
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_arguments = self.get_arguments(self.prev_tool_call_arr[0])
                    prev_args_json = json.dumps(prev_arguments, ensure_ascii=False) if prev_arguments else ""
                    
                    # Calculate the common prefix and send the diff
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    prefix = _find_common_prefix(prev_args_json, cur_args_json)
                    argument_diff = prefix[sent:]
                    
                    if argument_diff:
                        self.streamed_args_for_tool[self.current_tool_id] += argument_diff
                        
                        # Update prev_tool_call_arr
                        tool_call_obj["arguments"] = cur_arguments
                        self.prev_tool_call_arr = [tool_call_obj]
                        
                        return StreamingParseResult(
                            calls=[
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    parameters=argument_diff,
                                )
                            ]
                        )
                
                return StreamingParseResult()
                
        except Exception as e:
            logger.exception(f"Error trying to handle streaming tool call: {e}")
            logger.debug("Skipping chunk as a result of tool streaming extraction error")
            return StreamingParseResult()

    def structure_info(self) -> _GetInfoFunc:
        """
        Return structure information for constrained generation.
        
        For InternLM format, the structure is:
        - begin: <|action_start|><|plugin|>\n
        - end: <|action_end|>
        - trigger: the begin token
        """
        return lambda name: StructureInfo(
            begin='<|action_start|><|plugin|>\n{"name": "' + name + '", "parameters": ',
            end="}<|action_end|>",
            trigger="<|action_start|><|plugin|>",
        )

