import ast
import html
import json
import logging
import re
from typing import Any, Dict, List, Tuple

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.ebnf_composer import EBNFComposer

logger = logging.getLogger(__name__)


def _safe_val(raw: str) -> Any:
    raw = html.unescape(raw.strip())
    try:
        return json.loads(raw)
    except Exception:
        try:
            return ast.literal_eval(raw)
        except Exception:
            return raw


class Qwen3CoderDetector(BaseFormatDetector):
    """
    Detector for Qwen 3 models.
    Assumes function call format:
        <tool_call>
        <function=execute_bash>
        <parameter=command>
        pwd && ls
        </parameter>
        </function>
        </tool_call>
    """

    def __init__(self):
        super().__init__()
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)</parameter>|<parameter=(.*?)$", re.DOTALL
        )
        self._buf: str = ""
        
        # Streaming state variables
        self._current_function_name: str = ""
        self._current_parameters: Dict[str, Any] = {}
        self._streamed_parameters: Dict[str, str] = {}  # Track what parameter content we've streamed
        self._in_tool_call: bool = False
        self._function_name_sent: bool = False

    def _get_tool_indices(self, tools: List[Tool]) -> Dict[str, int]:
        """Get a mapping of tool names to their indices in the tools list."""
        return {
            tool.function.name: i for i, tool in enumerate(tools) if tool.function.name
        }

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        normal, calls = self._extract(text, tools)
        return StreamingParseResult(normal_text=normal, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buf += new_text
        normal = ""
        calls: List[ToolCallItem] = []
        
        # Build tool indices for validation
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)
        
        while True:
            # If we're not in a tool call and don't see a start token, return normal text
            if not self._in_tool_call and self.tool_call_start_token not in self._buf:
                normal += self._buf
                self._buf = ""
                break
                
            # Look for tool call start
            if not self._in_tool_call:
                s = self._buf.find(self.tool_call_start_token)
                if s == -1:
                    normal += self._buf
                    self._buf = ""
                    break
                    
                if s > 0:
                    normal += self._buf[:s]
                    self._buf = self._buf[s:]
                    
                self._in_tool_call = True
                self._function_name_sent = False
                self._current_function_name = ""
                self._current_parameters = {}
                self._streamed_parameters = {}
                
                # Remove the start token
                self._buf = self._buf[len(self.tool_call_start_token):]
                continue
            
            # We're in a tool call, try to parse function name if not sent yet
            if not self._function_name_sent:
                # Look for function name pattern: <function=name>
                function_match = re.search(r'<function=([^>]+)>', self._buf)
                if function_match:
                    function_name = function_match.group(1).strip()
                    
                    # Validate function name
                    if function_name in self._tool_indices:
                        self._current_function_name = function_name
                        self._function_name_sent = True
                        
                        # Initialize tool call tracking
                        if self.current_tool_id == -1:
                            self.current_tool_id = 0
                        
                        # Ensure tracking arrays are large enough
                        while len(self.prev_tool_call_arr) <= self.current_tool_id:
                            self.prev_tool_call_arr.append({})
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")
                            
                        # Store tool call info
                        self.prev_tool_call_arr[self.current_tool_id] = {
                            "name": function_name,
                            "arguments": {}
                        }
                        
                        # Send tool name with empty parameters
                        calls.append(ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=function_name,
                            parameters="",
                        ))
                        
                        # Remove the processed function declaration
                        self._buf = self._buf[function_match.end():]
                        continue
                    else:
                        # Invalid function name, reset state
                        logger.warning(f"Invalid function name: {function_name}")
                        self._reset_streaming_state()
                        normal += self._buf
                        self._buf = ""
                        break
                else:
                    # Function name not complete yet, wait for more text
                    break
            
            # Parse parameters incrementally
            if self._function_name_sent:
                # Only look for complete parameter patterns first
                param_matches = list(re.finditer(r'<parameter=([^>]+)>(.*?)</parameter>', self._buf, re.DOTALL))
                
                new_params = {}
                for match in param_matches:
                    param_name = match.group(1).strip()
                    param_value = match.group(2)
                    new_params[param_name] = _safe_val(param_value)
                
                # Calculate parameter diff to stream
                if new_params != self._current_parameters:
                    current_args_json = json.dumps(new_params, ensure_ascii=False)
                    
                    # Calculate what to send based on what we've already streamed
                    sent_length = len(self.streamed_args_for_tool[self.current_tool_id])
                    
                    if len(current_args_json) > sent_length:
                        argument_diff = current_args_json[sent_length:]
                        
                        calls.append(ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=argument_diff,
                        ))
                        
                        self.streamed_args_for_tool[self.current_tool_id] += argument_diff
                    
                    self._current_parameters = new_params
                    self.prev_tool_call_arr[self.current_tool_id]["arguments"] = new_params
                
                # Check if tool call is complete
                if self.tool_call_end_token in self._buf:
                    end_pos = self._buf.find(self.tool_call_end_token)
                    
                    # Process any remaining complete parameters before the end token
                    before_end = self._buf[:end_pos]
                    final_matches = list(re.finditer(r'<parameter=([^>]+)>(.*?)</parameter>', before_end, re.DOTALL))
                    
                    final_params = {}
                    for match in final_matches:
                        param_name = match.group(1).strip()
                        param_value = match.group(2)
                        final_params[param_name] = _safe_val(param_value)
                    
                    if final_params != self._current_parameters:
                        final_args_json = json.dumps(final_params, ensure_ascii=False)
                        sent_length = len(self.streamed_args_for_tool[self.current_tool_id])
                        
                        if len(final_args_json) > sent_length:
                            final_diff = final_args_json[sent_length:]
                            calls.append(ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=None,
                                parameters=final_diff,
                            ))
                    
                    # Complete the tool call
                    self._buf = self._buf[end_pos + len(self.tool_call_end_token):]
                    self._reset_streaming_state()
                    self.current_tool_id += 1
                    continue
                else:
                    # Tool call not complete yet, wait for more text
                    break
        
        return StreamingParseResult(normal_text=normal, calls=calls)

    def _reset_streaming_state(self):
        """Reset streaming state for the next tool call"""
        self._in_tool_call = False
        self._function_name_sent = False
        self._current_function_name = ""
        self._current_parameters = {}
        self._streamed_parameters = {}
        self.current_tool_name_sent = False


    def _extract(self, text: str, tools: List[Tool]) -> Tuple[str, List[ToolCallItem]]:
        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []
        cursor = 0
        while True:
            s = text.find(self.tool_call_start_token, cursor)
            if s == -1:
                normal_parts.append(text[cursor:])
                break
            normal_parts.append(text[cursor:s])
            e = text.find(self.tool_call_end_token, s)
            if e == -1:
                normal_parts.append(text[s:])
                break
            block = text[s : e + len(self.tool_call_end_token)]
            cursor = e + len(self.tool_call_end_token)
            calls.extend(self._parse_block(block, tools))
        return "".join(normal_parts), calls

    def _parse_block(self, block: str, tools: List[Tool]) -> List[ToolCallItem]:
        res: List[ToolCallItem] = []
        for m in self.tool_call_function_regex.findall(block):
            txt = m[0] if m[0] else m[1]
            if ">" not in txt:
                continue
            idx = txt.index(">")
            fname = txt[:idx].strip()
            body = txt[idx + 1 :]
            params: Dict[str, Any] = {}
            for pm in self.tool_call_parameter_regex.findall(body):
                ptxt = pm[0] if pm[0] else pm[1]
                if ">" not in ptxt:
                    continue
                pidx = ptxt.index(">")
                pname = ptxt[:pidx].strip()
                pval = ptxt[pidx + 1 :].lstrip("\n").rstrip("\n")
                params[pname] = _safe_val(pval)
            raw = {"name": fname, "arguments": params}
            try:
                # TODO: fix idx in function call, the index for a function
                # call will always be -1 in parse_base_json
                res.extend(self.parse_base_json(raw, tools))
            except Exception:
                logger.warning("invalid tool call for %s dropped", fname)
        return res

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def build_ebnf(self, tools: List[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            individual_call_start_token=self.tool_call_start_token.replace("\n", "\\n"),
            individual_call_end_token=self.tool_call_end_token.replace("\n", "\\n"),
            tool_call_separator="\\n",
            function_format="xml",
            call_rule_fmt='"<function={name}>\\n" {arguments_rule} "\\n</function>"',
            key_value_rule_fmt='"<parameter={key}>\\n" {valrule} "\\n</parameter>"',
        )
