import ast
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from json import JSONDecodeError, JSONDecoder
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Type, Union

import partial_json_parser
from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow
from pydantic import BaseModel

from sglang.srt.openai_api.protocol import (
    StructuralTagResponseFormat,
    StructuresResponseFormat,
    Tool,
    ToolChoice,
)

logger = logging.getLogger(__name__)

TOOLS_TAG_LIST = [
    "<|plugin|>",
    "<function=",
    "<tool_call>",
    "<|python_tag|>",
    "[TOOL_CALLS]",
    "<｜tool▁calls▁begin｜>",
]


class ToolCallItem(BaseModel):
    """Simple encapsulation of the parsed ToolCall result for easier usage in streaming contexts."""

    tool_index: int
    name: Optional[str] = None
    parameters: str  # JSON string


def _find_common_prefix(s1: str, s2: str) -> str:
    prefix = ""
    min_length = min(len(s1), len(s2))
    for i in range(0, min_length):
        if s1[i] == s2[i]:
            prefix += s1[i]
        else:
            break
    return prefix


def _partial_json_loads(input_str: str, flags: Allow) -> Tuple[Any, int]:
    try:
        return (partial_json_parser.loads(input_str, flags), len(input_str))
    except JSONDecodeError as e:
        if "Extra data" in e.msg:
            dec = JSONDecoder()
            return dec.raw_decode(input_str)
        raise


def _is_complete_json(input_str: str) -> bool:
    try:
        json.loads(input_str)
        return True
    except JSONDecodeError:
        return False


class StreamingParseResult:
    """Result of streaming incremental parsing."""

    def __init__(
        self, normal_text: str = "", calls: Optional[List[ToolCallItem]] = None
    ):
        self.normal_text = normal_text
        self.calls = calls or []


@dataclass
class StructureInfo:
    begin: str
    end: str
    trigger: str


_GetInfoFunc = Callable[[str], StructureInfo]
"""
Helper alias of function
Usually it is a function that takes a name string and returns a StructureInfo object,
which can be used to construct a structural_tag object
"""


class BaseFormatDetector(ABC):
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    def __init__(self):
        # initialize properties used for state when parsing tool calls in
        self._buffer = ""
        # streaming mode
        self.prev_tool_call_arr: List[Dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: List[str] = (
            []
        )  # map what has been streamed for each tool so far to a list
        self.bot_token = ""
        self.eot_token = ""

    def parse_base_json(self, action: Any, tools: List[Tool]) -> List[ToolCallItem]:
        tool_indices = {
            tool.function.name: i for i, tool in enumerate(tools) if tool.function.name
        }
        if not isinstance(action, list):
            action = [action]

        results = []
        for act in action:
            name = act.get("name")
            if name and name in tool_indices:
                results.append(
                    ToolCallItem(
                        tool_index=tool_indices[name],
                        name=name,
                        parameters=json.dumps(
                            act.get("parameters") or act.get("arguments", {}),
                            ensure_ascii=False,
                        ),
                    )
                )
            else:
                logger.warning(f"Model attempted to call undefined function: {name}")

        return results

    @abstractmethod
    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Parses the text in one go. Returns success=True if the format matches, otherwise False.
        Note that leftover_text here represents "content that this parser will not consume further".
        """
        action = json.loads(text)
        return StreamingParseResult(calls=self.parse_base_json(action, tools))

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing with tool validation.
        """
        # Append new text to buffer
        self._buffer += new_text
        current_text = self._buffer
        if not (self.bot_token in current_text or current_text.startswith("{")):
            self._buffer = ""
            if self.eot_token in new_text:
                new_text = new_text.replace(self.eot_token, "")
            return StreamingParseResult(normal_text=new_text)

        # Build tool indices if not already built
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = {
                tool.function.name: i
                for i, tool in enumerate(tools)
                if tool.function and tool.function.name
            }

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            tool_call_arr = []
            is_complete = []
            try:
                start_idx = (
                    len(self.bot_token)
                    if current_text.startswith(self.bot_token)
                    else 0
                )
                while start_idx < len(current_text):
                    (obj, end_idx) = _partial_json_loads(
                        current_text[start_idx:], flags
                    )
                    is_complete.append(
                        _is_complete_json(current_text[start_idx : start_idx + end_idx])
                    )
                    start_idx += end_idx + len("; ")

                    # Validate tool name if present
                    if "name" in obj and obj["name"] not in self._tool_indices:
                        # Invalid tool name - reset state
                        self._buffer = ""
                        self.current_tool_id = -1
                        self.current_tool_name_sent = False
                        if self.streamed_args_for_tool:
                            self.streamed_args_for_tool.pop()
                        return StreamingParseResult()

                    # Handle parameters/arguments consistency
                    if "parameters" in obj:
                        assert (
                            "arguments" not in obj
                        ), "model generated both parameters and arguments"
                        obj["arguments"] = obj["parameters"]
                    tool_call_arr.append(obj)

            except MalformedJSON:
                return StreamingParseResult()

            if len(tool_call_arr) == 0:
                return StreamingParseResult()

            current_tool_call: Dict = (
                tool_call_arr[self.current_tool_id] if len(tool_call_arr) > 0 else {}
            )

            # Handle new tool in array
            if len(tool_call_arr) > 0 and len(tool_call_arr) > self.current_tool_id + 1:
                if self.current_tool_id >= 0:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments:
                        cur_args_json = json.dumps(cur_arguments)
                        sent = len(self.streamed_args_for_tool[self.current_tool_id])
                        argument_diff = cur_args_json[sent:]

                        res = StreamingParseResult(
                            calls=[
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name="",
                                    parameters=argument_diff,
                                )
                            ],
                        )
                        self.streamed_args_for_tool[
                            self.current_tool_id
                        ] += argument_diff
                    else:
                        res = StreamingParseResult()
                else:
                    res = StreamingParseResult()

                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                return res

            # Handle tool name
            elif not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name and function_name in self._tool_indices:
                    res = StreamingParseResult(
                        calls=[
                            ToolCallItem(
                                tool_index=self._tool_indices[function_name],
                                name=function_name,
                                parameters="",
                            )
                        ],
                    )
                    self.current_tool_name_sent = True
                else:
                    res = StreamingParseResult()

            # Handle streaming arguments
            else:
                cur_arguments = current_tool_call.get("arguments")
                res = StreamingParseResult()

                if cur_arguments:
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments)
                    prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                        "arguments"
                    )

                    argument_diff = None
                    if is_complete[self.current_tool_id]:
                        argument_diff = cur_args_json[sent:]
                        self._buffer = ""
                        self.prev_tool_call_arr[self.current_tool_id].clear()
                        self.current_tool_name_sent = False
                        self.streamed_args_for_tool[self.current_tool_id] = ""

                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments)
                        if cur_args_json != prev_args_json:
                            prefix = _find_common_prefix(prev_args_json, cur_args_json)
                            argument_diff = prefix[sent:]

                    if argument_diff is not None:
                        res = StreamingParseResult(
                            calls=[
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    parameters=argument_diff,
                                )
                            ],
                        )
                        if not is_complete[self.current_tool_id]:
                            self.streamed_args_for_tool[
                                self.current_tool_id
                            ] += argument_diff

            self.prev_tool_call_arr = tool_call_arr
            return res

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult()

    @abstractmethod
    def has_tool_call(self, text: str) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()


class EBNFComposer:
    json_grammar_ebnf_str = r"""
        json ::= basic_array | basic_object
        basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
        basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
        basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
        basic_string ::= (([\"] basic_string_1 [\"]))
        basic_string_1 ::= "" | [^"\\\x00-\x1F] basic_string_1 | "\\" escape basic_string_1
        escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
        basic_boolean ::= "true" | "false"
        basic_null ::= "null"
        basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]"
        basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"
        ws ::= [ \n\t]*
        """

    pythonic_grammar_ebnf_str = r"""
        py_value ::= basic_number | basic_string | basic_array | "True" | "False" | "None"
        basic_any ::= basic_number | basic_string | basic_array | basic_object
        basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
        basic_string ::= (([\"] basic_string_1 [\"]))
        basic_string_1 ::= "" | [^"\\\x00-\x1F] basic_string_1 | "\\" escape basic_string_1
        escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
        basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]"
        basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"
        ws ::= [ \n\t]*
    """

    @staticmethod
    def get_value_rule(prop: dict, is_pythonic: bool) -> str:
        if "enum" in prop:
            return " | ".join([f'"{v}"' for v in prop["enum"]])
        return "py_value" if is_pythonic else "json"

    @staticmethod
    def build_ebnf(
        tools,
        *,
        tool_calls_rule: str,
        call_rule_fmt: str,
        arguments_rule_fmt: str,
        key_value_fmt: str,
        is_pythonic: bool = False,
    ):
        """
        Generalized EBNF builder for all detectors.
        Args:
            tools: list of Tool
            tool_calls_rule: the top-level rule string (e.g. "tool_calls ::= ...")
            call_rule_fmt: format string for call_{name} rule (expects {name}, {arguments_rule})
            arguments_rule_fmt: format string for arguments_{name} rule (expects {arg_rules})
            key_value_fmt: format for key-value pairs (expects {key}, {valrule})
            is_pythonic: if True, use pythonic value rules
        """
        lines = [
            "root ::= " + tool_calls_rule,
            "function_call ::= "
            + " | ".join([f"call_{t.function.name}" for t in tools]),
        ]
        for tool in tools:
            name = tool.function.name
            params = tool.function.parameters or {}
            props = list(params.get("properties", {}).keys())
            required = set(params.get("required", []))
            arg_rules = []
            for key in props:
                prop = params["properties"][key]
                valrule = EBNFComposer.get_value_rule(prop, is_pythonic)
                pair = key_value_fmt.format(key=key, valrule=valrule)
                if key not in required:
                    pair = f"[ {pair} ]"
                arg_rules.append(pair)
            arguments_rule = arguments_rule_fmt.format(
                arg_rules=(' "," '.join(arg_rules) if arg_rules else "")
            )
            lines.append(
                call_rule_fmt.format(name=name, arguments_rule=f"arguments_{name}")
            )
            lines.append(f"arguments_{name} ::= {arguments_rule}")

        if is_pythonic:
            lines.append(EBNFComposer.pythonic_grammar_ebnf_str)
        else:
            lines.append(EBNFComposer.json_grammar_ebnf_str)
        return "\n".join(lines)


class Qwen25Detector(BaseFormatDetector):
    """
    Detector for Qwen 2.5 models.
    Assumes function call format:
      <tool_call>{"name":"xxx", "arguments":{...}}</tool_call>
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Qwen 2.5 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        pattern = rf"{self.bot_token}(.*?){self.eot_token}"
        match_result_list = re.findall(pattern, text, re.DOTALL)
        calls = []
        for match_result in match_result_list:
            match_result = json.loads(match_result)
            calls.extend(self.parse_base_json(match_result, tools))
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<tool_call>{"name":"' + name + '", "arguments":',
            end="}</tool_call>",
            trigger="<tool_call>",
        )

    def build_ebnf(self, tools: List[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            # tool_calls_rule="tool_calls ::= ' ' function_call ' '",
            tool_calls_rule='"<tool_call>" function_call "</tool_call>"',
            call_rule_fmt='call_{name} ::= "{" "name" ":" "{name}" "," "arguments" ":" {arguments_rule} "}"',
            arguments_rule_fmt='"{" {arg_rules} "}"',
            key_value_fmt='"{key}" ":" {valrule}',
            is_pythonic=False,
        )


class MistralDetector(BaseFormatDetector):
    """
    Detector for Mistral models.
    Assumes function call format:
      [TOOL_CALLS] [{"name":"xxx", "arguments":{...}}]
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.bot_token = "[TOOL_CALLS] ["
        self.tool_call_regex = re.compile(r"\[{.*}\]", re.DOTALL)

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Mistral format tool call."""
        return self.bot_token in text

    def _clean_text(self, text: str) -> str:
        """
        clean text to only leave ''[TOOL_CALLS] [{"name": xxx, "arguments": {xxx}}]'
        for example,
            text = '[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}]\n\nToday\'s weather in Boston is :{function call result} (in Fahrenheit)\n\nIf you prefer Celsius, please let me know.'
            return '[TOOL_CALLS] [{"name": "get_current_weather", "arguments": {"location": "Boston, MA", "unit": "fahrenheit"}}]'
        The key pattern is [TOOL_CALLS] [...]
        """
        # TODO: check if Mistral supports multiple tool calls, currently assume only support one tool call
        find_results = re.findall(r"\[TOOL_CALLS\] \[.*?\]", text, re.DOTALL)
        if len(find_results) > 0:
            return find_results[0]
        else:
            return ""

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        text = self._clean_text(text)
        tool_content = text.replace("[TOOL_CALLS]", "").strip()
        raw_tool_calls = self.tool_call_regex.findall(tool_content)
        calls = []
        if len(raw_tool_calls) > 0:
            raw_tool_call = raw_tool_calls[0]
            function_call_arr = json.loads(raw_tool_call)
            for match_result in function_call_arr:
                calls.extend(self.parse_base_json(match_result, tools))
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='[TOOL_CALLS] [{"name":"' + name + '", "arguments":',
            end="}]",
            trigger="[TOOL_CALLS]",
        )

    def build_ebnf(self, tools: List[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            tool_calls_rule='"[TOOL_CALLS] [ " function_call ( " , " function_call )* " ]"',
            call_rule_fmt='call_{name} ::= "{" "name" ":" "{name}" "," "arguments" ":" {arguments_rule} "}"',
            arguments_rule_fmt='"{" {arg_rules} "}"',
            key_value_fmt='"{key}" ":" {valrule}',
            is_pythonic=False,
        )


class Llama32Detector(BaseFormatDetector):
    """
    Detector for Llama 3.2 models.
    Assumes function call format:
      <|python_tag|>{"name":"xxx", "arguments":{...}}
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<|python_tag|>"

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Llama 3.2 format tool call."""
        # depending on the prompt format the Llama model may or may not
        # prefix the output with the <|python_tag|> token
        return "<|python_tag|>" in text or text.startswith("{")

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse function calls from text, handling multiple JSON objects."""
        if "<|python_tag|>" not in text and not text.startswith("{"):
            return StreamingParseResult(normal_text=text, calls=[])

        if "<|python_tag|>" in text:
            normal_text, action_text = text.split("<|python_tag|>")
        else:
            normal_text, action_text = "", text

        # Split by semicolon and process each part
        json_parts = [part.strip() for part in action_text.split(";") if part.strip()]
        all_actions = []
        for part in json_parts:
            try:
                # Parse each individual JSON object
                action = json.loads(part)
                all_actions.append(action)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON part: {part}")
                logger.warning(f"JSON parse error: {str(e)}")
                continue
        calls = []
        # Only process if we found valid JSON objects
        if all_actions:
            calls = self.parse_base_json(all_actions, tools)
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<|python_tag|>{"name":"' + name + '", "arguments":',
            end="}",
            trigger="<|python_tag|>",
        )

    def build_ebnf(self, tools: List[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            tool_calls_rule='"<|python_tag|>" function_call',
            call_rule_fmt='call_{name} ::= "{" "name" ":" "{name}" "," "arguments" ":" {arguments_rule} "}"',
            arguments_rule_fmt='"{" {arg_rules} "}"',
            key_value_fmt='"{key}" ":" {valrule}',
            is_pythonic=False,
        )


class DeepSeekV3Detector(BaseFormatDetector):
    """
    Detector for DeepSeek models.
    Assumes function call format:
      '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n```json\n{"location": "Tokyo"}\n```<｜tool▁call▁end｜>\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n```json\n{"location": "Paris"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜tool▁calls▁begin｜>"
        self.eot_token = "<｜tool▁calls▁end｜>"
        self.func_call_regex = r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
        self.func_detail_regex = r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n```json\n(.*)\n```<｜tool▁call▁end｜>"
        self._last_arguments = ""

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        match_result_list = re.findall(self.func_call_regex, text, re.DOTALL)
        calls = []
        try:
            for match_result in match_result_list:
                # Get function name
                func_detail = re.search(self.func_detail_regex, match_result, re.DOTALL)
                func_name = func_detail.group(2)
                func_args = func_detail.group(3)
                func_args = json.loads(func_args)
                # construct match_result for parse_base_json
                match_result = {"name": func_name, "parameters": func_args}
                calls.extend(self.parse_base_json(match_result, tools))
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=">" + name + "\n```json\n",
            end="\n```<",
            trigger=">" + name + "\n```json\n",
        )

    def build_ebnf(self, tools: List[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            tool_calls_rule="'<｜tool▁calls▁begin｜>' tool_call (tool_call)* '<｜tool▁calls▁end｜>'",
            call_rule_fmt='call_{name} ::= "<｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\\n```json\\n" {arguments_rule} "\\n```<｜tool▁call▁end｜>"',
            arguments_rule_fmt='"{" {arg_rules} "}"',
            key_value_fmt='"{key}" ":" {valrule}',
            is_pythonic=False,
        )

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for DeepSeekV3 format.
        """
        self._buffer += new_text
        current_text = self._buffer

        if self.bot_token not in current_text:
            self._buffer = ""
            for e_token in [self.eot_token, "```", "<｜tool▁call▁end｜>"]:
                if e_token in new_text:
                    new_text = new_text.replace(e_token, "")
            return StreamingParseResult(normal_text=new_text)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = {
                tool.function.name: i
                for i, tool in enumerate(tools)
                if tool.function and tool.function.name
            }

        calls: list[ToolCallItem] = []
        try:
            partial_match = re.search(
                pattern=r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n```json\n(.*)",
                string=current_text,
                flags=re.DOTALL,
            )
            if partial_match:
                func_name = partial_match.group(2).strip()
                func_args_raw = partial_match.group(3).strip()

                if not self.current_tool_name_sent:
                    calls.append(
                        ToolCallItem(
                            tool_index=self._tool_indices.get(func_name, 0),
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                else:
                    argument_diff = (
                        func_args_raw[len(self._last_arguments) :]
                        if func_args_raw.startswith(self._last_arguments)
                        else func_args_raw
                    )

                    if argument_diff:
                        calls.append(
                            ToolCallItem(
                                tool_index=self._tool_indices.get(func_name, 0),
                                name=None,
                                parameters=argument_diff,
                            )
                        )
                        self._last_arguments += argument_diff

                    if _is_complete_json(func_args_raw):
                        result = StreamingParseResult(normal_text="", calls=calls)
                        self._buffer = ""
                        self._last_arguments = ""
                        self.current_tool_name_sent = False
                        return result

            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)


class PythonicDetector(BaseFormatDetector):
    """
    Detector for Llama-3.2 and Llama-4 models with pythonic tool call format.
    Assumes function call format:
      [tool1(arg1=val1, arg2=val2), tool2(arg1=val3)]
    Arguments are Python literals (not JSON).
    """

    def __init__(self):
        super().__init__()
        FUNC_CALL = r"""
            [a-zA-Z_][\w]*(\.[a-zA-Z_][\w]*)*       # Function name: dotted, Python-style identifiers
            \(                                      # Opening parenthesis for arguments
                (                                   # --- Optional repeated key=val pairs ending with comma ---
                    [a-zA-Z]+\w*=.*,\s*             #     Match key=val followed by comma and optional whitespace
                )*                                  # --- Zero or more such arguments
                (                                   # --- Optional last argument without trailing comma ---
                    [a-zA-Z]+\w*=.*\s*              #     Match final key=val (no comma)
                )?                                  # --- This part is optional
            \)                                      # Closing parenthesis
        """

        self.tool_call_regex = re.compile(
            rf"""
            \[                                      # Opening square bracket
                \s*
                {FUNC_CALL}                         # First function call
                (
                    \s*,\s*{FUNC_CALL}              # Additional function calls (comma-separated)
                )*                                  # Zero or more
                \s*
            \]                                      # Closing square bracket
            """,
            re.VERBOSE | re.DOTALL,
        )
        # self.tool_call_regex_xgrammar = r"\[\s*[a-zA-Z_][\w]*(\.[a-zA-Z_][\w]*)*\(((?:[a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s*)?)\)(\s*,\s*[a-zA-Z_][\w]*(\.[a-zA-Z_][\w]*)*\(((?:[a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s*)?)\)))*\s*\]"

    def has_tool_call(self, text: str) -> bool:
        return bool(self.tool_call_regex.match(text.strip()))

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        # Try parsing the text as a Python list of function calls
        text = text.strip()
        if not (text.startswith("[") and text.endswith("]")):
            # Not a pythonic tool call format
            return StreamingParseResult(normal_text=text, calls=[])
        try:
            module = ast.parse(text)
            parsed = getattr(module.body[0], "value", None)
            if not (
                isinstance(parsed, ast.List)
                and all(isinstance(e, ast.Call) for e in parsed.elts)
            ):
                return StreamingParseResult(normal_text=text, calls=[])
            calls = []
            tool_indices = {
                tool.function.name: i
                for i, tool in enumerate(tools)
                if tool.function.name
            }
            for call in parsed.elts:
                if not isinstance(call.func, ast.Name):
                    continue
                function_name = call.func.id
                arguments = {}
                for keyword in call.keywords:
                    arguments[keyword.arg] = self._get_parameter_value(keyword.value)
                calls.append(
                    ToolCallItem(
                        tool_index=tool_indices.get(function_name, -1),
                        name=function_name,
                        parameters=json.dumps(arguments, ensure_ascii=False),
                    )
                )
            return StreamingParseResult(normal_text="", calls=calls)
        except Exception:
            logger.exception("Error in pythonic tool call parsing.")
            return StreamingParseResult(normal_text=text, calls=[])

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for pythonic tool calls.
        Buffers input until a complete pythonic tool call (from [ to ]) is found,
        then parses and emits any detected calls.
        """
        self._buffer += new_text
        start = self._buffer.find("[")
        end = self._buffer.find("]", start)
        if start != -1 and end != -1:
            call_text = self._buffer[start : end + 1]
            result = self.detect_and_parse(call_text, tools)
            self._buffer = self._buffer[end + 1 :]
            return result
        return StreamingParseResult(normal_text="")

    def _get_parameter_value(self, val):
        if isinstance(val, ast.Constant):
            return val.value
        elif isinstance(val, ast.Dict):
            return {
                k.value: self._get_parameter_value(v)
                for k, v in zip(val.keys, val.values)
            }
        elif isinstance(val, ast.List):
            return [self._get_parameter_value(v) for v in val.elts]
        else:
            raise ValueError("Tool call arguments must be literals")

    def structure_info(self) -> _GetInfoFunc:
        def info(name: str):
            return StructureInfo(begin=f"[{name}(", end=")]", trigger=f"[{name}(")

        return info

    def build_ebnf(self, tools: List[Tool]) -> Optional[str]:
        return EBNFComposer.build_ebnf(
            tools,
            tool_calls_rule='"[" function_call ("," function_call)* "]"',
            call_rule_fmt='call_{name} ::= "{name}" "(" {arguments_rule} ")"',
            arguments_rule_fmt="{arg_rules}",
            key_value_fmt='"{key}" "=" {valrule}',
            is_pythonic=True,
        )


class FunctionCallParser:
    """
    Parser for function/tool calls in model outputs.

    This class handles both streaming and non-streaming parsing of function calls using a detector.
    In streaming scenarios, each time new_text is received, it calls detector.parse_streaming_increment
    and returns the resulting normal_text and calls to the upper layer (or SSE).
    """

    ToolCallParserEnum: Dict[str, Type[BaseFormatDetector]] = {
        "llama3": Llama32Detector,
        "qwen25": Qwen25Detector,
        "mistral": MistralDetector,
        "deepseekv3": DeepSeekV3Detector,
        "pythonic": PythonicDetector,
    }

    def __init__(self, tools: List[Tool], tool_call_parser: str):
        detector = None
        detector_class = self.ToolCallParserEnum.get(tool_call_parser)
        if detector_class:
            detector = detector_class()
        else:
            raise ValueError(f"Unsupported tool_call_parser: {tool_call_parser}")

        self.detector = detector
        self.tools = tools

    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains a tool call in the format supported by this parser.
        This delegates to the detector's implementation.

        Args:
            text: The text to check for tool calls

        Returns:
            True if the text contains a tool call, False otherwise
        """
        return self.detector.has_tool_call(text)

    def parse_non_stream(self, full_text: str) -> Tuple[str, list[ToolCallItem]]:
        """
        One-time parsing of the full text to extract tool calls.

        Args:
            full_text: The complete text to parse

        Returns:
            A tuple containing:
            - The remaining text after parsing that was not consumed by the detector (can be treated as normal text)
            - A list of tool calls parsed from the text
        """
        parsed_result = self.detector.detect_and_parse(full_text, self.tools)
        tool_call_list = parsed_result.calls
        if tool_call_list:
            return parsed_result.normal_text, tool_call_list
        else:
            return full_text, []

    def parse_stream_chunk(self, chunk_text: str) -> Tuple[str, list[ToolCallItem]]:
        """
        Streaming incremental parsing of chunks of text as they arrive.

        Args:
            chunk_text: The new chunk of text to parse

        Returns:
            A tuple containing:
            - The normal text that should be displayed to the user
            - A list of tool calls parsed from the chunk
        """
        final_normal_text = ""
        final_calls = []

        sp_result = self.detector.parse_streaming_increment(chunk_text, self.tools)
        if sp_result.normal_text:
            final_normal_text = sp_result.normal_text
        if sp_result.calls:
            final_calls.extend(sp_result.calls)
            final_normal_text = sp_result.normal_text

        return final_normal_text, final_calls

    def get_structure_tag(self) -> StructuralTagResponseFormat:
        """
        Generate a structural tag response format for all available tools.

        This creates the necessary structural tags that guide the model's output format.
        """
        tool_structures: List[StructuresResponseFormat] = list()
        tool_trigger_set: Set[str] = set()

        get_structure_info = self.detector.structure_info()
        for tool in self.tools:
            function = tool.function
            name = function.name
            assert name is not None
            info = get_structure_info(name)

            # accept all if not strict, otherwise only accept the schema
            schema = function.parameters if function.strict else {}

            tool_structures.append(
                StructuresResponseFormat(
                    begin=info.begin,
                    schema=schema,  # type: ignore
                    end=info.end,
                )
            )
            tool_trigger_set.add(info.trigger)

        return StructuralTagResponseFormat(
            type="structural_tag",
            structures=tool_structures,
            triggers=list(tool_trigger_set),
        )

    def get_structure_constraint(
        self, tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]]
    ) -> Optional[Tuple[str, Any]]:
        """
        Returns the appropriate structure constraint for tool calls based on the tool_choice.
        The constraint is used to guide the model's output format.

        Args:
            tool_choice: The tool choice setting from the request

        Returns:
            A tuple of (constraint_type, constraint_value) to be added to sampling parameters,
            or None if no constraint applies.
        """
        if tool_choice == "auto":
            strict_tag = self.get_structure_tag()
            return ("structural_tag", strict_tag)
        else:
            ebnf = self.get_ebnf(tool_choice)
            return ("ebnf", ebnf) if ebnf is not None else None

    def get_ebnf(
        self, tool_choice: Union[ToolChoice, Literal["required"]]
    ) -> Optional[str]:
        """
        Get the EBNF grammar for the specified tool choice.
        """
        filtered_tools = []
        if isinstance(tool_choice, ToolChoice):
            fn_name = tool_choice.function.name
            filtered_tools = [t for t in self.tools if t.function.name == fn_name]
        else:
            filtered_tools = self.tools
        return self.detector.build_ebnf(filtered_tools)
