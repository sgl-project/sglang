"""
GLM-5.3-Flash tool call detector.

The GLM-5.3-Flash model outputs tool calls using the special token 154843
(the 11-char string "<tool_call>") as the start marker. The model uses two
different formats:

1. JSON format (most common):
        <tool_call>Weather/get_weather_by_city>{"location": "Beijing"}

2. Special-token format (less common):
        <tool_call>Weather
        <arg_key>location</arg_key>
        <arg_value>Beijing</arg_value>
        </tool_call>

The model uses the 11-char special token (154843), NOT the 3-char "<![" (75459)
used by glm47. This detector handles both formats.
"""

import json
import logging
import re
from typing import Any, Dict, List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
)

logger = logging.getLogger(__name__)

# Special tokens used by GLM-5.3-Flash (token IDs 154843-154850)
TC_START = "<tool_call>"    # 154843 - tool_call start
TC_END = "</tool_call>"    # 154844 - tool_call end
AK_START = "<arg_key>"     # 154847 - arg_key start
AK_END = "</arg_key>"      # 154848 - arg_key end
AV_START = "<arg_value>"   # 154849 - arg_value start
AV_END = "</arg_value>"    # 154850 - arg_value end


class Glm53FlashDetector(BaseFormatDetector):
    """
    Detector for GLM-5.3-Flash tool call format.

    The model uses the 11-char special token (154843), not the 3-char "<![" (75459).
    Two output formats are supported:
    1. JSON:   <tool_call>func_name>{"key": "value"}
    2. Tags:   <tool_call>func_name<arg_key>key</arg_key><arg_value>val</arg_value></tool_call>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = TC_START
        self.eot_token = TC_END

        # Regex for tag format: TC_START + name + (AK_START key AK_END AV_START val AV_END)* + TC_END
        # Name stops at AK_START or end of line
        self.tag_regex = re.compile(
            re.escape(TC_START)
            + r"(.*?)"
            + r"((?:" + re.escape(AK_START) + r".*?" + re.escape(AK_END)
            + re.escape(AV_START) + r".*?" + re.escape(AV_END) + r"\s*)*)"
            + re.escape(TC_END),
            re.DOTALL,
        )
        # Also a simpler tag regex that stops name at AK_START
        self.tag_regex2 = re.compile(
            re.escape(TC_START)
            + r"((?:(?!" + re.escape(AK_START) + r").)*?)"
            + r"((?:" + re.escape(AK_START) + r".*?" + re.escape(AK_END)
            + re.escape(AV_START) + r".*?" + re.escape(AV_END) + r"\s*)*)"
            + re.escape(TC_END),
            re.DOTALL,
        )

        # Regex for JSON format: TC_START + name + > + JSON (until next TC_START or TC_END or end)
        self.json_regex = re.compile(
            re.escape(TC_START) + r"(.*?)(?:>|\n)(.*?)(?="
            + re.escape(TC_START) + r"|" + re.escape(TC_END) + r"|$)",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def _normalize_func_name(self, raw_name: str, tools: List[Tool]) -> str:
        """Normalize the function name, matching against known tool names."""
        name = raw_name.strip()
        if not name:
            return name

        tool_names = set()
        for t in tools:
            if hasattr(t, "function") and t.function:
                tool_names.add(t.function.name)

        if name in tool_names:
            return name

        if "/" in name:
            parts = [p.strip() for p in name.split("/") if p.strip()]
            for part in parts:
                if part in tool_names:
                    return part
            for tname in tool_names:
                for part in parts:
                    if tname in part or part in tname:
                        return tname

        if ":" in name:
            part = name.split(":")[0].strip()
            if part in tool_names:
                return part

        # If no match but tools are available, try case-insensitive matching
        name_lower = name.lower()
        for tname in tool_names:
            if tname.lower() == name_lower:
                return tname
            # Check if the name is a "display name" that relates to a tool
            # e.g., "Weather" -> "get_weather"
            if name_lower in tname.lower() or tname.lower().replace("get_", "") == name_lower:
                return tname

        # If still no match and there's only one tool, use it
        if len(tool_names) == 1:
            return next(iter(tool_names))

        return name

    def _parse_json_args(self, raw_args: str) -> Dict[str, Any]:
        """Parse JSON arguments from the raw argument string."""
        raw_args = raw_args.strip()
        if not raw_args:
            return {}
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except (json.JSONDecodeError, ValueError):
            start = raw_args.find("{")
            end = raw_args.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(raw_args[start : end + 1])
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    pass
            return {}

    def _parse_tag_args(self, raw_args: str) -> Dict[str, Any]:
        """Parse arguments from the tag format: AK_START key AK_END AV_START val AV_END"""
        arguments = {}
        pair_regex = re.compile(
            re.escape(AK_START) + r"(.*?)" + re.escape(AK_END)
            + r"\s*" + re.escape(AV_START) + r"(.*?)" + re.escape(AV_END),
            re.DOTALL,
        )
        for match in pair_regex.finditer(raw_args):
            key = match.group(1).strip()
            value = match.group(2).strip()
            try:
                arguments[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                arguments[key] = value
        return arguments

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-time parsing: Detects and parses tool calls in the provided text."""
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])

        normal_text_parts = []
        last_end = 0
        calls = []

        # Check if the text uses tag format (has AK_START tokens and TC_END)
        if AK_START in text and TC_END in text:
            # Tag format: TC_START + name + (AK_START key AK_END AV_START val AV_END)* + TC_END
            # Use simple regex to find TC_START...TC_END blocks
            block_regex = re.compile(re.escape(TC_START) + r"(.*?)" + re.escape(TC_END), re.DOTALL)
            for match in block_regex.finditer(text):
                if match.start() > last_end:
                    normal_text_parts.append(text[last_end : match.start()])
                last_end = match.end()

                inner = match.group(1)
                # Split name and args at AK_START
                if AK_START in inner:
                    name_part, args_part = inner.split(AK_START, 1)
                    raw_name = name_part.strip()
                    raw_args = AK_START + args_part
                else:
                    raw_name = inner.strip()
                    raw_args = ""

                func_name = self._normalize_func_name(raw_name, tools)
                if not func_name:
                    continue

                arguments = self._parse_tag_args(raw_args)
                call_item = {"name": func_name, "parameters": arguments}
                calls.extend(self.parse_base_json(call_item, tools))

            if last_end < len(text):
                remaining = text[last_end:]
                if self.bot_token in remaining:
                    json_result = self._parse_json_format(remaining, tools)
                    if json_result.normal_text:
                        normal_text_parts.append(json_result.normal_text)
                    calls.extend(json_result.calls)
                else:
                    normal_text_parts.append(remaining)
        else:
            # JSON format: TC_START + name + > + JSON
            json_result = self._parse_json_format(text, tools)
            if json_result.normal_text:
                normal_text_parts.append(json_result.normal_text)
            calls.extend(json_result.calls)

        normal_text = "".join(normal_text_parts).strip()
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _parse_json_format(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse tool calls in JSON format: TC_START + name + > + JSON"""
        normal_text_parts = []
        last_end = 0
        calls = []

        for match in self.json_regex.finditer(text):
            if match.start() > last_end:
                normal_text_parts.append(text[last_end : match.start()])
            last_end = match.end()

            raw_name = match.group(1).strip()
            raw_args = match.group(2)

            func_name = self._normalize_func_name(raw_name, tools)
            if not func_name:
                continue

            arguments = self._parse_json_args(raw_args)
            call_item = {"name": func_name, "parameters": arguments}
            calls.extend(self.parse_base_json(call_item, tools))

        if last_end < len(text):
            normal_text_parts.append(text[last_end:])

        normal_text = "".join(normal_text_parts).strip()
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Streaming incremental parsing for GLM-5.3-Flash format."""
        self._buffer += new_text
        current_text = self._buffer

        if self.bot_token not in current_text:
            is_potential_start = any(
                self.bot_token.startswith(current_text[-i:])
                for i in range(1, min(len(current_text), len(self.bot_token)) + 1)
            )
            if not is_potential_start:
                output_text = current_text
                self._buffer = ""
                return StreamingParseResult(normal_text=output_text)
            return StreamingParseResult(normal_text="")

        normal_text = ""
        first_idx = current_text.find(self.bot_token)
        if first_idx > 0:
            normal_text = current_text[:first_idx]
            current_text = current_text[first_idx:]
            self._buffer = current_text

        calls = []

        # Check for tag format first (has TC_END)
        if self.eot_token in current_text:
            tag_match = self.tag_regex2.search(current_text)
            if tag_match:
                raw_name = tag_match.group(1).strip()
                raw_args = tag_match.group(2)
                func_name = self._normalize_func_name(raw_name, tools)

                if func_name:
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.prev_tool_call_arr = []
                        self.streamed_args_for_tool = [""]
                        self.current_tool_name_sent = False

                    if not self.current_tool_name_sent:
                        self.current_tool_name_sent = True
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=func_name,
                                parameters="",
                            )
                        )

                    arguments = self._parse_tag_args(raw_args)
                    args_json = json.dumps(arguments, ensure_ascii=False)
                    self.streamed_args_for_tool[self.current_tool_id] = args_json
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=args_json,
                        )
                    )

                    self._buffer = current_text[tag_match.end():]
                    return StreamingParseResult(normal_text=normal_text, calls=calls)

        # Try JSON format
        match = re.search(
            re.escape(self.bot_token) + r"(.*?)(?:>|\n)",
            current_text,
            re.DOTALL,
        )
        if not match:
            return StreamingParseResult(normal_text=normal_text)

        raw_name = match.group(1).strip()
        func_name = self._normalize_func_name(raw_name, tools)

        if not func_name:
            self._buffer = ""
            return StreamingParseResult(normal_text=normal_text + current_text)

        if self.current_tool_id == -1:
            self.current_tool_id = 0
            self.prev_tool_call_arr = []
            self.streamed_args_for_tool = [""]
            self.current_tool_name_sent = False

        if not self.current_tool_name_sent:
            self.current_tool_name_sent = True
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=func_name,
                    parameters="",
                )
            )

        after_name = current_text[match.end():]
        json_start = after_name.find("{")
        if json_start >= 0:
            json_str = after_name[json_start:]
            try:
                parsed = json.loads(json_str)
                args_json = json.dumps(parsed, ensure_ascii=False)
                self.streamed_args_for_tool[self.current_tool_id] = args_json
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=None,
                        parameters=args_json,
                    )
                )
                self._buffer = after_name[json_start + len(json_str):]
            except json.JSONDecodeError:
                last_brace = after_name.rfind("}")
                if last_brace >= 0:
                    try:
                        partial = after_name[json_start : last_brace + 1]
                        parsed = json.loads(partial)
                        args_json = json.dumps(parsed, ensure_ascii=False)
                        self.streamed_args_for_tool[self.current_tool_id] = args_json
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=None,
                                parameters=args_json,
                            )
                        )
                        self._buffer = after_name[last_brace + 1:]
                    except json.JSONDecodeError:
                        pass

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def supports_structural_tag(self) -> bool:
        return False

    def get_structural_tag(self, *args, **kwargs):
        return None

    def structure_info(self):
        return None

    def get_structural_tag_name(self) -> str:
        return "glm5_3_flash"
