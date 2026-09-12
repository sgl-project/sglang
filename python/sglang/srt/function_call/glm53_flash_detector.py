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
from typing import Any, Dict, List, Optional, Tuple

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
            + re.escape(AV_START) + r".*?" + re.escape(AV_END) + r"\s*)+)"
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

        # Simplified tag format observed in the wild (AK_END/AV_START omitted):
        # TC_START + name + (AK_START key=value AV_END)+ + TC_END
        self.tag_regex3 = re.compile(
            re.escape(TC_START)
            + r"((?:(?!" + re.escape(AK_START) + r").)*?)"
            + r"((?:" + re.escape(AK_START)
            + r"(?:(?!" + re.escape(TC_END) + r").)*?"
            + re.escape(AV_END) + r"\s*)+)"
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


    def _fix_args_against_schema(
        self, arguments, func_name, tools
    ):
        """Fix nested schema unwrapping by the model."""
        if not func_name or not tools:
            return arguments

        tool_func = None
        for t in tools:
            if hasattr(t, "function") and t.function and t.function.name == func_name:
                tool_func = t.function
                break

        if not tool_func or not tool_func.parameters:
            return arguments

        schema = tool_func.parameters
        if not isinstance(schema, dict):
            return arguments

        props = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(required, list):
            required = []

        if isinstance(arguments, dict):
            missing_required = [k for k in required if k not in arguments]
            if not missing_required:
                return arguments

            for missing_key in missing_required:
                prop_schema = props.get(missing_key, {})
                if not isinstance(prop_schema, dict):
                    continue
                if prop_schema.get("type") != "array":
                    continue

                item_schema = prop_schema.get("items", {})
                item_props = set()
                if isinstance(item_schema, dict):
                    item_props = set(item_schema.get("properties", {}).keys())

                arg_keys = set(arguments.keys())
                if item_props and arg_keys and arg_keys.issubset(item_props):
                    logger.info(
                        "GLM-5.3-Flash: fixing unwrapped args for '%s' - "
                        "wrapping in {'%s': [args]}",
                        func_name, missing_key,
                    )
                    return {missing_key: [arguments]}

        if isinstance(arguments, list):
            for missing_key in required:
                prop_schema = props.get(missing_key, {})
                if isinstance(prop_schema, dict) and prop_schema.get("type") == "array":
                    logger.info(
                        "GLM-5.3-Flash: fixing bare array args for '%s' - "
                        "wrapping in {'%s': [args]}",
                        func_name, missing_key,
                    )
                    return {missing_key: arguments}
            # No matching array property found — return empty dict so the
            # API doesn't reject "arguments must be an object"
            logger.info(
                "GLM-5.3-Flash: list args for '%s' have no matching array "
                "property, returning {} to satisfy object requirement",
                func_name,
            )
            return {}

        return arguments

    def _parse_json_args(self, raw_args: str) -> Any:
        """Parse JSON arguments from the raw argument string.

        Returns a dict, a list (for bare array args), or empty dict on failure.
        When the model outputs a bare array [{...}, {...}] for a tool with
        an array-type required property, we return the list so
        _fix_args_against_schema can wrap it.
        """
        raw_args = raw_args.strip()
        if not raw_args:
            return {}
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
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
        if not arguments:
            # Simplified tag format: AK_START key=value AV_END (AK_END/AV_START omitted)
            simple_regex = re.compile(
                re.escape(AK_START) + r"(.*?)" + re.escape(AV_END),
                re.DOTALL,
            )
            for match in simple_regex.finditer(raw_args):
                kv = match.group(1).strip()
                if "=" in kv:
                    key, value = kv.split("=", 1)
                else:
                    key, value = kv, ""
                key = key.strip()
                value = value.strip()
                try:
                    arguments[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    arguments[key] = value
        if not arguments:
            # Last resort: embedded JSON object
            brace = raw_args.find("{")
            if brace >= 0:
                try:
                    parsed = json.loads(raw_args[brace:])
                    if isinstance(parsed, dict):
                        arguments = parsed
                except (json.JSONDecodeError, ValueError):
                    pass
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
                arguments = self._fix_args_against_schema(arguments, func_name, tools)
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

            raw_name = match.group(1).split("<arg_key")[0].strip()
            raw_args = match.group(2)

            func_name = self._normalize_func_name(raw_name, tools)
            if not func_name:
                continue

            arguments = self._parse_json_args(raw_args)
            arguments = self._fix_args_against_schema(arguments, func_name, tools)
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

        # Check for tag format first (has TC_END and at least one AK_START pair)
        if self.eot_token in current_text and AK_START in current_text:
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
                    arguments = self._fix_args_against_schema(arguments, func_name, tools)
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
                    # Advance tool_id for next tool call in the same response
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False
                    self.streamed_args_for_tool.append("")
                    return StreamingParseResult(normal_text=normal_text, calls=calls)

        # Simplified tag format (AK_END/AV_START omitted) - must beat the JSON
        # fallback, whose name regex would otherwise swallow "<arg_key" into
        # the function name (e.g. "Read<arg_key").
        if self.eot_token in current_text:
            simple_match = self.tag_regex3.search(current_text)
            if simple_match:
                raw_name = simple_match.group(1).strip()
                raw_args = simple_match.group(2)
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
                    arguments = self._fix_args_against_schema(arguments, func_name, tools)
                    args_json = json.dumps(arguments, ensure_ascii=False)
                    self.streamed_args_for_tool[self.current_tool_id] = args_json
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=args_json,
                        )
                    )
                    self._buffer = current_text[simple_match.end():]
                    # Advance tool_id for next tool call in the same response
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False
                    self.streamed_args_for_tool.append("")
                    return StreamingParseResult(normal_text=normal_text, calls=calls)

        # Try JSON format
        match = re.search(
            re.escape(self.bot_token) + r"(.*?)(?:>|\n)",
            current_text,
            re.DOTALL,
        )
        if not match:
            return StreamingParseResult(normal_text=normal_text)

        raw_name = match.group(1).split("<arg_key")[0].strip()
        func_name = self._normalize_func_name(raw_name, tools)

        if not func_name:
            self._buffer = ""
            return StreamingParseResult(normal_text=normal_text + current_text)

        # If AK_START is present but TC_END is not, the model is using tag
        # format in streaming mode and TC_END hasn't arrived yet. Don't
        # emit the name via the JSON handler — wait for TC_END so the tag
        # handler can process the complete tool call.
        if AK_START in current_text and self.eot_token not in current_text:
            return StreamingParseResult(normal_text=normal_text)

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
        # Look for JSON array first ([), then JSON object ({)
        # The model sometimes outputs a bare array [{...}, {...}] for
        # tools with array-type required properties (e.g. todo_write).
        # If we start from the first { inside the array, raw_decode only
        # parses the first object and the rest leaks as content.
        json_start = after_name.find("[")
        if json_start < 0:
            json_start = after_name.find("{")
        if json_start >= 0:
            json_str = after_name[json_start:]
            decoder = json.JSONDecoder()
            try:
                parsed, end_idx = decoder.raw_decode(json_str)
                parsed = self._fix_args_against_schema(parsed, func_name, tools)
                args_json = json.dumps(parsed, ensure_ascii=False)
                self.streamed_args_for_tool[self.current_tool_id] = args_json
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=None,
                        parameters=args_json,
                    )
                )
                # Advance tool_id for next tool call in the same response
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                self._buffer = after_name[json_start + end_idx:]
            except json.JSONDecodeError:
                # JSON incomplete - check if model moved to next tool call
                next_tc = after_name.find(self.bot_token, json_start)
                if next_tc >= 0:
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False
                    self.streamed_args_for_tool.append("")
                    self._buffer = after_name[next_tc:]
                # else: wait for more data (incomplete JSON)

        return StreamingParseResult(normal_text=normal_text, calls=calls)


    def finish(self, tools):
        """Flush remaining buffer content when stream ends.

        The base class finish() returns empty result, but we may have
        unprocessed tool calls left in the buffer when the model outputs
        multiple tool calls in a single response and the stream ends
        before all are processed.
        """
        calls = []
        if not self._buffer or self.bot_token not in self._buffer:
            # No tool call in buffer, return as normal text
            normal_text = self._buffer if self._buffer else ""
            self._buffer = ""
            return StreamingParseResult(normal_text=normal_text, calls=calls)

        # Process remaining tool calls in the buffer
        current = self._buffer
        self._buffer = ""

        # Keep processing until no more tool calls found
        while self.bot_token in current:
            first_idx = current.find(self.bot_token)
            if first_idx > 0:
                # There's normal text before the tool call - discard it
                current = current[first_idx:]

            # Try tag format first
            if self.eot_token in current and AK_START in current:
                tag_match = self.tag_regex2.search(current)
                if tag_match:
                    raw_name = tag_match.group(1).strip()
                    raw_args = tag_match.group(2)
                    func_name = self._normalize_func_name(raw_name, tools)
                    if func_name:
                        if self.current_tool_id == -1:
                            self.current_tool_id = 0
                        if not self.current_tool_name_sent:
                            self.current_tool_name_sent = True
                            calls.append(ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=func_name, parameters="",
                            ))
                        arguments = self._parse_tag_args(raw_args)
                        arguments = self._fix_args_against_schema(arguments, func_name, tools)
                        args_json = json.dumps(arguments, ensure_ascii=False)
                        self.streamed_args_for_tool[self.current_tool_id] = args_json
                        calls.append(ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None, parameters=args_json,
                        ))
                        self.current_tool_id += 1
                        self.current_tool_name_sent = False
                        self.streamed_args_for_tool.append("")
                        current = current[tag_match.end():]
                        continue

            # Try JSON format
            match = re.search(
                re.escape(self.bot_token) + r"(.*?)(?:>|\n)",
                current, re.DOTALL,
            )
            if not match:
                break

            raw_name = match.group(1).split("<arg_key")[0].strip()
            func_name = self._normalize_func_name(raw_name, tools)
            if not func_name:
                break

            if self.current_tool_id == -1:
                self.current_tool_id = 0
            if not self.current_tool_name_sent:
                self.current_tool_name_sent = True
                calls.append(ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=func_name, parameters="",
                ))

            after_name = current[match.end():]
            # Look for JSON array first ([), then JSON object ({)
            json_start = after_name.find("[")
            if json_start < 0:
                json_start = after_name.find("{")
            if json_start >= 0:
                json_str = after_name[json_start:]
                decoder = json.JSONDecoder()
                try:
                    parsed, end_idx = decoder.raw_decode(json_str)
                    parsed = self._fix_args_against_schema(parsed, func_name, tools)
                    args_json = json.dumps(parsed, ensure_ascii=False)
                    self.streamed_args_for_tool[self.current_tool_id] = args_json
                    calls.append(ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=None, parameters=args_json,
                    ))
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False
                    self.streamed_args_for_tool.append("")
                    current = after_name[json_start + end_idx:]
                    continue
                except json.JSONDecodeError:
                    pass

            break

        return StreamingParseResult(normal_text="", calls=calls)

    def supports_structural_tag(self) -> bool:
        return False

    def get_structural_tag(self, *args, **kwargs):
        return None

    def structure_info(self):
        return None

    def get_structural_tag_name(self) -> str:
        return "glm5_3_flash"
