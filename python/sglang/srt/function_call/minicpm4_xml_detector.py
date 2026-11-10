import json
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult, _GetInfoFunc
from sglang.srt.function_call.minicpm5_detector import parse_arguments
from sglang.srt.function_call.utils import get_schema_properties

logger = logging.getLogger(__name__)

_FUNC_OPEN_RE = re.compile(r"<function\s+name=['\"]([^'\"]+)['\"][^>]*>")
_PARAM_OPEN_RE = re.compile(r"<param\s+name=['\"]([^'\"]+)['\"][^>]*>")

CDATA_START = "<![CDATA["
CDATA_END = "]]>"
PARAM_CLOSE = "</param>"
FUNCTION_CLOSE = "</function>"
PARAM_OPEN = "<param"


def _convert_param_value(key: str, val_text: str, props: dict) -> object:
    arg_type = props[key].get("type") if key in props else None
    if arg_type is None or arg_type == "string":
        return val_text
    parsed_val, _ = parse_arguments(val_text)
    return parsed_val


def _extract_param_value(text: str, start: int) -> Tuple[str, int]:
    """Extract a param value and return (value, end_pos after value markup)."""
    if text.startswith(CDATA_START, start):
        cdata_content_start = start + len(CDATA_START)
        cdata_end = text.find(CDATA_END, cdata_content_start)
        if cdata_end != -1:
            value = text[cdata_content_start:cdata_end]
            end_pos = cdata_end + len(CDATA_END)
            if text.startswith(PARAM_CLOSE, end_pos):
                end_pos += len(PARAM_CLOSE)
            return value, end_pos

        # Broken CDATA: fall back to nearest-boundary extraction from content start.
        start = cdata_content_start

    slice_text = text[start:]
    candidates = []
    for marker, consume in (
        (PARAM_CLOSE, len(PARAM_CLOSE)),
        (PARAM_OPEN, 0),
        (FUNCTION_CLOSE, 0),
    ):
        idx = slice_text.find(marker)
        if idx != -1:
            candidates.append((idx, consume))

    if not candidates:
        func_close = text.find(FUNCTION_CLOSE, start)
        if func_close == -1:
            return text[start:], len(text)
        return text[start:func_close].strip(), func_close

    end_idx, consume = min(candidates, key=lambda x: x[0])
    value = slice_text[:end_idx].strip()
    end_pos = start + end_idx + consume
    return value, end_pos


def _find_function_close(text: str, start: int = 0) -> int:
    """Find </function> that is not inside a well-formed CDATA section."""
    pos = start
    while True:
        idx = text.find(FUNCTION_CLOSE, pos)
        if idx == -1:
            return -1

        cdata_start = text.rfind(CDATA_START, start, idx)
        if cdata_start == -1:
            return idx

        cdata_end = text.find(CDATA_END, cdata_start)
        if cdata_end == -1:
            return idx

        if cdata_end < idx:
            return idx

        pos = idx + len(FUNCTION_CLOSE)


def _parse_function_block(
    text: str, start: int, name_to_props: Dict[str, dict]
) -> Tuple[Optional[dict], int, bool]:
    """
    Parse one <function>...</function> block starting at `start`.

    Returns:
        (call_dict or None, end_pos, is_complete)
    """
    m = _FUNC_OPEN_RE.match(text, start)
    if not m:
        return None, start + 1, False

    func_name = (m.group(1) or "").strip()
    pos = m.end()
    arguments: Dict[str, object] = {}
    seen_keys: Set[str] = set()
    props = name_to_props.get(func_name, {})

    close_idx = _find_function_close(text, pos)
    if close_idx == -1:
        return None, len(text), False

    body_end = close_idx
    pos_in_body = pos

    while pos_in_body < body_end:
        remaining = text[pos_in_body:body_end]
        ws_len = len(remaining) - len(remaining.lstrip())
        if ws_len:
            pos_in_body += ws_len
            continue

        pm = _PARAM_OPEN_RE.match(text, pos_in_body)
        if pm:
            key = pm.group(1).strip()
            val_start = pm.end()
            val_text, val_end = _extract_param_value(text, val_start)
            pos_in_body = val_end

            if key in seen_keys:
                logger.warning(
                    "Ignoring duplicate param '%s' in tool call '%s'",
                    key,
                    func_name,
                )
                continue

            if props and key not in props:
                logger.warning(
                    "Ignoring unknown param '%s' in tool call '%s'",
                    key,
                    func_name,
                )
                continue

            seen_keys.add(key)
            arguments[key] = _convert_param_value(key, val_text, props)
            continue

        if text.startswith(PARAM_OPEN, pos_in_body):
            logger.warning(
                "Ignoring param without name in tool call '%s'",
                func_name,
            )
            close = text.find(PARAM_CLOSE, pos_in_body, body_end)
            if close == -1:
                break
            pos_in_body = close + len(PARAM_CLOSE)
            continue

        break

    end_pos = close_idx + len(FUNCTION_CLOSE)
    call = {"name": func_name, "parameters": arguments}
    return call, end_pos, True


class MiniCPM4XmlFormatDetector(BaseFormatDetector):
    """
    MiniCPM <function>/<param> tool calls scanned in order: a </function>
    inside CDATA does not close the call, unknown params are dropped, and
    calls missing a required param are dropped.

    Expected format example (multiple calls allowed):
      <function name="get_weather"><param name="city">北京</param><param name="date">2024-06-27</param></function>
      <function name="get_weather"><param name="city"><![CDATA[多行\n文本]]></param></function>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<function"
        self.eot_token = "</function>"

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        idx = text.find(self.bot_token)
        if idx == -1:
            return StreamingParseResult(normal_text=text, calls=[])

        normal_parts: List[str] = []
        calls = []
        name_to_tool = {t.function.name: t for t in tools if t.function.name}
        name_to_props: Dict[str, dict] = {}
        name_to_required: Dict[str, Set[str]] = {}
        try:
            for name, t in name_to_tool.items():
                params = t.function.parameters
                if not isinstance(params, dict):
                    params = {}
                name_to_props[name] = get_schema_properties(params)
                name_to_required[name] = set(params.get("required") or ())

            last_end = 0
            pos = 0
            while pos < len(text):
                start = text.find(self.bot_token, pos)
                if start == -1:
                    break

                if start > last_end:
                    normal_parts.append(text[last_end:start])

                call_obj, end_pos, is_complete = _parse_function_block(
                    text, start, name_to_props
                )

                if not is_complete:
                    normal_parts.append(text[start:])
                    last_end = len(text)
                    break

                func_name = call_obj["name"]
                if func_name not in name_to_tool:
                    normal_parts.append(text[start:end_pos])
                elif name_to_required[func_name].issubset(call_obj["parameters"]):
                    calls.extend(self.parse_base_json(call_obj, tools))
                else:
                    logger.warning(
                        "Dropped tool call '%s' due to missing required params",
                        func_name,
                    )

                last_end = end_pos
                pos = end_pos

            if last_end < len(text):
                normal_parts.append(text[last_end:])

            return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)
        except Exception as e:
            logger.error("Error in detect_and_parse: %s", e)
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        current_text = self._buffer

        start = current_text.find(self.bot_token)
        if start == -1:
            partial_len = self._ends_with_partial_token(self._buffer, self.bot_token)
            if partial_len:
                return StreamingParseResult()
            normal_text = self._buffer
            self._buffer = ""
            if self.current_tool_id > 0:
                normal_text = ""
            return StreamingParseResult(normal_text=normal_text)

        end = _find_function_close(current_text, start)
        if end != -1:
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = [""]
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")
            result = self.detect_and_parse(
                current_text[: end + len(self.eot_token)], tools=tools
            )
            if result.calls:
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": result.calls[0].name,
                    "arguments": json.loads(result.calls[0].parameters),
                }
                self.streamed_args_for_tool[self.current_tool_id] = result.calls[
                    0
                ].parameters
                result.calls[0].tool_index = self.current_tool_id
                self.current_tool_id += 1
            self._buffer = current_text[end + len(self.eot_token) :]
            return result

        normal_text = current_text[:start]
        self._buffer = current_text[start:]
        return StreamingParseResult(normal_text=normal_text)

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()
