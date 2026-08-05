import json
import logging
import re
from typing import Dict, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import safe_literal_eval

logger = logging.getLogger(__name__)

try:
    from lxml import etree as ET  # type: ignore

    _HAS_LXML = True
except Exception:  # pragma: no cover - environment may not have lxml
    import xml.etree.ElementTree as ET  # type: ignore

    _HAS_LXML = False

_FUNC_NAME_V1_REGEX = re.compile(r"<function\s+name=[\'\"]([^\'\"]+)[\'\"][^>]*>")
_PARAM_WITH_NAME_REGEX = re.compile(
    r"<param\s+name=[\'\"]([^\'\"]+)[\'\"]>([\s\S]*?)</param>", re.DOTALL
)
_PARAM_MISSING_NAME_REGEX = re.compile(r"<param(?![^>]*\bname=)[^>]*>", re.DOTALL)


def get_argument_type(
    func_name: str, arg_key: str, name_to_tool: Dict[str, Tool]
) -> Optional[str]:
    tool = name_to_tool.get(func_name)
    if not tool:
        return None
    params = tool.function.parameters or {}
    if not isinstance(params, dict):
        return None
    return params.get("properties", {}).get(arg_key, {}).get("type")


def parse_arguments(json_value):
    try:
        try:
            parsed_value = json.loads(json_value)
        except (json.JSONDecodeError, TypeError):
            parsed_value = safe_literal_eval(json_value)
        return parsed_value, True
    except (ValueError, SyntaxError, TypeError):
        return json_value, False


class MiniCPM5Detector(BaseFormatDetector):
    """
    Detector for MiniCPM-4 models (V3 schema) adapted to the new chat template.

    Expected format example (multiple calls allowed):
      <function name="get_weather"><param name="city">北京</param><param name="date">2024-06-27</param></function>
      <function name="get_weather"><param name="city"><![CDATA[多行\n文本]]></param></function>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<function"
        self.eot_token = "</function>"
        self.func_call_regex = r"<function.*?</function>"

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a MiniCPM-4 V3 XML-styled tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        idx = text.find(self.bot_token)
        if idx == -1:
            return StreamingParseResult(normal_text=text, calls=[])

        normal_parts = []
        calls = []
        name_to_tool = {t.function.name: t for t in tools if t.function.name}
        tool_names = set(name_to_tool.keys())
        name_to_allowed_props = {}
        name_to_required = {}
        for name, t in name_to_tool.items():
            params = t.function.parameters or {}
            props = (
                (params.get("properties", {}) or {}) if isinstance(params, dict) else {}
            )
            name_to_allowed_props[name] = set(props.keys())
            req = params.get("required", []) if isinstance(params, dict) else []
            try:
                name_to_required[name] = set(req)
            except Exception:
                name_to_required[name] = set()

        try:
            last_end = 0
            for m in re.finditer(self.func_call_regex, text, re.DOTALL):
                if m.start() > last_end:
                    normal_parts.append(text[last_end : m.start()])

                block = m.group(0)
                func_name = None
                arguments = {}
                parsed_ok = False
                param_invalid = False

                # Primary path: XML parsing (lxml preferred, stdlib fallback)
                try:
                    if _HAS_LXML:
                        try:
                            parser = ET.XMLParser(**{"strip_cdata": False})  # type: ignore[call-arg]
                        except TypeError:
                            parser = ET.XMLParser()
                        root = ET.fromstring(block, parser=parser)
                    else:
                        root = ET.fromstring(block)

                    if root.tag == "function":
                        func_node = root
                    else:
                        func_node = (
                            root.find("function") if hasattr(root, "find") else None
                        )

                    if func_node is not None:
                        func_name = (func_node.attrib.get("name") or "").strip()

                    args_node = (
                        func_node.find("arguments") if func_node is not None else None
                    )
                    param_nodes = []
                    if func_node is not None:
                        param_nodes = list(func_node.findall("param"))
                        if args_node is not None and not param_nodes:
                            param_nodes = list(args_node.findall("param"))

                    if func_node is not None:
                        seen_keys = set()
                        allowed_props = set()
                        if func_name in tool_names:
                            allowed_props = name_to_allowed_props.get(func_name, set())
                        has_invalid_param = False
                        for param in param_nodes:
                            key = param.attrib.get("name")
                            if not key:
                                has_invalid_param = True
                                break
                            if allowed_props and key not in allowed_props:
                                has_invalid_param = True
                                break
                            if key in seen_keys:
                                has_invalid_param = True
                                break
                            seen_keys.add(key)
                            val_text = param.text or ""
                            val_text = val_text.strip()
                            arg_type = get_argument_type(
                                func_name or "", key, name_to_tool
                            )
                            if arg_type != "string":
                                parsed_val, _ = parse_arguments(val_text)
                                arguments[key] = parsed_val
                            else:
                                arguments[key] = val_text
                        if has_invalid_param:
                            arguments.clear()
                            param_invalid = True
                    parsed_ok = bool(func_name)
                except Exception:
                    parsed_ok = False

                if not parsed_ok:
                    # Fallback path: regex extraction
                    try:
                        m_fn = _FUNC_NAME_V1_REGEX.search(block)
                        if m_fn:
                            func_name = (m_fn.group(1) or "").strip()
                        has_invalid_param = (
                            _PARAM_MISSING_NAME_REGEX.search(block) is not None
                        )
                        seen_keys = set()
                        allowed_props = set()
                        if func_name in tool_names:
                            allowed_props = name_to_allowed_props.get(func_name, set())
                        for pm in _PARAM_WITH_NAME_REGEX.finditer(block):
                            key = pm.group(1).strip()
                            if allowed_props and key not in allowed_props:
                                has_invalid_param = True
                                break
                            if key in seen_keys:
                                has_invalid_param = True
                                break
                            seen_keys.add(key)
                            val_text = pm.group(2) or ""
                            if val_text.startswith("<![CDATA[") and val_text.endswith(
                                "]]>"
                            ):
                                val_text = val_text[len("<![CDATA[") : -len("]]>")]
                            val_text = val_text.strip()
                            arg_type = get_argument_type(
                                func_name or "", key, name_to_tool
                            )
                            if arg_type != "string":
                                parsed_val, _ = parse_arguments(val_text)
                                arguments[key] = parsed_val
                            else:
                                arguments[key] = val_text
                        if has_invalid_param:
                            arguments.clear()
                            param_invalid = True
                        parsed_ok = bool(func_name)
                    except Exception:
                        parsed_ok = False

                if not func_name or func_name not in tool_names or param_invalid:
                    parsed_ok = False
                else:
                    req_props = name_to_required.get(func_name, set())
                    if req_props and not req_props.issubset(arguments.keys()):
                        parsed_ok = False

                if parsed_ok:
                    tool_call_obj = {"name": func_name, "parameters": arguments}
                    calls.extend(self.parse_base_json(tool_call_obj, tools))
                else:
                    normal_parts.append(block)

                last_end = m.end()

            if last_end < len(text):
                normal_parts.append(text[last_end:])

            return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            return StreamingParseResult(normal_text=text)

    def _append_tool_call(self, call, all_calls: List) -> None:
        if self.current_tool_id == -1:
            self.current_tool_id = 0
            self.prev_tool_call_arr = []
            self.streamed_args_for_tool = [""]

        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")

        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": call.name,
            "arguments": json.loads(call.parameters),
        }
        self.streamed_args_for_tool[self.current_tool_id] = call.parameters
        call.tool_index = self.current_tool_id
        self.current_tool_id += 1
        all_calls.append(call)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        normal_parts = []
        all_calls = []

        while True:
            current_text = self._buffer
            start = current_text.find(self.bot_token)
            if start == -1:
                partial_len = self._ends_with_partial_token(
                    current_text, self.bot_token
                )
                if partial_len > 0:
                    self._buffer = current_text[-partial_len:]
                    emit = current_text[:-partial_len]
                else:
                    self._buffer = ""
                    emit = "" if self.current_tool_id > 0 else current_text
                if emit:
                    normal_parts.append(emit)
                break

            if start > 0:
                normal_parts.append(current_text[:start])
                current_text = current_text[start:]

            end = current_text.find(self.eot_token)
            if end == -1:
                self._buffer = current_text
                break

            block = current_text[: end + len(self.eot_token)]
            self._buffer = current_text[end + len(self.eot_token) :]

            result = self.detect_and_parse(block, tools=tools)
            for call in result.calls:
                self._append_tool_call(call, all_calls)

            if self.bot_token not in self._buffer:
                partial_len = self._ends_with_partial_token(
                    self._buffer, self.bot_token
                )
                if partial_len == 0:
                    emit = "" if self.current_tool_id > 0 else self._buffer
                    if emit:
                        normal_parts.append(emit)
                    self._buffer = ""
                break

        return StreamingParseResult(normal_text="".join(normal_parts), calls=all_calls)

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()
