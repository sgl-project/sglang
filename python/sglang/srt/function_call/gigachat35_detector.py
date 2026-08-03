import json
import logging
import re
from typing import Any, Dict, List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)

_GCML_OPEN = "<｜GCML｜tool_calls>"
_GCML_CLOSE = "</｜GCML｜tool_calls>"
_GCML_INVOKE_RE = re.compile(
    r"<｜GCML｜invoke\s+name=\"(?P<name>[^\"]*)\"\s*>"
    r"(?P<body>.*?)"
    r"</｜GCML｜invoke>",
    re.DOTALL,
)
_GCML_PARAM_RE = re.compile(
    r"<｜GCML｜parameter\s+name=\"(?P<name>[^\"]*)\"\s+"
    r"string=\"(?P<is_string>true|false)\"\s*>"
    r"(?P<value>.*?)"
    r"</｜GCML｜parameter>",
    re.DOTALL,
)
_TRAILING_MARKER_RE = re.compile(r"(?:<\|message_sep\|>|</s>)+\s*$")


def _strip_trailing_markers(text: str) -> str:
    return _TRAILING_MARKER_RE.sub("", text)


def _parse_gcml_invoke_body(body: str) -> Dict[str, Any]:
    """Parse the <｜GCML｜parameter ...> entries inside one invoke block."""
    args: Dict[str, Any] = {}
    for m in _GCML_PARAM_RE.finditer(body):
        raw = m.group("value")
        if m.group("is_string") == "true":
            args[m.group("name")] = raw
            continue
        try:
            args[m.group("name")] = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            args[m.group("name")] = raw
    return args


class GigaChat35Detector(BaseFormatDetector):
    def __init__(self) -> None:
        super().__init__()
        self.bot_token = _GCML_OPEN
        self.eot_token = _GCML_CLOSE
        self._tool_region_started: bool = False
        self._emitted_invokes: int = 0

    def has_tool_call(self, text: str) -> bool:
        """True if the text contains the GCML tool-call opening marker."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Non-streaming parse of a complete model output."""
        if self.bot_token not in text:
            return StreamingParseResult(
                normal_text=_strip_trailing_markers(text), calls=[]
            )

        leading, _, after_open = text.partition(self.bot_token)
        invokes_body, _, _ = after_open.partition(self.eot_token)

        actions = [
            {
                "name": m.group("name"),
                "arguments": _parse_gcml_invoke_body(m.group("body")),
            }
            for m in _GCML_INVOKE_RE.finditer(invokes_body)
        ]
        if not actions:
            return StreamingParseResult(
                normal_text=_strip_trailing_markers(text), calls=[]
            )

        calls = self.parse_base_json(actions, tools)
        normal_text = _strip_trailing_markers(leading).rstrip("\n")
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        current_text = self._buffer

        if not self._tool_region_started and self.bot_token not in current_text:
            if self._ends_with_partial_token(current_text, self.bot_token):
                return StreamingParseResult()
            self._buffer = ""
            return StreamingParseResult(normal_text=current_text)

        if not self._tool_region_started:
            leading, _, rest = current_text.partition(self.bot_token)
            self._tool_region_started = True
            self._buffer = self.bot_token + rest
            current_text = self._buffer
            if leading:
                return StreamingParseResult(normal_text=leading)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        _, _, after_open = current_text.partition(self.bot_token)
        invokes_body, _, _ = after_open.partition(self.eot_token)
        matches = list(_GCML_INVOKE_RE.finditer(invokes_body))
        calls: List[ToolCallItem] = []
        for i in range(self._emitted_invokes, len(matches)):
            m = matches[i]
            name = m.group("name")
            if name not in self._tool_indices:
                logger.warning(f"[GigaChat35] undefined function call: {name}")
                continue
            args = _parse_gcml_invoke_body(m.group("body"))
            args_json = json.dumps(args, ensure_ascii=False)
            calls.append(ToolCallItem(tool_index=i, name=name, parameters=args_json))
            while len(self.prev_tool_call_arr) <= i:
                self.prev_tool_call_arr.append({})
            self.prev_tool_call_arr[i] = {"name": name, "arguments": args}
            while len(self.streamed_args_for_tool) <= i:
                self.streamed_args_for_tool.append("")
            self.streamed_args_for_tool[i] = args_json

        self._emitted_invokes = len(matches)
        return StreamingParseResult(calls=calls)

    def supports_structural_tag(self) -> bool:
        """GigaChat 3.5 GCML does not use structural tags."""
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError(
            "GigaChat35Detector does not support structural_tag format."
        )
