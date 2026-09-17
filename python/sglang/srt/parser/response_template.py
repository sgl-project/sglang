# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SGLang serving adapters for checkpoint-defined `response_template` grammars."""

from __future__ import annotations

import copy
import json
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

import regex as re

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult as ToolStreamingParseResult,
)
from sglang.srt.function_call.core_types import (
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.parser.response_template_config import (
    resolve_detector_response_template,
    validate_response_template_for_serving,
)

logger = logging.getLogger(__name__)


class _ReasoningResult:
    """Duck-types `reasoning_parser.StreamingParseResult`."""

    def __init__(self, normal_text: str = "", reasoning_text: str = ""):
        self.normal_text = normal_text
        self.reasoning_text = reasoning_text


def derive_tool_extraction_template(
    full: dict[str, Any],
    *,
    sink_field: str = "normal",
    tool_field: str = "tool_calls",
) -> dict[str, Any]:
    """Build a tool-stage template that only extracts tool-call regions."""
    anchor: dict[str, Any] = {}
    if "start_anchor_pattern" in full:
        anchor["start_anchor_pattern"] = full["start_anchor_pattern"]
    elif "start_anchor" in full:
        anchor["start_anchor"] = full["start_anchor"]

    fields: dict[str, Any] = {
        sink_field: {"content": "text", "content_args": {"strip": False}},
    }
    field = copy.deepcopy(full["fields"][tool_field])
    if field.get("content") == "xml-inline":
        field.setdefault("content_args", {})["strict"] = True
    fields[tool_field] = field
    return {"defaults": {}, **anchor, "fields": fields}


def derive_reasoning_extraction_template(
    full: dict[str, Any],
    *,
    tool_field: str = "tool_calls",
) -> dict[str, Any]:
    """Treat tool regions as opaque wire text while routing reasoning."""
    template = copy.deepcopy(full)
    field = template.get("fields", {}).get(tool_field)
    if field is None:
        return template
    field["content"] = "text"
    field["content_args"] = {"strip": False}
    field["transform"] = "{content}"
    field.pop("transform_each", None)
    return template


def _first_field_literal(field: dict[str, Any], key: str) -> str:
    raw = field.get(key)
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list) and raw and isinstance(raw[0], str):
        return raw[0]
    return ""


def _field_open_re(field: dict[str, Any]) -> re.Pattern[str] | None:
    if "open_pattern" in field:
        return re.compile(field["open_pattern"], re.DOTALL)
    raw = field.get("open")
    if isinstance(raw, str):
        return re.compile(re.escape(raw), re.DOTALL)
    if isinstance(raw, list) and raw and all(isinstance(s, str) for s in raw):
        return re.compile("|".join(re.escape(s) for s in raw), re.DOTALL)
    return None


class AdapterMode(Enum):
    REASONING = "reasoning"
    TOOL = "tool"


class ResponseTemplateStreamAdapter:
    """Streaming `response_template` parse with mode-specific event routing."""

    def __init__(
        self,
        template: dict[str, Any],
        *,
        mode: AdapterMode,
        prefix: str | None = None,
        tool_field: str = "tool_calls",
        thinking_field: str = "thinking",
        content_field: str = "content",
        passthrough_field: str = "normal",
    ):
        self._tool_field = tool_field
        self._thinking_field = thinking_field
        self._content_field = content_field
        self._passthrough_field = passthrough_field
        self._mode = mode
        if mode == AdapterMode.REASONING:
            parser_template = derive_reasoning_extraction_template(
                template,
                tool_field=tool_field,
            )
        else:
            parser_template = derive_tool_extraction_template(
                template,
                sink_field=passthrough_field,
                tool_field=tool_field,
            )
        self._parser_template = parser_template
        self._prefix = "" if prefix is None else prefix
        self._stream_parser = None
        self._committed_input_offset = 0
        self._passthrough = False
        self._pending_reasoning = ""
        self._pending_tool_raw = ""
        self._pending_tool_body = ""
        self._pending_tool_streamed = False
        self._failed_streamed_tool = False
        self._finalized = False
        self._prefix_opens_tool_region: bool | None = None
        self._deferred_prefix_open_pending = True

    @property
    def passthrough(self) -> bool:
        return self._passthrough

    def _make_parser(self, tools: Sequence[Any] | None = None):
        from sglang.srt.parser.chat_parsing.response_parser import ResponseParser

        parser_tools = [
            tool if isinstance(tool, dict) else tool.model_dump()
            for tool in tools or []
        ]
        return ResponseParser(
            self._parser_template,
            prefix=self._prefix,
            tools=parser_tools,
        )

    def _initial_events(self, parser: Any) -> list[dict]:
        if self._mode != AdapterMode.TOOL:
            return []
        active_open = None
        for event in parser.initial_events:
            if event.get("field") != self._tool_field:
                continue
            if event["type"] == "region_open":
                active_open = {**event, "from_prefix": True}
            elif event["type"] == "region_close":
                active_open = None
        if active_open is not None:
            self._deferred_prefix_open_pending = False
        return [] if active_open is None else [active_open]

    def _mark_deferred_prefix_open(self, events: list[dict]) -> list[dict]:
        if not self._deferred_prefix_open_pending or not self._prefix:
            return events
        for index, event in enumerate(events):
            raw = event.get("raw", "")
            if (
                event.get("field") == self._tool_field
                and event["type"] == "region_open"
                and raw
                and self._prefix.endswith(raw)
            ):
                events[index] = {**event, "from_prefix": True}
                self._deferred_prefix_open_pending = False
                break
        return events

    def _recover_failed_tool_input(self, current_text: str) -> str:
        if self._stream_parser is None:
            return current_text
        parser_input = self._stream_parser.input_text
        return parser_input[self._committed_input_offset :]

    def prefix_opens_tool_region(self) -> bool:
        if self._prefix_opens_tool_region is not None:
            return self._prefix_opens_tool_region
        try:
            result = bool(self._initial_events(self._make_parser()))
        except Exception:
            result = False
        self._prefix_opens_tool_region = result
        return result

    def prefix_ends_with_tool_open(self, pattern: re.Pattern[str]) -> bool:
        return any(
            match.end() == len(self._prefix) for match in pattern.finditer(self._prefix)
        )

    def _raw_tool_events(self, events: Sequence[dict]) -> str:
        parts: List[str] = []
        for event in events:
            if event.get("field") != self._tool_field or event.get("from_prefix"):
                continue
            if event["type"] == "region_chunk":
                parts.append(event["text"])
            else:
                parts.append(event.get("raw", ""))
        return "".join(parts)

    def detect_and_parse(
        self,
        text: str,
        tools: Sequence[Any] | None = None,
    ) -> tuple[list[dict], bool]:
        try:
            parser = self._make_parser(tools)
            events = self._initial_events(parser)
            events += self._mark_deferred_prefix_open(parser.feed(text))
            _, final_events = parser.finalize()
            events += final_events
            return events, True
        except Exception as exc:
            logger.warning(
                "response_template parsing failed; passing through: %s",
                exc,
            )
            return [], False

    def feed(
        self,
        text: str,
        tools: Sequence[Any] | None = None,
    ) -> tuple[list[dict], bool]:
        if self._failed_streamed_tool:
            if not text:
                return [], True
            return [
                {
                    "type": "region_malformed",
                    "field": self._tool_field,
                    "text": text,
                }
            ], True
        if self._passthrough or self._finalized:
            return [], False
        initial_events: list[dict] = []
        try:
            if self._stream_parser is None:
                self._stream_parser = self._make_parser(tools)
                initial_events = self._initial_events(self._stream_parser)
                self._committed_input_offset = len(self._stream_parser.input_text)
            events = initial_events + self._mark_deferred_prefix_open(
                self._stream_parser.feed(text)
            )
            self._committed_input_offset = self._stream_parser.consumed_offset
            return events, True
        except Exception as exc:
            logger.warning(
                "response_template streaming parse failed; preserving wire bytes: %s",
                exc,
            )
            if self._mode == AdapterMode.TOOL:
                if self._pending_tool_streamed:
                    malformed = (
                        self._pending_tool_body + self._recover_failed_tool_input(text)
                    )
                    self._pending_tool_raw = ""
                    self._pending_tool_body = ""
                    self._failed_streamed_tool = True
                    return [
                        {
                            "type": "region_malformed",
                            "field": self._tool_field,
                            "text": malformed,
                        }
                    ], True
                self._passthrough = True
                passthrough = self._raw_tool_events(
                    initial_events
                ) + self._recover_failed_tool_input(text)
                self._pending_tool_raw = ""
                return [
                    {
                        "type": "region_chunk",
                        "field": self._passthrough_field,
                        "text": passthrough,
                    }
                ], True
            self._passthrough = True
            return [], False

    def finalize(
        self,
        tools: Sequence[Any] | None = None,
    ) -> tuple[list[dict], bool]:
        if self._failed_streamed_tool:
            self._finalized = True
            return [], True
        if self._passthrough or self._finalized:
            return [], not self._passthrough
        initial_events: list[dict] = []
        try:
            if self._stream_parser is None:
                self._stream_parser = self._make_parser(tools)
                initial_events = self._initial_events(self._stream_parser)
                self._committed_input_offset = len(self._stream_parser.input_text)
            _, events = self._stream_parser.finalize()
            self._committed_input_offset = self._stream_parser.consumed_offset
            self._finalized = True
            return initial_events + events, True
        except Exception as exc:
            logger.warning(
                "response_template finalize failed; preserving wire bytes: %s",
                exc,
            )
            if self._mode == AdapterMode.TOOL:
                if self._pending_tool_streamed:
                    malformed = (
                        self._pending_tool_body + self._recover_failed_tool_input("")
                    )
                    self._pending_tool_raw = ""
                    self._pending_tool_body = ""
                    self._failed_streamed_tool = True
                    self._finalized = True
                    if not malformed:
                        return [], True
                    return [
                        {
                            "type": "region_malformed",
                            "field": self._tool_field,
                            "text": malformed,
                        }
                    ], True
                self._passthrough = True
                passthrough = self._raw_tool_events(
                    initial_events
                ) + self._recover_failed_tool_input("")
                self._pending_tool_raw = ""
                if not passthrough:
                    return [], False
                return [
                    {
                        "type": "region_chunk",
                        "field": self._passthrough_field,
                        "text": passthrough,
                    }
                ], True
            self._passthrough = True
            return [], False

    def route_reasoning_events(
        self, events: List[dict], *, stream_reasoning: bool
    ) -> tuple[str, str]:
        reasoning_parts: List[str] = []
        normal_parts: List[str] = []
        for event in events:
            field = event.get("field")
            etype = event["type"]
            if field == self._thinking_field:
                if etype == "region_chunk":
                    if stream_reasoning:
                        reasoning_parts.append(event["text"])
                    else:
                        self._pending_reasoning += event["text"]
                elif etype == "region_close" and not stream_reasoning:
                    reasoning_parts.append(self._pending_reasoning)
                    self._pending_reasoning = ""
            elif field == self._content_field:
                if etype == "region_chunk":
                    normal_parts.append(event["text"])
            elif field == self._tool_field:
                if etype == "region_open":
                    if not event.get("from_prefix"):
                        normal_parts.append(event.get("raw", ""))
                elif etype == "region_close":
                    normal_parts.append(event.get("raw", ""))
                elif etype == "region_chunk":
                    normal_parts.append(event["text"])
        return "".join(normal_parts), "".join(reasoning_parts)

    def route_tool_events(
        self,
        events: List[dict],
        *,
        on_tool_open: Callable[[str], bool] | None = None,
        on_tool_close: Callable[[Any], bool],
        on_tool_malformed: Callable[[str], None] | None = None,
    ) -> str:
        normal_parts: List[str] = []
        for event in events:
            field = event.get("field")
            etype = event["type"]
            if field == self._passthrough_field and etype == "region_chunk":
                normal_parts.append(event["text"])
            elif field == self._tool_field:
                if etype == "region_open":
                    raw_open = event.get("raw", "")
                    self._pending_tool_raw = (
                        "" if event.get("from_prefix") else raw_open
                    )
                    self._pending_tool_body = ""
                    if on_tool_open is not None:
                        self._pending_tool_streamed = on_tool_open(raw_open)
                elif etype == "region_chunk":
                    self._pending_tool_raw += event["text"]
                    self._pending_tool_body += event["text"]
                elif etype == "region_malformed":
                    if on_tool_malformed is not None:
                        on_tool_malformed(event["text"])
                elif etype == "region_close":
                    raw_close = event.get("raw", "")
                    if (
                        not on_tool_close(event["value"])
                        and not self._pending_tool_streamed
                    ):
                        normal_parts.append(self._pending_tool_raw + raw_close)
                    self._pending_tool_raw = ""
                    self._pending_tool_body = ""
                    self._pending_tool_streamed = False
        return "".join(normal_parts)


class _ResponseTemplateParserInputMixin:
    """Preserve response-template delimiters until parsing."""

    @staticmethod
    def configure_request_for_parsing(request: Any) -> None:
        request.skip_special_tokens = False
        template_kwargs = dict(getattr(request, "chat_template_kwargs", None) or {})
        template_kwargs["spaces_between_special_tokens"] = False
        request.chat_template_kwargs = template_kwargs


class ResponseTemplateReasoningDetector(_ResponseTemplateParserInputMixin):
    """Reasoning detector driven by a `response_template` grammar."""

    response_template: dict | None = None
    thinking_field: str = "thinking"
    content_field: str = "content"
    tool_field: str = "tool_calls"

    def __init__(
        self,
        stream_reasoning: bool = True,
        tokenizer=None,
        response_template: dict | None = None,
        prefix: str | None = None,
        **_kwargs,
    ):
        fallback = (
            self.response_template if response_template is None else response_template
        )
        template = resolve_detector_response_template(tokenizer, fallback)
        if template is None:
            raise ValueError("response_template is required")
        validate_response_template_for_serving(template)
        self.response_template = template
        self.stream_reasoning = stream_reasoning
        thinking = template.get("fields", {}).get(self.thinking_field, {})
        self.think_start_token = _first_field_literal(thinking, "open") or getattr(
            self, "_default_think_start", ""
        )
        self.think_end_token = _first_field_literal(thinking, "close") or getattr(
            self, "_default_think_end", ""
        )
        self.think_start_self_label = ""
        self.thinks_internally = False
        self.reasoning_default = "explicit_enable_thinking"
        self._adapter = ResponseTemplateStreamAdapter(
            template,
            mode=AdapterMode.REASONING,
            thinking_field=self.thinking_field,
            content_field=self.content_field,
            tool_field=self.tool_field,
            prefix=prefix,
        )

    def detect_and_parse(self, text: str) -> _ReasoningResult:
        events, ok = self._adapter.detect_and_parse(text)
        if not ok:
            return _ReasoningResult(normal_text=text)
        normal_text, reasoning_text = self._adapter.route_reasoning_events(
            events, stream_reasoning=True
        )
        return _ReasoningResult(normal_text=normal_text, reasoning_text=reasoning_text)

    def parse_streaming_increment(self, new_text: str) -> _ReasoningResult:
        if self._adapter.passthrough:
            return _ReasoningResult(normal_text=new_text)
        events, ok = self._adapter.feed(new_text)
        if not ok:
            return _ReasoningResult(normal_text=new_text)
        normal_text, reasoning_text = self._adapter.route_reasoning_events(
            events, stream_reasoning=self.stream_reasoning
        )
        return _ReasoningResult(normal_text=normal_text, reasoning_text=reasoning_text)

    def finish(self) -> _ReasoningResult:
        if self._adapter.passthrough:
            return _ReasoningResult()
        events, ok = self._adapter.finalize()
        if not ok:
            return _ReasoningResult()
        normal_text, reasoning_text = self._adapter.route_reasoning_events(
            events, stream_reasoning=self.stream_reasoning
        )
        if not self.stream_reasoning and self._adapter._pending_reasoning:
            reasoning_text = self._adapter._pending_reasoning + reasoning_text
            self._adapter._pending_reasoning = ""
        return _ReasoningResult(normal_text=normal_text, reasoning_text=reasoning_text)


class ResponseTemplateToolDetector(
    _ResponseTemplateParserInputMixin,
    BaseFormatDetector,
):
    """Tool-call detector driven by a `response_template` grammar."""

    response_template: dict | None = None
    tool_field: str = "tool_calls"
    passthrough_field: str = "normal"
    reject_strict_without_constraints = True
    reject_parallel_auto_without_constraints = True

    def __init__(
        self,
        tokenizer=None,
        response_template: dict | None = None,
        prefix: str | None = None,
    ):
        super().__init__()
        fallback = (
            self.response_template if response_template is None else response_template
        )
        template = resolve_detector_response_template(tokenizer, fallback)
        if template is None:
            raise ValueError("response_template is required")
        validate_response_template_for_serving(template)
        self.response_template = template
        self._adapter = ResponseTemplateStreamAdapter(
            template,
            mode=AdapterMode.TOOL,
            tool_field=self.tool_field,
            passthrough_field=self.passthrough_field,
            prefix=prefix,
        )
        tool_spec = template["fields"][self.tool_field]
        self._tool_open_re = _field_open_re(tool_spec)
        self._tool_name_template = None
        if not tool_spec.get("transform_each", False):
            transform = tool_spec.get("transform")
            if isinstance(transform, dict):
                function = transform.get("function")
                if isinstance(function, dict):
                    name_template = function.get("name")
                    if isinstance(name_template, str):
                        self._tool_name_template = name_template
        self.bot_token = ""
        self.eot_token = ""
        if isinstance(tool_spec.get("close"), str):
            self.eot_token = tool_spec["close"]
        elif isinstance(tool_spec.get("close"), list) and tool_spec["close"]:
            self.eot_token = tool_spec["close"][0]
        self.incomplete_tool_call_indices: set[int] = set()
        self.has_incomplete_tool_call = False

    def has_tool_call(self, text: str) -> bool:
        return bool(
            self._tool_open_re is not None
            and self._tool_open_re.search(text) is not None
        ) or (
            self._tool_open_re is not None
            and (
                self._adapter.prefix_opens_tool_region()
                or self._adapter.prefix_ends_with_tool_open(self._tool_open_re)
            )
        )

    def _tool_name_from_open(self, raw: str) -> str | None:
        if self._tool_open_re is None or self._tool_name_template is None:
            return None
        match = self._tool_open_re.fullmatch(raw)
        if match is None:
            return None
        placeholder = re.fullmatch(
            r"\{(?P<capture>\w+(?:\.\w+)*)\}",
            self._tool_name_template,
        )
        if placeholder is None:
            return self._tool_name_template
        capture = placeholder.group("capture")
        if "." in capture:
            return None
        return match.groupdict().get(capture)

    def _to_tool_call_item(
        self, value: Any, tool_indices: Dict[str, int], tool_index: int
    ) -> Optional[ToolCallItem]:
        try:
            function = value["function"]
            name = function["name"]
            arguments = function["arguments"]
        except (TypeError, KeyError):
            logger.warning("response_template: malformed tool call value: %r", value)
            return None

        if not isinstance(name, str):
            logger.warning("response_template: malformed tool name: %r", name)
            return None

        if name not in tool_indices:
            logger.warning("Model attempted to call undefined function: %s", name)
            if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                return None

        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return ToolCallItem(tool_index=tool_index, name=name, parameters=arguments)

    def _to_tool_call_items(
        self,
        value: Any,
        tool_indices: Dict[str, int],
        first_tool_index: int,
    ) -> Optional[List[ToolCallItem]]:
        values = value if isinstance(value, list) else [value]
        if not values:
            return None
        items = [
            self._to_tool_call_item(item, tool_indices, first_tool_index + offset)
            for offset, item in enumerate(values)
        ]
        if any(item is None for item in items):
            return None
        return items

    def _emit_tool_call(
        self, item: ToolCallItem, pending_calls: List[ToolCallItem]
    ) -> None:
        name_was_streamed = self.current_tool_name_sent
        if self.current_tool_id == -1:
            self.current_tool_id = 0
            self.prev_tool_call_arr = []
            self.streamed_args_for_tool = [""]
        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")

        try:
            parsed_args = json.loads(item.parameters)
        except json.JSONDecodeError:
            parsed_args = item.parameters
        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": item.name,
            "arguments": parsed_args,
        }
        self.streamed_args_for_tool[self.current_tool_id] = item.parameters
        if name_was_streamed:
            item = item.model_copy(update={"name": None})
        pending_calls.append(item)
        self.incomplete_tool_call_indices.discard(self.current_tool_id)
        self.current_tool_id += 1
        self.current_tool_name_sent = False
        self.has_incomplete_tool_call = bool(self.incomplete_tool_call_indices)

    def _emit_tool_name(self, name: str, pending_calls: List[ToolCallItem]) -> None:
        if self.current_tool_id == -1:
            self.current_tool_id = 0
            self.prev_tool_call_arr = []
            self.streamed_args_for_tool = [""]
        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")
        pending_calls.append(
            ToolCallItem(
                tool_index=self.current_tool_id,
                name=name,
                parameters="",
            )
        )
        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": name,
            "arguments": "",
        }
        self.incomplete_tool_call_indices.add(self.current_tool_id)
        self.current_tool_name_sent = True
        self.has_incomplete_tool_call = True

    def _emit_malformed_tool_arguments(
        self,
        text: str,
        pending_calls: List[ToolCallItem],
    ) -> None:
        if self.current_tool_id < 0 or not self.current_tool_name_sent:
            return
        previous = self.streamed_args_for_tool[self.current_tool_id]
        arguments = previous + text
        self.streamed_args_for_tool[self.current_tool_id] = arguments
        self.prev_tool_call_arr[self.current_tool_id]["arguments"] = arguments
        if text:
            pending_calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=None,
                    parameters=text,
                )
            )
        self.incomplete_tool_call_indices.add(self.current_tool_id)
        self.has_incomplete_tool_call = True

    def _route_tool_events(
        self, events: List[dict], tool_indices: Dict[str, int]
    ) -> tuple[str, List[ToolCallItem]]:
        calls: List[ToolCallItem] = []

        def on_close(value: Any) -> bool:
            items = self._to_tool_call_items(value, tool_indices, len(calls))
            if items is None:
                return False
            calls.extend(items)
            return True

        normal_text = self._adapter.route_tool_events(events, on_tool_close=on_close)
        return normal_text, calls

    def detect_and_parse(
        self, text: str, tools: List[Tool]
    ) -> ToolStreamingParseResult:
        if not self.has_tool_call(text):
            return ToolStreamingParseResult(normal_text=text, calls=[])

        events, ok = self._adapter.detect_and_parse(text, tools)
        if not ok:
            return ToolStreamingParseResult(normal_text=text, calls=[])

        normal_text, calls = self._route_tool_events(
            events, self._get_tool_indices(tools)
        )
        return ToolStreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> ToolStreamingParseResult:
        if self._adapter.passthrough:
            return ToolStreamingParseResult(normal_text=new_text, calls=[])

        events, ok = self._adapter.feed(new_text, tools)
        if not ok:
            return ToolStreamingParseResult(normal_text=new_text, calls=[])
        return self._emit_streaming_events(events, tools)

    def finish(self, tools: List[Tool]) -> ToolStreamingParseResult:
        if self._adapter.passthrough:
            return ToolStreamingParseResult()
        events, ok = self._adapter.finalize(tools)
        if not ok:
            return ToolStreamingParseResult()
        return self._emit_streaming_events(events, tools)

    def _emit_streaming_events(
        self, events: List[dict], tools: List[Tool]
    ) -> ToolStreamingParseResult:
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        pending_calls: List[ToolCallItem] = []

        def on_open(raw: str) -> bool:
            name = self._tool_name_from_open(raw)
            if name is None:
                return False
            if (
                name not in self._tool_indices
                and not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get()
            ):
                return False
            self._emit_tool_name(name, pending_calls)
            return True

        def on_close(value: Any) -> bool:
            tool_index = self.current_tool_id if self.current_tool_id >= 0 else 0
            items = self._to_tool_call_items(
                value,
                self._tool_indices,
                tool_index,
            )
            if items is None:
                return False
            for item in items:
                self._emit_tool_call(item, pending_calls)
            return True

        normal_text = self._adapter.route_tool_events(
            events,
            on_tool_open=on_open,
            on_tool_close=on_close,
            on_tool_malformed=lambda text: self._emit_malformed_tool_arguments(
                text,
                pending_calls,
            ),
        )
        return ToolStreamingParseResult(normal_text=normal_text, calls=pending_calls)

    def supports_structural_tag(self) -> bool:
        return False

    def parses_required_natively(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError(
            "structure_info not used with the response_template tool parser"
        )
