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
from collections.abc import Callable, Sequence
from typing import Any

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
from sglang.srt.parser.chat_parsing.response_parser import ResponseParser
from sglang.srt.parser.chat_parsing.response_templates import (
    ResponseTemplate,
    load_response_template,
)

logger = logging.getLogger(__name__)

_THINKING_FIELD = "thinking"
_CONTENT_FIELD = "content"
_TOOL_FIELD = "tool_calls"
_PASSTHROUGH_FIELD = "normal"
_SUPPORTED_RESPONSE_TEMPLATE_FIELDS = frozenset(
    {_THINKING_FIELD, _CONTENT_FIELD, _TOOL_FIELD}
)


def validate_response_template_for_serving(template: dict) -> ResponseTemplate:
    """Validate the template and reject semantic fields serving cannot route."""
    loaded = load_response_template(template)
    fields = set(loaded.fields)
    unsupported = (fields - _SUPPORTED_RESPONSE_TEMPLATE_FIELDS) | (
        set(loaded.defaults) - {"role"}
    )
    if unsupported:
        raise ValueError(
            "response_template contains unsupported semantic fields: "
            f"{sorted(unsupported)}. Supported fields are: "
            f"{sorted(_SUPPORTED_RESPONSE_TEMPLATE_FIELDS)}"
        )
    for name in (_THINKING_FIELD, _CONTENT_FIELD):
        field = loaded.fields.get(name)
        if field is None:
            continue
        if (
            field.content != "text"
            or field.transform is not None
            or field.join is not None
            or field.content_args.get("strip") is True
        ):
            raise ValueError(
                f"response_template field {name!r} uses semantics that cannot "
                "be streamed by the OpenAI serving adapter"
            )
    return loaded


def resolve_response_template(
    tokenizer: Any | None,
    fallback: dict | None,
) -> dict | None:
    """Load `response_template` from the tokenizer, then use the fallback."""
    if tokenizer is None:
        return fallback

    template = getattr(tokenizer, "response_template", None)
    if isinstance(template, dict):
        return template

    init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
    template = init_kwargs.get("response_template")
    if isinstance(template, dict):
        return template

    return fallback


class _ReasoningResult:
    """Duck-types `reasoning_parser.StreamingParseResult`."""

    def __init__(self, normal_text: str = "", reasoning_text: str = ""):
        self.normal_text = normal_text
        self.reasoning_text = reasoning_text


def _streaming_template(template: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(template)
    fields = result["fields"]
    if "content" not in fields and all(
        "open" in field or "open_pattern" in field for field in fields.values()
    ):
        fields["content"] = {
            "content": "text",
            "content_args": {"strip": False},
        }
    for field in fields.values():
        field["optional"] = True
    tool_field = fields.get(_TOOL_FIELD)
    if isinstance(tool_field, dict) and tool_field.get("content") == "xml-inline":
        tool_field.setdefault("content_args", {})["strict"] = True
    return result


def _tool_extraction_template(template: dict[str, Any]) -> dict[str, Any]:
    template = _streaming_template(template)
    anchor_name = (
        "start_anchor_pattern" if "start_anchor_pattern" in template else "start_anchor"
    )
    return {
        anchor_name: template[anchor_name],
        "fields": {
            _PASSTHROUGH_FIELD: {
                "content": "text",
                "content_args": {"strip": False},
            },
            _TOOL_FIELD: template["fields"][_TOOL_FIELD],
        },
    }


class ResponseTemplateStreamAdapter:
    """Stream a response template and route its generic region events."""

    def __init__(
        self,
        template: dict | ResponseTemplate,
        *,
        prefix: str | None = None,
    ):
        self._parser_template = load_response_template(template)
        self._prefix = prefix or ""
        self._stream_parser: ResponseParser | None = None
        self._pending_reasoning = ""
        self._pending_tool_start: int | None = None
        self._pending_tool_body_start: int | None = None
        self._pending_tool_streamed = False
        self._finalized = False

    def _make_parser(self, tools: Sequence[Any] | None = None) -> ResponseParser:
        parser_tools = [
            tool if isinstance(tool, dict) else tool.model_dump()
            for tool in tools or []
        ]
        return ResponseParser(
            self._parser_template,
            prefix=self._prefix,
            tools=parser_tools,
        )

    @staticmethod
    def _active_initial_tool_event(parser: ResponseParser) -> list[dict]:
        for event in reversed(parser.initial_events):
            if event.get("field") != _TOOL_FIELD:
                continue
            if event["type"] in {"region_open", "region_close", "region_malformed"}:
                return [event] if event["type"] == "region_open" else []
        return []

    @staticmethod
    def _tool_name(value: Any) -> str | None:
        function = value.get("function") if isinstance(value, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        return name if isinstance(name, str) else None

    def _generated_text(
        self,
        event: dict,
        *,
        start: int | None = None,
        end: int | None = None,
    ) -> str:
        assert self._stream_parser is not None
        start = event["start"] if start is None else start
        end = event["end"] if end is None else end
        start = max(start, self._stream_parser.prefix_end)
        return self._stream_parser.input_text[start:end] if start < end else ""

    def has_tool_region(self, text: str) -> bool:
        try:
            parser = self._make_parser()
            events = self._active_initial_tool_event(parser) + parser.feed(text)
        except (KeyError, RuntimeError, TypeError, ValueError):
            return False
        return any(
            event["type"] == "region_open" and event.get("field") == _TOOL_FIELD
            for event in events
        )

    def detect_and_parse(
        self,
        text: str,
        tools: Sequence[Any] | None = None,
    ) -> list[dict]:
        parser = self._make_parser(tools)
        self._stream_parser = parser
        events = self._active_initial_tool_event(parser) + parser.feed(text)
        _, final_events = parser.finalize()
        return events + final_events

    def feed(
        self,
        text: str,
        tools: Sequence[Any] | None = None,
    ) -> list[dict]:
        if self._finalized:
            return []
        initial_events: list[dict] = []
        if self._stream_parser is None:
            self._stream_parser = self._make_parser(tools)
            initial_events = self._active_initial_tool_event(self._stream_parser)
        return initial_events + self._stream_parser.feed(text)

    def finalize(
        self,
        tools: Sequence[Any] | None = None,
    ) -> list[dict]:
        if self._finalized:
            return []
        initial_events: list[dict] = []
        if self._stream_parser is None:
            self._stream_parser = self._make_parser(tools)
            initial_events = self._active_initial_tool_event(self._stream_parser)
        _, events = self._stream_parser.finalize()
        self._finalized = True
        return initial_events + events

    def route_reasoning_events(
        self, events: list[dict], *, stream_reasoning: bool
    ) -> tuple[str, str]:
        reasoning_parts: list[str] = []
        normal_parts: list[str] = []
        for event in events:
            field = event.get("field")
            etype = event["type"]
            if field == _THINKING_FIELD:
                if etype == "region_chunk":
                    text = self._generated_text(event)
                    if stream_reasoning:
                        reasoning_parts.append(text)
                    else:
                        self._pending_reasoning += text
                elif etype == "region_close" and not stream_reasoning:
                    reasoning_parts.append(self._pending_reasoning)
                    self._pending_reasoning = ""
            elif field == _CONTENT_FIELD and etype == "region_chunk":
                normal_parts.append(self._generated_text(event))
            elif field == _TOOL_FIELD:
                if etype == "region_malformed":
                    start = event["end"] - len(event["raw_close"])
                    normal_parts.append(self._generated_text(event, start=start))
                elif etype in {"region_open", "region_chunk", "region_close"}:
                    normal_parts.append(self._generated_text(event))
        return "".join(normal_parts), "".join(reasoning_parts)

    def route_tool_events(
        self,
        events: list[dict],
        *,
        on_tool_open: Callable[[str], bool] | None = None,
        on_tool_close: Callable[[Any], bool],
        on_tool_malformed: Callable[[str, bool], None] | None = None,
    ) -> str:
        normal_parts: list[str] = []
        malformed_starts = {
            event["start"]
            for event in events
            if event.get("field") == _TOOL_FIELD and event["type"] == "region_malformed"
        }
        for event in events:
            field = event.get("field")
            etype = event["type"]
            if field == _PASSTHROUGH_FIELD and etype == "region_chunk":
                normal_parts.append(self._generated_text(event))
            elif field == _TOOL_FIELD:
                if etype == "region_open":
                    self._pending_tool_start = event["start"]
                    self._pending_tool_body_start = event["end"]
                    name = self._tool_name(event.get("provisional_value"))
                    self._pending_tool_streamed = bool(
                        event["start"] not in malformed_starts
                        and name is not None
                        and on_tool_open is not None
                        and on_tool_open(name)
                    )
                elif etype == "region_malformed":
                    if on_tool_malformed is not None:
                        body_start = event["start"] + len(event["raw_open"])
                        body_end = event["end"] - len(event["raw_close"])
                        on_tool_malformed(
                            self._generated_text(
                                event,
                                start=body_start,
                                end=body_end,
                            ),
                            event["closed"],
                        )
                    if not self._pending_tool_streamed:
                        normal_parts.append(self._generated_text(event))
                    self._clear_pending_tool()
                elif etype == "region_close":
                    if not on_tool_close(event["value"]):
                        if (
                            self._pending_tool_streamed
                            and on_tool_malformed is not None
                        ):
                            on_tool_malformed(
                                self._generated_text(
                                    event,
                                    start=self._pending_tool_body_start,
                                    end=event["start"],
                                ),
                                True,
                            )
                        elif self._pending_tool_start is not None:
                            normal_parts.append(
                                self._generated_text(
                                    event,
                                    start=self._pending_tool_start,
                                )
                            )
                    self._clear_pending_tool()
        return "".join(normal_parts)

    def _clear_pending_tool(self) -> None:
        self._pending_tool_start = None
        self._pending_tool_body_start = None
        self._pending_tool_streamed = False


class _ResponseTemplateParserInputMixin:
    """Preserve response-template delimiters until parsing."""

    response_template: dict | None = None

    def _load_response_template(
        self,
        tokenizer: Any | None,
        response_template: dict | None,
    ) -> tuple[dict, ResponseTemplate]:
        fallback = (
            self.response_template if response_template is None else response_template
        )
        template = resolve_response_template(tokenizer, fallback)
        if template is None:
            raise ValueError("response_template is required")
        self.response_template = template
        return template, validate_response_template_for_serving(template)

    @staticmethod
    def configure_request_for_parsing(request: Any) -> None:
        request.skip_special_tokens = False
        template_kwargs = dict(getattr(request, "chat_template_kwargs", None) or {})
        template_kwargs["spaces_between_special_tokens"] = False
        request.chat_template_kwargs = template_kwargs


class ResponseTemplateReasoningDetector(_ResponseTemplateParserInputMixin):
    """Reasoning detector driven by a `response_template` grammar."""

    def __init__(
        self,
        stream_reasoning: bool = True,
        tokenizer=None,
        response_template: dict | None = None,
        prefix: str | None = None,
        **_kwargs,
    ):
        template, loaded = self._load_response_template(
            tokenizer,
            response_template,
        )
        self.stream_reasoning = stream_reasoning
        thinking = loaded.fields.get(_THINKING_FIELD)
        self.think_start_token = (
            thinking.open_literals[0] if thinking and thinking.open_literals else ""
        )
        self.think_end_token = (
            thinking.close_literals[0] if thinking and thinking.close_literals else ""
        )
        self.think_start_self_label = ""
        self.thinks_internally = False
        self.reasoning_default = "explicit_enable_thinking"
        self._adapter = ResponseTemplateStreamAdapter(
            _streaming_template(template),
            prefix=prefix,
        )

    def detect_and_parse(self, text: str) -> _ReasoningResult:
        events = self._adapter.detect_and_parse(text)
        normal_text, reasoning_text = self._adapter.route_reasoning_events(
            events, stream_reasoning=True
        )
        return _ReasoningResult(normal_text=normal_text, reasoning_text=reasoning_text)

    def parse_streaming_increment(self, new_text: str) -> _ReasoningResult:
        events = self._adapter.feed(new_text)
        normal_text, reasoning_text = self._adapter.route_reasoning_events(
            events, stream_reasoning=self.stream_reasoning
        )
        return _ReasoningResult(normal_text=normal_text, reasoning_text=reasoning_text)

    def finish(self) -> _ReasoningResult:
        events = self._adapter.finalize()
        normal_text, reasoning_text = self._adapter.route_reasoning_events(
            events, stream_reasoning=self.stream_reasoning
        )
        return _ReasoningResult(normal_text=normal_text, reasoning_text=reasoning_text)


class ResponseTemplateToolDetector(
    _ResponseTemplateParserInputMixin,
    BaseFormatDetector,
):
    """Tool-call detector driven by a `response_template` grammar."""

    def __init__(
        self,
        tokenizer=None,
        response_template: dict | None = None,
        prefix: str | None = None,
    ):
        super().__init__()
        template, loaded = self._load_response_template(
            tokenizer,
            response_template,
        )
        self._adapter = ResponseTemplateStreamAdapter(
            _tool_extraction_template(template),
            prefix=prefix,
        )
        tool_spec = loaded.fields[_TOOL_FIELD]
        if tool_spec.close_literals:
            self.eot_token = tool_spec.close_literals[0]
        self.incomplete_tool_call_indices: set[int] = set()
        self._tool_indices: dict[str, int] | None = None

    @property
    def has_incomplete_tool_call(self) -> bool:
        return bool(self.incomplete_tool_call_indices)

    def has_tool_call(self, text: str) -> bool:
        return self._adapter.has_tool_region(text)

    def _to_tool_call_item(
        self, value: Any, tool_indices: dict[str, int], tool_index: int
    ) -> ToolCallItem | None:
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
        tool_indices: dict[str, int],
        first_tool_index: int,
    ) -> list[ToolCallItem] | None:
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
        self, item: ToolCallItem, pending_calls: list[ToolCallItem]
    ) -> None:
        name_was_streamed = self.current_tool_name_sent
        self._ensure_tool_slot()
        assert self.current_tool_id >= 0
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

    def _ensure_tool_slot(self) -> None:
        if self.current_tool_id == -1:
            self.current_tool_id = 0
            self.prev_tool_call_arr = []
            self.streamed_args_for_tool = [""]
        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")

    def _emit_tool_name(self, name: str, pending_calls: list[ToolCallItem]) -> None:
        self._ensure_tool_slot()
        assert self.current_tool_id >= 0
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

    def _emit_malformed_tool_arguments(
        self,
        text: str,
        pending_calls: list[ToolCallItem],
        closed: bool,
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
        if closed:
            self.current_tool_id += 1
            self.current_tool_name_sent = False

    def _route_tool_events(
        self, events: list[dict], tool_indices: dict[str, int]
    ) -> tuple[str, list[ToolCallItem]]:
        calls: list[ToolCallItem] = []

        def on_close(value: Any) -> bool:
            items = self._to_tool_call_items(value, tool_indices, len(calls))
            if items is None:
                return False
            calls.extend(items)
            return True

        normal_text = self._adapter.route_tool_events(events, on_tool_close=on_close)
        return normal_text, calls

    def detect_and_parse(
        self, text: str, tools: list[Tool]
    ) -> ToolStreamingParseResult:
        events = self._adapter.detect_and_parse(text, tools)
        normal_text, calls = self._route_tool_events(
            events, self._get_tool_indices(tools)
        )
        return ToolStreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> ToolStreamingParseResult:
        events = self._adapter.feed(new_text, tools)
        return self._emit_streaming_events(events, tools)

    def finish(self, tools: list[Tool]) -> ToolStreamingParseResult:
        events = self._adapter.finalize(tools)
        return self._emit_streaming_events(events, tools)

    def _emit_streaming_events(
        self, events: list[dict], tools: list[Tool]
    ) -> ToolStreamingParseResult:
        if self._tool_indices is None:
            self._tool_indices = self._get_tool_indices(tools)
        tool_indices = self._tool_indices

        pending_calls: list[ToolCallItem] = []

        def on_open(name: str) -> bool:
            if name not in tool_indices and not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                return False
            self._emit_tool_name(name, pending_calls)
            return True

        def on_close(value: Any) -> bool:
            tool_index = max(self.current_tool_id, 0)
            items = self._to_tool_call_items(
                value,
                tool_indices,
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
            on_tool_malformed=lambda text, closed: self._emit_malformed_tool_arguments(
                text,
                pending_calls,
                closed,
            ),
        )
        return ToolStreamingParseResult(normal_text=normal_text, calls=pending_calls)

    def supports_structural_tag(self) -> bool:
        return False

    def validate_structure_constraint_request(
        self,
        tool_choice: Any,
        parallel_tool_calls: bool,
        strict_requested: bool,
    ) -> None:
        if tool_choice == "auto" and not parallel_tool_calls:
            raise ValueError(
                f"{type(self).__name__} cannot enforce "
                "parallel_tool_calls=False with automatic tool choice"
            )
        if tool_choice == "auto" and strict_requested:
            raise ValueError(
                f"{type(self).__name__} does not support strict tool constraints"
            )

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError(
            "structure_info not used with the response_template tool parser"
        )
