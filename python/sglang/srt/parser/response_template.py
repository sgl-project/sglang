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
from sglang.srt.parser.chat_parsing.content_parsers import _apply_transform
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
        ):
            raise ValueError(
                f"response_template field {name!r} uses semantics that cannot "
                "be streamed by the OpenAI serving adapter"
            )
    return loaded


def resolve_response_template(
    tokenizer: Any | None,
    template: dict | None = None,
) -> dict | None:
    """Return `template` if given, otherwise the tokenizer's metadata."""
    if template is not None or tokenizer is None:
        return template
    template = getattr(tokenizer, "response_template", None)
    if not isinstance(template, dict):
        init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
        template = init_kwargs.get("response_template")
    return template if isinstance(template, dict) else None


def configure_response_template_request(request: Any) -> None:
    """Preserve response-template delimiters during detokenization."""
    request.skip_special_tokens = False
    template_kwargs = dict(getattr(request, "chat_template_kwargs", None) or {})
    template_kwargs["spaces_between_special_tokens"] = False
    request.chat_template_kwargs = template_kwargs


def _load_serving_template(
    tokenizer: Any | None, template: dict | None, fallback: dict | None
) -> tuple[dict, ResponseTemplate]:
    """Use the explicit template, then checkpoint metadata, then the detector's
    built-in fallback."""
    template = resolve_response_template(tokenizer, template) or fallback
    if template is None:
        raise ValueError("response_template is required")
    return template, validate_response_template_for_serving(template)


def _streaming_template(template: dict[str, Any]) -> ResponseTemplate:
    result = copy.deepcopy(template)
    fields = result["fields"]
    if _CONTENT_FIELD not in fields and all(
        "open" in field or "open_pattern" in field for field in fields.values()
    ):
        fields[_CONTENT_FIELD] = {}
    for field in fields.values():
        field["optional"] = True
    return load_response_template(result)


def _tool_extraction_template(template: dict[str, Any]) -> ResponseTemplate:
    tool_field = copy.deepcopy(template["fields"][_TOOL_FIELD])
    tool_field["optional"] = True
    anchor_name = (
        "start_anchor_pattern" if "start_anchor_pattern" in template else "start_anchor"
    )
    return load_response_template(
        {
            anchor_name: template[anchor_name],
            "fields": {_PASSTHROUGH_FIELD: {}, _TOOL_FIELD: tool_field},
        }
    )


class _ReasoningResult:
    """Duck-types `reasoning_parser.StreamingParseResult`."""

    def __init__(self, normal_text: str = "", reasoning_text: str = ""):
        self.normal_text = normal_text
        self.reasoning_text = reasoning_text


class ResponseTemplateStreamAdapter:
    """Parse one response with a template and route its generic region events."""

    def __init__(self, template: ResponseTemplate, *, prefix: str = ""):
        self._template = template
        self._prefix = prefix
        self._parser: ResponseParser | None = None
        self._prefix_end = 0
        self._finalized = False
        tool_field = template.fields.get(_TOOL_FIELD)
        self._tool_has_closer = (
            tool_field is not None and tool_field.close_re is not None
        )
        self._pending_reasoning = ""
        self._pending_tool_streamed = False

    def _start(self, tools: Sequence[Any] | None) -> list[dict]:
        """Create the parser and replay a tool region left open by the prefix."""
        if self._parser is not None:
            return []
        self._parser = ResponseParser(
            self._template,
            prefix=self._prefix,
            tools=[
                tool if isinstance(tool, dict) else tool.model_dump()
                for tool in tools or []
            ],
        )
        self._prefix_end = len(self._parser.input_text)
        for event in reversed(self._parser.initial_events):
            if event.get("field") == _TOOL_FIELD and event["type"] != "region_chunk":
                return [event] if event["type"] == "region_open" else []
        return []

    def feed(self, text: str, tools: Sequence[Any] | None = None) -> list[dict]:
        if self._finalized:
            return []
        events = self._start(tools)
        return events + self._parser.feed(text)

    def finalize(self, tools: Sequence[Any] | None = None) -> list[dict]:
        if self._finalized:
            return []
        events = self._start(tools)
        self._finalized = True
        return events + self._parser.finalize()[1]

    def parse(self, text: str, tools: Sequence[Any] | None = None) -> list[dict]:
        return self.feed(text, tools) + self.finalize(tools)

    def _generated_text(self, event: dict) -> str:
        start = max(event["start"], self._prefix_end)
        return self._parser.input_text[start : event["end"]]

    def _open_tool_name(self, captures: dict | None) -> str | None:
        """Name already fixed by the opener, before the region body exists."""
        field = self._template.fields.get(_TOOL_FIELD)
        transform = None if field is None or field.transform_each else field.transform
        function = transform.get("function") if isinstance(transform, dict) else None
        if not isinstance(function, dict):
            return None
        try:
            name = _apply_transform(function.get("name"), captures or {})
        except (KeyError, ValueError):
            return None
        return name if isinstance(name, str) else None

    def route_reasoning_events(
        self, events: list[dict], *, stream_reasoning: bool
    ) -> _ReasoningResult:
        reasoning_parts: list[str] = []
        normal_parts: list[str] = []
        for event in events:
            field = event.get("field")
            etype = event["type"]
            if field == _THINKING_FIELD:
                if etype == "region_chunk":
                    if stream_reasoning:
                        reasoning_parts.append(event["text"])
                    else:
                        self._pending_reasoning += event["text"]
                elif etype == "region_close" and not stream_reasoning:
                    reasoning_parts.append(self._pending_reasoning)
                    self._pending_reasoning = ""
            elif field == _CONTENT_FIELD and etype == "region_chunk":
                normal_parts.append(event["text"])
            elif field == _TOOL_FIELD:
                # Hand tool regions back verbatim so the tool detector can parse them.
                normal_parts.append(
                    event["text"]
                    if etype == "region_chunk"
                    else self._generated_text(event)
                )
        return _ReasoningResult("".join(normal_parts), "".join(reasoning_parts))

    def route_tool_events(
        self,
        events: list[dict],
        *,
        on_tool_open: Callable[[str], bool] | None = None,
        on_tool_close: Callable[[Any], bool],
        on_tool_dropped: Callable[[], None] | None = None,
    ) -> str:
        """Route tool regions to the callbacks and return the passthrough text.

        A tool call is emitted only when it ends with its closer, parses, and is
        accepted by `on_tool_close`. Otherwise it is dropped, and a name that was
        already streamed stays an incomplete call."""
        normal_parts: list[str] = []
        unusable_opens: set[int] = set()
        open_index = None
        for index, event in enumerate(events):
            if event.get("field") == _TOOL_FIELD:
                if event["type"] == "region_open":
                    open_index = index
                elif self._is_unusable_tool_end(event):
                    unusable_opens.add(open_index)
        for index, event in enumerate(events):
            field = event.get("field")
            etype = event["type"]
            if field == _PASSTHROUGH_FIELD and etype == "region_chunk":
                normal_parts.append(event["text"])
            elif field != _TOOL_FIELD or etype == "region_chunk":
                continue
            elif etype == "region_open":
                name = self._open_tool_name(event.get("captures"))
                self._pending_tool_streamed = bool(
                    index not in unusable_opens
                    and name is not None
                    and on_tool_open is not None
                    and on_tool_open(name)
                )
            else:
                if self._is_unusable_tool_end(event):
                    logger.warning(
                        "response_template: dropping malformed or cut-off tool call"
                    )
                    accepted = False
                else:
                    accepted = on_tool_close(event["value"])
                if not accepted and self._pending_tool_streamed and on_tool_dropped:
                    on_tool_dropped()
                self._pending_tool_streamed = False
        return "".join(normal_parts)

    def _is_unusable_tool_end(self, event: dict) -> bool:
        """Whether a tool region ended malformed, or at end of stream without its
        closer because the call was cut off."""
        return event["type"] == "region_malformed" or (
            event["type"] == "region_close"
            and self._tool_has_closer
            and event["start"] == len(self._parser.input_text)
        )


class ResponseTemplateReasoningDetector:
    """Reasoning detector driven by a `response_template` grammar."""

    response_template: dict | None = None
    _default_think_start = ""
    _default_think_end = ""
    thinks_internally = False
    reasoning_default = "explicit_enable_thinking"

    def __init__(
        self,
        stream_reasoning: bool = True,
        tokenizer=None,
        response_template: dict | None = None,
        prefix: str = "",
        **_kwargs,
    ):
        template, loaded = _load_serving_template(
            tokenizer, response_template, self.response_template
        )
        self.response_template = template
        self.stream_reasoning = stream_reasoning
        thinking = loaded.fields.get(_THINKING_FIELD)
        self.think_start_token = (
            thinking.open_literals[0]
            if thinking and thinking.open_literals
            else self._default_think_start
        )
        self.think_end_token = (
            thinking.close_literals[0]
            if thinking and thinking.close_literals
            else self._default_think_end
        )
        self.think_start_self_label = ""
        self._template = _streaming_template(template)
        self._prefix = prefix
        self._adapter = ResponseTemplateStreamAdapter(self._template, prefix=prefix)

    def detect_and_parse(self, text: str) -> _ReasoningResult:
        adapter = ResponseTemplateStreamAdapter(self._template, prefix=self._prefix)
        return adapter.route_reasoning_events(
            adapter.parse(text), stream_reasoning=True
        )

    def parse_streaming_increment(self, new_text: str) -> _ReasoningResult:
        return self._adapter.route_reasoning_events(
            self._adapter.feed(new_text), stream_reasoning=self.stream_reasoning
        )

    def finish(self) -> _ReasoningResult:
        return self._adapter.route_reasoning_events(
            self._adapter.finalize(), stream_reasoning=self.stream_reasoning
        )


class ResponseTemplateToolDetector(BaseFormatDetector):
    """Tool-call detector driven by a `response_template` grammar."""

    response_template: dict | None = None

    def __init__(
        self,
        tokenizer=None,
        response_template: dict | None = None,
        prefix: str = "",
    ):
        super().__init__()
        template, loaded = _load_serving_template(
            tokenizer, response_template, self.response_template
        )
        self.response_template = template
        if _TOOL_FIELD not in loaded.fields:
            raise ValueError(f"response_template has no {_TOOL_FIELD!r} field")
        self._template = _tool_extraction_template(template)
        self._prefix = prefix
        self._adapter = ResponseTemplateStreamAdapter(self._template, prefix=prefix)
        self.tool_close_literals = loaded.fields[_TOOL_FIELD].close_literals or []
        if self.tool_close_literals:
            self.eot_token = self.tool_close_literals[0]
        self.incomplete_tool_call_indices: set[int] = set()
        self._tool_indices: dict[str, int] | None = None

    def has_tool_call(self, text: str) -> bool:
        adapter = ResponseTemplateStreamAdapter(self._template, prefix=self._prefix)
        return any(
            event["type"] == "region_open" and event.get("field") == _TOOL_FIELD
            for event in adapter.feed(text)
        )

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
        pending_calls.append(
            ToolCallItem(tool_index=self.current_tool_id, name=name, parameters="")
        )
        self.prev_tool_call_arr[self.current_tool_id] = {"name": name, "arguments": ""}
        self.incomplete_tool_call_indices.add(self.current_tool_id)
        self.current_tool_name_sent = True

    def _drop_streamed_tool_call(self) -> None:
        """Leave the streamed name as an incomplete call and move to the next."""
        self.current_tool_id += 1
        self.current_tool_name_sent = False

    def detect_and_parse(
        self, text: str, tools: list[Tool]
    ) -> ToolStreamingParseResult:
        adapter = ResponseTemplateStreamAdapter(self._template, prefix=self._prefix)
        tool_indices = self._get_tool_indices(tools)
        calls: list[ToolCallItem] = []

        def on_close(value: Any) -> bool:
            items = self._to_tool_call_items(value, tool_indices, len(calls))
            if items is None:
                return False
            calls.extend(items)
            return True

        normal_text = adapter.route_tool_events(
            adapter.parse(text, tools), on_tool_close=on_close
        )
        return ToolStreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> ToolStreamingParseResult:
        return self._emit_streaming_events(self._adapter.feed(new_text, tools), tools)

    def finish(self, tools: list[Tool]) -> ToolStreamingParseResult:
        return self._emit_streaming_events(self._adapter.finalize(tools), tools)

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
            items = self._to_tool_call_items(
                value, tool_indices, max(self.current_tool_id, 0)
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
            on_tool_dropped=self._drop_streamed_tool_call,
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


def tool_close_token_ids(
    tool_call_parser: str | None, tokenizer: Any | None
) -> frozenset[int]:
    """Token ids of the tool-call closers of a response-template tool parser.

    Detokenization keeps these when they stop generation, so a tool call that
    reaches end of stream without its closer was cut off."""
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    detector_class = FunctionCallParser.ToolCallParserEnum.get(tool_call_parser)
    if (
        tokenizer is None
        or detector_class is None
        or not issubclass(detector_class, ResponseTemplateToolDetector)
    ):
        return frozenset()
    try:
        detector = detector_class(tokenizer=tokenizer)
    except ValueError:
        return frozenset()
    token_ids = (
        tokenizer.encode(literal, add_special_tokens=False)
        for literal in detector.tool_close_literals
    )
    return frozenset(ids[0] for ids in token_ids if len(ids) == 1)
