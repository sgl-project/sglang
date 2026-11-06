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
import functools
import json
import logging
from collections.abc import Callable, Sequence
from typing import Any

import msgspec

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import ToolStrictLevel, envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult as ToolStreamingParseResult,
)
from sglang.srt.function_call.core_types import (
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.response_template_structural_tag import (
    holds_arguments,
    message_start_anchors,
    message_start_openers,
    typed_parameters,
)
from sglang.srt.parser.chat_parsing.content_parsers import _apply_transform
from sglang.srt.parser.chat_parsing.response_parser import (
    ResponseParser,
    _json_valued,
)
from sglang.srt.parser.chat_parsing.response_templates import (
    ResponseTemplate,
    ResponseTemplateField,
    load_response_template,
)

logger = logging.getLogger(__name__)

_THINKING_FIELD = "thinking"
_CONTENT_FIELD = "content"
_TOOL_FIELD = "tool_calls"
_PASSTHROUGH_FIELD = "normal"
_RAW_MEMBERS_KEY = "members"
_SUPPORTED_RESPONSE_TEMPLATE_FIELDS = frozenset(
    {_THINKING_FIELD, _CONTENT_FIELD, _TOOL_FIELD}
)
# Where JSON-schema output replaces the template's framing, following the
# reasoning gate of ReasonerGrammarBackend; at the answer opener when the
# grammar frames the JSON as the answer.
GRAMMAR_AFTER_REASONING = "after_reasoning"
GRAMMAR_AT_ANSWER = "at_answer"
GRAMMAR_FROM_START = "from_start"


class _ServingTemplates(msgspec.Struct, frozen=True):
    """A validated template and the variants the serving detectors parse with."""

    source: ResponseTemplate
    streaming: ResponseTemplate
    reasoning_only: ResponseTemplate | None
    reasoning_to_answer: ResponseTemplate | None
    tool_extraction: ResponseTemplate | None
    tool_to_answer: ResponseTemplate | None


def validate_response_template_for_serving(template: dict) -> ResponseTemplate:
    """Validate the template and every variant the serving detectors build."""
    return _derive_serving_templates(template).source


def _derive_serving_templates(template: dict) -> _ServingTemplates:
    # Detectors are built several times per request; derive each template once.
    return _derive_serving_templates_from_json(json.dumps(template))


@functools.lru_cache(maxsize=8)
def _derive_serving_templates_from_json(template_json: str) -> _ServingTemplates:
    template = json.loads(template_json)
    loaded = load_response_template(template)
    _check_serving_fields(loaded)
    content = loaded.fields.get(_CONTENT_FIELD)
    answer_opens = content is not None and content.open_re is not None
    return _ServingTemplates(
        source=loaded,
        streaming=_streaming_template(template),
        reasoning_only=(
            _reasoning_only_template(template, ends_at_answer=False)
            if _THINKING_FIELD in loaded.fields
            else None
        ),
        # Only an answer with an opener can end the reasoning-only parse.
        reasoning_to_answer=(
            _reasoning_only_template(template, ends_at_answer=True)
            if _THINKING_FIELD in loaded.fields and answer_opens
            else None
        ),
        tool_extraction=(
            _tool_extraction_template(template)
            if _TOOL_FIELD in loaded.fields
            else None
        ),
        tool_to_answer=(
            _tool_extraction_template(template, ends_at_answer=True)
            if _TOOL_FIELD in loaded.fields and answer_opens
            else None
        ),
    )


def _check_serving_fields(loaded: ResponseTemplate) -> None:
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
    tool_field = loaded.fields.get(_TOOL_FIELD)
    if tool_field is not None and tool_field.open_re is None:
        raise ValueError(
            f"response_template field {_TOOL_FIELD!r} needs 'open' or "
            "'open_pattern' so tool calls can be told apart from other text"
        )


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


def grammar_holds_calls(tool_grammar: bool, grammar_start: str | None) -> bool:
    """Whether a template grammar holds the output's calls: the tool-call
    grammar, or the JSON answer's, which admits calls before the answer."""
    return tool_grammar or grammar_start == GRAMMAR_AT_ANSWER


def _load_serving_templates(
    tokenizer: Any | None, template: dict | None, fallback: dict | None
) -> tuple[dict, _ServingTemplates]:
    """Use the explicit template, then checkpoint metadata, then the detector's
    built-in fallback."""
    template = resolve_response_template(tokenizer, template)
    if template is None:
        template = fallback
    if template is None:
        raise ValueError("response_template is required")
    return template, _derive_serving_templates(template)


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


def _reasoning_only_template(
    template: dict[str, Any], *, ends_at_answer: bool
) -> ResponseTemplate:
    fields = {_THINKING_FIELD: copy.deepcopy(template["fields"][_THINKING_FIELD])}
    if ends_at_answer:
        # Text between thinking and the answer, like a tool call, stays normal.
        fields[_PASSTHROUGH_FIELD] = {}
        fields[_CONTENT_FIELD] = copy.deepcopy(template["fields"][_CONTENT_FIELD])
        if _TOOL_FIELD in template["fields"]:
            # A call's own text can hold markers that open no region.
            fields[_TOOL_FIELD] = copy.deepcopy(template["fields"][_TOOL_FIELD])
    for field in fields.values():
        field["optional"] = True
    anchor_name = (
        "start_anchor_pattern" if "start_anchor_pattern" in template else "start_anchor"
    )
    return load_response_template(
        {anchor_name: template[anchor_name], "fields": fields}
    )


def _tool_extraction_template(
    template: dict[str, Any], *, ends_at_answer: bool = False
) -> ResponseTemplate:
    tool_field = copy.deepcopy(template["fields"][_TOOL_FIELD])
    tool_field["optional"] = True
    # The detector transforms each member itself, so a malformed member drops alone;
    # the envelope keeps delimiter captures valid and hides raw members from schema
    # coercion, which must only see the call each member's transform produces.
    if tool_field.pop("transform_each", False):
        tool_field["transform"] = {_RAW_MEMBERS_KEY: "{content}"}
    fields = {_PASSTHROUGH_FIELD: {}, _TOOL_FIELD: tool_field}
    if ends_at_answer:
        # The answer opens only outside the reasoning, which stays text.
        for name in (_THINKING_FIELD, _CONTENT_FIELD):
            if name in template["fields"]:
                fields[name] = {
                    **copy.deepcopy(template["fields"][name]),
                    "optional": True,
                }
    anchor_name = (
        "start_anchor_pattern" if "start_anchor_pattern" in template else "start_anchor"
    )
    return load_response_template(
        {anchor_name: template[anchor_name], "fields": fields}
    )


def _passthrough_chunk(text: str) -> dict:
    return {"type": "region_chunk", "field": _PASSTHROUGH_FIELD, "text": text}


class _ReasoningResult:
    """Duck-types `reasoning_parser.StreamingParseResult`."""

    def __init__(self, normal_text: str = "", reasoning_text: str = ""):
        self.normal_text = normal_text
        self.reasoning_text = reasoning_text


class ResponseTemplateStreamAdapter:
    """Parse one response with a template and route its generic region events."""

    def __init__(
        self,
        template: ResponseTemplate,
        *,
        prefix: str = "",
        grammar_start: str | None = None,
        each_transform: Any = None,
        constrained: bool = False,
        held_by: ResponseTemplateField | None = None,
    ):
        self._template = template
        self._constrained = constrained
        # Only constrained parsing reads held values as JSON.
        self._held_by = held_by if constrained else None
        self._prefix = prefix
        self._parser: ResponseParser | None = None
        self._prefix_end = 0
        self._finalized = False
        tool_field = template.fields.get(_TOOL_FIELD)
        self._tool_close_re = None if tool_field is None else tool_field.close_re
        self._each_transform = each_transform
        self._pending_reasoning = ""
        self._pending_tool_streamed = False
        self._open_tool_captures: dict = {}
        self._open_region: dict | None = None
        # Where the output ends, before finalize completes a cut-off closer.
        self._output_end: int | None = None
        self._grammar_start = grammar_start
        # Once JSON-schema output has started, text passes through unparsed.
        self._verbatim = grammar_start == GRAMMAR_FROM_START
        # Whether the output wrote the answer's opener.
        self.answer_opened = False

    def _start(self, tools: Sequence[Any] | None) -> list[dict]:
        """Create the parser and replay a tool region left open by the prefix."""
        if self._parser is not None:
            return []
        self._parser = ResponseParser(
            self._template,
            prefix=self._prefix,
            tools=[
                _typed_tool(
                    tool if isinstance(tool, dict) else tool.model_dump(),
                    self._held_by,
                )
                for tool in tools or []
            ],
            constrained=self._constrained,
        )
        self._prefix_end = len(self._parser.input_text)
        self._track(self._parser.initial_events)
        if (
            self._grammar_start == GRAMMAR_AT_ANSWER
            and self._open_region is not None
            and self._open_region["field"] == _CONTENT_FIELD
        ):
            # The prompt opened the answer, so all generated text is answer
            # text, and none of the prompt's passes through.
            self._verbatim = True
            return []
        for event in reversed(self._parser.initial_events):
            if event.get("field") == _TOOL_FIELD and event["type"] != "region_chunk":
                return [event] if event["type"] == "region_open" else []
        return []

    def feed(self, text: str, tools: Sequence[Any] | None = None) -> list[dict]:
        if self._finalized:
            return []
        if self._verbatim:
            return [_passthrough_chunk(text)] if text else []
        events = self._start(tools)
        if self._verbatim:
            return [_passthrough_chunk(text)] if text else []
        return self._end_reasoning(events + self._track(self._parser.feed(text)))

    def finalize(self, tools: Sequence[Any] | None = None) -> list[dict]:
        if self._finalized:
            return []
        events = self._start(tools)
        self._finalized = True
        if self._verbatim:
            return events
        self._output_end = len(self._parser.input_text)
        events += self._parser.feed(self._closer_rest())
        return self._end_reasoning(events + self._parser.finalize()[1])

    def _track(self, events: list[dict]) -> list[dict]:
        for event in events:
            if event["type"] == "region_open":
                self._open_region = event
                if event["field"] == _CONTENT_FIELD and event["end"] > event["start"]:
                    self.answer_opened = True
            elif event["type"] in ("region_close", "region_malformed"):
                self._open_region = None
        return events

    def _closer_rest(self) -> str:
        """The rest of a closer the output stopped in after ending its region,
        when only the next message's start anchor is missing."""
        if self._open_region is None:
            return ""
        field = self._template.fields[self._open_region["field"]]
        text = self._parser.input_text[self._open_region["end"] :]
        for closer in field.close_literals or ():
            for anchor in message_start_anchors(self._template):
                if not anchor or not closer.endswith(anchor) or closer == anchor:
                    continue
                for cut in range(len(closer) - 1, len(closer) - len(anchor) - 1, -1):
                    if text.endswith(closer[:cut]):
                        return closer[cut:]
        return ""

    def _end_reasoning(self, events: list[dict]) -> list[dict]:
        # Stop parsing at the reasoning end, where the grammar backend starts
        # the JSON schema, and pass the rest through unchanged.
        if self._grammar_start not in (GRAMMAR_AFTER_REASONING, GRAMMAR_AT_ANSWER):
            return events
        if _CONTENT_FIELD in self._template.fields:
            end_field, end_types = _CONTENT_FIELD, ("region_open",)
        else:
            end_field, end_types = _THINKING_FIELD, ("region_close", "region_malformed")
        for index, event in enumerate(events):
            if event.get("field") == end_field and event["type"] in end_types:
                self._verbatim = True
                text = self._parser.input_text[event["end"] :]
                return events[: index + 1] + (
                    [_passthrough_chunk(text)] if text else []
                )
        return events

    def parse(self, text: str, tools: Sequence[Any] | None = None) -> list[dict]:
        return self.feed(text, tools) + self.finalize(tools)

    def _generated_text(self, event: dict) -> str:
        start = max(event["start"], self._prefix_end)
        return self._parser.input_text[start : event["end"]]

    def _output_text(self, event: dict) -> str:
        """The generated text of `event`, without closer bytes finalize added."""
        end = event["end"] if self._output_end is None else self._output_end
        return self._parser.input_text[
            max(event["start"], self._prefix_end) : min(event["end"], end)
        ]

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
            elif (
                field in (_CONTENT_FIELD, _PASSTHROUGH_FIELD)
                and etype == "region_chunk"
            ):
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
            elif field == _CONTENT_FIELD and etype == "region_open":
                # The answer's opener stays text, like the answer after it.
                normal_parts.append(self._generated_text(event))
            elif field == _THINKING_FIELD:
                # Unseparated output keeps its reasoning as it was written.
                normal_parts.append(
                    event["text"]
                    if etype == "region_chunk"
                    else self._output_text(event)
                )
            elif field != _TOOL_FIELD or etype == "region_chunk":
                continue
            elif etype == "region_open":
                self._open_tool_captures = event.get("captures") or {}
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
                    accepted = on_tool_close(self._transform_members(event["value"]))
                if not accepted and self._pending_tool_streamed and on_tool_dropped:
                    on_tool_dropped()
                self._pending_tool_streamed = False
        return "".join(normal_parts)

    def _transform_members(self, value: Any) -> Any:
        """Apply a `transform_each` template to each member of a tool region,
        dropping the members it cannot transform, and cast their arguments to
        the tool schemas as the parser does for transformed regions."""
        if self._each_transform is None:
            return value
        value = value[_RAW_MEMBERS_KEY]
        if not isinstance(value, list):
            logger.warning("response_template: malformed tool call value: %r", value)
            return []
        json_values = _json_valued(self._template.fields[_TOOL_FIELD])
        members = []
        for member in value:
            try:
                scope = {**self._open_tool_captures, **member}
                call = _apply_transform(self._each_transform, scope)
                members.append(
                    self._parser._coerce_tool_calls(call, json_values=json_values)
                )
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    "response_template: malformed tool call value: %r", member
                )
        return members

    def _is_unusable_tool_end(self, event: dict) -> bool:
        """Whether a tool region ended malformed, or at end of stream without its
        closer because the call was cut off."""
        if event["type"] == "region_malformed":
            return True
        text = self._parser.input_text
        # End-of-stream finalization and a matched zero-width closer such as `\Z`
        # emit the same close event, so check whether the closer matches there.
        return (
            event["type"] == "region_close"
            and self._tool_close_re is not None
            and event["start"] == len(text)
            and self._tool_close_re.match(text, len(text)) is None
        )


class ResponseTemplateReasoningDetector:
    """Reasoning detector driven by a `response_template` grammar."""

    response_template: dict | None = None
    _default_think_start = ""
    _default_think_end = ""
    thinks_internally = False
    reasoning_default = "explicit_enable_thinking"
    strict_arguments = False

    def __init__(
        self,
        stream_reasoning: bool = True,
        tokenizer=None,
        response_template: dict | None = None,
        prefix: str = "",
        force_nonempty_content: bool = False,
        grammar_start: str | None = None,
        tools: Sequence[Any] | None = None,
        constrained: bool = False,
        **_kwargs,
    ):
        template, templates = _load_serving_templates(
            tokenizer, response_template, self.response_template
        )
        self.response_template = template
        self.stream_reasoning = stream_reasoning
        self.think_excluded_tokens = None
        self._force_nonempty_content = force_nonempty_content
        self._open_reasoning = ""
        thinking = templates.source.fields.get(_THINKING_FIELD)
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
        self._thinking_field = thinking
        self._content_field = templates.source.fields.get(_CONTENT_FIELD)
        self._tool_field = templates.source.fields.get(_TOOL_FIELD)
        self._message_start = message_start_anchors(templates.source)
        # Thinking openers of output that starts the assistant message.
        self.message_start_think_openers = (
            message_start_openers(thinking, self._message_start) if thinking else []
        )
        self.think_start_self_label = ""
        reasoning_gated = (GRAMMAR_AFTER_REASONING, GRAMMAR_AT_ANSWER)
        if grammar_start in reasoning_gated and thinking is None:
            grammar_start = GRAMMAR_FROM_START
        self._grammar_start = grammar_start
        if grammar_start == GRAMMAR_AT_ANSWER:
            self._template = templates.reasoning_to_answer
        elif grammar_start == GRAMMAR_AFTER_REASONING:
            self._template = templates.reasoning_only
        else:
            self._template = templates.streaming
        self._prefix = prefix
        self._tools = tools
        self._constrained = constrained
        self._adapter = self._new_adapter()
        # Whether the parsed output opened the answer that the grammar admits
        # instead of calls, so that none of its text is a call.
        self.answer_started = False

    def _new_adapter(self) -> ResponseTemplateStreamAdapter:
        return ResponseTemplateStreamAdapter(
            self._template,
            prefix=self._prefix,
            grammar_start=self._grammar_start,
            constrained=self._constrained,
            held_by=self._tool_field if self.strict_arguments else None,
        )

    def get_think_end_token_ids(self, tokenizer) -> list[int]:
        return tokenizer.encode(self.think_end_token, add_special_tokens=False)

    def get_self_opened_think_start_token_ids(self, tokenizer) -> list[int] | None:
        return None

    def starts_message(self, prefix: str) -> bool:
        """Whether output after `prefix` starts the assistant message, and so
        may skip the reasoning."""
        return _starts_message(self._template, prefix)

    def answer_opener(self, prefix: str) -> str:
        """The literal opener of an answer right after `prefix`, or "" unless
        `prefix` starts the assistant message."""
        if self._content_field is None or not self.starts_message(prefix):
            return ""
        openers = message_start_openers(self._content_field, self._message_start)
        return openers[0] if openers else ""

    def answer_rest_structural_tag(self, prefix: str) -> dict[str, Any] | None:
        """A structural tag for the rest of an answer that `prefix` leaves open,
        or `None` if none applies."""
        return None

    def json_answer_structural_tag(
        self,
        json_schema: Any,
        *,
        tools: list[dict[str, Any]],
        parallel_tool_calls: bool = True,
        message_start: bool = False,
    ) -> dict[str, Any] | None:
        """A structural tag for JSON framed as the answer, after the reasoning
        or, at the `message_start`, from the first token, or `None` to apply the
        JSON schema directly. Calls to `tools` may come first."""
        return None

    def json_output_started(self, text: str) -> bool:
        """Whether generated `text` reaches the start of JSON-schema output."""
        adapter = self._new_adapter()
        adapter.parse(text, self._tools)
        return adapter._verbatim

    def get_answer_frame(self) -> None:
        return None

    def detect_and_parse(self, text: str) -> _ReasoningResult:
        adapter = self._new_adapter()
        result = adapter.route_reasoning_events(
            adapter.parse(text, self._tools), stream_reasoning=True
        )
        self._record_answer(adapter)
        if self._force_nonempty_content and not result.normal_text:
            result.normal_text, result.reasoning_text = result.reasoning_text, ""
        return result

    def parse_streaming_increment(self, new_text: str) -> _ReasoningResult:
        events = self._adapter.feed(new_text, self._tools)
        self._record_answer(self._adapter)
        if self._force_nonempty_content:
            self._track_open_reasoning(events)
        return self._adapter.route_reasoning_events(
            events, stream_reasoning=self.stream_reasoning
        )

    def finish(self) -> _ReasoningResult:
        events = self._adapter.finalize(self._tools)
        self._record_answer(self._adapter)
        result = self._adapter.route_reasoning_events(
            events, stream_reasoning=self.stream_reasoning
        )
        # Under force_nonempty_content, reasoning left open at the end of the
        # stream is also returned as content.
        if self._force_nonempty_content:
            unclosed_reasoning = self._track_open_reasoning(events)
            if unclosed_reasoning is not None:
                result.normal_text += unclosed_reasoning
                result.reasoning_text = ""
        return result

    def _record_answer(self, adapter: ResponseTemplateStreamAdapter) -> None:
        # A grammar that holds the calls admits them or an answer, so no call
        # follows the answer's opener.
        self.answer_started = self._constrained and adapter.answer_opened

    def _track_open_reasoning(self, events: list[dict]) -> str | None:
        """Accumulate open thinking text; return the text of a region `events` close."""
        closed = None
        for event in events:
            if event.get("field") != _THINKING_FIELD:
                continue
            if event["type"] == "region_chunk":
                self._open_reasoning += event["text"]
            elif event["type"] in ("region_close", "region_malformed"):
                closed, self._open_reasoning = self._open_reasoning, ""
        return closed


class ResponseTemplateToolDetector(BaseFormatDetector):
    """Tool-call detector driven by a `response_template` grammar."""

    response_template: dict | None = None
    constrains_every_turn = False
    strict_arguments = False

    def __init__(
        self,
        tokenizer=None,
        response_template: dict | None = None,
        prefix: str = "",
        tools: list[Tool] | None = None,
        constrained: bool = False,
    ):
        super().__init__()
        template, templates = _load_serving_templates(
            tokenizer, response_template, self.response_template
        )
        self.response_template = template
        if templates.tool_extraction is None:
            raise ValueError(f"response_template has no {_TOOL_FIELD!r} field")
        # Under a grammar that admits calls or an answer, no call follows the
        # answer's opener, so the parse ends there.
        self._ends_at_answer = constrained and templates.tool_to_answer is not None
        self._template = (
            templates.tool_to_answer
            if self._ends_at_answer
            else templates.tool_extraction
        )
        self._prefix = prefix
        self._tools = tools
        self._constrained = constrained
        self._tool_field = templates.source.fields[_TOOL_FIELD]
        self._each_transform = (
            self._tool_field.transform if self._tool_field.transform_each else None
        )
        self._adapter = self._new_adapter()
        self._thinking_field = templates.source.fields.get(_THINKING_FIELD)
        self._content_field = templates.source.fields.get(_CONTENT_FIELD)
        self._message_start = message_start_anchors(templates.source)
        self.tool_close_literals = self._tool_field.close_literals or []
        if self.tool_close_literals:
            self.eot_token = self.tool_close_literals[0]
        self.incomplete_tool_call_indices: set[int] = set()
        self._tool_indices: dict[str, int] | None = None

    def _new_adapter(self) -> ResponseTemplateStreamAdapter:
        return ResponseTemplateStreamAdapter(
            self._template,
            prefix=self._prefix,
            grammar_start=GRAMMAR_AT_ANSWER if self._ends_at_answer else None,
            each_transform=self._each_transform,
            constrained=self._constrained,
            held_by=self._tool_field if self.strict_arguments else None,
        )

    def has_tool_call(self, text: str) -> bool:
        adapter = self._new_adapter()
        return any(
            event["type"] == "region_open" and event.get("field") == _TOOL_FIELD
            for event in adapter.parse(text)
        )

    def _to_tool_call(
        self, value: Any, tool_indices: dict[str, int]
    ) -> tuple[str, str] | None:
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
        return name, arguments

    def _to_tool_call_items(
        self,
        value: Any,
        tool_indices: dict[str, int],
        first_tool_index: int,
    ) -> list[ToolCallItem]:
        """The region's accepted calls, numbered from `first_tool_index`."""
        values = value if isinstance(value, list) else [value]
        calls = [
            call
            for call in (self._to_tool_call(item, tool_indices) for item in values)
            if call is not None
        ]
        return [
            ToolCallItem(
                tool_index=first_tool_index + offset, name=name, parameters=arguments
            )
            for offset, (name, arguments) in enumerate(calls)
        ]

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
        adapter = self._new_adapter()
        tool_indices = self._get_tool_indices(tools)
        calls: list[ToolCallItem] = []

        def on_close(value: Any) -> bool:
            items = self._to_tool_call_items(value, tool_indices, len(calls))
            if not items:
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
            if not items:
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

    def open_call_closer(self, text: str) -> str | None:
        """The closer for a call that `text` leaves open at its end."""
        if not self.tool_close_literals:
            return None
        adapter = self._new_adapter()
        for event in reversed(adapter.parse(text, self._tools)):
            if event.get("field") == _TOOL_FIELD and event["type"] != "region_chunk":
                open_at_end = event["type"] == "region_close" and event["start"] == len(
                    adapter._parser.input_text
                )
                return self.tool_close_literals[0] if open_at_end else None
        return None

    def get_structural_tag(
        self,
        tools: list[Tool] | None = None,
        tool_choice: Any = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
        message_start: bool = False,
    ):
        return None

    def validate_structure_constraint_request(
        self,
        tool_choice: Any,
        parallel_tool_calls: bool,
        strict_requested: bool,
        *,
        tools: list[Tool],
        thinking_mode: bool = False,
    ):
        """Raise for a requested constraint the template cannot hold. Return
        the template's own tag for this request's tool calls, or `None` to
        leave them to the generic constraint."""
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


def _typed_tool(tool: dict, held_by: ResponseTemplateField | None = None) -> dict:
    """`tool`, in either API's shape, as template grammars constrain it, strict
    where `held_by` holds its arguments."""
    function = tool.get("function", tool)
    if not isinstance(function, dict):
        return tool
    function = dict(function)
    parameters = function.get("parameters")
    if isinstance(parameters, dict):
        function["parameters"] = parameters = typed_parameters(parameters)
    if envs.SGLANG_TOOL_STRICT_LEVEL.get() >= ToolStrictLevel.PARAMETER or (
        held_by is not None
        and not function.get("strict")
        and isinstance(parameters, dict)
        and holds_arguments(held_by, parameters)
    ):
        function["strict"] = True
    return {**tool, "function": function} if "function" in tool else function


def _starts_message(template: ResponseTemplate, prefix: str) -> bool:
    ends = [match.end() for match in template.start_anchor_re.finditer(prefix)]
    return bool(ends) and ends[-1] == len(prefix)


def tool_close_tokens(
    tool_call_parser: str | None, tokenizer: Any | None
) -> dict[int, str]:
    """Single-token tool-call closers of a response-template tool parser, by id.

    Detokenization keeps these when they stop generation, so a tool call that
    reaches end of stream without its closer was cut off."""
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    detector_class = FunctionCallParser.ToolCallParserEnum.get(tool_call_parser)
    if (
        tokenizer is None
        or detector_class is None
        or not issubclass(detector_class, ResponseTemplateToolDetector)
    ):
        return {}
    try:
        detector = detector_class(tokenizer=tokenizer)
    except ValueError:
        return {}
    closers = {}
    for literal in detector.tool_close_literals:
        ids = tokenizer.encode(literal, add_special_tokens=False)
        if len(ids) == 1:
            closers[ids[0]] = literal
    return closers


def tool_close_token_ids(
    tool_call_parser: str | None, tokenizer: Any | None
) -> frozenset[int]:
    return frozenset(tool_close_tokens(tool_call_parser, tokenizer))
