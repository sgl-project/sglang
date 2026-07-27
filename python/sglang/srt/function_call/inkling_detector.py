import json
import logging
import re
from collections.abc import Mapping
from typing import List, Optional

from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow
from xgrammar import StructuralTag

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import _is_complete_json, _partial_json_loads
from sglang.srt.parser.inkling_tokenizer import (
    CONTENT_INVOKE_TOOL_JSON,
    END_MESSAGE,
    INKLING_CONTROL_TOKENS,
    INKLING_SPECIAL_TOKEN_IDS,
    MESSAGE_MODEL,
)

logger = logging.getLogger(__name__)


class InklingDetector(BaseFormatDetector):
    """
    Detector for Inkling structured tool calls.

    Format:
        <|message_model|>name<|content_invoke_tool_json|>{"name":"...","args":{...}}<|end_message|>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = CONTENT_INVOKE_TOOL_JSON
        self.eot_token = END_MESSAGE
        self.tool_call_regex = re.compile(
            re.escape(self.bot_token) + r"\s*(.*?)\s*" + re.escape(self.eot_token),
            re.DOTALL,
        )
        self._current_header_name: str | None = None

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=self._clean_normal_text(text))

        try:
            calls: list[ToolCallItem] = []
            for match in self.tool_call_regex.finditer(text):
                try:
                    payload = json.loads(match.group(1).strip())
                except json.JSONDecodeError as exc:
                    logger.warning("Invalid Inkling tool call JSON: %s", exc)
                    continue
                if not isinstance(payload, Mapping):
                    logger.warning("Invalid Inkling tool call payload: %s", payload)
                    continue
                _, header_name = self._split_trailing_tool_header(text[: match.start()])
                call = self._tool_call_item(
                    payload, tools, len(calls), header_name=header_name
                )
                if call is not None:
                    calls.append(call)

            if not calls:
                # Every candidate call was rejected (bad payload or a
                # header/payload name mismatch). Match the framework contract
                # every other detector follows: normal_text is only the content
                # BEFORE the tool marker — the rejected tool-call region is
                # dropped, never regurgitated as visible content.
                prefix, _ = self._split_trailing_tool_header(
                    text[: text.find(self.bot_token)]
                )
                return StreamingParseResult(normal_text=self._clean_normal_text(prefix))

            normal_prefix, _ = self._split_trailing_tool_header(
                text[: text.find(self.bot_token)]
            )
            normal_text = self._clean_normal_text(normal_prefix)
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as exc:
            logger.error("Error in Inkling detect_and_parse: %s", exc, exc_info=True)
            prefix, _ = self._split_trailing_tool_header(
                text[: text.find(self.bot_token)]
            )
            return StreamingParseResult(normal_text=self._clean_normal_text(prefix))

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        # Drain every complete call in the delta: this detector has no
        # stream-end flush, so anything left in self._buffer is lost.
        self._buffer += new_text
        all_calls: list[ToolCallItem] = []
        normal_parts: list[str] = []
        while True:
            result, made_progress = self._parse_buffered_increment(tools)
            if result.normal_text:
                normal_parts.append(result.normal_text)
            if result.calls:
                all_calls.extend(result.calls)
            if not made_progress:
                break
        return StreamingParseResult(
            normal_text="".join(normal_parts),
            calls=all_calls,
        )

    def _parse_buffered_increment(
        self, tools: List[Tool]
    ) -> tuple[StreamingParseResult, bool]:
        # One drain step: emit a text run or one complete call; the bool is
        # whether the buffer advanced (the caller loops while it does).
        current_text = self._buffer

        if self.bot_token not in current_text:
            header_start = self._pending_tool_header_start(current_text)
            if header_start is not None:
                safe_text = current_text[:header_start]
                self._buffer = current_text[header_start:]
                return (
                    StreamingParseResult(
                        normal_text=self._clean_normal_text(safe_text)
                    ),
                    False,
                )
            # Hold back a partial prefix of ANY token _clean_normal_text
            # strips — emitting a split control token leaks its first half as
            # visible text (the completed token would have been stripped).
            partial_len = max(
                self._ends_with_partial_token(current_text, token)
                for token in INKLING_CONTROL_TOKENS
            )
            if partial_len:
                safe_text = current_text[:-partial_len]
                self._buffer = current_text[-partial_len:]
            else:
                safe_text = current_text
                self._buffer = ""
            return (
                StreamingParseResult(normal_text=self._clean_normal_text(safe_text)),
                False,
            )

        bot_pos = current_text.find(self.bot_token)
        if bot_pos > 0:
            normal_text, self._current_header_name = self._split_trailing_tool_header(
                current_text[:bot_pos]
            )
            self._buffer = current_text[bot_pos:]
            normal_text = self._clean_normal_text(normal_text)
            if normal_text:
                # prefix stripped, call now at buffer head -> keep draining
                return StreamingParseResult(normal_text=normal_text), True
            current_text = self._buffer

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        start_idx = len(self.bot_token)
        while start_idx < len(current_text) and current_text[start_idx].isspace():
            start_idx += 1

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
        try:
            payload, end_idx = _partial_json_loads(current_text[start_idx:], flags)
        except (MalformedJSON, json.JSONDecodeError):
            return StreamingParseResult(), False
        if not isinstance(payload, Mapping):
            return StreamingParseResult(), False

        calls: list[ToolCallItem] = []
        name = payload.get("name")
        if (
            not self.current_tool_name_sent
            and isinstance(name, str)
            and (self._current_header_name is None or self._current_header_name == name)
        ):
            self._ensure_current_tool()
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=name,
                    parameters="",
                )
            )
            self.current_tool_name_sent = True
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": name,
                "arguments": {},
            }

        json_text = current_text[start_idx : start_idx + end_idx]
        if not _is_complete_json(json_text):
            return StreamingParseResult(calls=calls), False

        call = self._tool_call_item(
            payload,
            tools,
            self.current_tool_id,
            header_name=self._current_header_name,
        )
        if call is None:
            # Drop only the rejected call's span, not the whole buffer, or a
            # trailing valid call dies; clear the header so it can't leak.
            self._abandon_current_tool()
            self._buffer = self._remaining_after_call(current_text, start_idx + end_idx)
            self._current_header_name = None
            return StreamingParseResult(calls=calls), True

        if self.current_tool_id == -1:
            self._ensure_current_tool()

        args = json.loads(call.parameters)
        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": call.name,
            "arguments": args,
        }
        sent = self.streamed_args_for_tool[self.current_tool_id]
        remaining_args = call.parameters[len(sent) :]
        if remaining_args:
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=None,
                    parameters=remaining_args,
                )
            )
            self.streamed_args_for_tool[self.current_tool_id] += remaining_args

        self._buffer = self._remaining_after_call(current_text, start_idx + end_idx)
        self.current_tool_id += 1
        self.current_tool_name_sent = False
        self._current_header_name = None
        return StreamingParseResult(calls=calls), True

    def structure_info(self) -> _GetInfoFunc:
        def info(name: str) -> StructureInfo:
            trigger = f"{MESSAGE_MODEL}{name}{self.bot_token}"
            return StructureInfo(
                begin=f'{trigger}{{"name":"{name}","args":',
                end=f"}}{self.eot_token}",
                trigger=trigger,
            )

        return info

    def get_auto_tool_call_structural_tag(
        self, tools: Optional[List[Tool]] = None
    ) -> StructuralTag:
        """Constrain JSON after Inkling's tool-payload trigger token.

        Automatic tool choice still permits unconstrained assistant text. Once
        the model emits ``CONTENT_INVOKE_TOOL_JSON``, XGrammar requires a
        complete ``{"name": string, "args": object}`` payload followed by
        ``END_MESSAGE``. This mirrors the TML sampling default used by the OAI
        API and intentionally does not restrict names to the request's tools.
        """
        del tools
        return StructuralTag.model_validate(
            {
                "type": "structural_tag",
                "format": {
                    "type": "token_triggered_tags",
                    "trigger_tokens": [
                        INKLING_SPECIAL_TOKEN_IDS[CONTENT_INVOKE_TOOL_JSON]
                    ],
                    "tags": [
                        {
                            "type": "tag",
                            "begin": {
                                "type": "token",
                                "token": INKLING_SPECIAL_TOKEN_IDS[
                                    CONTENT_INVOKE_TOOL_JSON
                                ],
                            },
                            "content": {
                                "type": "json_schema",
                                "json_schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "args": {"type": "object"},
                                    },
                                    "required": ["name", "args"],
                                    "additionalProperties": False,
                                },
                            },
                            "end": {
                                "type": "token",
                                "token": INKLING_SPECIAL_TOKEN_IDS[END_MESSAGE],
                            },
                        }
                    ],
                },
            }
        )

    def _tool_call_item(
        self,
        payload: Mapping[str, object],
        tools: List[Tool],
        call_index: int,
        *,
        header_name: str | None = None,
    ) -> ToolCallItem | None:
        name = payload.get("name")
        args = payload.get("args")
        if not isinstance(name, str) or not isinstance(args, Mapping):
            logger.warning("Invalid Inkling tool call payload: %s", payload)
            return None
        if header_name is not None and header_name != name:
            logger.warning(
                "Inkling tool header %r does not match payload name %r",
                header_name,
                name,
            )
            return None

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)
        if name not in self._tool_indices:
            # Surface the call anyway (OpenAI behavior for hallucinated tools):
            # the harness sees a structured tool_call, returns a tool error, and
            # the model can self-correct — instead of the serialized invocation
            # degrading into terminal answer text.
            logger.warning("Surfacing Inkling call to undeclared tool: %s", name)

        return ToolCallItem(
            tool_index=call_index,
            name=name,
            parameters=json.dumps(args, ensure_ascii=False),
        )

    def _ensure_current_tool(self) -> None:
        if self.current_tool_id == -1:
            self.current_tool_id = 0
        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")

    def _abandon_current_tool(self) -> None:
        """Discard the in-flight call after a rejected payload.

        Resetting ``current_tool_id`` to -1 here would collide the NEXT valid
        call with tool index 0 (``_ensure_current_tool`` maps -1 -> 0) and
        slice its arguments against index 0's already-streamed args. Keep the
        counter: an unannounced slot is simply reused; an announced slot is
        abandoned by advancing past it.
        """
        if self.current_tool_name_sent:
            self.current_tool_id += 1
        self.current_tool_name_sent = False
        self._current_header_name = None

    def _split_trailing_tool_header(self, text: str) -> tuple[str, str | None]:
        message_pos = self._pending_tool_header_start(text)
        if message_pos is None:
            return text, None
        header = text[message_pos + len(MESSAGE_MODEL) :]
        return text[:message_pos], header.strip() or None

    def _pending_tool_header_start(self, text: str) -> int | None:
        """Position of a trailing ``<|message_model|>`` whose header (the text
        after it) contains no complete special token yet — i.e. a possible
        tool-call header still forming."""
        message_pos = text.rfind(MESSAGE_MODEL)
        if message_pos < 0:
            return None
        header = text[message_pos + len(MESSAGE_MODEL) :]
        if any(token in header for token in INKLING_CONTROL_TOKENS):
            return None
        return message_pos

    def _remaining_after_call(self, text: str, end_idx: int) -> str:
        remaining = text[end_idx:]
        if remaining.startswith(self.eot_token):
            return remaining[len(self.eot_token) :]
        if self.eot_token in remaining:
            return remaining.split(self.eot_token, 1)[1]
        return remaining

    def _clean_normal_text(self, text: str) -> str:
        for token in INKLING_CONTROL_TOKENS:
            text = text.replace(token, "")
        return text
