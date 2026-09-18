import json
import logging
from typing import List

import orjson
from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import _partial_json_loads

logger = logging.getLogger(__name__)


class CohereCommand4Detector(BaseFormatDetector):
    """Detector for ``<|START_ACTION|>[...JSON array...]<|END_ACTION|>``."""

    def __init__(self):
        super().__init__()
        self.bot_token = "<|START_ACTION|>"
        self.eot_token = "<|END_ACTION|>"
        # Per the chat template the array items are separated by ``,`` only --
        # the surrounding newlines/whitespace are also valid JSON whitespace.
        self.tool_call_separator = ","

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    @staticmethod
    def _normalize_calls(arr) -> List[dict]:
        """Translate Cohere's per-item shape ``{tool_call_id, tool_name,
        parameters}`` into the shape ``parse_base_json`` expects (``name`` /
        ``parameters``). Drops ``tool_call_id`` since the OpenAI Chat
        Completions schema assigns its own id."""
        if isinstance(arr, dict):
            arr = [arr]
        if not isinstance(arr, list):
            return []
        out: List[dict] = []
        for act in arr:
            if not isinstance(act, dict):
                continue
            normalized = dict(act)
            if "name" not in normalized and "tool_name" in normalized:
                normalized["name"] = normalized.pop("tool_name")
            normalized.pop("tool_call_id", None)
            out.append(normalized)
        return out

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Non-streaming parse."""
        idx = text.find(self.bot_token)
        if idx == -1:
            return StreamingParseResult(normal_text=text)
        normal_text = text[:idx]
        body_start = idx + len(self.bot_token)
        eot_idx = text.find(self.eot_token, body_start)
        body = text[body_start:eot_idx] if eot_idx != -1 else text[body_start:]

        # body should be ``[ {...}, {...} ]`` (with arbitrary whitespace).
        # Prefer the full-text JSON parser when the block is complete; fall
        # back to ``_partial_json_loads`` to be forgiving when generation was
        # truncated before ``<|END_ACTION|>``.
        arr = None
        try:
            arr = orjson.loads(body)
        except (orjson.JSONDecodeError, TypeError, ValueError):
            try:
                arr, _ = _partial_json_loads(body, Allow.ALL)
            except (MalformedJSON, json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"Cohere tool-call body did not parse as JSON: {e}; "
                    "returning surrounding text as normal output."
                )
                return StreamingParseResult(normal_text=normal_text)

        normalized = self._normalize_calls(arr)
        return StreamingParseResult(
            normal_text=normal_text,
            calls=self.parse_base_json(normalized, tools),
        )

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Buffered streaming. Tool-call blocks are short (typically <2KB) so
        we accumulate until the closing ``<|END_ACTION|>`` arrives and emit
        the whole block at once. Anything before ``<|START_ACTION|>`` streams
        through as normal text.
        """
        self._buffer += new_text
        current = self._buffer

        bot_pos = current.find(self.bot_token)
        if bot_pos == -1:
            # Defensive: keep any trailing characters that might be the start
            # of a partial bot_token in the buffer for the next chunk.
            partial = self._ends_with_partial_token(current, self.bot_token)
            if partial:
                head = current[:-partial]
                self._buffer = current[-partial:]
                return StreamingParseResult(normal_text=head)
            self._buffer = ""
            return StreamingParseResult(normal_text=current)

        # ``bot_token`` is somewhere in the buffer. Stream out anything before
        # it as normal text exactly once.
        if bot_pos > 0:
            head = current[:bot_pos]
            self._buffer = current[bot_pos:]
            current = self._buffer
            return StreamingParseResult(normal_text=head)

        # Buffer starts with bot_token. Wait for the closing token, then
        # parse and emit the full call list. Anything past <|END_ACTION|>
        # (typically <|END_OF_TURN_TOKEN|>) stays in the buffer for the next
        # increment to handle.
        eot_pos = current.find(self.eot_token, len(self.bot_token))
        if eot_pos == -1:
            return StreamingParseResult()

        block_end = eot_pos + len(self.eot_token)
        result = self.detect_and_parse(current[:block_end], tools)
        self._buffer = current[block_end:]
        return result

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        def _info(name: str) -> StructureInfo:
            return StructureInfo(
                begin=(
                    '<|START_ACTION|>[{"tool_call_id": "0", "tool_name": "'
                    + name
                    + '", "parameters": '
                ),
                end="}]<|END_ACTION|>",
                trigger="<|START_ACTION|>",
            )

        return _info
