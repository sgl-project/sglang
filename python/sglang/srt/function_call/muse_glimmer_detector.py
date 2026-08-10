import json
import logging
import re
from typing import Dict, List, Optional, Set

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)

# Channel framing, shared with the reasoning-side MuseGlimmerDetector.
MESSAGE = "<|message|>"
EOM = "<|eom|>"
EOT = "<|eot|>"
START = "<|start|>"

# ATEM payload markers.
FUNCTION_CALLS_OPEN = "<atem:function_calls>"
FUNCTION_CALLS_CLOSE = "</atem:function_calls>"
INVOKE_CLOSE = "</atem:invoke>"

_RECIPIENT_RE = re.compile(r"to=([^\s<]+)")
_INVOKE_OPEN_RE = re.compile(r'<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)"[^>]*?>')
_PARAM_RE = re.compile(
    r'<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>(?P<value>.*?)'
    r"</atem:parameter>",
    re.DOTALL,
)

# Recipients whose bodies are prose, never tool calls.
_NON_TOOL_RECIPIENTS = frozenset({"self", "user"})


def _is_tool_channel(recipient: Optional[str]) -> bool:
    """True when this channel routes to a tool."""
    return recipient is not None and recipient not in _NON_TOOL_RECIPIENTS


# Longest marker that could straddle a chunk boundary while streaming.
_MAX_MARKER = max(len(m) for m in (MESSAGE, EOM, EOT, START, FUNCTION_CALLS_OPEN))


def _decode_value(raw: str):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _normalize_name(emitted: str, registered: Set[str]) -> str:
    """Strip the chat template's doubled namespace."""
    if not registered or emitted in registered:
        return emitted
    if "." not in emitted:
        return emitted
    head, _, tail = emitted.partition(".")
    if head == tail and head in registered:
        return head
    leaf = emitted.rsplit(".", 1)[-1]
    matches = [n for n in registered if n.rsplit(".", 1)[-1] == leaf]
    if len(matches) == 1:
        return matches[0]
    return emitted


def _could_start_header(text: str) -> bool:
    """Whether the tail could still grow into a header."""
    stripped = text.lstrip()
    if not stripped:
        return True
    if not (stripped.startswith("to=") or "to=".startswith(stripped[:3])):
        return False
    if MESSAGE in stripped:
        return True
    recipient, angle, marker = stripped[3:].partition("<")
    if any(c.isspace() for c in recipient):
        return False
    return not angle or MESSAGE.startswith("<" + marker)


class MuseGlimmerDetector(BaseFormatDetector):
    """Format detector for Muse Glimmer's ATEM tool-call blocks."""

    def __init__(self):
        super().__init__()
        # Streaming channel state.
        self._recipient: Optional[str] = None
        self._in_body = False
        self._at_stream_start = True
        # Name of the invoke whose arguments are still arriving, if any.
        self._open_invoke: Optional[str] = None

    def has_tool_call(self, text: str) -> bool:
        return FUNCTION_CALLS_OPEN in text or "<atem:invoke" in text

    def _registered_names(self, tools: Optional[List[Tool]]) -> Set[str]:
        return {t.function.name for t in tools or [] if t.function and t.function.name}

    def _emit_call(
        self, name: str, args: Dict, registered: Set[str]
    ) -> Optional[ToolCallItem]:
        """Build one ToolCallItem, honoring the unknown-tool policy."""
        name = _normalize_name(name, registered)
        if name not in registered:
            logger.warning("Model attempted to call undefined function: %s", name)
            if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                return None
        self.current_tool_id += 1
        parameters = json.dumps(args, ensure_ascii=False)
        self.prev_tool_call_arr.append({"name": name, "arguments": args})
        self.streamed_args_for_tool.append(parameters)
        return ToolCallItem(
            tool_index=self.current_tool_id, name=name, parameters=parameters
        )

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        result = self.parse_streaming_increment(text, tools)
        end = self.finish(tools)
        return StreamingParseResult(
            normal_text=result.normal_text + end.normal_text,
            calls=result.calls + end.calls,
        )

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        registered = self._registered_names(tools)
        calls: List[ToolCallItem] = []
        normal_parts: List[str] = []

        if self._buffer:
            if self._in_body:
                self._consume_body(
                    self._buffer,
                    registered,
                    calls,
                    normal_parts,
                    final=True,
                )
            else:
                # Truncated header: no body ever arrived, keep it as text.
                normal_parts.append(self._buffer)
            self._buffer = ""

        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def _held_back(self, text: str) -> int:
        """Length of the trailing suffix that could still grow into a marker."""
        markers = (MESSAGE, EOM, EOT, START, FUNCTION_CALLS_OPEN, "<atem:invoke")
        for k in range(min(len(text), _MAX_MARKER - 1), 0, -1):
            if any(m.startswith(text[-k:]) for m in markers):
                return k
        return 0

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        registered = self._registered_names(tools)
        calls: List[ToolCallItem] = []
        normal_parts: List[str] = []

        while self._buffer:
            if not self._in_body:
                # Resolve the channel header before anything can be emitted.
                if self._at_stream_start and _could_start_header(self._buffer):
                    pass
                else:
                    ws = len(self._buffer) - len(self._buffer.lstrip())
                    head = self._buffer[ws : ws + len(START)]
                    if not START.startswith(head):
                        # Unframed prose: no header is coming.
                        self._in_body = True
                        self._recipient = None
                        self._at_stream_start = False
                        continue
                    if ws:
                        normal_parts.append(self._buffer[:ws])
                        self._buffer = self._buffer[ws:]
                    if len(head) < len(START):
                        break  # Partial "<|start|>", wait for the rest.

                idx = self._buffer.find(MESSAGE)
                if idx == -1:
                    break
                header = self._buffer[:idx]
                m = _RECIPIENT_RE.search(header)
                self._recipient = m.group(1) if m else "user"
                self._buffer = self._buffer[idx + len(MESSAGE) :]
                self._in_body = True
                self._at_stream_start = False
                continue

            # Inside a body: find the terminator, if it has arrived.
            end_at, end_len = -1, 0
            for tok in (EOM, EOT):
                i = self._buffer.find(tok)
                if i != -1 and (end_at == -1 or i < end_at):
                    end_at, end_len = i, len(tok)

            if end_at == -1:
                keep = self._held_back(self._buffer)
                chunk = self._buffer[: len(self._buffer) - keep]
                if not chunk:
                    break
                consumed = self._consume_body(
                    chunk, registered, calls, normal_parts, final=False
                )
                if consumed == 0:
                    break
                self._buffer = self._buffer[consumed:]
                continue

            self._consume_body(
                self._buffer[:end_at],
                registered,
                calls,
                normal_parts,
                final=True,
            )
            self._buffer = self._buffer[end_at + end_len :]
            self._in_body = False
            self._recipient = None
            self._open_invoke = None

        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def _consume_body(
        self,
        chunk: str,
        registered: Set[str],
        calls: List[ToolCallItem],
        normal_parts: List[str],
        final: bool,
    ) -> int:
        if not _is_tool_channel(self._recipient):
            normal_parts.append(chunk)
            return len(chunk)

        pos = 0
        while pos < len(chunk):
            if self._open_invoke is None:
                m = _INVOKE_OPEN_RE.search(chunk, pos)
                if m is None:
                    return pos if not final else len(chunk)
                self._open_invoke = m.group("name")
                pos = m.end()
                continue

            close_at = chunk.find(INVOKE_CLOSE, pos)
            if close_at == -1:
                return pos if not final else len(chunk)
            body = chunk[pos:close_at]
            args = {
                pm.group("key"): _decode_value(pm.group("value"))
                for pm in _PARAM_RE.finditer(body)
            }
            item = self._emit_call(self._open_invoke, args, registered)
            if item is not None:
                calls.append(item)
            self._open_invoke = None
            pos = close_at + len(INVOKE_CLOSE)

        return len(chunk)

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'{FUNCTION_CALLS_OPEN}\n<atem:invoke name="{name}">',
            end=f"{INVOKE_CLOSE}\n{FUNCTION_CALLS_CLOSE}",
            trigger=FUNCTION_CALLS_OPEN,
        )
