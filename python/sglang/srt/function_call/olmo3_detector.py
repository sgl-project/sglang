# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/olmo3_tool_parser.py

import ast
import json
import logging
from typing import Any, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import safe_ast_parse

logger = logging.getLogger(__name__)

FUNCTION_CALLS_BEGIN = "<function_calls>"
FUNCTION_CALLS_END = "</function_calls>"

# Olmo 3 writes JSON literals as often as Python ones.
_JSON_LITERALS = {"true": True, "false": False, "null": None}


class _NotACallBlock(ValueError):
    """The block is not a sequence of ``name(key=value, ...)`` calls."""


class Olmo3Detector(BaseFormatDetector):
    """Detector for the Olmo 3 tool-call format (``--tool-call-parser olmo3``).

    Olmo 3 Instruct and Think models write newline-separated pythonic calls
    inside a ``<function_calls>`` block, and accept JSON literals (``true``,
    ``false``, ``null``) next to Python ones::

        <function_calls>get_weather(city="Paris", unit=null)
        get_time(timezone="Europe/Paris")</function_calls>

    This is the format of vLLM's ``olmo3`` tool parser. A block is emitted
    atomically in streaming mode, once its calls parse, so streaming and
    non-streaming requests return the same calls. A ``</function_calls>``
    inside a string argument does not end the block.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = FUNCTION_CALLS_BEGIN
        self.eot_token = FUNCTION_CALLS_END
        self.tool_call_separator = "\n"

    def has_tool_call(self, text: str) -> bool:
        return FUNCTION_CALLS_BEGIN in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        calls: list[ToolCallItem] = []
        normal_parts: list[str] = []
        cursor = 0

        while cursor < len(text):
            start = text.find(FUNCTION_CALLS_BEGIN, cursor)
            if start < 0:
                normal_parts.append(text[cursor:])
                break
            normal_parts.append(text[cursor:start])
            body_start = start + len(FUNCTION_CALLS_BEGIN)
            end, parsed = _find_block_end(text=text, body_start=body_start)
            if parsed is None:
                # No closing tag closes a valid block: accept the rest of the
                # output as the block if it parses (the model stopped without
                # closing it); otherwise it is not a tool call.
                try:
                    parsed = _parse_calls(text[body_start:])
                except _NotACallBlock:
                    normal_parts.append(text[start:])
                    break
                cursor = len(text)
            else:
                cursor = end + len(FUNCTION_CALLS_END)
            items = self._build_items(calls=parsed, tools=tools, first_index=len(calls))
            calls.extend(item for item, _ in items)

        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        calls: list[ToolCallItem] = []
        normal_parts: list[str] = []

        while self._buffer:
            start = self._buffer.find(FUNCTION_CALLS_BEGIN)
            if start < 0:
                keep = _partial_marker_suffix_length(
                    text=self._buffer, marker=FUNCTION_CALLS_BEGIN
                )
                normal_parts.append(self._buffer[: len(self._buffer) - keep])
                self._buffer = self._buffer[len(self._buffer) - keep :]
                break
            if start > 0:
                normal_parts.append(self._buffer[:start])
                self._buffer = self._buffer[start:]

            end, parsed = _find_block_end(
                text=self._buffer, body_start=len(FUNCTION_CALLS_BEGIN)
            )
            if parsed is None:
                break  # wait for the rest of the block
            self._buffer = self._buffer[end + len(FUNCTION_CALLS_END) :]
            calls.extend(self._emit(calls=parsed, tools=tools))

        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Resolve a block left open when the stream ended.

        Mirrors ``detect_and_parse``: an unclosed block whose calls parse is
        emitted, anything else is released as text.
        """
        pending = self._buffer
        self._buffer = ""
        if not pending.startswith(FUNCTION_CALLS_BEGIN):
            return StreamingParseResult(normal_text=pending)
        body = pending[len(FUNCTION_CALLS_BEGIN) :]
        # A closing tag may be buffered when the block before it never parsed.
        try:
            parsed = _parse_calls(body)
        except _NotACallBlock:
            return StreamingParseResult(normal_text=pending)
        return StreamingParseResult(calls=self._emit(calls=parsed, tools=tools))

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError(
            "Olmo 3 pythonic calls cannot be represented by structural tags"
        )

    def _build_items(
        self, calls: list[ast.Call], tools: List[Tool], first_index: int
    ) -> list[tuple[ToolCallItem, dict]]:
        """Tool-call items for the calls that name a declared tool and have
        JSON-serializable literal arguments; any other call is skipped alone."""
        items = []
        for call in calls:
            name = call.func.id
            try:
                arguments = {
                    keyword.arg: _literal_value(keyword.value)
                    for keyword in call.keywords
                }
                # allow_nan=False: 1e999 parses to inf, which is not JSON.
                parameters = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
            except (ValueError, TypeError, RecursionError) as error:
                logger.warning(f"Skipping Olmo 3 tool call {name}: {error}")
                continue
            if not self.parse_base_json(
                action={"name": name, "arguments": {}}, tools=tools
            ):
                continue  # unknown tool, already logged
            item = ToolCallItem(
                tool_index=first_index + len(items), name=name, parameters=parameters
            )
            items.append((item, arguments))
        return items

    def _emit(self, calls: list[ast.Call], tools: List[Tool]) -> list[ToolCallItem]:
        """Record a parsed block's calls in the streaming state and return them."""
        items = self._build_items(
            calls=calls, tools=tools, first_index=self.current_tool_id + 1
        )
        for item, arguments in items:
            self.current_tool_id += 1
            self.prev_tool_call_arr.append({"name": item.name, "arguments": arguments})
            self.streamed_args_for_tool.append(item.parameters)
        return [item for item, _ in items]


def _find_block_end(text: str, body_start: int) -> tuple[int, Optional[list]]:
    """Find the ``</function_calls>`` that closes the block.

    Returns ``(end, calls)`` for the first closing tag whose body parses as
    calls, ``(-1, None)`` when no closing tag yields a valid block. Trying
    every candidate lets a string argument contain the closing tag.
    """
    end = text.find(FUNCTION_CALLS_END, body_start)
    while end >= 0:
        try:
            return end, _parse_calls(text[body_start:end])
        except _NotACallBlock:
            end = text.find(FUNCTION_CALLS_END, end + 1)
    return -1, None


def _parse_calls(source: str) -> list[ast.Call]:
    """Parse the body of a ``<function_calls>`` block into call nodes.

    The body is parsed as Python statements, so a string argument may contain
    newlines and a call may span several lines. Olmo 3 sometimes indents its
    calls, which is a syntax error for a module; in that case fall back to
    joining the stripped lines into a list, as vLLM's ``olmo3`` parser does.
    """
    source = source.strip()
    if not source:
        raise _NotACallBlock("empty block")
    try:
        nodes = [
            statement.value if isinstance(statement, ast.Expr) else statement
            for statement in safe_ast_parse(source).body
        ]
    except (SyntaxError, ValueError, RecursionError):
        joined = ", ".join(line.strip() for line in source.splitlines() if line.strip())
        try:
            parsed = safe_ast_parse(f"[{joined}]").body
        except (SyntaxError, ValueError, RecursionError) as error:
            raise _NotACallBlock(str(error)) from error
        if len(parsed) != 1 or not isinstance(parsed[0], ast.Expr):
            raise _NotACallBlock("not a list of calls")
        value = parsed[0].value
        if not isinstance(value, ast.List):
            raise _NotACallBlock("not a list of calls")
        nodes = list(value.elts)

    for node in nodes:
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and not node.args
            and all(keyword.arg is not None for keyword in node.keywords)
        ):
            raise _NotACallBlock("every statement must be name(key=value, ...)")
    return nodes


def _literal_value(node: ast.AST) -> Any:
    """Convert one argument node to a JSON-serializable value."""
    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(node.value, (bool, int, float, str)):
            return node.value
        raise ValueError(
            f"constant has no JSON representation: {type(node.value).__name__}"
        )
    if isinstance(node, ast.Name) and node.id in _JSON_LITERALS:
        return _JSON_LITERALS[node.id]
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) in (int, float)
    ):
        value = node.operand.value
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal_value(element) for element in node.elts]
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                raise ValueError("dictionary keys must be string literals")
            result[key.value] = _literal_value(value)
        return result
    raise ValueError(f"tool call arguments must be literals, got {type(node).__name__}")


def _partial_marker_suffix_length(text: str, marker: str) -> int:
    """Length of the suffix that may become ``marker`` in the next chunk."""
    for size in range(min(len(text), len(marker) - 1), 0, -1):
        if text.endswith(marker[:size]):
            return size
    return 0
