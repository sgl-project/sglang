"""
Detector for LFM2 (Liquid Foundation Model 2) function call format.

Format Structure (Pythonic style):
```
<|tool_call_start|>[function_name(arg1="value1", arg2="value2")]<|tool_call_end|>
```

Multiple tool calls:
```
<|tool_call_start|>[func1(arg="val"), func2(arg="val")]<|tool_call_end|>
```

Also supports JSON format:
```
<|tool_call_start|>[{"name": "func_name", "arguments": {...}}]<|tool_call_end|>
```
"""

import ast
import json
import keyword as _python_keyword
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import safe_ast_parse

logger = logging.getLogger(__name__)


_PYTHONIC_NAME_LITERALS = {
    "True": True,
    "False": False,
    "None": None,
    "true": True,
    "false": False,
    "null": None,
}

_QUOTE_FOLLOWERS = {",", ")", "]", "}", ":"}
_RESERVED_KW_SUFFIX = "_pyreservedkw_"


def _rename_reserved_kwargs(text: str) -> Tuple[str, bool]:
    """Rename Python-keyword parameter names so the text parses.

    Tools legitimately name parameters ``from``/``in``/``class``, but
    ``memory_get(from=1)`` is a Python ``SyntaxError``. Rename ``from=`` to
    ``from_pyreservedkw_=`` (outside string literals, keyword-argument
    position only), parse, then restore via
    :func:`_restore_reserved_kwarg_names`. Returns (rewritten_text, changed).
    """
    out: List[str] = []
    quote: Optional[str] = None
    changed = False
    last_sig = ""
    index, length = 0, len(text)
    while index < length:
        char = text[index]
        if quote is not None:
            out.append(char)
            if char == "\\" and index + 1 < length:
                out.append(text[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            out.append(char)
            last_sig = char
            index += 1
            continue
        if char.isalpha() or char == "_":
            end = index
            while end < length and (text[end].isalnum() or text[end] == "_"):
                end += 1
            name = text[index:end]
            look = end
            while look < length and text[look].isspace():
                look += 1
            if (
                _python_keyword.iskeyword(name)
                and look < length
                and text[look] == "="
                and (look + 1 >= length or text[look + 1] != "=")
                and last_sig in {"(", ","}
            ):
                out.append(name + _RESERVED_KW_SUFFIX)
                changed = True
            else:
                out.append(name)
            last_sig = name[-1]
            index = end
            continue
        out.append(char)
        if not char.isspace():
            last_sig = char
        index += 1
    return "".join(out), changed


def _restore_reserved_kwarg_names(arguments: dict) -> dict:
    """Exact inverse of :func:`_rename_reserved_kwargs` on a decoded dict."""
    restored = {}
    for key, value in arguments.items():
        if (
            isinstance(key, str)
            and key.endswith(_RESERVED_KW_SUFFIX)
            and _python_keyword.iskeyword(key[: -len(_RESERVED_KW_SUFFIX)])
        ):
            restored[key[: -len(_RESERVED_KW_SUFFIX)]] = value
        else:
            restored[key] = value
    return restored


def _is_escaped(text: str, index: int) -> bool:
    """Whether the char at ``index`` follows an odd run of backslashes."""
    backslashes = 0
    j = index - 1
    while j >= 0 and text[j] == "\\":
        backslashes += 1
        j -= 1
    return backslashes % 2 == 1


def _escape_nested_quotes_in_strings(text: str) -> Tuple[str, bool]:
    """Close a broken string literal at the only closing quote that works.

    Shell commands nest unescaped same-style quotes inside a string argument
    (``command='sed -n '360,450p' f.py'`` or a quoted ``python3 -c`` payload),
    which Python reads as juxtaposed garbage, so the call is dropped even
    though the intent is unambiguous. A string is broken when its first
    unescaped quote cannot syntactically close it (what follows is none of
    ``,)]}:``). For a broken string, every syntactically plausible closing
    quote is tried — interior quotes escaped, the rest kept verbatim — and
    the result validated with ``ast.parse``. Exactly one parsing candidate
    means recovery; zero or several means genuine ambiguity and the text is
    returned unchanged rather than guessed at.

    Returns (rewritten_text, changed).
    """

    def unescaped_quotes(start: int, quote: str) -> List[int]:
        positions = []
        j = start
        while j < len(text):
            if text[j] == "\\":
                j += 2
                continue
            if text[j] == quote:
                positions.append(j)
            j += 1
        return positions

    def is_closer(pos: int) -> bool:
        k = pos + 1
        while k < len(text) and text[k].isspace():
            k += 1
        return k < len(text) and text[k] in _QUOTE_FOLLOWERS

    # A late-closing reading can swallow a whole sibling call into the
    # string value (``f(a='x 'y'), g(...)`` parsing as one call with
    # ``g(...)`` inside ``a``) — worse than dropping it, since the tool then
    # runs with corrupted arguments. Counting brackets is immune to the
    # broken quote, so the block's call count is the invariant.
    expected_calls = len(_split_top_level_calls(text, respect_strings=False))
    prefix: List[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char not in {"'", '"'}:
            prefix.append(char)
            index += 1
            continue
        quotes = unescaped_quotes(index + 1, char)
        if not quotes:
            return text, False
        if is_closer(quotes[0]):
            prefix.append(text[index : quotes[0] + 1])
            index = quotes[0] + 1
            continue
        winners = []
        for close in (j for j in quotes if is_closer(j)):
            interior: List[str] = []
            for j in range(index + 1, close):
                if text[j] == char and not _is_escaped(text, j):
                    interior.append("\\")
                interior.append(text[j])
            candidate = "".join(
                ["".join(prefix), char, "".join(interior), char, text[close + 1 :]]
            )
            try:
                module = safe_ast_parse(_escape_ctrl_chars_in_strings(candidate))
            except (SyntaxError, ValueError):
                continue
            if expected_calls > 1 and _top_level_call_count(module) < expected_calls:
                continue
            winners.append(candidate)
        if len(winners) == 1:
            return winners[0], True
        return text, False
    return text, False


def _escape_ctrl_chars_in_strings(text: str) -> str:
    """Escape raw control chars inside string literals of pythonic text.

    Models frequently place raw newlines inside a string argument (multi-line
    shell commands), which is invalid Python, and a NUL byte anywhere makes
    ``ast.parse`` raise ``ValueError``. Escaping ``\\n``/``\\r``/``\\t``/
    ``\\x00`` only inside string literals makes the text parseable while the
    escape sequences evaluate back to the exact original value.
    """
    out: List[str] = []
    quote: Optional[str] = None
    index, length = 0, len(text)
    while index < length:
        char = text[index]
        if quote is None:
            if char in {"'", '"'}:
                quote = char
            out.append(char)
        elif char == "\\" and index + 1 < length:
            out.append(char)
            out.append(text[index + 1])
            index += 2
            continue
        elif char == quote:
            quote = None
            out.append(char)
        elif char == "\n":
            out.append("\\n")
        elif char == "\r":
            out.append("\\r")
        elif char == "\t":
            out.append("\\t")
        elif char == "\x00":
            out.append("\\x00")
        else:
            out.append(char)
        index += 1
    return "".join(out)


def _normalize_leading_zero_ints(text: str) -> str:
    """Strip leading zeros from decimal int literals (``month=07`` -> ``7``).

    Zero-padded integers are a ``SyntaxError`` no other rewrite recovers.
    Only rewrites outside string literals; tokens that are already valid
    Python (``0x``/``0o``/``0b``, floats, exponents, all-zero literals,
    fractional parts like ``1.07``) are left untouched.
    """
    out: List[str] = []
    quote: Optional[str] = None
    index, length = 0, len(text)
    while index < length:
        char = text[index]
        if quote is not None:
            out.append(char)
            if char == "\\" and index + 1 < length:
                out.append(text[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            out.append(char)
            index += 1
            continue
        if char.isalpha() or char == "_":
            end = index
            while end < length and (text[end].isalnum() or text[end] == "_"):
                end += 1
            out.append(text[index:end])
            index = end
            continue
        if char.isdigit():
            end = index
            while end < length and (text[end].isdigit() or text[end] == "_"):
                end += 1
            token = text[index:end]
            digits = token.replace("_", "")
            follower = text[end] if end < length else ""
            preceded_by_dot = index > 0 and text[index - 1] == "."
            if (
                digits[0] == "0"
                and digits.strip("0")
                and not preceded_by_dot
                and follower not in {".", "e", "E", "j", "J"}
            ):
                out.append(str(int(digits)))
            else:
                out.append(token)
            index = end
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _recovery_candidates(content: str) -> List[Tuple[str, bool]]:
    """Progressive rewrites for content that failed to parse.

    Each rewrite is a no-op on already-valid text; the first candidate whose
    result parses wins. Nested-quote recovery is re-escaped, since requoting
    can move raw newlines inside the string. The flag marks candidates that
    went through reserved-keyword renaming; only their decoded arguments get
    the original parameter names restored.
    """
    escaped = _escape_ctrl_chars_in_strings(_normalize_leading_zero_ints(content))
    candidates: List[Tuple[str, bool]] = [(escaped, False)]
    requoted, requote_changed = _escape_nested_quotes_in_strings(escaped)
    if requote_changed:
        candidates.append((_escape_ctrl_chars_in_strings(requoted), False))
    for text, _ in list(candidates):
        renamed, kw_renamed = _rename_reserved_kwargs(text)
        if kw_renamed:
            candidates.append((renamed, True))
            # A call can stack both quirks; renaming first lets requote
            # validate candidates the keyword SyntaxError otherwise blocks.
            requoted_after, requote_after_changed = _escape_nested_quotes_in_strings(
                renamed
            )
            if requote_after_changed:
                candidates.append((_escape_ctrl_chars_in_strings(requoted_after), True))
    return candidates


def _split_top_level_calls(text: str, *, respect_strings: bool = True) -> List[str]:
    """Split a pythonic call block into top-level call segments.

    ``[a(x=1), b(y=2)]`` becomes ``["a(x=1)", "b(y=2)"]``: one enclosing
    bracket pair is stripped and only commas at bracket depth 0 separate
    segments. With ``respect_strings=False`` only brackets are counted,
    which a broken quote cannot desynchronize; string arguments always sit
    at depth >= 1, so their commas still never split.
    """
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    segments: List[str] = []
    start = 0
    depth = 0
    quote: Optional[str] = None
    index = 0
    while index < len(text):
        char = text[index]
        if respect_strings and quote is not None:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if respect_strings and char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            segments.append(text[start:index])
            start = index + 1
        index += 1
    segments.append(text[start:])
    return [segment.strip() for segment in segments if segment.strip()]


def _top_level_call_count(module: ast.Module) -> int:
    """Number of calls in a parsed ``[a(...), b(...)]`` block."""
    if not module.body:
        return 0
    value = getattr(module.body[0], "value", None)
    if isinstance(value, ast.List):
        return sum(1 for element in value.elts if isinstance(element, ast.Call))
    return 1 if isinstance(value, ast.Call) else 0


def _salvage_calls_from_unparsable_block(text: str) -> List[Tuple[ast.Call, bool]]:
    """Recover individual calls from a block ``ast.parse`` cannot handle.

    When the block as a whole is a SyntaxError no rewrite recovers, there
    is no call list at all and one bad call drops every parseable sibling,
    leaving an agent loop with no tool result. Split with both scanning
    strategies and parse each segment on its own through the rewrite
    ladder. A wrongly split segment simply fails to parse and is dropped,
    so this can only under-recover, never attribute arguments to the wrong
    call. Each call carries the reserved-keyword flag of the candidate it
    parsed from.
    """
    best: List[Tuple[ast.Call, bool]] = []
    for respect_strings in (True, False):
        segments = _split_top_level_calls(text, respect_strings=respect_strings)
        if len(segments) < 2:
            continue
        calls: List[Tuple[ast.Call, bool]] = []
        for segment in segments:
            for candidate, kw_renamed in [(segment, False)] + _recovery_candidates(
                segment
            ):
                try:
                    module = safe_ast_parse(candidate)
                except (SyntaxError, ValueError):
                    continue
                parsed = getattr(module.body[0], "value", None) if module.body else None
                if isinstance(parsed, ast.Call):
                    calls.append((parsed, kw_renamed))
                break
        if len(calls) > len(best):
            best = calls
    return best


class Lfm2Detector(BaseFormatDetector):
    """
    Detector for LFM2 (Liquid Foundation Model 2) function call format.

    Supports both Pythonic and JSON formats:

    Pythonic:
    ```
    <|tool_call_start|>[calculator(expression="5 * 7")]<|tool_call_end|>
    ```

    JSON:
    ```
    <|tool_call_start|>[{"name": "calculator", "arguments": {"expression": "5 * 7"}}]<|tool_call_end|>
    ```
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.bot_token = "<|tool_call_start|>"
        self.eot_token = "<|tool_call_end|>"
        self.tool_call_separator = ""
        self._streamed_parameters: Dict[int, str] = {}
        self._stream_call_offset = 0

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains an LFM2 format tool call."""
        return self.bot_token in text

    def _get_parameter_value(self, val: ast.AST) -> Any:
        """
        Extract Python literal value from AST node.

        Handles constants, dicts, and lists recursively.
        Reuses pattern from PythonicDetector.
        """
        if isinstance(val, ast.Constant):
            if val.value is None or isinstance(val.value, (str, int, float)):
                return val.value
            # bytes/Ellipsis/complex have no JSON form; raising ValueError
            # here lets the per-call handler skip this call instead of a
            # TypeError inside json.dumps dropping every sibling call.
            raise ValueError(
                f"Constant has no JSON representation: {type(val.value).__name__}"
            )
        elif isinstance(val, ast.Dict):
            return {
                self._get_parameter_value(k): self._get_parameter_value(v)
                for k, v in zip(val.keys, val.values)
                if k is not None  # Handle {**kwargs} case where key is None
            }
        elif isinstance(val, ast.List):
            return [self._get_parameter_value(v) for v in val.elts]
        elif isinstance(val, ast.Tuple):
            return tuple(self._get_parameter_value(v) for v in val.elts)
        elif isinstance(val, ast.Set):
            # JSON has no set type; decode as a list in source order.
            return [self._get_parameter_value(v) for v in val.elts]
        elif isinstance(val, ast.JoinedStr) and all(
            isinstance(part, ast.Constant) for part in val.values
        ):
            # A placeholder-free f-string (f'hello') is a plain string
            # constant, but Python parses it as JoinedStr; f-strings with
            # real placeholders still fall through to the raise below.
            return "".join(str(part.value) for part in val.values)
        elif isinstance(val, ast.Name):
            # Python True/False/None are ast.Constant on modern Python, but
            # accept their legacy node shape plus LFM2's JSON-literal spellings.
            try:
                return _PYTHONIC_NAME_LITERALS[val.id]
            except KeyError:
                raise ValueError(f"Unsupported name reference: {val.id}") from None
        elif isinstance(val, ast.UnaryOp) and isinstance(val.op, (ast.USub, ast.UAdd)):
            # Handle signed numbers like -5 and +5
            inner = self._get_parameter_value(val.operand)
            if isinstance(inner, (int, float)) and not isinstance(inner, bool):
                return -inner if isinstance(val.op, ast.USub) else inner
            raise ValueError(f"Cannot apply sign to non-numeric value: {inner}")
        else:
            raise ValueError(
                f"Tool call arguments must be literals, got: {type(val).__name__}"
            )

    def _get_function_name(self, func: ast.AST) -> Optional[str]:
        """Extract a flat or dotted function name from a Pythonic call node."""
        parts: List[str] = []
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value

        if not isinstance(func, ast.Name):
            return None

        parts.append(func.id)
        return ".".join(reversed(parts))

    def _parse_pythonic_call(
        self,
        call: ast.Call,
        call_index: int,
        tool_indices: Dict[str, int],
        *,
        restore_reserved_kwarg: bool = False,
    ) -> Optional[ToolCallItem]:
        """
        Parse a single AST Call node into a ToolCallItem.

        Args:
            call: AST Call node representing a function call
            call_index: Index of this call in the list of calls
            tool_indices: Mapping of tool names to their indices
            restore_reserved_kwarg: Whether the parsed text went through
                reserved-keyword renaming

        Returns:
            ToolCallItem if successful, None if the call should be skipped
        """
        function_name = self._get_function_name(call.func)
        if function_name is None:
            logger.warning(
                f"Tool call function must be a name or dotted name, got: {type(call.func).__name__}"
            )
            return None

        # Validate that the function exists in the tools
        if function_name not in tool_indices:
            logger.warning(
                f"Model attempted to call undefined function: {function_name}"
            )
            if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                return None  # Skip unknown tools (default legacy behavior)

        if call.args:
            # Only keyword arguments carry parameter names; positional
            # values used to be dropped silently, emitting a
            # successful-looking call with arguments missing. Reject
            # instead (parseable sibling calls are kept).
            logger.warning(f"Tool call {function_name} has positional arguments")
            return None

        # Parse arguments
        arguments = {}
        for keyword in call.keywords:
            if keyword.arg is None:
                # **-unpacking is ast.keyword(arg=None); the kwargs used to
                # be skipped silently, emitting the call with arguments
                # missing. Merge dict literals with Python's
                # later-binding-wins semantics and reject anything else.
                try:
                    unpacked = self._get_parameter_value(keyword.value)
                except ValueError as e:
                    logger.warning(f"Failed to parse **-unpacked arguments: {e}")
                    return None
                if not isinstance(unpacked, dict):
                    logger.warning("**-unpacked arguments must be a dict literal")
                    return None
                arguments.update(unpacked)
                continue
            try:
                arguments[keyword.arg] = self._get_parameter_value(keyword.value)
            except ValueError as e:
                logger.warning(f"Failed to parse argument {keyword.arg}: {e}")
                return None

        if restore_reserved_kwarg:
            # Unconditional restore would rewrite a parameter literally
            # named e.g. ``in_pyreservedkw_`` to ``in`` on the normal path.
            arguments = _restore_reserved_kwarg_names(arguments)

        try:
            # allow_nan=False: a non-finite float (e.g. the literal 1e999
            # overflowing to inf) would otherwise serialize as Infinity,
            # which is not valid JSON for downstream clients.
            parameters = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError) as e:
            logger.warning(f"Arguments of {function_name} are not valid JSON: {e}")
            return None

        return ToolCallItem(
            tool_index=call_index,  # Use the call index in the response, not tool position
            name=function_name,
            parameters=parameters,
        )

    def _parse_pythonic_content(
        self, content: str, tools: List[Tool]
    ) -> Tuple[List[ToolCallItem], str]:
        """
        Parse Pythonic format tool calls using AST.

        Args:
            content: The content between tool call tags (without the tags)
            tools: List of available tools

        Returns:
            Tuple of (list of parsed calls, error message if any)
        """
        content = content.strip()
        tool_indices = self._get_tool_indices(tools)

        try:
            kw_renamed = False
            try:
                module = safe_ast_parse(content)
            except (SyntaxError, ValueError):
                # Recoverable model quirks are rewritten value-preservingly;
                # the first rewrite that parses wins. Unrecoverable text
                # re-raises the original error.
                for candidate, kw_renamed in _recovery_candidates(content):
                    try:
                        module = safe_ast_parse(candidate)
                        break
                    except (SyntaxError, ValueError):
                        continue
                else:
                    # The block as a whole is unrecoverable. Split it into
                    # top-level segments and parse each on its own so one bad
                    # call does not drop every parseable sibling.
                    salvaged = _salvage_calls_from_unparsable_block(content)
                    if not salvaged:
                        raise
                    calls = []
                    for call_index, (call, segment_kw_renamed) in enumerate(salvaged):
                        item = self._parse_pythonic_call(
                            call,
                            call_index,
                            tool_indices,
                            restore_reserved_kwarg=segment_kw_renamed,
                        )
                        if item is not None:
                            calls.append(item)
                    return calls, ""
            parsed = getattr(module.body[0], "value", None) if module.body else None

            if parsed is None:
                return [], "Empty or invalid Python expression"

            # Handle both single call and list of calls
            if isinstance(parsed, ast.List):
                call_nodes = parsed.elts
            elif isinstance(parsed, ast.Call):
                call_nodes = [parsed]
            else:
                return (
                    [],
                    f"Expected function call or list, got: {type(parsed).__name__}",
                )

            # Validate all elements are calls
            if not all(isinstance(e, ast.Call) for e in call_nodes):
                return [], "Not all elements in list are function calls"

            calls = []
            for call_index, call in enumerate(call_nodes):
                item = self._parse_pythonic_call(
                    call,
                    call_index,
                    tool_indices,
                    restore_reserved_kwarg=kw_renamed,
                )
                if item is not None:
                    calls.append(item)

            return calls, ""

        except (SyntaxError, ValueError) as e:
            return [], f"Python syntax error: {e}"
        except Exception as e:
            logger.exception("Unexpected error in pythonic tool call parsing")
            return [], f"Unexpected error: {e}"

    def _parse_json_content(
        self, content: str, tools: List[Tool]
    ) -> Tuple[List[ToolCallItem], str]:
        """
        Parse JSON format tool calls.

        Uses parse_base_json from BaseFormatDetector for consistent handling
        of SGLANG_FORWARD_UNKNOWN_TOOLS and tool validation.

        Args:
            content: The content between tool call tags (without the tags)
            tools: List of available tools

        Returns:
            Tuple of (list of parsed calls, error message if any)
        """
        content = content.strip()

        try:
            parsed = json.loads(content)
            # parse_base_json handles list/dict normalization, tool validation,
            # and SGLANG_FORWARD_UNKNOWN_TOOLS consistently with other detectors
            calls = self.parse_base_json(parsed, tools)
            return calls, ""

        except json.JSONDecodeError as e:
            return [], f"JSON parse error: {e}"

    def _parse_tool_calls_content(
        self, content: str, tools: List[Tool]
    ) -> List[ToolCallItem]:
        """
        Parse the content between tool call tags.
        Handles both JSON and Pythonic formats.
        """
        content = content.strip()

        # First, try JSON format (faster check)
        if content.startswith("[{") or content.startswith("{"):
            calls, error = self._parse_json_content(content, tools)
            if calls:
                return calls
            # If JSON parsing failed but it looked like JSON, log the error
            if error:
                logger.debug(f"JSON parsing failed: {error}, trying Pythonic format")

        # Try Pythonic format
        calls, error = self._parse_pythonic_content(content, tools)
        if calls:
            return calls

        if error:
            logger.warning(f"Failed to parse tool calls: {error}")

        return []

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text

        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        # Find all <|tool_call_start|>...<|tool_call_end|> blocks
        pattern = rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}"
        match_result_list = re.findall(pattern, text, re.DOTALL)

        calls = []
        for match_result in match_result_list:
            parsed_calls = self._parse_tool_calls_content(match_result, tools)
            calls.extend(parsed_calls)

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _strip_special_tokens(self, text: str) -> str:
        """Remove special tokens from text."""
        return text.replace(self.bot_token, "").replace(self.eot_token, "")

    def _stream_pythonic_prefix(
        self, content: str, tools: List[Tool]
    ) -> List[ToolCallItem]:
        """Parse a provisional completion, emitting only its real JSON prefix.

        Synthetic closing quotes/brackets make AST parsing possible, but are
        never sent to the client. Scalars wait for a delimiter: e.g. ``1``
        could still become ``1e-3``. Incomplete escape sequences also wait.
        """
        content = content.strip().removeprefix("[")
        segments = _split_top_level_calls(content)
        calls = []
        for index, segment in enumerate(segments):
            stack = []
            quote = None
            cursor = 0
            while cursor < len(segment):
                char = segment[cursor]
                if quote:
                    if char == "\\":
                        cursor += 2
                        continue
                    if char == quote:
                        quote = None
                elif char in "\"'":
                    quote = char
                elif char in "([{":
                    stack.append({"(": ")", "[": "]", "{": "}"}[char])
                elif char in ")]}":
                    if not stack or stack.pop() != char:
                        break
                cursor += 1
            segment = segment[:cursor]
            if not stack:
                parsed, _ = self._parse_pythonic_content(segment, tools)
                if parsed:
                    item = parsed[0]
                    item.tool_index = index
                    calls.append(item)
                continue
            # Do not guess a partial number, identifier, keyword, or dict key.
            if not quote and segment.rstrip()[-1:] not in ",)]}":
                continue
            suffix = "".join(reversed(stack))
            candidate = segment
            if quote:
                # An incomplete Python escape cannot be committed to JSON.
                escape = re.search(r"(?<!\\)(?:\\\\)*\\([^\\]*)$", candidate)
                if escape:
                    tail = escape.group(1)
                    width = {"x": 2, "u": 4, "U": 8}.get(tail[:1])
                    if (
                        not tail
                        or (width and len(tail) < width + 1)
                        or (tail.startswith("N{") and "}" not in tail)
                        or (tail[:1] in "01234567" and len(tail) < 3)
                    ):
                        candidate = candidate[: escape.end() - len(tail) - 1]
                candidate += quote
            # ``{'key'`` is still ambiguous between a dict and a set;
            # ``(1`` between grouping and a tuple. Wait rather than commit
            # a JSON container that later input could change.
            if ")" in stack[1:]:
                continue
            try:
                provisional, renamed = _rename_reserved_kwargs(candidate + suffix)
                tree = safe_ast_parse(_escape_ctrl_chars_in_strings(provisional))
                if any(isinstance(node, ast.Set) for node in ast.walk(tree)):
                    continue
            except (SyntaxError, ValueError):
                continue
            call = tree.body[0].value if tree.body else None
            if not isinstance(call, ast.Call):
                continue
            item = self._parse_pythonic_call(
                call,
                index,
                self._get_tool_indices(tools),
                restore_reserved_kwarg=renamed,
            )
            if item is None:
                continue
            # One JSON closer per open container (including the kwargs dict).
            item.parameters = item.parameters[: -(len(stack) + bool(quote))]
            calls.append(item)
        return calls

    def _argument_deltas(self, calls: List[ToolCallItem]) -> List[ToolCallItem]:
        deltas = []
        for call in calls:
            previous = self._streamed_parameters.get(call.tool_index)
            if previous is not None and not call.parameters.startswith(previous):
                # Recovery of malformed output cannot retract already sent bytes.
                continue
            delta = call.parameters[len(previous or "") :]
            if delta or previous is None:
                deltas.append(
                    ToolCallItem(
                        tool_index=self._stream_call_offset + call.tool_index,
                        name=call.name if previous is None else None,
                        parameters=delta,
                    )
                )
                self._streamed_parameters[call.tool_index] = call.parameters
        return deltas

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for LFM2 tool calls.

        Pythonic argument prefixes are emitted before the end marker. JSON
        blocks retain the existing complete-block parsing behavior.
        """
        self._buffer += new_text

        # Check for partial bot_token at the end
        partial_bot = self._ends_with_partial_token(self._buffer, self.bot_token)
        partial_eot = self._ends_with_partial_token(self._buffer, self.eot_token)

        # Find bot_token position
        bot_pos = self._buffer.find(self.bot_token)

        if bot_pos == -1:
            # No tool call start found
            if partial_bot:
                # Might be partial bot_token, hold back that part
                safe_text = self._buffer[:-partial_bot]
                self._buffer = self._buffer[-partial_bot:]
                return StreamingParseResult(normal_text=safe_text)
            else:
                # No tool call, emit all as normal text
                normal_text = self._strip_special_tokens(self._buffer)
                self._buffer = ""
                return StreamingParseResult(normal_text=normal_text)

        # We have bot_token - extract any normal text before it
        normal_text_before = self._buffer[:bot_pos] if bot_pos > 0 else ""

        # Look for the end token
        eot_pos = self._buffer.find(self.eot_token, bot_pos + len(self.bot_token))

        if eot_pos == -1:
            content = self._buffer[bot_pos + len(self.bot_token) :]
            if partial_eot:
                content = content[:-partial_eot]
            self._buffer = self._buffer[bot_pos:]
            calls = []
            if not content.lstrip().startswith(("{", "[{")):
                calls = self._argument_deltas(
                    self._stream_pythonic_prefix(content, tools)
                )
            return StreamingParseResult(normal_text=normal_text_before, calls=calls)

        # We have a complete tool call block
        tool_call_block = self._buffer[bot_pos : eot_pos + len(self.eot_token)]
        remaining = self._buffer[eot_pos + len(self.eot_token) :]

        # Parse the complete block
        result = self.detect_and_parse(tool_call_block, tools)
        result.calls = self._argument_deltas(result.calls)
        if self._streamed_parameters:
            self._stream_call_offset += max(self._streamed_parameters) + 1
        self._streamed_parameters.clear()

        # Update buffer with remaining text
        self._buffer = remaining

        # Add any normal text before the tool call
        if normal_text_before:
            result.normal_text = normal_text_before + (result.normal_text or "")

        return result

    def supports_structural_tag(self) -> bool:
        """
        Return False because LFM2 uses Pythonic format which is not JSON-compatible.

        structural_tag only supports JSON-compatible content between begin and end,
        so it cannot parse Pythonic function call syntax like `func(arg="val")`.
        """
        return False

    def structure_info(self) -> _GetInfoFunc:
        """
        Return structure info for constrained generation.

        Note: This is provided for completeness but won't be used since
        supports_structural_tag() returns False.
        """
        return lambda name: StructureInfo(
            begin="<|tool_call_start|>[" + name + "(",
            end=")]<|tool_call_end|>",
            trigger="<|tool_call_start|>",
        )
