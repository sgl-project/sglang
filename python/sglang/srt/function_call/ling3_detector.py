import re
from functools import lru_cache
from typing import Any, Literal

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.base_format_detector import (
    StructuralTag,
    get_model_structural_tag,
)
from sglang.srt.function_call.core_types import (
    NativeToolConstraintError,
    StreamingParseResult,
)
from sglang.srt.function_call.glm4_moe_detector import (
    Glm4MoeDetector,
    _get_argument_schema,
    _is_json_null,
    _json_schema_allows_null,
    get_argument_type,
)


@lru_cache(maxsize=1)
def _ling3_native_structural_tag_available() -> bool:
    if get_model_structural_tag is None:
        return False
    try:
        get_model_structural_tag(
            model="glm_4_7", tools=[], tool_choice="auto", reasoning=False
        )
        return True
    except ValueError:
        # Older XGrammar versions may import successfully without registering
        # the GLM XML format. Constrained requests must fail closed in that case.
        return False


# An argument value is raw text up to </arg_value>, as in the native grammar, so a
# </tool_call> inside it does not close the call; an open value runs to the end.
# Plain text never starts at <arg_value>, so each span has one parse (no ReDoS).
_TOOL_CALL_BODY = (
    r"(?:<arg_value>(?:(?!</arg_value>).)*(?:</arg_value>|$)"
    r"|(?!</tool_call>|<arg_value>).)*"
)
# Opener, name and separator each have one way to split the text: whitespace runs
# are taken whole, and the name stops at the first delimiter or nested opener.
_NAME_STOP = r"\\n|\n|<arg_key>|</tool_call>|<tool_call>"
_TOOL_CALL_HEAD = (
    rf"<tool_call>\s*(?!\s)((?:(?!\s|{_NAME_STOP}).(?:(?!{_NAME_STOP}).)*)?)"
    r"(?:(?:\\n|\n)\s*(?!\s)|(?=<arg_key>)|(?=</tool_call>))"
)


class Ling3Detector(Glm4MoeDetector):
    """
    Detector for Ling3 tool calls.

    Ling3 uses the GLM-4.5 XML format, but model outputs may either put a newline
    after the function name, emit the first argument tag immediately, or close a
    no-argument tool call immediately.

    Constrained tool calls use XGrammar's native GLM XML schema serializer,
    which also accepts a newline after the function name.
    """

    _STREAMING_PARTIAL_PATTERN = re.compile(
        rf"{_TOOL_CALL_HEAD}({_TOOL_CALL_BODY})(</tool_call>|$)", re.DOTALL
    )

    def __init__(self):
        super().__init__()
        # An unfinished trailing call is matched too, so findall does not restart
        # inside its open value; it yields "" for it, which the caller skips.
        self.func_call_regex = (
            rf"(<tool_call>{_TOOL_CALL_BODY}</tool_call>)|<tool_call>{_TOOL_CALL_BODY}$"
        )
        self.func_detail_regex = re.compile(
            rf"{_TOOL_CALL_HEAD}(<arg_key>{_TOOL_CALL_BODY})?</tool_call>", re.DOTALL
        )

    def supports_structural_tag(self) -> bool:
        # The serving layer also uses this capability to select the native
        # parser for required/named tool calls, including streaming responses.
        return _ling3_native_structural_tag_available()

    def get_structural_tag_name(self) -> str:
        # Ling and GLM share <arg_key>/<arg_value> argument serialization.
        return "glm_4_7"

    def get_structural_tag(
        self,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | Literal["auto", "required"] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> StructuralTag | None:
        try:
            if not self.supports_structural_tag():
                raise NativeToolConstraintError(
                    "Ling3 native XML tool constraints require XGrammar with "
                    "glm_4_7 support; refusing JSON fallback."
                )

            from xgrammar.structural_tag import (
                TagsWithSeparatorFormat,
                TriggeredTagsFormat,
            )

            tag = super().get_structural_tag(
                tools=tools,
                tool_choice=tool_choice,
                thinking_mode=thinking_mode,
                parallel_tool_calls=parallel_tool_calls,
            )
            # The builtin builder owns tool-choice normalization, strict schemas,
            # and the optional reasoning prefix. Adapt only the tool-call suffix.
            suffix = tag.format.elements[-1] if thinking_mode else tag.format
            if isinstance(suffix, (TriggeredTagsFormat, TagsWithSeparatorFormat)):
                suffix.stop_after_first = not parallel_tool_calls
            return tag
        except NativeToolConstraintError:
            raise
        except Exception as exc:
            raise NativeToolConstraintError(
                "Failed to build Ling3 native XML tool constraints; check the "
                "tool schema and XGrammar glm_4_7 support. Refusing JSON fallback."
            ) from exc

    def _parse_argument_pairs(
        self, pairs: list[tuple[str, str]], func_name: str, tools: list[Tool]
    ) -> dict[str, Any]:
        arguments = super()._parse_argument_pairs(pairs, func_name, tools)
        for key, value in pairs:
            key = key.strip()
            if get_argument_type(func_name, key, tools) != "string":
                continue
            # XML strings are raw text, not JSON literals, as in streaming;
            # only a bare null under a nullable schema is a typed null.
            if _is_json_null(value) and _json_schema_allows_null(
                _get_argument_schema(func_name, key, tools)
            ):
                arguments[key] = None
            else:
                arguments[key] = value
        return arguments

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        # The parent consumes one call per invocation; drain a coalesced delta now,
        # since no further delta may arrive to flush the remaining calls.
        self._buffer += new_text
        result = StreamingParseResult()
        while self._buffer:
            if not self.current_tool_name_sent:
                start = self._buffer.find(self.bot_token)
                if start > 0:
                    result.normal_text += self._buffer[:start]
                    self._buffer = self._buffer[start:]
            previous_buffer = self._buffer
            parsed = super().parse_streaming_increment("", tools)
            result.normal_text += parsed.normal_text
            result.calls.extend(parsed.calls)
            if self._buffer == previous_buffer:
                # An incomplete marker or call needs another input delta.
                break
        return result
