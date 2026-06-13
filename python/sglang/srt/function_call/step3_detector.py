"""Parser and detector for the Step3 tool-call wire format.

Example raw output (the ``｜`` bars are U+FF5C FULLWIDTH VERTICAL LINE)::

    Some content.
    <｜tool_calls_begin｜>
    <｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="get_weather">
    <steptml:parameter name="city">Shanghai</steptml:parameter>
    <steptml:parameter name="count">5</steptml:parameter>
    </steptml:invoke><｜tool_call_end｜>
    <｜tool_calls_end｜>
    More content.

One ``<｜tool_calls_begin｜>`` block per message; multiple
``<｜tool_call_begin｜>…<｜tool_call_end｜>`` entries inside it. Everything
after ``<｜tool_calls_end｜>`` is normal content.

Tolerances (recorded on the detector compatibility policy, see the compatibility package): entries whose type part is not ``function`` are skipped
per-call (historical behavior); unparseable text between tags is skipped up to
the next recognizable tag; parameter values are whitespace-stripped and
converted via the tool's JSON schema; ``null`` becomes JSON null for any
parameter type.
"""

from typing import Dict, Generator, Optional

from sglang.srt.function_call.compatibility import CompatibilityEvent, CompatibilityMode
from sglang.srt.function_call.core_types import _GetInfoFunc
from sglang.srt.function_call.parsing import TagToolCallParser
from sglang.srt.function_call.tag_format_detector import TagToolCallDetector


class Step3TextParser(TagToolCallParser):
    BOT = "<｜tool_calls_begin｜>"
    EOT = "<｜tool_calls_end｜>"
    CALL_BEGIN = "<｜tool_call_begin｜>"
    CALL_END = "<｜tool_call_end｜>"
    TOOL_SEP = "<｜tool_sep｜>"
    INVOKE_PREFIX = '<steptml:invoke name="'
    INVOKE_SUFFIX = '">'
    INVOKE_END = "</steptml:invoke>"
    PARAM_PREFIX = '<steptml:parameter name="'
    PARAM_NAME_SUFFIX = '">'
    PARAM_END = "</steptml:parameter>"

    def _process(self) -> Generator:
        yield from self._content_until(self.BOT)

        tool_call_index = 0
        while True:
            yield from self._whitespace()
            tried = yield from self._literal(
                (self.CALL_BEGIN, self.EOT), should_raise=False
            )
            if tried is None:
                # Unparseable text between calls: skip to the next call or
                # to the end of the block.
                yield from self._skip_garbage(
                    until=(self.CALL_BEGIN, self.EOT),
                    detail="between tool-call entries",
                )
                continue
            if tried == self.EOT:
                break

            type_part = yield from self._take_any(until=self.TOOL_SEP)
            if type_part.strip() != "function":
                # Historical behavior: skip non-function entries per-call.
                self._note(
                    CompatibilityEvent.SKIPPED_NON_FUNCTION_ENTRY,
                    detail=f"type {type_part.strip()[:40]!r}",
                )
                yield from self._take_any(until=self.CALL_END)
                continue

            tried = yield from self._literal(self.INVOKE_PREFIX, should_raise=False)
            if tried is None:
                yield from self._skip_garbage(
                    until=self.CALL_END,
                    detail="function entry without <steptml:invoke>",
                    consume_suffix=True,
                )
                continue

            function_name = yield from self._take_any(until=self.INVOKE_SUFFIX)
            dropped = yield from self._drop_unknown_invoke(
                function_name, self.CALL_END
            )
            if dropped:
                continue
            self._emit_name(tool_call_index, function_name)
            function = self._get_function(function_name)

            is_first_parameter = True
            while True:
                yield from self._whitespace()
                tried = yield from self._literal(
                    (self.INVOKE_END, self.PARAM_PREFIX),
                    should_raise=False,
                )
                if tried == self.INVOKE_END:
                    self._emit_call_close(tool_call_index)
                    # Anything between </steptml:invoke> and the call end
                    # marker is dropped (the historical parser used .find).
                    yield from self._skip_garbage(
                        until=self.CALL_END,
                        detail="between </steptml:invoke> and call end",
                        consume_suffix=True,
                    )
                    break
                if tried is None:
                    yield from self._skip_garbage(
                        until=(self.PARAM_PREFIX, self.INVOKE_END),
                        detail="inside <steptml:invoke>",
                    )
                    continue

                parameter_name = yield from self._take_any(
                    until=self.PARAM_NAME_SUFFIX
                )
                self._emit_param_key(
                    tool_call_index, parameter_name, is_first_parameter
                )
                data_type = self._param_data_type(function, parameter_name)
                yield from self._param_value(
                    tool_call_index=tool_call_index,
                    until=self.PARAM_END,
                    data_type=data_type,
                    always_nullable=True,
                    strip="all",
                )
                is_first_parameter = False

            tool_call_index += 1

        # After the closing token, everything is normal content.
        self._commit()
        yield from self._take_any(key=self._content_field, commit_each=True)

class Step3Detector(TagToolCallDetector):
    """Detector for the Step3 function call format documented above.

    Parsing is implemented by ``Step3TextParser``; streaming bookkeeping
    comes from ``TagToolCallDetector`` and error recovery from the package
    compatibility mode (the ``compatibility`` package).
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜tool_calls_begin｜>"
        self.eot_token = "<｜tool_calls_end｜>"
        self.tool_call_begin = "<｜tool_call_begin｜>"
        self.tool_call_end = "<｜tool_call_end｜>"
        self.tool_sep = "<｜tool_sep｜>"

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Step3 format tool call."""
        return self.bot_token in text

    def _make_grammar(
        self, functions: Optional[Dict], compatibility: CompatibilityMode
    ) -> Step3TextParser:
        return Step3TextParser(functions=functions, compatibility=compatibility)

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()
