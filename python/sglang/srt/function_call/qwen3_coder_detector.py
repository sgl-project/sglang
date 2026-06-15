"""Parser and detector for the Qwen3-Coder tool-call wire format.

Example raw output::

    Some content.
    <tool_call>
    <function=get_weather>
    <parameter=city>
    Beijing
    </parameter>
    <parameter=count>3</parameter>
    </function>
    </tool_call>

One function per ``<tool_call>`` block; multiple blocks per message; text
before, between, and after blocks is normal content.

Tolerances (recorded on the detector compatibility policy, see the compatibility package; matching
the historical parser): one newline on each side of
a parameter value is structural and stripped; a parameter value may also be
terminated by the next ``<parameter=`` or by ``</function>`` when the model
omits ``</parameter>``; unparsable text inside a block is skipped up to the
next recognizable tag; ``null`` becomes JSON null for any parameter type;
values are converted via the tool's JSON schema.
"""

import json
from typing import Dict, Generator, Optional

from sglang.srt.function_call.compatibility import CompatibilityEvent, CompatibilityMode
from sglang.srt.function_call.core_types import ToolCallItem, _GetInfoFunc
from sglang.srt.function_call.parsing import TagToolCallParser
from sglang.srt.function_call.tag_format_detector import TagToolCallDetector


class Qwen3CoderTextParser(TagToolCallParser):
    BLOCK_OPEN = "<tool_call>"
    BLOCK_CLOSE = "</tool_call>"
    FUNC_PREFIX = "<function="
    FUNC_END = "</function>"
    PARAM_PREFIX = "<parameter="
    PARAM_END = "</parameter>"

    def _process(self) -> Generator:
        tool_call_index = 0
        while True:
            # Content before / between / after blocks.
            yield from self._content_until(self.BLOCK_OPEN)
            yield from self._whitespace()
            tried = yield from self._literal(self.FUNC_PREFIX, should_raise=False)
            if tried is None:
                # No function tag where expected: drop the block body.
                yield from self._skip_garbage(
                    until=self.BLOCK_CLOSE,
                    detail="<tool_call> block without <function=>",
                    consume_suffix=True,
                )
                continue

            function_name = yield from self._take_any(until=">")
            self._emit_name(tool_call_index, function_name)
            function = self._get_function(function_name)

            is_first_parameter = True
            while True:
                yield from self._whitespace()
                tried = yield from self._literal(
                    (self.FUNC_END, self.PARAM_PREFIX),
                    should_raise=False,
                )
                if tried == self.FUNC_END:
                    self._emit_call_close(tool_call_index)
                    break
                if tried is None:
                    yield from self._skip_garbage(
                        until=(self.PARAM_PREFIX, self.FUNC_END),
                        detail="inside <function=>",
                    )
                    continue

                parameter_name = yield from self._take_any(until=">")
                self._emit_param_key(
                    tool_call_index, parameter_name, is_first_parameter
                )
                data_type = self._param_data_type(function, parameter_name)
                # The value normally ends at </parameter>; tolerate a missing
                # close tag by also stopping at the next parameter or the end
                # of the function, consuming only the real terminator.
                yield from self._param_value(
                    tool_call_index=tool_call_index,
                    until=(self.PARAM_END, self.PARAM_PREFIX, self.FUNC_END),
                    data_type=data_type,
                    always_nullable=True,
                    strip="one_newline",
                    should_consume_suffix=False,
                )
                tried = yield from self._literal(self.PARAM_END, should_raise=False)
                if tried is None:
                    self._note(
                        CompatibilityEvent.MISSING_CLOSE_TAG,
                        detail=f"parameter {parameter_name[:40]!r}",
                    )
                is_first_parameter = False

            yield from self._whitespace()
            # Tolerate stray text between </function> and </tool_call>.
            yield from self._skip_garbage(
                until=self.BLOCK_CLOSE,
                detail="between </function> and </tool_call>",
                consume_suffix=True,
            )
            tool_call_index += 1


class Qwen3CoderDetector(TagToolCallDetector):
    """Detector for Qwen3-Coder models; the wire format is documented above.

    Parsing is implemented by ``Qwen3CoderTextParser``; streaming bookkeeping
    comes from ``TagToolCallDetector`` and error recovery from the package
    compatibility mode (the ``compatibility`` package).
    """

    def __init__(self):
        super().__init__()
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def _make_grammar(
        self, functions: Optional[Dict], compatibility: CompatibilityMode
    ) -> Qwen3CoderTextParser:
        # Historical Qwen3-Coder behavior: unknown tool names are forwarded,
        # not gated (mirrors _nonstream_call_items below).
        return Qwen3CoderTextParser(
            functions=functions,
            compatibility=compatibility,
            gate_unknown_tools=False,
        )

    def supports_structural_tag(self) -> bool:
        return True

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def _nonstream_call_items(self, name, args_dict, index, tools):
        # Historical Qwen3-Coder behavior: non-streaming tool_index is the
        # call ordinal and unknown tool names are forwarded rather than
        # filtered through parse_base_json.
        return [
            ToolCallItem(
                tool_index=index,
                name=name,
                parameters=json.dumps(args_dict, ensure_ascii=False),
            )
        ]

    def get_structural_tag_name(self) -> str:
        return "qwen_3_coder"
