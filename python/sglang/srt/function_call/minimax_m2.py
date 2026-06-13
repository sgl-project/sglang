from typing import Dict, Generator, Optional

from sglang.srt.function_call.compatibility import CompatibilityMode
from sglang.srt.function_call.core_types import _GetInfoFunc
from sglang.srt.function_call.parsing import TagToolCallParser
from sglang.srt.function_call.tag_format_detector import TagToolCallDetector


class M2TextParser(TagToolCallParser):
    BLOCK_OPEN = "<minimax:tool_call>"
    BLOCK_CLOSE = "</minimax:tool_call>"
    INVOKE_PREFIX = '<invoke name="'
    INVOKE_SUFFIX = '">'
    INVOKE_END = "</invoke>"
    PARAM_PREFIX = '<parameter name="'
    PARAM_NAME_SUFFIX = '">'
    PARAM_END = "</parameter>"

    def _process(self) -> Generator:
        tool_call_index = 0
        while True:
            # Content before / between / after blocks.
            yield from self._content_until(self.BLOCK_OPEN)
            while True:
                yield from self._whitespace()
                tried = yield from self._literal(
                    (self.INVOKE_PREFIX, self.BLOCK_CLOSE),
                    should_raise=False,
                )
                if tried is None:
                    # Unparseable text inside the block: skip to the next
                    # invoke or to the block close.
                    yield from self._skip_garbage(
                        until=(self.INVOKE_PREFIX, self.BLOCK_CLOSE),
                        detail="inside <minimax:tool_call>",
                    )
                    continue
                if tried == self.BLOCK_CLOSE:
                    break

                function_name = yield from self._take_any(until=self.INVOKE_SUFFIX)
                dropped = yield from self._drop_unknown_invoke(
                    function_name, self.INVOKE_END
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
                        break
                    if tried is None:
                        # Unparseable text inside the invoke: skip to the
                        # next parameter or to the end of the invoke.
                        yield from self._skip_garbage(
                            until=(self.PARAM_PREFIX, self.INVOKE_END),
                            detail="inside <invoke>",
                        )
                        continue

                    parameter_name = yield from self._take_any(
                        until=self.PARAM_NAME_SUFFIX
                    )
                    self._emit_param_key(
                        tool_call_index, parameter_name, is_first_parameter
                    )
                    data_type = self._param_data_type(function, parameter_name)
                    # One newline on each side of the value is structural,
                    # not part of the value.
                    yield from self._literal("\n", should_raise=False)
                    yield from self._param_value(
                        tool_call_index=tool_call_index,
                        until=("\n" + self.PARAM_END, self.PARAM_END),
                        data_type=data_type,
                        always_nullable=True,
                    )
                    is_first_parameter = False

                tool_call_index += 1

class MinimaxM2Detector(TagToolCallDetector):
    """Detector for MiniMax M2 models; the wire format is documented above.

    Parsing is implemented by ``M2TextParser``; streaming bookkeeping comes
    from ``TagToolCallDetector`` and error recovery from the package
    compatibility mode (the ``compatibility`` package).
    """

    def __init__(self):
        super().__init__()
        self.tool_call_start_token: str = "<minimax:tool_call>"
        self.tool_call_end_token: str = "</minimax:tool_call>"

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def _make_grammar(
        self, functions: Optional[Dict], compatibility: CompatibilityMode
    ) -> M2TextParser:
        return M2TextParser(functions=functions, compatibility=compatibility)

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError
