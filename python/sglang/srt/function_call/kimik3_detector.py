import json
from typing import List, Literal, Optional, Union

from xgrammar import StructuralTag

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.kimik3_format import TOOLS_CLOSE, TOOLS_OPEN
from sglang.srt.function_call.kimik3_native import KimiK3NativeParser, KimiK3ParseError
from sglang.srt.function_call.kimik3_structural_tag import (
    get_kimik3_auto_tool_call_structural_tag,
    get_kimik3_structural_tag,
)


class KimiK3Detector(BaseFormatDetector):
    """Detector for the Kimi K3 XTML tool-call format.

    K3 emits tool calls in a ``tools`` channel built from dedicated special
    tokens; the plain reply lives in a preceding ``response`` channel:

    ```
    <|open|>response<|sep|>text<|close|>response<|sep|>
    <|open|>tools<|sep|>
      <|open|>call tool="name" index="1"<|sep|>
        <|open|>argument key="k" type="string"<|sep|>raw text<|close|>argument<|sep|>
      <|close|>call<|sep|>
    <|close|>tools<|sep|>
    ```

    ``type="string"`` argument values are raw text; other types are
    JSON-decoded. Attribute values reverse the template's ``&amp;``/``&quot;``
    escaping.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = TOOLS_OPEN
        self.eot_token = TOOLS_CLOSE
        self._native = KimiK3NativeParser(stream_arguments=True)

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def supports_structural_tag(self) -> bool:
        return True

    def parses_required_natively(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError(
            "Kimi K3 uses its model-native structural tag implementation"
        )

    def get_auto_tool_call_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        return get_kimik3_auto_tool_call_structural_tag(
            tools or [],
            thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> StructuralTag:
        return get_kimik3_structural_tag(
            tools=tools or [],
            tool_choice=tool_choice,
            thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )

    def detect_and_parse(
        self, text: str, tools: List[Tool], *, allow_incomplete: bool = False
    ) -> StreamingParseResult:
        # Use the same state machine for complete and incrementally delivered
        # output. Never downgrade a native framing error to ordinary text.
        parser = KimiK3NativeParser()
        result = parser.feed(text)
        tail = parser.finish(allow_incomplete=allow_incomplete)
        self._validate_tool_names(result.calls + tail.calls, tools)
        return StreamingParseResult(
            normal_text=result.normal_text + tail.normal_text,
            calls=[
                ToolCallItem(
                    tool_index=i,
                    name=call.name,
                    parameters=json.dumps(
                        call.arguments, ensure_ascii=False, allow_nan=False
                    ),
                )
                for i, call in enumerate(result.calls + tail.calls)
            ],
        )

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._native.allowed_tools = self._declared_tools(tools) if tools else None
        result = self._native.feed(new_text)
        calls = []
        for delta in result.deltas:
            if delta.name is not None:
                self.current_tool_id += 1
                # The serving terminal flush treats an object here as complete
                # expected arguments. Keep a sentinel until native call close
                # is validated; a length-cut header must remain just "{".
                self.prev_tool_call_arr.append({"name": delta.name, "arguments": ""})
                self.streamed_args_for_tool.append("")
            self.streamed_args_for_tool[self.current_tool_id] += delta.parameters
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=delta.name,
                    parameters=delta.parameters,
                )
            )
        # Record only validated complete argument objects. Deltas remain
        # provisional until the serving layer validates the terminal.
        for offset, call in enumerate(reversed(result.calls)):
            index = self.current_tool_id - offset
            # A chunk may end with the next call's header but no completed call.
            if self._native.state in {"call", "argument"}:
                index -= 1
            self.prev_tool_call_arr[index]["arguments"] = call.arguments
        return StreamingParseResult(normal_text=result.normal_text, calls=calls)

    @staticmethod
    def _declared_tools(tools):
        return {
            tool.get("function", {}).get("name")
            if isinstance(tool, dict)
            else tool.function.name
            for tool in tools
        }

    @staticmethod
    def _validate_tool_names(calls, tools: List[Tool]) -> None:
        # Low-level detector callers may omit a schema list. Serving always
        # supplies the declared tools; reject an unknown name before its delta.
        if calls and tools:
            declared = KimiK3Detector._declared_tools(tools)
            if any(call.name not in declared for call in calls):
                raise KimiK3ParseError("unknown_tool")

    def finish(
        self, tools: List[Tool], *, allow_incomplete: bool = False
    ) -> StreamingParseResult:
        result = self._native.finish(allow_incomplete=allow_incomplete)
        return StreamingParseResult(normal_text=result.normal_text)
