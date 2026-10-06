import json
import re
from typing import List, Optional

from xgrammar import StructuralTag
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    OrFormat,
    RegexFormat,
    SequenceFormat,
    TagFormat,
    TagsWithSeparatorFormat,
)

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.parser.harmony_parser import prefix_hold

_CALL_HINT = "to=functions."
_CALL_RE = re.compile(
    r"(?:<\|start\|>)?\s*(?:assistant)?\s*to=functions\.(?P<name>[^\s<]+)\s*"
    r"<\|channel\|>\s*commentary\s*(?:<\|constrain\|>\s*\w+\s*)?<\|message\|>"
    r"(?P<args>.*?)(?P<end><\|call\|>|<\|end\|>|$)",
    re.DOTALL,
)

# Grammar strings match token text, where "▁assistant" is " assistant" whichever
# way the tokenizer decodes.
_GRAMMAR_CALL_REST = r" ?commentary ?(<\|constrain\|>)? *json ?<\|message\|> ?"
_GRAMMAR_FINAL_BEGIN = "<|start|> assistant<|channel|> final<|message|>"


class LlmJpHarmonyDetector(BaseFormatDetector):
    """Detector for LLM-jp-4.1 tool calls.

    LLM-jp-4.1 uses the gpt-oss Harmony format, but its tokenizer decodes a space
    after every special token, and parallel calls close every call but the last
    with ``<|end|>``::

        <|start|> assistant to=functions.get_weather<|channel|> commentary <|constrain|>  json<|message|> {"city": "Tokyo"}<|end|>
        <|start|> assistant to=functions.get_weather<|channel|> commentary <|constrain|>  json<|message|> {"city": "Osaka"}<|call|>

    ``<|call|>`` is an EOS token, so it is trimmed and the last call ends at the end
    of the output.
    """

    def has_tool_call(self, text: str) -> bool:
        return _CALL_HINT in text

    def _to_call_item(
        self, name: str, arguments: str, tools: List[Tool], tool_index: int
    ) -> Optional[ToolCallItem]:
        if (
            name not in self._get_tool_indices(tools)
            and not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get()
        ):
            return None
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return None
        return ToolCallItem(
            tool_index=tool_index,
            name=name,
            parameters=json.dumps(parsed, ensure_ascii=False),
        )

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        tool_indices = self._get_tool_indices(tools)
        calls, normal, pos = [], [], 0
        for match in _CALL_RE.finditer(text):
            normal.append(text[pos : match.start()])
            pos = match.end()
            item = self._to_call_item(
                match["name"],
                match["args"],
                tools,
                tool_indices.get(match["name"], -1),
            )
            if item is not None:
                calls.append(item)
        normal.append(text[pos:])
        return StreamingParseResult(normal_text="".join(normal).strip(), calls=calls)

    def _record_streamed_call(self, item: ToolCallItem) -> ToolCallItem:
        # Calls are sent whole, so the streamed arguments equal the final ones.
        item.tool_index = len(self.prev_tool_call_arr)
        self.prev_tool_call_arr.append(
            {"name": item.name, "arguments": json.loads(item.parameters)}
        )
        self.streamed_args_for_tool.append(item.parameters)
        return item

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        # The reasoning parser passes on only tool-call messages, each opened by
        # <|start|>, so everything from a <|start|> on is held until its call ends.
        self._buffer += new_text
        normal, calls = [], []
        while self._buffer:
            match = _CALL_RE.search(self._buffer)
            if match is None or not match["end"]:
                start = self._buffer.find("<|start|>")
                if start < 0:
                    emit, self._buffer = prefix_hold(self._buffer, ["<|start|>"])
                    normal.append(emit)
                else:
                    normal.append(self._buffer[:start])
                    self._buffer = self._buffer[start:]
                break
            normal.append(self._buffer[: match.start()])
            self._buffer = self._buffer[match.end() :]
            item = self._to_call_item(match["name"], match["args"], tools, 0)
            if item is not None:
                calls.append(self._record_streamed_call(item))
        return StreamingParseResult(normal_text="".join(normal), calls=calls)

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        # The last call has no terminator: <|call|> is an EOS token.
        text, self._buffer = self._buffer, ""
        match = _CALL_RE.search(text)
        if match is None:
            return StreamingParseResult(normal_text=text)
        item = self._to_call_item(match["name"], match["args"], tools, 0)
        calls = [] if item is None else [self._record_streamed_call(item)]
        return StreamingParseResult(normal_text=text[: match.start()], calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError("llm-jp-harmony uses get_structural_tag")

    def get_structural_tag(
        self,
        tools: Optional[List[Tool]] = None,
        tool_choice="auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        # Applied after the reasoning, which the reasoning parser owns.
        tools = tools or []
        if isinstance(tool_choice, ToolChoice):
            tools = [t for t in tools if t.function.name == tool_choice.function.name]
        if not tools:
            return None
        calls = TagsWithSeparatorFormat(
            tags=[_call_tag(tool) for tool in tools],
            separator="<|end|>",
            at_least_one=True,
            stop_after_first=not parallel_tool_calls,
        )
        if tool_choice != "auto":
            return StructuralTag(format=calls)
        # <|return|> is an EOS token, not text, so a final message cannot be a Tag
        # with an end string.
        final = SequenceFormat(
            elements=[ConstStringFormat(value=_GRAMMAR_FINAL_BEGIN), AnyTextFormat()]
        )
        return StructuralTag(format=OrFormat(elements=[calls, final]))

    def get_auto_tool_call_structural_tag(
        self,
        tools: Optional[List[Tool]] = None,
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        # Unconstrained auto output would ignore parallel_tool_calls=false.
        if parallel_tool_calls:
            return None
        return self.get_structural_tag(
            tools=tools,
            tool_choice="auto",
            thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )


def _call_tag(tool: Tool) -> TagFormat:
    # The call ends at EOS (<|call|>) or at the <|end|> separator.
    return TagFormat(
        begin=f"<|start|> assistant to=functions.{tool.function.name}<|channel|>",
        content=SequenceFormat(
            elements=[
                RegexFormat(pattern=_GRAMMAR_CALL_REST),
                JSONSchemaFormat(
                    json_schema=tool.function.parameters or {"type": "object"}
                ),
            ]
        ),
        end="",
    )
