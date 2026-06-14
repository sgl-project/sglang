from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Union

from typing_extensions import Self

from sglang.srt.function_call.compatibility import CompatibilityEvent, CompatibilityMode
from sglang.srt.function_call.compatibility.param_types import (
    AtomDataType,
    FunctionCallParameterDataType,
)
from sglang.srt.function_call.core_types import _GetInfoFunc
from sglang.srt.function_call.parsing import (
    TagToolCallParser,
    default_tool_call_output_key,
    json_dumps,
)
from sglang.srt.function_call.tag_format_detector import TagToolCallDetector


class M3TextParser(TagToolCallParser):
    """
    Parser for MiniMax M3 models.

    M3 uses a namespace token `]<]minimax[>[` as delimiter before each tag.
    Parameters use actual XML tag names (not `<parameter name="...">`), and can be nested.
    Complex arguments, as nested XML tags, are buffered and emitted as complete JSON;
    Simple arguments, are streamed character-by-character if possible.

    Example raw output::

        ]<]minimax[>[<tool_call>
        ]<]minimax[>[<invoke name="func1">]<]minimax[>[<p1>value1]<]minimax[>[</p1>]<]minimax[>[<p2>]<]minimax[>[<item>]<]minimax[>[<k>val]<]minimax[>[</k>]<]minimax[>[</item>]<]minimax[>[</p2>]<]minimax[>[</invoke>
        ]<]minimax[>[</tool_call>
    """

    def __init__(
        self,
        *,
        compatibility: CompatibilityMode,
        functions: Optional[Dict] = None,
        tool_call_xml_tag_name: str = "tool_call",
        tool_call_namespace_token: str = "]<]minimax[>[",
        content_field: str = "content",
        tool_call_output_key: Callable[
            [int, Dict], Dict
        ] = default_tool_call_output_key,
    ):
        self._tool_call_namespace_token = tool_call_namespace_token
        self._tool_call_start = (
            f"{self._tool_call_namespace_token}<{tool_call_xml_tag_name}>"
        )
        self._tool_call_end = (
            f"{self._tool_call_namespace_token}</{tool_call_xml_tag_name}>"
        )
        self._invoke_prefix = f'{self._tool_call_namespace_token}<invoke name="'
        self._invoke_suffix = '">'
        self._end_of_invoke = f"{self._tool_call_namespace_token}</invoke>"
        self._parameter_prefix = f"{self._tool_call_namespace_token}<"
        self._parameter_suffix = f"{self._tool_call_namespace_token}</"
        super().__init__(
            functions=functions,
            content_field=content_field,
            tool_call_output_key=tool_call_output_key,
            compatibility=compatibility,
        )

    def _process(self) -> Generator[dict, None, None]:
        if not self._functions:
            yield from self._take_any(key=self._content_field, commit_each=True)
        else:
            yield from self._take_any(
                until=self._tool_call_start,
                key=self._content_field,
                should_consume_suffix=False,
                commit_each=True,
            )
            self._commit()
            yield from self._literal(self._tool_call_start)
            yield from self._literal("\n", should_raise=False)

            # NOTE: Only ONE `<tool_call>` block is supported by design.
            # Multiple parallel calls must share a single wrapper and use
            # multiple `<invoke>` tags inside it. A second `<tool_call>` after
            # the first `</tool_call>` will cause `update()` to raise
            # PatternMismatched (pattern exhausted).
            tool_call_index = 0
            while True:
                if tool_call_index:
                    yield from self._literal("\n", should_raise=False)
                tried = yield from self._literal(
                    (self._invoke_prefix, self._tool_call_end),
                    should_raise=False,
                )
                if tried == self._tool_call_end:
                    # Block closed cleanly: text up to here is accounted for;
                    # anything later that makes ``update()`` raise (e.g. a
                    # second block) is re-surfaced from this point.
                    self._commit()
                    break
                if tried is None:
                    # Unrecognizable text: leave it uncommitted so the
                    # imminent pattern-exhausted failure re-surfaces the whole
                    # unparsed region (including this block's markers).
                    break

                function_name = yield from self._take_any(until=self._invoke_suffix)
                dropped = yield from self._drop_unknown_invoke(
                    function_name, self._end_of_invoke
                )
                if dropped:
                    continue
                self._emit_name(tool_call_index, function_name)
                function = self._get_function(function_name)
                is_first_parameter = True
                while True:
                    tried = yield from self._literal(
                        (self._end_of_invoke, self._parameter_prefix),
                        should_raise=False,
                    )
                    if tried == self._end_of_invoke:
                        self._emit_call_close(tool_call_index)
                        break
                    if tried is None:
                        # Unparsable text: drop the rest of this invoke and
                        # close its arguments, hoping the format recovers
                        # after the invoke ends.
                        yield from self._recover_invoke(
                            tool_call_index, self._end_of_invoke
                        )
                        break
                    parameter_name = yield from self._take_any(until=">")
                    self._emit_param_key(
                        tool_call_index, parameter_name, is_first_parameter
                    )
                    parameter_suffix = "{}{}>".format(
                        self._parameter_suffix, parameter_name
                    )
                    parameter_data_type = (
                        FunctionCallParameterDataType.get_schema_of_parameter(
                            function, parameter_name
                        )
                    )
                    tried = yield from self._literal(
                        self._parameter_prefix,
                        should_raise=False,
                        should_consume=False,
                    )
                    if tried is not None:
                        param_body_str = yield from self._take_any(
                            until=parameter_suffix
                        )
                        if param_body_str:
                            # nested XML -> object
                            # NOTE: The namespace token has the highest semantic
                            # priority. Once the model emits `]<]minimax[>[<` here,
                            # we MUST treat the body as nested XML, even if the
                            # schema says this parameter should be a primitive.
                            # The model is asserting "this is a JSON level
                            # transition" — schema mismatches are reported back via
                            # the agent loop, not silently rewritten here.
                            if not parameter_data_type.candidates_allow_structure:
                                self._note(
                                    CompatibilityEvent.STRUCTURE_OVERRODE_SCHEMA,
                                    detail=f"parameter {parameter_name[:40]!r}",
                                )
                            param = self._parse_parameter(
                                param_body_str, parameter_data_type
                            )
                        else:
                            param = parameter_data_type.convert(
                                "", compatibility=self.compatibility
                            )
                        self._emit_args(tool_call_index, json_dumps(param))
                    else:
                        # no more nested XML -> string / number / boolean
                        yield from self._take_data_type_as_json(
                            until=parameter_suffix,
                            key=lambda value: self._tool_call_output_key(
                                tool_call_index, {"arguments": value}
                            ),
                            data_type=parameter_data_type,
                            always_nullable=False,
                            should_consume_suffix=True,
                        )
                    is_first_parameter = False
                tool_call_index += 1

    def _parse_parameter(
        self, body: str, parameter_data_type: FunctionCallParameterDataType
    ) -> dict:
        chunks = body.split(self._tool_call_namespace_token)
        # NOTE: Array detection is intentionally a strict "schema says array
        # AND first child is <item>" check. We do NOT promote a uniform
        # `<x><x>...` body to an array on schema mismatch — leave it as
        # `{"x": [...]}` so the agent loop can spot and correct the model.
        if (
            AtomDataType.array in parameter_data_type.candidates
            and len(chunks) > 1
            and chunks[1].startswith("<item>")
        ):
            root = []
        else:
            root = {}
        stack: List[_StackItem] = [
            _StackItem(
                tag=None,
                value=root,
                texts=None,
                data_type=parameter_data_type,
                compatibility=self.compatibility,
            )
        ]

        # Ignore the first chunk inside the parameter.
        # It should be empty, since we've tried `self._parameter_prefix` and failed before entering this function.
        for chunk_index in range(1, len(chunks)):
            chunk = chunks[chunk_index]
            # There are 7 = 3 + 3 + 1 non-empty categories of chunks.
            if chunk.startswith("</"):
                gt_offset = chunk.find(">", 2)
                if gt_offset == -1:
                    # 1. `</tag`
                    tag = chunk[2:]
                    value = None
                elif gt_offset == len(chunk) - 1:
                    # 2. `</tag>`
                    tag = chunk[2:-1]
                    value = None
                else:
                    # 3. `</tag>value`
                    tag = chunk[2:gt_offset]
                    value = chunk[gt_offset + 1 :]
                matched = False
                while len(stack) > 1:
                    item = stack.pop()
                    stack[-1].append(item)
                    if item.tag == tag:
                        matched = True
                        break
                    self._note(
                        CompatibilityEvent.MISMATCHED_CLOSING_TAG,
                        detail=f"</{tag[:40]}> closed open <{str(item.tag)[:40]}>",
                    )
                if not matched:
                    self._note(
                        CompatibilityEvent.MISMATCHED_CLOSING_TAG,
                        detail=f"</{tag[:40]}> matched no open tag",
                    )
                if value:
                    stack[-1].append_text(value)
            elif chunk.startswith("<"):
                gt_offset = chunk.find(">", 1)
                if gt_offset == -1:
                    # 4. `<tag`
                    tag = chunk[1:]
                    value = None
                elif gt_offset == len(chunk) - 1:
                    # 5. `<tag>`
                    tag = chunk[1:-1]
                    value = None
                else:
                    # 6. `<tag>value`
                    tag = chunk[1:gt_offset]
                    value = chunk[gt_offset + 1 :]
                sub_data_type = stack[-1].get_data_type_of_property(tag)
                if (
                    sub_data_type
                    and AtomDataType.array in sub_data_type.candidates
                    and len(chunks) > chunk_index + 1
                    and chunks[chunk_index + 1].startswith("<item>")
                ):
                    sub = []
                elif sub_data_type and AtomDataType.object in sub_data_type.candidates:
                    sub = {}
                else:
                    sub = None
                stack.append(
                    _StackItem(
                        tag=tag,
                        value=sub,
                        texts=[value] if value else None,
                        data_type=sub_data_type,
                        compatibility=self.compatibility,
                    )
                )
            elif chunk:
                # 7. `value`
                stack[-1].append_text(chunk)

        if len(stack) > 1:
            self._note(
                CompatibilityEvent.UNCLOSED_TAGS_AT_END,
                detail=f"{len(stack) - 1} unclosed tag(s)",
            )
        while len(stack) > 1:
            item = stack.pop()
            stack[-1].append(item)

        return stack[0].get_value()


@dataclass(slots=True)
class _StackItem:
    tag: Optional[str]
    value: Optional[Union[Dict, List]]
    texts: Optional[List[str]]
    data_type: Optional[FunctionCallParameterDataType]
    compatibility: CompatibilityMode

    def _note(self, event: CompatibilityEvent, detail: str = "") -> None:
        self.compatibility.note(event, detail)

    def get_value(self) -> Any:
        if self.value is None:
            if self.texts:
                value = "".join(self.texts)
            else:
                value = ""
            if self.data_type:
                value = self.data_type.convert(value, compatibility=self.compatibility)
            return value
        elif self.texts and isinstance(self.value, dict):
            extra_text_key = "$text"
            while extra_text_key in self.value:
                extra_text_key = "$" + extra_text_key
            self._note(
                CompatibilityEvent.MIXED_TEXT_CAPTURED,
                detail=f"under {extra_text_key!r} in <{str(self.tag)[:40]}>",
            )
            self.value[extra_text_key] = "".join(self.texts)
            return self.value
        else:
            return self.value

    def append(self, item: Self) -> None:
        if self.value is None:
            self.value = {item.tag: item.get_value()}
        elif isinstance(self.value, dict):
            if item.tag in self.value:
                # NOTE: Duplicate tag inside an object is collapsed into a
                # list to preserve all values, even if the schema declares
                # the key as a singleton. We don't drop the data and we
                # don't try to "fix" the schema mismatch silently — the agent
                # loop should surface the inconsistency back to the model.
                self._note(
                    CompatibilityEvent.DUPLICATE_TAG_AS_LIST,
                    detail=f"tag <{str(item.tag)[:40]}>",
                )
                value = self.value[item.tag]
                if isinstance(value, list):
                    value.append(item.get_value())
                else:
                    self.value[item.tag] = [value, item.get_value()]
            else:
                self.value[item.tag] = item.get_value()
        elif isinstance(self.value, list):
            # We expect `item.tag` to be `"item"`, but if it's not, we should still accept it.
            if item.tag != "item":
                self._note(
                    CompatibilityEvent.NON_ITEM_ARRAY_CHILD,
                    detail=f"tag <{str(item.tag)[:40]}>",
                )
            self.value.append(item.get_value())
        else:
            raise NotImplementedError()

    def append_text(self, value: str) -> None:
        if isinstance(self.value, list):
            if self.data_type:
                data_type = self.data_type.get_data_type_of_item(index=len(self.value))
                if data_type:
                    value = data_type.convert(value, compatibility=self.compatibility)
            self.value.append(value)
        elif self.texts is None:
            self.texts = [value]
        else:
            self.texts.append(value)

    def get_data_type_of_property(
        self, tag: str
    ) -> Optional[FunctionCallParameterDataType]:
        if self.data_type:
            if isinstance(self.value, list):
                # We expect `tag` to be `"item"`, but if it's not, we should still accept it.
                return self.data_type.get_data_type_of_item(index=len(self.value))
            elif isinstance(self.value, dict):
                return self.data_type.get_data_type_of_property(tag)
            else:
                return None


class MinimaxM3Detector(TagToolCallDetector):
    """Detector for the MiniMax M3 namespace-tagged tool-call format.

    Parsing is implemented by ``M3TextParser``; streaming bookkeeping comes
    from ``TagToolCallDetector`` and error recovery from the package
    compatibility mode (the ``compatibility`` package). Reasoning is handled by SGLang's
    separate ``--reasoning-parser minimax-m3`` stage, so the parser emits only
    ``content`` / ``tool_calls`` deltas.
    """

    def has_tool_call(self, text: str) -> bool:
        return "]<]minimax[>[<tool_call>" in text

    def _make_grammar(
        self, functions: Optional[Dict], compatibility: CompatibilityMode
    ) -> M3TextParser:
        return M3TextParser(functions=functions, compatibility=compatibility)

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError
