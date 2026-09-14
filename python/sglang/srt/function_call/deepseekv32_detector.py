import json
import logging
import re

from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import _find_common_prefix, _partial_json_loads

logger = logging.getLogger(__name__)
_JSON_DECODER = json.JSONDecoder()


class MalformedDSMLToolCall(ValueError):
    """A complete invoke body that is not well-formed DSML: a parameter tag
    without the DSML marker or the ``string`` attribute, a truncated or
    non-object JSON body, stray text. Raised only under strict parsing
    (``SGLANG_ENABLE_STRICT_DSML_TOOL_CALLS``); the callers drop the call and
    forward the text as content."""


class DeepSeekV32Detector(BaseFormatDetector):
    """
    Detector for DeepSeek V3.2 model function call format.

    The DeepSeek V3.2 format uses XML-like DSML tags to delimit function calls.
    Supports two parameter formats:

    Format 1 - XML Parameter Tags:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>
        ...
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Format 2 - Direct JSON:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        {
            "param_name": "value"
        }
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Examples:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>

    <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        { "city": "San Francisco" }
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜DSML｜function_calls>` and `</｜DSML｜function_calls>`
    - Individual Tool Call: Wrapped between `<｜DSML｜invoke name="...">` and `</｜DSML｜invoke>`
    - Parameters: Either XML tags or direct JSON format
    - Supports multiple tool calls

    Strict parsing (``SGLANG_ENABLE_STRICT_DSML_TOOL_CALLS=1``, off by default):
    a complete invoke body must be well-formed (every non-blank character inside
    a matched parameter tag, or one JSON object), like the vendor parser
    requires. A malformed body drops the call and forwards the text as content
    with a warning instead of parsing as ``{}``; in streaming the tool name
    travels in the same delta as the first argument bytes and the arguments
    stay incomplete JSON until the closer validates the body, so a client
    never holds a name with empty arguments or a complete call the one-shot
    path would refuse, and a malformed invoke turns the rest of its calls
    block into content.

    Reference: DeepSeek V3.2 format specification
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜DSML｜function_calls>"
        self.eot_token = "</｜DSML｜function_calls>"
        self.invoke_end_token = "</｜DSML｜invoke>"
        self.parameter_regex = r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</｜DSML｜parameter>'
        self.partial_parameter_regex = (
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*)$'
        )
        self.function_calls_regex = (
            r"<｜DSML｜function_calls>(.*?)</｜DSML｜function_calls>"
        )
        # Long-form `<｜DSML｜invoke name="x">...</｜DSML｜invoke>` and the
        # self-closing `<｜DSML｜invoke name="x"/>` shape V4 emits for zero-arg
        # tools. The `end` group is empty when the closer hasn't streamed in.
        self.invoke_regex = (
            r'<｜DSML｜invoke\s+name="(?P<name>[^"]+)"\s*'
            r"(?:(?P<self_close>/>)"
            r"|>(?P<body>.*?)(?P<end>(?:</｜DSML｜invoke>|$)))"
        )
        self.prefix_parameter_end_call = ["</", "｜DSML｜", "parameter"]
        self.prefix_invoke_end_call = ["</", "｜DSML｜", "inv", "oke"]
        self.current_tool_id = -1
        self.strict = envs.SGLANG_ENABLE_STRICT_DSML_TOOL_CALLS.get()
        # Strict streaming state. _pending_params: partial arguments of a call
        # whose name has not gone out yet (nothing is tracked for it, so a cut
        # stream leaves nothing to back-fill). _poisoned_block: a malformed
        # invoke was dropped and the rest of its calls block is content.
        # _trimmed_separator: the "\n\n" trimmed off the preamble, re-emitted
        # if the block is dropped in a later pass.
        self._pending_params = None
        self._poisoned_block = False
        self._trimmed_separator = ""

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek v32 format tool call."""
        return self.bot_token in text or "<｜DSML｜invoke" in text

    @staticmethod
    def _unpack_invoke_match(m: "re.Match[str]") -> tuple[str, str, bool]:
        """Returns (name, body, is_complete) for an invoke_regex match.

        Self-closing invokes have empty body and are always complete.
        Long-form bodies are always strings (possibly empty); they're
        incomplete when matched against `$` because the closing tag
        hasn't streamed in yet.
        """
        name = m.group("name").strip()
        if m.group("self_close"):
            return name, "", True
        return name, m.group("body"), bool(m.group("end"))

    @staticmethod
    def _require_json_object(body: str) -> None:
        try:
            json.loads(body)
        except (json.JSONDecodeError, ValueError) as e:
            raise MalformedDSMLToolCall(f"invalid JSON body: {e}") from e

    @staticmethod
    def _hold_json_close(prefix: str) -> str:
        """The longest start of ``prefix`` that does not begin with a complete
        JSON value. Strict streaming sends this instead of ``prefix`` on a
        partial pass, so what a client holds for a call stays incomplete JSON
        until the closer arrives and the body passes validation; the character
        that completes the object, and any text the body carries after it, go
        out with the closer or not at all."""
        try:
            _, end = _JSON_DECODER.raw_decode(prefix)
        except json.JSONDecodeError:
            return prefix
        return prefix[: end - 1]

    def _parse_parameters_from_xml(
        self, invoke_content: str, allow_partial: bool = False
    ) -> str:
        """
        Parse parameters from either XML-like format or JSON format to str.

        Supports two formats:
        1. XML parameter tags: <｜DSML｜parameter name="..." string="...">value</｜DSML｜parameter>
        2. Direct JSON: { "key": "value" }

        Under strict parsing a complete body raises MalformedDSMLToolCall unless
        it is one JSON object or every non-blank character sits inside a
        matched parameter tag.
        """
        # First, try to parse as direct JSON (new format)
        invoke_content_stripped = invoke_content.strip()
        if invoke_content_stripped.startswith("{"):
            if allow_partial:
                # Remove incomplete invoke end call prefix in case they are captured by param
                for token in reversed(self.prefix_invoke_end_call):
                    invoke_content_stripped = invoke_content_stripped.rstrip(token)
                return invoke_content_stripped
            elif invoke_content_stripped.endswith("}"):
                if self.strict:
                    self._require_json_object(invoke_content_stripped)
                return invoke_content_stripped

        # Fall back to XML parameter tag parsing (original format)
        parameters = {}
        # Find all complete parameter matches
        param_matches = list(
            re.finditer(self.parameter_regex, invoke_content, re.DOTALL)
        )

        last_match_end = 0
        unparsed = []  # body text outside the matched parameter tags
        for match in param_matches:
            unparsed.append(invoke_content[last_match_end : match.start()])
            param_name = match.group(1)
            param_type = match.group(2)
            param_value = match.group(3)
            last_match_end = match.end()

            # Convert value based on type
            if param_type == "true":  # string type
                parameters[param_name] = param_value.strip()
            else:
                # Try to parse as JSON for other types
                try:
                    parameters[param_name] = json.loads(param_value.strip())
                except (json.JSONDecodeError, ValueError):
                    parameters[param_name] = param_value.strip()

        # If allowed, try to parse a partial parameter at the end
        if allow_partial:
            remaining_content = invoke_content[last_match_end:]

            # Remove incomplete parameter_end_call prefix in case they are captured by param
            for token in reversed(self.prefix_parameter_end_call):
                remaining_content = remaining_content.rstrip(token)

            # Match start of a parameter tag + value (potentially incomplete)
            # Regex: <tag name="..." string="...">VALUE... (no end tag)
            partial_match = re.search(
                self.partial_parameter_regex, remaining_content, re.DOTALL
            )

            if partial_match and (param_value := partial_match.group(3)):
                param_name = partial_match.group(1)
                if partial_match.group(2) == "true":
                    parameters[param_name] = param_value.strip()
                else:
                    try:
                        parameters[param_name] = _partial_json_loads(
                            param_value, Allow.ALL
                        )[0]
                    except (json.JSONDecodeError, MalformedJSON, ValueError):
                        parameters[param_name] = param_value.strip()
        elif self.strict:
            # A tag without the DSML marker or the string attribute, a truncated
            # JSON body or prose would otherwise parse as `{}`, an executable
            # zero-argument call; an empty or blank body still is one.
            unparsed.append(invoke_content[last_match_end:])
            leftover = "".join(unparsed).strip()
            if leftover:
                raise MalformedDSMLToolCall(
                    f"unparsed text inside the invoke body: {leftover[:80]!r}"
                )

        return json.dumps(parameters, ensure_ascii=False)

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].removesuffix("\n\n") if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        calls = []
        try:
            sections = re.findall(self.function_calls_regex, text, re.DOTALL)
            if not sections:
                return StreamingParseResult(normal_text=normal_text, calls=[])

            # Find all invoke blocks
            for function_calls_content in sections:
                for invoke_match in re.finditer(
                    self.invoke_regex, function_calls_content, re.DOTALL
                ):
                    func_name, invoke_content, _ = self._unpack_invoke_match(
                        invoke_match
                    )
                    try:
                        func_args = self._parse_parameters_from_xml(invoke_content)
                    except MalformedDSMLToolCall as e:
                        # Atomic: no call of the turn survives, the client gets
                        # the text and finish_reason "stop".
                        logger.warning(
                            "Malformed DSML tool call for %s dropped; forwarding the turn as text: %s",
                            func_name,
                            e,
                        )
                        return StreamingParseResult(normal_text=text)
                    # construct match_result for parse_base_json
                    match_result = {
                        "name": func_name,
                        "parameters": json.loads(func_args),
                    }
                    calls.extend(self.parse_base_json(match_result, tools))

            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def _ensure_tool_slots(self) -> None:
        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for DeepSeekV32 format.
        Supports multiple consecutive invoke blocks and argument streaming.
        """
        self._buffer += new_text
        current_text = self._buffer

        if self._poisoned_block:
            return self._forward_poisoned_block(current_text, tools)

        # Check if buffer contains any DSML markers or ends with potential tag prefix
        # This handles partial/streaming DSML content
        dsml_markers = ["｜DSML｜", "<｜", "</｜"]
        potentially_dsml = any(marker in current_text for marker in dsml_markers)

        # Also check if text ends with start of a tag (to handle "<" arriving separately)
        dsml_prefixes = ["<", "<｜", "</", "</｜"]
        ends_with_prefix = any(
            current_text.rstrip().endswith(prefix) for prefix in dsml_prefixes
        )

        if (
            not self.has_tool_call(current_text)
            and not potentially_dsml
            and not ends_with_prefix
        ):
            if self.strict and not current_text.strip():
                # Whitespace ahead of a possible calls block (the "\n\n" the
                # encoder puts before it) is held, not sent as its own content
                # delta; the preamble trim consumes it or finish() releases it.
                return StreamingParseResult()
            self._buffer = ""
            for e_token in [self.eot_token, self.invoke_end_token]:
                if e_token in current_text:
                    current_text = current_text.replace(e_token, "")
            return StreamingParseResult(normal_text=current_text)

        all_calls: list[ToolCallItem] = []
        # Only recovered for the first call: the DSML guard above never releases a
        # buffer that still holds a marker, so later prose stays buffered.
        # raw_head is the same text before the "\n\n" trim, for a strict drop.
        preamble = ""
        raw_head = ""
        try:
            # Loop to handle multiple consecutive invoke blocks
            while True:
                # Try to match an invoke block (may be partial)
                invoke_match = re.search(
                    pattern=self.invoke_regex,
                    string=current_text,
                    flags=re.DOTALL,
                )
                if not invoke_match:
                    break

                func_name, invoke_content, is_tool_end = self._unpack_invoke_match(
                    invoke_match
                )

                # Where this call's wire text starts: the calls-block opener when
                # it is still buffered (first call of a block), else the invoke.
                block_start = invoke_match.start()
                bot_pos = current_text.rfind(self.bot_token, 0, block_start)
                if bot_pos != -1:
                    block_start = bot_pos

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]
                    # Same trailing-newline trim as detect_and_parse, so both agree.
                    raw_head = current_text[:block_start]
                    preamble = raw_head.removesuffix("\n\n")
                    self._trimmed_separator = raw_head[len(preamble) :]

                if not self.strict:
                    self._ensure_tool_slots()
                    # 1. Send tool name if not sent yet
                    if not self.current_tool_name_sent:
                        all_calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=func_name,
                                parameters="",
                            )
                        )
                        self.current_tool_name_sent = True

                # 2. Parse current parameters (partial or complete)
                try:
                    current_params = self._parse_parameters_from_xml(
                        invoke_content, allow_partial=not is_tool_end
                    )
                except MalformedDSMLToolCall as e:
                    # The untrimmed preamble when this pass recovered it, else
                    # the separator an earlier pass trimmed off the one it sent.
                    head = raw_head or self._trimmed_separator
                    self._trimmed_separator = ""
                    return self._drop_malformed_streaming_call(
                        func_name=func_name,
                        error=e,
                        preamble=head,
                        raw_block=current_text[block_start:],
                        earlier_calls=all_calls,
                        tools=tools,
                    )

                # 3. Calculate and send incremental arguments
                if self.current_tool_name_sent:
                    sent_len = len(self.streamed_args_for_tool[self.current_tool_id])
                    prev_params = self.prev_tool_call_arr[self.current_tool_id].get(
                        "arguments"
                    )
                else:
                    # Strict: nothing is tracked for the call until its name goes out.
                    sent_len = 0
                    prev_params = self._pending_params

                argument_diff = None

                if is_tool_end:
                    # If complete, send everything remaining
                    argument_diff = current_params[sent_len:]
                elif prev_params is not None and current_params != prev_params:
                    # If partial, send stable prefix diff
                    prefix = _find_common_prefix(current_params, prev_params)
                    if self.strict:
                        # A body can be a complete JSON object followed by
                        # text the closer will reject; the arguments stay
                        # incomplete until the body is validated end to end.
                        prefix = self._hold_json_close(prefix)
                    if len(prefix) > sent_len:
                        argument_diff = prefix[sent_len:]

                if not self.current_tool_name_sent:
                    # Strict: the name goes out in the same item as the first
                    # non-empty argument delta (the whole body at the closer,
                    # "{}" at the least), so a client never holds a name with
                    # "" arguments, which it would execute as a zero-argument
                    # call if the stream is cut or the close is malformed.
                    if not argument_diff and not is_tool_end:
                        self._pending_params = current_params
                        break
                    self._ensure_tool_slots()
                    all_calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters=argument_diff,
                        )
                    )
                    self.current_tool_name_sent = True
                    self._pending_params = None
                elif argument_diff:
                    all_calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=argument_diff,
                        )
                    )

                if argument_diff:
                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff

                # Update the stored arguments
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": func_name,
                    "arguments": current_params,
                }

                # Check if tool call is complete (has closing tag)
                if is_tool_end:
                    # Remove the completed tool call from buffer
                    self._buffer = current_text[invoke_match.end() :]
                    current_text = self._buffer  # Update for next iteration
                    self._trimmed_separator = ""

                    # Move to next tool call
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False

                    # Continue loop to check for more invoke blocks
                    continue
                else:
                    # Tool call not complete yet, don't return anything
                    # Wait for more chunks until we see </｜DSML｜invoke>
                    break

            # No more invoke blocks found
            return StreamingParseResult(normal_text=preamble, calls=all_calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            # Re-emit verbatim rather than swallowing the turn; the preamble is
            # still inside current_text unless a completed call advanced past it.
            # Calls are dropped on purpose: the failure can land between a tool's
            # name and its arguments, and a half-formed call is worse than none.
            self._buffer = ""
            self._pending_params = None
            self._poisoned_block = False
            self._trimmed_separator = ""
            if not current_text.startswith(preamble):
                current_text = preamble + current_text
            return StreamingParseResult(normal_text=current_text)

    def _drop_malformed_streaming_call(
        self,
        func_name: str,
        error: MalformedDSMLToolCall,
        preamble: str,
        raw_block: str,
        earlier_calls: list[ToolCallItem],
        tools: list[Tool],
    ) -> StreamingParseResult:
        """Strict streaming drop: the call's wire text becomes content, the
        rest of its calls block is poisoned, calls completed earlier in the
        pass are kept, and nothing is left for the serving layer to back-fill.

        A name that already went out cannot be recalled; its tracked arguments
        are pinned to the streamed prefix (unparsable JSON, never `{}`).
        """
        logger.warning(
            "Malformed DSML tool call for %s dropped; forwarding the block as text: %s",
            func_name,
            error,
        )
        self._pending_params = None
        if self.current_tool_name_sent:
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": func_name,
                "arguments": self.streamed_args_for_tool[self.current_tool_id],
            }
            self.current_tool_id += 1
            self.current_tool_name_sent = False
        else:
            del self.prev_tool_call_arr[self.current_tool_id :]
            del self.streamed_args_for_tool[self.current_tool_id :]
        self._poisoned_block = True
        forwarded = self._forward_poisoned_block(raw_block, tools)
        return StreamingParseResult(
            normal_text=preamble + forwarded.normal_text,
            calls=earlier_calls + forwarded.calls,
        )

    def _forward_poisoned_block(
        self, current_text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        """Content up to and including the calls-block closer; the text after
        it goes back through parse_streaming_increment. Without the closer,
        everything but a suffix that could be the start of it is forwarded."""
        end = current_text.find(self.eot_token)
        if end == -1:
            hold = self._ends_with_partial_token(current_text, self.eot_token)
            keep = len(current_text) - hold
            self._buffer = current_text[keep:]
            return StreamingParseResult(normal_text=current_text[:keep])
        cut = end + len(self.eot_token)
        self._buffer = ""
        self._poisoned_block = False
        normal_text, calls = current_text[:cut], []
        if current_text[cut:]:
            rest = self.parse_streaming_increment(current_text[cut:], tools)
            normal_text += rest.normal_text
            calls = rest.calls
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def finish(self, tools: list[Tool]) -> StreamingParseResult:
        """Strict: release text held for a marker that can no longer come
        (whitespace, a stray "<", the tail of a poisoned block). Text holding a
        DSML marker (an unterminated calls block) stays dropped, as in
        detect_and_parse."""
        if not self.strict:
            return StreamingParseResult()
        held, self._buffer = self._buffer, ""
        self._pending_params = None
        self._trimmed_separator = ""
        if self._poisoned_block:
            self._poisoned_block = False
            return StreamingParseResult(normal_text=held)
        if (
            held
            and not self.has_tool_call(held)
            and not any(marker in held for marker in ("｜DSML｜", "<｜", "</｜"))
        ):
            return StreamingParseResult(normal_text=held)
        return StreamingParseResult()

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'<｜DSML｜invoke name="{name}">',
            end="</｜DSML｜invoke>",
            trigger="<｜DSML｜invoke",
        )

    def get_structural_tag_name(self) -> str:
        return "deepseek_v3_2"
