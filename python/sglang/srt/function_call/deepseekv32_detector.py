import json
import logging
import re

from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import _find_common_prefix, _partial_json_loads

logger = logging.getLogger(__name__)


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
        # False while nothing but whitespace has been emitted since the last DSML
        # tag: that whitespace is the layout separating the tag from the prose
        # after it, not content. True to start with, because the preamble is
        # content from the first character.
        self._gap_started = True
        # Whether any normal text has left the detector this turn. A gap has
        # something to separate itself from only if it has.
        self._emitted_anything = False

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

    def _parse_parameters_from_xml(
        self, invoke_content: str, allow_partial: bool = False
    ) -> str:
        """
        Parse parameters from either XML-like format or JSON format to str.

        Supports two formats:
        1. XML parameter tags: <｜DSML｜parameter name="..." string="...">value</｜DSML｜parameter>
        2. Direct JSON: { "key": "value" }
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
                return invoke_content_stripped

        # Fall back to XML parameter tag parsing (original format)
        parameters = {}
        # Find all complete parameter matches
        param_matches = list(
            re.finditer(self.parameter_regex, invoke_content, re.DOTALL)
        )

        last_match_end = 0
        for match in param_matches:
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

        return json.dumps(parameters, ensure_ascii=False)

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        if idx == -1:
            # No section opener, but the turn may have been cut off inside one.
            return StreamingParseResult(normal_text=self._cut_at_markup(text), calls=[])

        # The whitespace in front of the opener indents it; the line break the
        # call stood on comes back from `_join_gap` on the other side.
        normal_text = text[:idx].rstrip()

        calls = []
        try:
            sections = list(re.finditer(self.function_calls_regex, text, re.DOTALL))
            if not sections:
                return StreamingParseResult(normal_text=normal_text, calls=[])

            normal_text += self._section_gaps(text, sections)

            # Find all invoke blocks
            for section in sections:
                function_calls_content = section.group(1)
                for invoke_match in re.finditer(
                    self.invoke_regex, function_calls_content, re.DOTALL
                ):
                    func_name, invoke_content, _ = self._unpack_invoke_match(
                        invoke_match
                    )
                    func_args = self._parse_parameters_from_xml(invoke_content)
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

    @staticmethod
    def _cut_at_markup(text: str) -> str:
        """Everything up to the first DSML tag, opening or closing, and its indent.

        A turn can stop anywhere -- a length cap, an abort -- including in the
        middle of a tag the model was still writing. What follows the `<｜` (or
        `</｜`) was markup by then, not content, and must not reach the client;
        what precedes it is prose no completed call ever consumed. The whitespace
        in front of the tag goes with it, exactly as `_strip_section_markers`
        drops the indent of a tag that did finish arriving.
        """
        markup = re.search(r"\s*</?｜", text)
        return text if markup is None else text[: markup.start()]

    @staticmethod
    def _join_gap(layout: str, *, after_text: bool) -> str:
        """What a gap's leading whitespace collapses to once its call is gone.

        The call sat on its own line, so removing it has to leave the line break
        it was standing on -- and nothing more. Deleting the whitespace outright
        runs `'Let me look it up.'` into `'It is sunny in SF.'`, and takes the
        indent off the line that follows, which is content: a list item stops
        starting its line, a code line loses the four spaces the newline was
        carrying. Collapsing to two would open a paragraph break the model never
        wrote.

        Nothing at all when no text has been emitted yet: the turn opened with
        the call, so there is nothing on the other side to separate from. And a
        run with no newline in it never had a line to keep -- the call was inline
        -- so it stands as the model wrote it.
        """
        if not after_text:
            return ""
        line_start = layout.rfind("\n")
        return layout if line_start == -1 else "\n" + layout[line_start + 1 :]

    def _gap_text(self, text: str) -> str:
        """Resolve the layout whitespace at the head of a gap, once per gap.

        Tracked on the instance rather than trimmed in place because a gap
        arrives over several deltas: only the whitespace before the gap's first
        real character is layout, and a newline in the middle of the prose has
        to survive. Whitespace with nothing behind it yet is left for a later
        delta to resolve, since what it collapses to depends on the text that
        follows it.

        Call this on text that is being emitted, never on text `_split_flushable`
        may still hold back: a fragment that grows into a tag would otherwise
        count as the gap's first real character and strand the layout behind it.
        """
        if self._gap_started:
            self._emitted_anything = self._emitted_anything or bool(text)
            return text

        prose = text.lstrip()
        if not prose:
            return ""

        layout = text[: len(text) - len(prose)]
        self._gap_started = True
        joined = self._join_gap(layout, after_text=self._emitted_anything) + prose
        self._emitted_anything = True
        return joined

    def _section_gaps(self, text: str, sections: list["re.Match[str]"]) -> str:
        """The normal text between and after the tool call sections of `text`.

        The one-shot counterpart of what the streaming path emits from
        `_gap_text` and `_lead_text`, trimmed to match: leading layout goes
        through `_join_gap`, trailing layout goes wherever another section
        follows it and can claim the whitespace as its indent. The last gap ends
        the turn, so its trailing newlines are the model's own.
        """
        emitted = bool(text[: sections[0].start()].strip())
        gaps = []
        for i, section in enumerate(sections):
            is_last = i + 1 == len(sections)
            end = len(text) if is_last else sections[i + 1].start()
            gap = self._cut_at_markup(text[section.end() : end])
            prose = gap.lstrip()
            if prose in ("<", "</"):
                # Nothing in this gap but the first characters of the next tag,
                # which the turn stopped inside of. Layout, not the model's `<`.
                prose = ""
            if not prose:
                continue
            layout = gap[: len(gap) - len(gap.lstrip())]
            joined = self._join_gap(layout, after_text=emitted) + prose
            gaps.append(joined if is_last else joined.rstrip())
            emitted = True
        return "".join(gaps)

    def _markers(self) -> tuple[str, ...]:
        """Every DSML tag this detector can meet, longest-lived first.

        Read off the instance rather than cached at construction: subclasses set
        their own `bot_token` / `eot_token` after `super().__init__()` (V4 says
        `tool_calls` where V3.2 says `function_calls`).
        """
        return (
            self.bot_token,
            self.eot_token,
            self.invoke_end_token,
            "<｜DSML｜invoke",
        )

    def _strip_section_markers(self, text: str) -> str:
        """Drop whole DSML tags, along with the newlines that indent them.

        The indent belongs to the tag, not to the prose around it, which is why
        `detect_and_parse` never shows it either. Safe to swallow here only
        because `_split_flushable` refuses to release a trailing newline, so an
        indent is still in the buffer when its tag lands however the chunks fell.
        """
        for token in (self.eot_token, self.invoke_end_token, self.bot_token):
            text = re.sub(rf"\s*{re.escape(token)}", "", text)
        return text

    def _strip_leading_closers(self, text: str) -> str:
        """Drop the closers of a call that already streamed, and their indent.

        Only the leading run: a tag further in is one the turn was cut off
        inside, and `_cut_at_markup` has to see it where it stands rather than
        have it spliced out from under the prose in front of it.
        """
        closers = "|".join(
            re.escape(t) for t in (self.invoke_end_token, self.eot_token)
        )
        return re.sub(rf"^(?:\s*(?:{closers}))+", "", text)

    def _split_flushable(self, text: str) -> tuple[str, str]:
        """Split `text` into (safe to emit now, hold back for the next chunk).

        Held back: a suffix that is a strict prefix of a real marker, and a
        trailing run of newlines, which may yet turn out to be the indent of a
        tag rather than prose. Emitting an indent is not something a later chunk
        can take back, and that is what would make the output depend on where
        the deltas split. Whatever is held is released by `finish()` once the
        stream ends and the marker can no longer arrive.

        Nothing else is held: prose that merely contains `<` ("5 < 6") streams
        out immediately.
        """
        hold_from = len(text)
        start = text.rfind("<")
        if start != -1:
            tail = text[start:]
            if any(
                len(tail) < len(marker) and marker.startswith(tail)
                for marker in self._markers()
            ):
                hold_from = start
        # Whatever is held takes its indent with it, otherwise the newlines in
        # front of a half-arrived tag go out before the tag is known.
        hold_from = len(text[:hold_from].rstrip())
        return text[:hold_from], text[hold_from:]

    def finish(self, tools: list[Tool]) -> StreamingParseResult:
        """Release text held for a marker that can no longer arrive.

        `_split_flushable` withholds anything that could still turn out to be a
        DSML tag. Once the stream is over nothing more is coming, so a held
        fragment was ordinary text all along and is owed to the client -- unless
        the tag did arrive and the turn ended inside it, which is what
        `_cut_at_markup` separates.
        """
        held, self._buffer = self._buffer, ""
        bot_pos = held.find(self.bot_token)
        if bot_pos != -1 and self.current_tool_id == -1:
            # The turn stopped inside the opening section, before any invoke was
            # matched, so what is held is the preamble and nothing else. Trim it
            # exactly as `detect_and_parse` trims it.
            return StreamingParseResult(normal_text=held[:bot_pos].rstrip())

        # Closers first: the buffer can still carry those of a call that did
        # finish, and the prose after them is owed to the client.
        held = self._cut_at_markup(self._strip_leading_closers(held))
        if not self._gap_started:
            # Nothing but layout has been emitted since the last tag, so a lone
            # `<` sitting in it is the next tag cut short rather than the model's
            # own `5 < 6` -- which is why that reading is confined to here.
            held = held.partition("<")[0]
        return StreamingParseResult(normal_text=self._gap_text(held))

    def _lead_text(self, current_text: str, *, invoke_start: int) -> tuple[str, int]:
        """The normal text in front of the call, and where that call begins.

        Trailing whitespace always goes: it indents the call, which is all the
        separator between two parallel invokes ever is, and keeping it would not
        be stable anyway -- it lands in this lead or in the preceding flush
        depending only on where the deltas happened to split. What the call was
        standing on comes back on the other side, from `_join_gap`.
        """
        call_start = current_text.rfind(self.bot_token, 0, invoke_start)
        if call_start == -1:
            call_start = invoke_start
        had_text = self._emitted_anything
        lead = self._gap_text(self._strip_section_markers(current_text[:call_start]))
        trimmed = lead.rstrip()
        # This is the one caller that trims what `_gap_text` handed back, so it
        # is the one that has to say what actually reached the client: a lead of
        # nothing but the call's own indent leaves the turn still empty.
        self._emitted_anything = had_text or bool(trimmed)
        return trimmed, call_start

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for DeepSeekV32 format.
        Supports multiple consecutive invoke blocks and argument streaming.
        """
        self._buffer += new_text
        current_text = self._buffer

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
            for e_token in [self.eot_token, self.invoke_end_token]:
                if e_token in current_text:
                    current_text = current_text.replace(e_token, "")
            flushable, self._buffer = self._split_flushable(current_text)
            return StreamingParseResult(normal_text=self._gap_text(flushable))

        all_calls: list[ToolCallItem] = []
        normal_text_parts: list[str] = []
        # The lead of the call being assembled is still sitting at the head of
        # current_text; only advancing past a completed call consumes it. The
        # error path below needs to know, or it re-emits what it dumps verbatim.
        lead_is_still_in_current_text = False
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

                is_first_call = self.current_tool_id == -1

                # Initialize state if this is the first tool call
                if is_first_call:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                # Whatever precedes this call is content the model meant to show.
                # Guarded on `current_tool_name_sent` so it runs exactly once per
                # call: it is only False the first time a call is seen, so an
                # invoke that spans chunks can't re-emit the same lead.
                if not self.current_tool_name_sent:
                    lead, call_start = self._lead_text(
                        current_text, invoke_start=invoke_match.start()
                    )
                    if lead:
                        normal_text_parts.append(lead)
                        lead_is_still_in_current_text = True
                    # The lead has left the detector, so drop it from the buffer.
                    # A completed call advances past it below; an incomplete one
                    # never would, and `finish()` would emit it a second time at
                    # the end of a turn cut short inside this invoke.
                    self._buffer = current_text[call_start:]

                # Ensure arrays are large enough for current tool
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

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
                current_params = self._parse_parameters_from_xml(
                    invoke_content, allow_partial=not is_tool_end
                )

                # 3. Calculate and send incremental arguments
                sent_len = len(self.streamed_args_for_tool[self.current_tool_id])
                prev_params = self.prev_tool_call_arr[self.current_tool_id].get(
                    "arguments"
                )

                argument_diff = None

                if is_tool_end:
                    # If complete, send everything remaining
                    argument_diff = current_params[sent_len:]
                elif prev_params is not None:
                    # If partial, send stable prefix diff
                    if current_params != prev_params:
                        prefix = _find_common_prefix(current_params, prev_params)
                        if len(prefix) > sent_len:
                            argument_diff = prefix[sent_len:]

                if argument_diff:
                    all_calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=argument_diff,
                        )
                    )
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
                    lead_is_still_in_current_text = False
                    # Whatever whitespace follows the call indents its closers.
                    self._gap_started = False

                    # Move to next tool call
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False

                    # Continue loop to check for more invoke blocks
                    continue
                else:
                    # Tool call not complete yet, don't return anything
                    # Wait for more chunks until we see </｜DSML｜invoke>
                    break

            # No more invoke blocks found. Anything still buffered is trailing
            # prose: the guard at the top can never release it, because the
            # section closer keeps `potentially_dsml` true for the rest of the
            # turn. Flush it here instead, holding back only a suffix that could
            # still grow into a marker.
            if not self.has_tool_call(current_text):
                flushable, self._buffer = self._split_flushable(
                    self._strip_section_markers(current_text)
                )
                normal_text_parts.append(self._gap_text(flushable))

            return StreamingParseResult(
                normal_text="".join(normal_text_parts), calls=all_calls
            )

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            # Re-emit verbatim rather than swallowing the turn. Leads whose call
            # completed are gone from current_text and have to be kept; the lead
            # of the call that failed is still in there and would come out twice.
            # Calls are dropped on purpose: the failure can land between a tool's
            # name and its arguments, and a half-formed call is worse than none.
            self._buffer = ""
            if lead_is_still_in_current_text and normal_text_parts:
                normal_text_parts.pop()
            return StreamingParseResult(
                normal_text="".join(normal_text_parts) + current_text
            )

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'<｜DSML｜invoke name="{name}">',
            end="</｜DSML｜invoke>",
            trigger="<｜DSML｜invoke",
        )

    def get_structural_tag_name(self) -> str:
        return "deepseek_v3_2"
