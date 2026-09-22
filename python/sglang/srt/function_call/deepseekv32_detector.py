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

    # Tag names after the DSML marker; subclasses override for newer formats.
    dsml_token = "｜DSML｜"
    tool_calls_block_name = "function_calls"
    invoke_tag_name = "invoke"
    parameter_tag_name = "parameter"
    strip_string_param_value: bool = True

    def __init__(self):
        super().__init__()
        block = f"{self.dsml_token}{self.tool_calls_block_name}"
        invoke = f"{self.dsml_token}{self.invoke_tag_name}"
        parameter = f"{self.dsml_token}{self.parameter_tag_name}"
        self.bot_token = f"<{block}>"
        self.eot_token = f"</{block}>"
        self.invoke_start_token = f"<{invoke}"
        self.invoke_end_token = f"</{invoke}>"
        self.parameter_regex = (
            rf'<{parameter}\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</{parameter}>'
        )
        self.partial_parameter_regex = (
            rf'<{parameter}\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*)$'
        )
        self.function_calls_regex = rf"<{block}>(.*?)</{block}>"
        # Long-form `<｜DSML｜invoke name="x">...</｜DSML｜invoke>` and the
        # self-closing `<｜DSML｜invoke name="x"/>` shape V4 emits for zero-arg
        # tools. The `end` group is empty when the closer hasn't streamed in.
        self.invoke_regex = (
            rf'<{invoke}\s+name="(?P<name>[^"]+)"\s*'
            r"(?:(?P<self_close>/>)"
            rf"|>(?P<body>.*?)(?P<end>(?:</{invoke}>|$)))"
        )
        # Consumed right-to-left by rstrip (a character set, not a suffix), so the
        # invoke name is split to limit how much of a partial value gets eaten.
        self.prefix_parameter_end_call = [
            "</",
            self.dsml_token,
            self.parameter_tag_name,
        ]
        self.prefix_invoke_end_call = [
            "</",
            self.dsml_token,
            self.invoke_tag_name[:-3],
            self.invoke_tag_name[-3:],
        ]
        self.current_tool_id = -1
        # Set once the pre-call prose has been streamed; see finish().
        self._preamble_emitted = False
        # Any DSML tag, so leftovers never reach user-visible content.
        # Escaped because subclasses may override the marker. The marker is a
        # single special token, so any occurrence is model markup rather than
        # prose: the leading `<` and the trailing `>` are both optional, since
        # a model can emit a truncated or mangled tag such as
        # `<｜DSML｜tool_calls|` (seen from a live server).
        #
        # The tag body is matched tag-shaped (a name plus optional
        # `attr="value"` pairs) rather than "anything up to the next `>`", so a
        # mangled marker mid-sentence can only ever consume the tag itself and
        # never the surrounding prose (e.g. `5 > 3` after it must survive).
        self.residual_markup_regex = (
            rf"<?/?{re.escape(self.dsml_token)}"
            rf'(?:\w*(?:\s+\w+="[^"\n]*")*\s*/?>|\w*\|?)?'
        )

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek v32 format tool call.

        A bare marker counts: it is a single special token, so its presence
        means the model emitted protocol markup even when the surrounding tag
        is mangled. Serving skips this detector entirely when this returns
        False, so the marker has to be recognised here for the markup to be
        removed from ``message.content``.
        """
        return self.dsml_token in text

    def _strip_residual_markup(self, text: str) -> str:
        """Remove leftover DSML tags so they never reach ``message.content``.

        Some generations contain markup this detector cannot turn into a
        structured call, such as a malformed invoke body or an unknown tool
        name. Echoing it back verbatim makes an OpenAI-compatible client read
        the turn as ordinary prose and silently drop the requested work, so
        strip it instead.
        """
        if self.dsml_token not in text:
            return text
        return re.sub(self.residual_markup_regex, "", text)

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
                parameters[param_name] = (
                    param_value.strip()
                    if self.strip_string_param_value
                    else param_value
                )
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
                    parameters[param_name] = (
                        param_value.strip()
                        if self.strip_string_param_value
                        else param_value
                    )
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
        if not self.has_tool_call(text):
            return StreamingParseResult(normal_text=text, calls=[])

        # The section wrapper is not always present: generations sometimes
        # emit a bare `<｜DSML｜invoke …>` block, or open the section and never
        # close it. Anchor on whichever marker comes first so those calls are
        # still recovered instead of returned as text.
        anchors = [
            pos
            for pos in (
                text.find(self.bot_token),
                text.find(self.invoke_start_token),
            )
            if pos != -1
        ]
        if not anchors:
            # A marker with no well-formed opener: nothing is parseable, but
            # the markup still must not reach the client as content.
            return StreamingParseResult(
                normal_text=self._strip_residual_markup(text), calls=[]
            )

        idx = min(anchors)
        normal_text = text[:idx].removesuffix("\n\n")

        calls = []
        try:
            sections = re.findall(self.function_calls_regex, text, re.DOTALL)
            if not sections:
                # No complete section: scan invoke blocks from the first
                # marker so an unterminated section still yields its calls.
                sections = [text[idx:]]

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
                        parameters = json.loads(func_args)
                    except (json.JSONDecodeError, ValueError) as e:
                        # One malformed invoke must not discard its siblings.
                        logger.warning(
                            f"Skipping unparsable DSML invoke for '{func_name}': {e}"
                        )
                        continue
                    # construct match_result for parse_base_json
                    match_result = {
                        "name": func_name,
                        "parameters": parameters,
                    }
                    calls.extend(self.parse_base_json(match_result, tools))

            return StreamingParseResult(
                normal_text=self._strip_residual_markup(normal_text), calls=calls
            )
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # Parsing failed: return the text without DSML markup rather than
            # leaking raw tags to the client.
            return StreamingParseResult(normal_text=self._strip_residual_markup(text))

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
            self._buffer = ""
            for e_token in [self.eot_token, self.invoke_end_token]:
                if e_token in current_text:
                    current_text = current_text.replace(e_token, "")
            return StreamingParseResult(normal_text=current_text)

        all_calls: list[ToolCallItem] = []
        # Only recovered for the first call: the DSML guard above never releases a
        # buffer that still holds a marker, so later prose stays buffered.
        preamble = ""
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

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]
                    call_start = invoke_match.start()
                    bot_pos = current_text.rfind(self.bot_token, 0, call_start)
                    if bot_pos != -1:
                        call_start = bot_pos
                    # Same trailing-newline trim as detect_and_parse, so both agree.
                    preamble = current_text[:call_start].removesuffix("\n\n")
                    # The buffer can still hold a partial copy of this text
                    # (a chunk boundary inside the preamble), so remember that
                    # it has already gone out and finish() must not resend it.
                    self._preamble_emitted = True

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
                    # Past a completed call the buffer no longer holds any of
                    # the emitted preamble, so whatever remains at end of
                    # stream is new text that finish() may release.
                    self._preamble_emitted = False
                    current_text = self._buffer  # Update for next iteration

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
            return StreamingParseResult(
                normal_text=self._strip_residual_markup(preamble), calls=all_calls
            )

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            # Re-emit verbatim rather than swallowing the turn; the preamble is
            # still inside current_text unless a completed call advanced past it.
            # Calls are dropped on purpose: the failure can land between a tool's
            # name and its arguments, and a half-formed call is worse than none.
            self._buffer = ""
            # The buffer is gone, so nothing of the emitted preamble can still
            # be in it; a later finish() must be free to release new prose.
            self._preamble_emitted = False
            if not current_text.startswith(preamble):
                current_text = preamble + current_text
            return StreamingParseResult(normal_text=current_text)

    def finish(self, tools: list[Tool]) -> StreamingParseResult:
        """Flush buffered state once the stream has ended.

        The DSML guard in ``parse_streaming_increment`` holds back any buffer
        containing the marker, waiting for a closer that can no longer arrive.
        Without this override that text is silently discarded, so a turn whose
        prose follows the tool calls comes back empty.

        Two rules keep this from making things worse:

        * Text is released only from *before* the first well-formed opener.
          Anything from the opener onwards was either already emitted as a
          call or is a half-written protocol block whose payload must not be
          shown as prose.
        * A completing call emits only the remainder of its arguments, with no
          name, so the serving layer extends the in-flight tool call instead of
          opening a second one at the same index.
        """
        buffered = self._buffer
        self._buffer = ""
        if not buffered:
            return StreamingParseResult()

        calls: list[ToolCallItem] = []
        try:
            # Complete an in-flight call whose closing tag never arrived, but
            # only when the re-parse actually extends what was streamed; a
            # generation cut mid-arguments re-parses to "{}", which would
            # leave the client holding unclosed JSON.
            invoke_match = re.search(self.invoke_regex, buffered, re.DOTALL)
            if invoke_match is not None and 0 <= self.current_tool_id < len(
                self.streamed_args_for_tool
            ):
                _, invoke_content, _ = self._unpack_invoke_match(invoke_match)
                final_args = self._parse_parameters_from_xml(invoke_content)
                sent = self.streamed_args_for_tool[self.current_tool_id]
                # A body that was cut mid-arguments re-parses to "{}" (no
                # complete parameter tag, and direct JSON is only accepted
                # once it closes). Completing the call with that would
                # dispatch a zero-argument call the model never asked for, so
                # discriminate on the body rather than on what was streamed:
                # an empty body legitimately means no arguments.
                body = invoke_content.strip()
                lost_arguments = final_args == "{}" and body not in ("", "{}")
                if lost_arguments:
                    logger.warning(
                        "DSML stream ended mid-arguments for tool_index %d; "
                        "leaving the streamed prefix %r unterminated rather "
                        "than completing it as an empty call",
                        self.current_tool_id,
                        sent,
                    )
                elif final_args.startswith(sent) and len(final_args) > len(sent):
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=final_args[len(sent) :],
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] = final_args
                elif not final_args.startswith(sent):
                    logger.warning(
                        "DSML re-parse %r does not extend the streamed prefix "
                        "%r; leaving it unterminated",
                        final_args,
                        sent,
                    )

            # Release only what precedes the first well-formed opener.
            cut = len(buffered)
            for token in (self.bot_token, self.invoke_start_token):
                pos = buffered.find(token)
                if pos != -1:
                    cut = min(cut, pos)
            leading = buffered[:cut]
            if self._preamble_emitted:
                # A chunk boundary can leave a partial copy of the already
                # streamed preamble in the buffer; resending it would duplicate
                # text in the client's content.
                leading = ""
            normal_text = self._strip_residual_markup(leading)
            # A generation cut inside the marker leaves a partial prefix that
            # can never complete now the stream is over; it is protocol
            # residue, not prose.
            for length in range(len(self.dsml_token) - 1, 0, -1):
                partial = self.dsml_token[:length]
                for opener in (f"<{partial}", f"</{partial}", partial):
                    if normal_text.endswith(opener):
                        normal_text = normal_text[: -len(opener)]
                        break
                else:
                    continue
                break
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in finish: {e}")
            # Emitting the raw buffer here would leak markup; drop it instead.
            return StreamingParseResult()

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'{self.invoke_start_token} name="{name}">',
            end=self.invoke_end_token,
            trigger=self.invoke_start_token,
        )

    def get_structural_tag_name(self) -> str:
        return "deepseek_v3_2"
