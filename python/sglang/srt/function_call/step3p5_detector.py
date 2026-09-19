import re
from dataclasses import dataclass
from typing import List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import StreamingParseResult, ToolCallItem
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector


@dataclass
class _NormalizedSegment:
    text: str
    literal: bool = False


class _Step3p5XMLNormalizer:
    """Normalize high-confidence Step3p5 XML mistakes incrementally.

    Step models use the same canonical XML protocol as Qwen3-Coder, but they
    occasionally omit a delimiter or a closing tag. Repairs are deliberately
    schema-gated and only happen when a later tag provides an unambiguous
    boundary. A stream truncated at EOS is left untouched.
    """

    _FIXED_TAGS = (
        "<tool_call>",
        "</tool_call>",
        "</function>",
        "</parameter>",
    )
    _PARTIAL_PREFIXES = _FIXED_TAGS + ("<function", "<parameter")

    def __init__(self, tools: List[Tool]):
        self._buffer = ""
        self._input_length = 0
        self._function_parameters = self._collect_tool_schema(tools)

        self._inside_tool_call = False
        self._synthetic_tool_call = False
        self._function_open = False
        self._parameter_open = False
        self._current_function: Optional[str] = None
        self._suppress_current_tool = False
        self._current_tool_raw_start: Optional[int] = None
        self._redundant_synthetic_close_pending = False
        self._pending_synthetic_whitespace = ""

    @staticmethod
    def _collect_tool_schema(tools: List[Tool]) -> dict[str, set[str]]:
        schemas: dict[str, set[str]] = {}
        for tool in tools:
            try:
                if tool.type != "function" or not tool.function.name:
                    continue
                parameters = tool.function.parameters
                if isinstance(parameters, dict):
                    properties = parameters.get("properties", {})
                    parameter_names = (
                        set(properties) if isinstance(properties, dict) else set()
                    )
                else:
                    parameter_names = set()
                schemas[tool.function.name] = parameter_names
            except AttributeError:
                continue
        return schemas

    def _is_partial_tag(self, text: str) -> bool:
        return any(prefix.startswith(text) for prefix in self._PARTIAL_PREFIXES)

    def _looks_like_function_candidate(self, text: str) -> bool:
        suffix = text[len("<function") :]
        if not suffix:
            return True

        if self._inside_tool_call and not self._function_open:
            return suffix[0] in "= \t\r\n" or suffix[0].isalpha()

        if suffix[0] == "=":
            candidate_name = re.split(r"[\s<>]", suffix[1:], maxsplit=1)[0]
            return not candidate_name or any(
                name.startswith(candidate_name) for name in self._function_parameters
            )

        if suffix[0] in " \t\r\n":
            candidate_name = re.split(r"[\s<>]", suffix.lstrip(), maxsplit=1)[0]
            if not candidate_name:
                return True
            if ">" in text:
                return candidate_name in self._function_parameters
            return any(
                name.startswith(candidate_name) for name in self._function_parameters
            )

        candidate_name = re.split(r"[\s<>]", suffix, maxsplit=1)[0]
        return any(
            name.startswith(candidate_name) for name in self._function_parameters
        )

    @staticmethod
    def _looks_like_parameter_candidate(text: str) -> bool:
        suffix = text[len("<parameter") :]
        return not suffix or suffix.startswith("=")

    @staticmethod
    def _append(
        output: list[_NormalizedSegment], text: str, *, literal: bool = False
    ) -> None:
        if not text:
            return
        if output and output[-1].literal == literal:
            output[-1].text += text
        else:
            output.append(_NormalizedSegment(text=text, literal=literal))

    @staticmethod
    def _split_missing_angle_body(body: str) -> tuple[str, str]:
        """Split a missing-``>`` tag into its name and preserved suffix."""
        match = re.match(r"([^\s<]+)(.*)", body, re.DOTALL)
        if match is None:
            return "", body
        return match.group(1), match.group(2)

    def _normalize_function_tag(
        self, raw_tag: str, complete: bool
    ) -> tuple[Optional[str], Optional[str]]:
        if complete:
            canonical = re.fullmatch(r"<function=([^<>]+)>", raw_tag)
            if canonical:
                return raw_tag, canonical.group(1)

            missing_equals = re.fullmatch(
                r"<function(?:\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*>", raw_tag
            )
            if missing_equals:
                name = missing_equals.group(1)
                if name in self._function_parameters:
                    return f"<function={name}>", name
            return None, None

        if not raw_tag.startswith("<function="):
            return None, None
        name, suffix = self._split_missing_angle_body(raw_tag[len("<function=") :])
        if name in self._function_parameters:
            return f"<function={name}>{suffix}", name
        return None, None

    def _normalize_parameter_tag(
        self, raw_tag: str, complete: bool
    ) -> tuple[Optional[str], Optional[str]]:
        if complete:
            canonical = re.fullmatch(r"<parameter=([^<>]+)>", raw_tag)
            if canonical:
                return raw_tag, canonical.group(1)
            return None, None

        if not raw_tag.startswith("<parameter=") or not self._current_function:
            return None, None

        name, suffix = self._split_missing_angle_body(raw_tag[len("<parameter=") :])
        if name.startswith("parameter="):
            name = name[len("parameter=") :]
        name = name.rstrip(":,;")
        declared = self._function_parameters.get(self._current_function, set())
        if name in declared:
            return f"<parameter={name}>{suffix}", name
        return None, None

    def _take_named_tag(self, kind: str) -> Optional[tuple[str, bool, int]]:
        """Return ``(raw_tag, complete, consumed)`` or wait for more input."""
        gt_pos = self._buffer.find(">", 1)
        next_lt = self._buffer.find("<", 1)
        if gt_pos != -1 and (next_lt == -1 or gt_pos < next_lt):
            return self._buffer[: gt_pos + 1], True, gt_pos + 1
        if next_lt != -1:
            return self._buffer[:next_lt], False, next_lt

        # A missing right angle is only repairable once a later tag supplies
        # the boundary. Until then this may simply be a split canonical tag.
        expected = f"<{kind}"
        if self._buffer.startswith(expected):
            return None
        return self._buffer[0], True, 1

    def _close_parameter(self, output: list[_NormalizedSegment]) -> None:
        if self._parameter_open:
            self._append(output, "</parameter>")
            self._parameter_open = False

    def _close_function(self, output: list[_NormalizedSegment]) -> None:
        self._close_parameter(output)
        if self._function_open:
            self._append(output, "</function>")
            self._function_open = False
            self._current_function = None

    def _close_tool_call(self, output: list[_NormalizedSegment]) -> None:
        self._close_function(output)
        if self._inside_tool_call:
            self._append(output, "</tool_call>")
        self._inside_tool_call = False
        self._synthetic_tool_call = False
        self._suppress_current_tool = False
        self._current_tool_raw_start = None
        self._redundant_synthetic_close_pending = False
        self._pending_synthetic_whitespace = ""

    def _handle_tool_start(
        self, output: list[_NormalizedSegment], raw_start: int
    ) -> None:
        if self._inside_tool_call:
            self._close_tool_call(output)
        self._append(output, "<tool_call>")
        self._inside_tool_call = True
        self._current_tool_raw_start = raw_start

    def _handle_function_start(
        self,
        output: list[_NormalizedSegment],
        raw_tag: str,
        complete: bool,
        raw_start: int,
    ) -> None:
        if self._suppress_current_tool:
            return

        if self._parameter_open or self._function_open:
            # A function-looking fragment inside an existing function may be
            # literal parameter content. Step's protocol permits one function
            # per tool-call wrapper, so treating it as a second function would
            # be an unsafe repair.
            self._append(output, raw_tag)
            return

        normalized, name = self._normalize_function_tag(raw_tag, complete)
        if normalized is None or name is None:
            if self._inside_tool_call:
                # The wrapper has already been consumed by the downstream
                # parser. Suppress malformed internals until its close rather
                # than emitting argument fragments with tool_index=-1.
                self._suppress_current_tool = True
            else:
                self._append(output, raw_tag, literal=True)
            return

        if not self._inside_tool_call:
            # A missing outer wrapper is repaired only for a declared tool.
            # Canonical unknown functions inside a real wrapper retain the
            # existing Qwen parser behavior, but a bare unknown call is not
            # promoted from text into a structured action.
            if name not in self._function_parameters:
                self._append(output, raw_tag, literal=True)
                return
            self._append(output, "<tool_call>")
            self._inside_tool_call = True
            self._synthetic_tool_call = True
            self._current_tool_raw_start = raw_start

        if self._function_open:
            self._close_function(output)
        self._suppress_current_tool = False
        self._append(output, normalized)
        self._function_open = True
        self._current_function = name

    def _handle_parameter_start(
        self, output: list[_NormalizedSegment], raw_tag: str, complete: bool
    ) -> None:
        if self._suppress_current_tool:
            return
        if not self._function_open:
            self._append(output, raw_tag, literal=True)
            return
        normalized, name = self._normalize_parameter_tag(raw_tag, complete)
        if normalized is None or name is None:
            # Preserve unsupported XML-like text as parameter content instead
            # of silently deleting it or manufacturing a parameter name.
            self._append(output, raw_tag)
            return
        self._close_parameter(output)
        self._append(output, normalized)
        self._parameter_open = True

    def feed(self, text: str) -> list[_NormalizedSegment]:
        self._buffer += text
        self._input_length += len(text)
        output: list[_NormalizedSegment] = []

        while self._buffer:
            if self._redundant_synthetic_close_pending:
                if self._buffer.startswith("</tool_call>"):
                    pass
                elif "</tool_call>".startswith(self._buffer):
                    break
                elif self._buffer[0].isspace():
                    whitespace_len = len(self._buffer) - len(self._buffer.lstrip())
                    self._pending_synthetic_whitespace += self._buffer[:whitespace_len]
                    self._buffer = self._buffer[whitespace_len:]
                    continue
                else:
                    self._append(
                        output,
                        self._pending_synthetic_whitespace,
                        literal=True,
                    )
                    self._pending_synthetic_whitespace = ""
                    self._redundant_synthetic_close_pending = False

            lt_pos = self._buffer.find("<")
            if lt_pos == -1:
                self._append(
                    output,
                    self._buffer,
                    literal=not self._inside_tool_call,
                )
                self._redundant_synthetic_close_pending = False
                self._buffer = ""
                break
            if lt_pos > 0:
                self._append(
                    output,
                    self._buffer[:lt_pos],
                    literal=not self._inside_tool_call,
                )
                self._redundant_synthetic_close_pending = False
                self._buffer = self._buffer[lt_pos:]
                continue

            if self._buffer.startswith("<tool_call>"):
                if self._parameter_open or self._function_open:
                    # Tool XML can be literal shell/code-edit payload. The
                    # downstream Qwen parser already buffers the enclosing
                    # parameter until its close, so preserve it verbatim.
                    self._append(output, "<tool_call>")
                    self._buffer = self._buffer[len("<tool_call>") :]
                    continue
                raw_start = self._input_length - len(self._buffer)
                self._handle_tool_start(output, raw_start)
                self._buffer = self._buffer[len("<tool_call>") :]
                continue

            if self._buffer.startswith("</tool_call>"):
                if self._suppress_current_tool:
                    # Let Qwen3CoderDetector leave its already-open wrapper,
                    # but do not manufacture a function or argument object.
                    self._append(output, "</tool_call>")
                    self._inside_tool_call = False
                    self._synthetic_tool_call = False
                    self._suppress_current_tool = False
                    self._function_open = False
                    self._parameter_open = False
                    self._current_function = None
                    self._current_tool_raw_start = None
                elif self._inside_tool_call:
                    self._close_tool_call(output)
                elif self._redundant_synthetic_close_pending:
                    self._redundant_synthetic_close_pending = False
                    self._pending_synthetic_whitespace = ""
                else:
                    self._append(output, "</tool_call>", literal=True)
                self._buffer = self._buffer[len("</tool_call>") :]
                continue

            if self._buffer.startswith("</function>"):
                if not self._suppress_current_tool and self._function_open:
                    self._close_parameter(output)
                    self._append(output, "</function>")
                    self._function_open = False
                    self._current_function = None
                    if self._synthetic_tool_call:
                        self._append(output, "</tool_call>")
                        self._inside_tool_call = False
                        self._synthetic_tool_call = False
                        self._current_tool_raw_start = None
                        self._redundant_synthetic_close_pending = True
                        self._pending_synthetic_whitespace = ""
                elif not self._suppress_current_tool:
                    self._append(output, "</function>", literal=True)
                self._buffer = self._buffer[len("</function>") :]
                continue

            if self._buffer.startswith("</parameter>"):
                if not self._suppress_current_tool and self._parameter_open:
                    self._append(output, "</parameter>")
                    self._parameter_open = False
                elif not self._suppress_current_tool:
                    self._append(output, "</parameter>", literal=True)
                self._buffer = self._buffer[len("</parameter>") :]
                continue

            if self._buffer.startswith("<function"):
                if not self._looks_like_function_candidate(self._buffer):
                    self._append(
                        output,
                        "<",
                        literal=not self._inside_tool_call,
                    )
                    self._buffer = self._buffer[1:]
                    continue
                taken = self._take_named_tag("function")
                if taken is None:
                    break
                raw_tag, complete, consumed = taken
                raw_start = self._input_length - len(self._buffer)
                self._handle_function_start(output, raw_tag, complete, raw_start)
                self._buffer = self._buffer[consumed:]
                continue

            if self._buffer.startswith("<parameter"):
                if not self._looks_like_parameter_candidate(self._buffer):
                    self._append(
                        output,
                        "<",
                        literal=not self._inside_tool_call,
                    )
                    self._buffer = self._buffer[1:]
                    continue
                taken = self._take_named_tag("parameter")
                if taken is None:
                    break
                raw_tag, complete, consumed = taken
                self._handle_parameter_start(output, raw_tag, complete)
                self._buffer = self._buffer[consumed:]
                continue

            if self._is_partial_tag(self._buffer):
                break

            # Not Step tool XML. Preserve the byte and let the canonical Qwen
            # parser apply its established text/value behavior.
            self._append(output, "<", literal=not self._inside_tool_call)
            self._buffer = self._buffer[1:]

        return output

    @property
    def incomplete_tool_raw_start(self) -> Optional[int]:
        if self._inside_tool_call and self._buffer:
            return self._current_tool_raw_start
        return None

    def flush_unrepairable(self) -> list[_NormalizedSegment]:
        """Return an EOS-truncated suffix verbatim without claiming repair."""
        suffix = self._pending_synthetic_whitespace + self._buffer
        self._pending_synthetic_whitespace = ""
        self._redundant_synthetic_close_pending = False
        self._buffer = ""
        return [_NormalizedSegment(suffix, literal=True)] if suffix else []


class Step3p5Detector(Qwen3CoderDetector):
    """Step3.5/3.7 tool parser with schema-gated malformed XML recovery."""

    _FUNCTION_CANDIDATE = re.compile(r"<function(?:=|\s+|[A-Za-z_])", re.DOTALL)

    def __init__(self):
        super().__init__()
        self._step_normalizer: Optional[_Step3p5XMLNormalizer] = None

    def has_tool_call(self, text: str) -> bool:
        # Serving checks this before non-stream parsing. Recognize a bare
        # function candidate so the Step-only missing-wrapper repair is reachable.
        return super().has_tool_call(text) or bool(
            self._FUNCTION_CANDIDATE.search(text)
        )

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        normalizer = _Step3p5XMLNormalizer(tools)
        segments = normalizer.feed(text)

        # A partial opening tag has no reliable boundary. Do not turn it into
        # an empty/invalid tool call merely because the non-streaming request
        # reached EOS.
        incomplete_start = normalizer.incomplete_tool_raw_start
        if incomplete_start is not None:
            # Keep completed calls before the truncated suffix. Re-running the
            # prefix makes the rollback local to the unfinished wrapper rather
            # than turning the entire response back into ordinary text.
            prefix_result = (
                self.detect_and_parse(text[:incomplete_start], tools)
                if incomplete_start > 0
                else StreamingParseResult()
            )
            prefix_result.normal_text += text[incomplete_start:]
            return prefix_result

        # Non-streaming owns the complete output and can safely close a
        # schema-confirmed Step call at EOS. Streaming deliberately waits for
        # a later chunk because it has no explicit finish callback.
        trailing: list[_NormalizedSegment] = []
        if normalizer._inside_tool_call and normalizer._function_open:
            normalizer._close_tool_call(trailing)
        else:
            trailing = normalizer.flush_unrepairable()
        segments += trailing

        try:
            result = self._parse_segments(Qwen3CoderDetector(), segments, tools)
        except Exception:
            return StreamingParseResult(normal_text=text)

        calls_by_index: dict[int, ToolCallItem] = {}
        for call in result.calls:
            state = calls_by_index.setdefault(
                call.tool_index,
                ToolCallItem(tool_index=call.tool_index, parameters=""),
            )
            if call.name:
                state.name = call.name
            state.parameters += call.parameters
        result.calls = [calls_by_index[index] for index in sorted(calls_by_index)]
        return result

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        if self._step_normalizer is None:
            self._step_normalizer = _Step3p5XMLNormalizer(tools)
        segments = self._step_normalizer.feed(new_text)
        return self._parse_segments(self, segments, tools)

    @staticmethod
    def _parse_segments(
        parser: Qwen3CoderDetector,
        segments: list[_NormalizedSegment],
        tools: List[Tool],
    ) -> StreamingParseResult:
        normal_text: list[str] = []
        calls: list[ToolCallItem] = []
        for segment in segments:
            if segment.literal:
                normal_text.append(segment.text)
                continue
            result = Qwen3CoderDetector.parse_streaming_increment(
                parser, segment.text, tools
            )
            normal_text.append(result.normal_text)
            calls.extend(result.calls)
        return StreamingParseResult(normal_text="".join(normal_text), calls=calls)
