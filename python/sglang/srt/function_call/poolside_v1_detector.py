import ast
import json
import logging
import re
from typing import Any, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


class PoolsideV1Detector(BaseFormatDetector):
    """
    Detector for poolside Laguna-XS.2 (poolside_v1 series) tool-call wire format.

    Wire format:
        <tool_call>{name}\\n
        <arg_key>{key}</arg_key>\\n
        <arg_value>{val}</arg_value>\\n
        ...
        </tool_call>

    String values are emitted as raw text; non-strings are JSON-encoded by
    the chat template. The parser does schema-based type coercion to round-trip
    them: schema type `string` keeps the raw value; other types attempt
    `json.loads` and fall back to `ast.literal_eval`, then to the raw string.
    """

    def __init__(self):
        super().__init__()

        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.arg_key_start = "<arg_key>"
        self.arg_key_end = "</arg_key>"
        self.arg_value_start = "<arg_value>"
        self.arg_value_end = "</arg_value>"

        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.arg_pair_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )

        # Streaming FSM state
        self.parsed_pos: int = 0
        self.is_inside_tool_call: bool = False
        self.json_started: bool = False
        self.current_func_name: Optional[str] = None
        self.current_pending_key: Optional[str] = None
        self.skip_until_call_end: bool = (
            False  # set when current call's name is unknown
        )

    # ---------- Helpers ----------

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    @staticmethod
    def _get_param_schema(func_name: str, tools: Optional[List[Tool]]) -> dict:
        if not tools:
            return {}
        for tool in tools:
            try:
                if (
                    tool.type == "function"
                    and tool.function.name == func_name
                    and isinstance(tool.function.parameters, dict)
                ):
                    return tool.function.parameters.get("properties", {})
            except AttributeError:
                continue
        return {}

    _STRING_TYPES = frozenset({"string", "str", "text", "enum"})

    @staticmethod
    def _convert_param_value(raw: str, schema: dict, key: str) -> Any:
        """Coerce a raw arg_value string per schema; fall back to raw on failure.

        Decoder selection by schema type:
          - string-like types         → identity (raw text)
          - no schema entry           → json.loads only (conservative; don't
                                        ast-eval untyped values)
          - everything else (int,
            number, bool, object, …)  → json.loads, then ast.literal_eval
        """
        spec = schema.get(key) if isinstance(schema, dict) else None
        param_type = str(spec.get("type", "")).lower() if isinstance(spec, dict) else ""
        if param_type in PoolsideV1Detector._STRING_TYPES:
            return raw

        decoders = (json.loads,) if not param_type else (json.loads, ast.literal_eval)
        for decoder in decoders:
            try:
                return decoder(raw)
            except (ValueError, SyntaxError):
                continue
        return raw

    @staticmethod
    def _strip_edge_newlines(value: str) -> str:
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        return value

    def _find_name_boundary(self, text: str) -> int:
        """
        Earliest of `\\n`, `<arg_key>`, `</tool_call>`. -1 if none — caller buffers.
        """
        candidates = []
        nl = text.find("\n")
        if nl != -1:
            candidates.append(nl)
        ak = text.find(self.arg_key_start)
        if ak != -1:
            candidates.append(ak)
        te = text.find(self.tool_call_end_token)
        if te != -1:
            candidates.append(te)
        return min(candidates) if candidates else -1

    # ---------- Non-streaming ----------

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if self.tool_call_start_token not in text:
            return StreamingParseResult(normal_text=text)

        tool_indices = self._get_tool_indices(tools)
        first_idx = text.find(self.tool_call_start_token)
        normal_text = text[:first_idx] if first_idx > 0 else ""

        calls: List[ToolCallItem] = []
        for body in self.tool_call_regex.findall(text):
            # _find_name_boundary searches for `\n` / `<arg_key>` / `</tool_call>`,
            # but the regex already stripped `</tool_call>`, so a no-arg call
            # without a trailing newline (`<tool_call>now</tool_call>`) gives
            # boundary == -1. Treat that case as "name == entire body".
            boundary = self._find_name_boundary(body)
            name = (body if boundary == -1 else body[:boundary]).strip()
            if not name or name not in tool_indices:
                continue

            schema = self._get_param_schema(name, tools)
            args: dict = {}
            for raw_key, raw_val in self.arg_pair_regex.findall(body):
                key = raw_key.strip()
                val = self._strip_edge_newlines(raw_val)
                args[key] = self._convert_param_value(val, schema, key)

            calls.append(
                ToolCallItem(
                    tool_index=tool_indices[name],
                    name=name,
                    parameters=json.dumps(args, ensure_ascii=False),
                )
            )

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    # ---------- Streaming ----------

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        if not self._buffer:
            return StreamingParseResult()

        tool_indices = self._get_tool_indices(tools)
        calls: List[ToolCallItem] = []
        normal_text_chunks: List[str] = []

        partial_tag_prefixes = (
            self.tool_call_start_token,
            self.tool_call_end_token,
            self.arg_key_start,
            self.arg_key_end,
            self.arg_value_start,
            self.arg_value_end,
        )

        try:
            while True:
                slice_ = self._buffer[self.parsed_pos :]
                if not slice_:
                    break

                # 1. <tool_call>
                if slice_.startswith(self.tool_call_start_token):
                    self.parsed_pos += len(self.tool_call_start_token)
                    self.is_inside_tool_call = True
                    self.skip_until_call_end = False
                    self.current_tool_name_sent = False
                    self.json_started = False
                    self.current_func_name = None
                    self.current_pending_key = None
                    continue

                # 2. Tool name (inside tool_call, name not yet emitted)
                if self.is_inside_tool_call and not self.current_tool_name_sent:
                    boundary = self._find_name_boundary(slice_)
                    if boundary == -1:
                        break
                    name = slice_[:boundary].strip()
                    # Mark name as "sent" UNCONDITIONALLY — this is the actual
                    # liveness guarantee for branch 2. A malformed
                    # `<tool_call><arg_key>...` gives boundary=0 at `<arg_key>`
                    # with no consumable newline, so consume=0 below; without
                    # the flag flip the next iteration re-enters branch 2 with
                    # an unchanged slice and spins. Setting the flag means
                    # branch 2 is one-shot per `<tool_call>` regardless of
                    # whether the name resolved, and subsequent iterations
                    # fall through to branches 4/5 (drain) or 3 (close).
                    self.current_tool_name_sent = True
                    if name in tool_indices:
                        self.current_tool_id += 1
                        self.current_func_name = name
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")
                        # Streaming uses the per-response sequential index
                        # (current_tool_id), NOT the tools-list slot index —
                        # OpenAI clients group chunks by `index`, so all chunks
                        # for one call must share the same value as later
                        # parameter emissions in branches 3/5.
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=name,
                                parameters="",
                            )
                        )
                    else:
                        # Unknown tool name (or empty) — drain until </tool_call>.
                        self.skip_until_call_end = True
                    # Consume only the name (and a single delimiting newline if present);
                    # leave <arg_key>/</tool_call> for the next iteration.
                    consume = boundary
                    if (
                        boundary < len(slice_)
                        and slice_[boundary : boundary + 1] == "\n"
                    ):
                        consume += 1
                    self.parsed_pos += consume
                    continue

                # 3. </tool_call>
                if slice_.startswith(self.tool_call_end_token):
                    if not self.skip_until_call_end and self.current_tool_name_sent:
                        if not self.json_started:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id, parameters="{}"
                                )
                            )
                            self.streamed_args_for_tool[self.current_tool_id] += "{}"
                        else:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id, parameters="}"
                                )
                            )
                            self.streamed_args_for_tool[self.current_tool_id] += "}"
                    self.parsed_pos += len(self.tool_call_end_token)
                    self.is_inside_tool_call = False
                    self.skip_until_call_end = False
                    self.current_tool_name_sent = False
                    self.json_started = False
                    self.current_func_name = None
                    self.current_pending_key = None
                    continue

                # 4. <arg_key>...</arg_key>
                if slice_.startswith(self.arg_key_start):
                    end = slice_.find(self.arg_key_end)
                    if end == -1:
                        break  # incomplete tag; wait
                    # Only capture when we're actively assembling a known
                    # call's params: inside a tool_call AND the name was
                    # already resolved (current_tool_id is valid). Without
                    # `is_inside_tool_call`, a stray `<arg_key>` before any
                    # `<tool_call>` would set pending_key, then branch 5
                    # would crash writing to streamed_args_for_tool[-1] on
                    # an empty list (and that crash would be masked by the
                    # outer except, hiding the FSM-state bug).
                    if (
                        not self.skip_until_call_end
                        and self.is_inside_tool_call
                        and self.current_tool_name_sent
                    ):
                        self.current_pending_key = slice_[
                            len(self.arg_key_start) : end
                        ].strip()
                    self.parsed_pos += end + len(self.arg_key_end)
                    continue

                # 5. <arg_value>...</arg_value>
                if slice_.startswith(self.arg_value_start):
                    end = slice_.find(self.arg_value_end)
                    if end == -1:
                        break  # incomplete tag; wait (no partial value emission)
                    raw = slice_[len(self.arg_value_start) : end]
                    raw = self._strip_edge_newlines(raw)
                    if (
                        not self.skip_until_call_end
                        and self.current_pending_key is not None
                    ):
                        schema = self._get_param_schema(self.current_func_name, tools)
                        converted = self._convert_param_value(
                            raw, schema, self.current_pending_key
                        )
                        kv = (
                            f"{json.dumps(self.current_pending_key)}: "
                            f"{json.dumps(converted, ensure_ascii=False)}"
                        )
                        if not self.json_started:
                            fragment = "{" + kv
                            self.json_started = True
                        else:
                            fragment = ", " + kv
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id, parameters=fragment
                            )
                        )
                        self.streamed_args_for_tool[self.current_tool_id] += fragment
                        self.current_pending_key = None
                    self.parsed_pos += end + len(self.arg_value_end)
                    continue

                # 6. Plain '<' that might be a partial known tag → buffer.
                if slice_.startswith("<"):
                    if any(
                        tag.startswith(slice_) and tag != slice_
                        for tag in partial_tag_prefixes
                    ):
                        break
                    # A bare '<' that cannot be a known tag.
                    if not self.is_inside_tool_call:
                        normal_text_chunks.append("<")
                    self.parsed_pos += 1
                    continue

                # 7. Free-form text up to the next `<`. Outside a tool call
                #    → emit; inside → discard (newlines between tags).
                #    Trailing-prefix holdback isn't needed here: branch 6
                #    handles any `<…` boundary, and tag prefixes all start
                #    with `<`, so a slice with no `<` cannot end on one.
                next_lt = slice_.find("<")
                segment = slice_ if next_lt == -1 else slice_[:next_lt]
                if not self.is_inside_tool_call:
                    normal_text_chunks.append(segment)
                self.parsed_pos += len(segment)

        except Exception as e:
            logger.error(f"Error in PoolsideV1Detector.parse_streaming_increment: {e}")
            return StreamingParseResult()

        if self.parsed_pos > 0:
            self._buffer = self._buffer[self.parsed_pos :]
            self.parsed_pos = 0

        return StreamingParseResult(
            calls=calls,
            normal_text="".join(normal_text_chunks),
        )

    # ---------- Constrained generation ----------

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f"<tool_call>{name}\n",
            end="</tool_call>",
            trigger="<tool_call>",
        )
