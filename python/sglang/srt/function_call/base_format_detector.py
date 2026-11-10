import inspect
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

import orjson
from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow

try:
    from xgrammar import StructuralTag

    try:
        from xgrammar import get_model_structural_tag
    except ImportError:
        # XGrammar 0.1.x names the model-tag helper get_builtin_structural_tag,
        # uses other model keys and has no tool_choice argument. Keep SGLang's
        # stable detector API while preserving required/named tool-choice semantics.
        from xgrammar import get_builtin_structural_tag

        def get_model_structural_tag(
            *,
            model: str,
            tools: List[Dict[str, Any]],
            tool_choice: Union[Dict[str, Any], Literal["auto", "required"]],
            reasoning: bool,
        ) -> Optional[StructuralTag]:
            deepseek_v4 = model == "deepseek_v4"
            model = {
                "qwen_3_coder": "qwen_coder",
                # XGrammar 0.1.32 has the same DSML body grammar under the
                # V3.2 template. V4 only renames the outer function_calls tag.
                "deepseek_v4": "deepseek_v3_2",
            }.get(model, model)
            if model not in {
                "llama",
                "qwen",
                "qwen_coder",
                "kimi",
                "deepseek_r1",
                "harmony",
                "deepseek_v3_2",
                "minimax",
            }:
                # Match the old optional-helper behavior for detector formats
                # that the pinned builtin API does not yet implement (for
                # example DeepSeek V4), rather than raising during a request.
                return None

            require_tool = tool_choice == "required"
            if isinstance(tool_choice, dict):
                selected_name = (
                    tool_choice.get("function", {}).get("name")
                    if isinstance(tool_choice.get("function"), dict)
                    else None
                )
                if selected_name:
                    tools = [
                        tool
                        for tool in tools
                        if tool.get("function", {}).get("name") == selected_name
                    ]
                    require_tool = True

            structural_tag = get_builtin_structural_tag(
                model=model,
                tools=tools,
                reasoning=reasoning,
            )
            if not require_tool and not deepseek_v4:
                return structural_tag

            payload = structural_tag.model_dump()

            def update_payload(value: Any) -> Any:
                if isinstance(value, dict):
                    if value.get("type") == "triggered_tags":
                        if require_tool:
                            value["at_least_one"] = True
                    for key, child in value.items():
                        value[key] = update_payload(child)
                elif isinstance(value, list):
                    for index, child in enumerate(value):
                        value[index] = update_payload(child)
                elif deepseek_v4 and isinstance(value, str):
                    return value.replace(
                        "<｜DSML｜function_calls>",
                        "<｜DSML｜tool_calls>",
                    ).replace(
                        "</｜DSML｜function_calls>",
                        "</｜DSML｜tool_calls>",
                    )
                return value

            update_payload(payload)
            return StructuralTag.model_validate(payload)

except ImportError:
    StructuralTag = Any
    get_model_structural_tag = None

# XGrammar >= 0.2.7 caps these tool-call formats at one call when
# parallel_tool_calls=False.
_TOOL_DISPATCH_FORMATS = (
    "triggered_tags",
    "token_triggered_tags",
    "tags_with_separator",
)


def _stop_after_first_call(value: Any) -> None:
    if isinstance(value, dict):
        if value.get("type") in _TOOL_DISPATCH_FORMATS:
            value["stop_after_first"] = True
        for child in value.values():
            _stop_after_first_call(child)
    elif isinstance(value, list):
        for child in value:
            _stop_after_first_call(child)


def _with_parallel_tool_calls(get_tag):
    # Workaround for XGrammar < 0.2.7, which rejects parallel_tool_calls;
    # drop once requirements.txt pins >= 0.2.7. Harmony keeps the parallel tag,
    # because there the cap would also forbid ordinary messages.
    def get_tag_with_parallel_tool_calls(*, parallel_tool_calls: bool = True, **kwargs):
        structural_tag = get_tag(**kwargs)
        if (
            parallel_tool_calls
            or structural_tag is None
            or kwargs.get("model") == "harmony"
        ):
            return structural_tag
        payload = structural_tag.model_dump()
        _stop_after_first_call(payload)
        return StructuralTag.model_validate(payload)

    return get_tag_with_parallel_tool_calls


if (
    get_model_structural_tag is not None
    and "parallel_tool_calls"
    not in inspect.signature(get_model_structural_tag).parameters
):
    get_model_structural_tag = _with_parallel_tool_calls(get_model_structural_tag)

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.environ import envs
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import (
    _find_common_prefix,
    _is_complete_json,
    _partial_json_loads,
)

logger = logging.getLogger(__name__)


class BaseFormatDetector(ABC):
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    def __init__(self):
        # Streaming state management
        # Buffer for accumulating incomplete patterns that arrive across multiple streaming chunks
        self._buffer = ""
        # Stores complete tool call info (name and arguments) for each tool being parsed.
        # Used by serving layer for completion handling when streaming ends.
        # Format: [{"name": str, "arguments": dict}, ...]
        self.prev_tool_call_arr: List[Dict] = []
        # Index of currently streaming tool call. Starts at -1 (no active tool),
        # increments as each tool completes. Tracks which tool's arguments are streaming.
        self.current_tool_id: int = -1
        # Flag for whether current tool's name has been sent to client.
        # Tool names sent first with empty parameters, then arguments stream incrementally.
        self.current_tool_name_sent: bool = False
        # Tracks raw JSON string content streamed to client for each tool's arguments.
        # Critical for serving layer to calculate remaining content when streaming ends.
        # Each index corresponds to a tool_id. Example: ['{"location": "San Francisco"', '{"temp": 72']
        self.streamed_args_for_tool: List[str] = []

        # Token configuration (override in subclasses)
        self.bot_token = ""
        self.eot_token = ""
        self.tool_call_separator = ", "

    def _get_tool_indices(self, tools: List[Tool]) -> Dict[str, int]:
        """
        Get a mapping of tool names to their indices in the tools list.

        This utility method creates a dictionary mapping function names to their
        indices in the tools list, which is commonly needed for tool validation
        and ToolCallItem creation.

        Args:
            tools: List of available tools

        Returns:
            Dictionary mapping tool names to their indices
        """
        return {
            tool.function.name: i for i, tool in enumerate(tools) if tool.function.name
        }

    def parse_base_json(self, action: Any, tools: List[Tool]) -> List[ToolCallItem]:
        tool_indices = self._get_tool_indices(tools)
        if not isinstance(action, list):
            action = [action]

        results = []
        for act in action:
            name = act.get("name")
            if not (name and name in tool_indices):
                logger.warning(f"Model attempted to call undefined function: {name}")
                if not envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
                    continue  # Skip unknown tools (default legacy behavior)

            results.append(
                ToolCallItem(
                    tool_index=tool_indices.get(name, -1),
                    name=name,
                    parameters=json.dumps(
                        act.get("parameters") or act.get("arguments", {}),
                        ensure_ascii=False,
                    ),
                )
            )

        return results

    @abstractmethod
    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Parses the text in one go. Returns success=True if the format matches, otherwise False.
        Note that leftover_text here represents "content that this parser will not consume further".
        """
        action = orjson.loads(text)
        return StreamingParseResult(calls=self.parse_base_json(action, tools))

    def _ends_with_partial_token(self, buffer: str, bot_token: str) -> int:
        """
        Check if buffer ends with a partial bot_token.
        Return the length of the partial bot_token.

        For some format, the bot_token is not a token in model's vocabulary, such as
        `[TOOL_CALLS] [` in Mistral.
        """
        for i in range(1, min(len(buffer) + 1, len(bot_token))):
            if bot_token.startswith(buffer[-i:]):
                return i
        return 0

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing with tool validation.

        This base implementation works best with formats where:
        1. bot_token is followed immediately by JSON (e.g., bot_token + JSON_array)
        2. JSON can be parsed incrementally using partial_json_loads
        3. Multiple tool calls are separated by "; " or ", "

        Examples of incompatible formats (need custom implementation, may reuse some logic from this class):
        - Each tool call is wrapped in a separate block: See Qwen25Detector
        - Multiple separate blocks: [TOOL_CALLS] [...] \n [TOOL_CALLS] [...]
        - Tool call is Pythonic style

        For incompatible formats, detectors should override this method with custom logic.
        """
        # Append new text to buffer
        self._buffer += new_text
        current_text = self._buffer

        # The current_text has tool_call if it is the start of a new tool call sequence
        # or it is the start of a new tool call after a tool call separator, when there is a previous tool call
        if not (
            self.has_tool_call(current_text)
            or (
                self.current_tool_id > 0
                and current_text.startswith(self.tool_call_separator)
            )
        ):
            # Only clear buffer if we're sure no tool call is starting
            if not self._ends_with_partial_token(self._buffer, self.bot_token):
                normal_text = self._buffer
                self._buffer = ""
                if self.eot_token in normal_text:
                    normal_text = normal_text.replace(self.eot_token, "")
                return StreamingParseResult(normal_text=normal_text)
            else:
                # Might be partial bot_token, keep buffering
                return StreamingParseResult()

        # Build tool indices if not already built
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

        try:
            try:
                # Priority check: if we're processing a subsequent tool (current_tool_id > 0),
                # first check if text starts with the tool separator. This is critical for
                # parallel tool calls because the bot_token (e.g., '[') can also
                # appear inside array parameters of the current tool, and we must not
                # mistakenly identify that as the start of a new tool.
                used_separator_branch = False
                if self.current_tool_id > 0 and current_text.startswith(
                    self.tool_call_separator
                ):
                    start_idx = len(self.tool_call_separator)
                    used_separator_branch = True
                else:
                    tool_call_pos = current_text.find(self.bot_token)
                    if tool_call_pos != -1:
                        start_idx = tool_call_pos + len(self.bot_token)
                    else:
                        start_idx = 0

                if start_idx >= len(current_text):
                    return StreamingParseResult()

                try:
                    obj, end_idx = _partial_json_loads(current_text[start_idx:], flags)
                except (MalformedJSON, json.JSONDecodeError):
                    # Separator landed on non-JSON markup; fall back to
                    # bot_token which skips past all inter-object markup.
                    # e.g. Qwen25: separator "," matches between eot/bot tags.
                    if used_separator_branch and self.bot_token in current_text:
                        start_idx = current_text.find(self.bot_token) + len(
                            self.bot_token
                        )
                        if start_idx >= len(current_text):
                            return StreamingParseResult()
                        obj, end_idx = _partial_json_loads(
                            current_text[start_idx:], flags
                        )
                    else:
                        raise

                is_current_complete = _is_complete_json(
                    current_text[start_idx : start_idx + end_idx]
                )

                # Validate tool name if present
                if "name" in obj and obj["name"] not in self._tool_indices:
                    # Invalid tool name - reset state
                    self._buffer = ""
                    self.current_tool_id = -1
                    self.current_tool_name_sent = False
                    if self.streamed_args_for_tool:
                        self.streamed_args_for_tool.pop()
                    return StreamingParseResult()

                # Handle parameters/arguments consistency
                # NOTE: we assume here that the obj is always partial of a single tool call
                if "parameters" in obj:
                    assert "arguments" not in obj, (
                        "model generated both parameters and arguments"
                    )
                    obj["arguments"] = obj["parameters"]

                current_tool_call = obj

            except (MalformedJSON, json.JSONDecodeError):
                return StreamingParseResult()

            if not current_tool_call:
                return StreamingParseResult()

            # Case 1: Handle tool name streaming
            # This happens when we encounter a tool but haven't sent its name yet
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")

                if function_name and function_name in self._tool_indices:
                    # If this is a new tool (current_tool_id was -1), initialize it
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.streamed_args_for_tool.append("")
                    # If this is a subsequent tool, ensure streamed_args_for_tool is large enough
                    elif self.current_tool_id >= len(self.streamed_args_for_tool):
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")

                    # Send the tool name with empty parameters
                    res = StreamingParseResult(
                        calls=[
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=function_name,
                                parameters="",
                            )
                        ],
                    )
                    self.current_tool_name_sent = True
                else:
                    res = StreamingParseResult()

            # Case 2: Handle streaming arguments
            # This happens when we've already sent the tool name and now need to stream arguments incrementally
            else:
                cur_arguments = current_tool_call.get("arguments")
                res = StreamingParseResult()

                if cur_arguments is not None:
                    # Calculate how much of the arguments we've already streamed
                    sent = len(self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_arguments = None
                    if self.current_tool_id < len(self.prev_tool_call_arr):
                        prev_arguments = self.prev_tool_call_arr[
                            self.current_tool_id
                        ].get("arguments")

                    argument_diff = None

                    # If the current tool's JSON is complete, send all remaining arguments
                    if is_current_complete:
                        argument_diff = cur_args_json[sent:]
                        completing_tool_id = (
                            self.current_tool_id
                        )  # Save the ID of the tool that's completing

                        # Only remove the processed portion, keep unprocessed content
                        self._buffer = current_text[start_idx + end_idx :]

                    # If the tool is still being parsed, send incremental changes
                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                        if cur_args_json != prev_args_json:
                            prefix = _find_common_prefix(prev_args_json, cur_args_json)
                            argument_diff = prefix[sent:]

                    # Update prev_tool_call_arr with current state
                    if self.current_tool_id >= 0:
                        # Ensure prev_tool_call_arr is large enough
                        while len(self.prev_tool_call_arr) <= self.current_tool_id:
                            self.prev_tool_call_arr.append({})
                        self.prev_tool_call_arr[self.current_tool_id] = (
                            current_tool_call
                        )

                    # Advance to next tool if complete
                    if is_current_complete:
                        self.current_tool_name_sent = False
                        self.current_tool_id += 1

                    # Send the argument diff if there's something new
                    if argument_diff is not None:
                        # Use the correct tool_index: completing_tool_id for completed tools, current_tool_id for ongoing
                        tool_index_to_use = (
                            completing_tool_id
                            if is_current_complete
                            else self.current_tool_id
                        )
                        res = StreamingParseResult(
                            calls=[
                                ToolCallItem(
                                    tool_index=tool_index_to_use,
                                    parameters=argument_diff,
                                )
                            ],
                        )
                        self.streamed_args_for_tool[tool_index_to_use] += argument_diff

            return res

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult()

    @abstractmethod
    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains function call markers specific to this format.
        """
        raise NotImplementedError()

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Called once when the stream ends; flush any buffered state.

        Detectors that hold text back while waiting for a marker that can no
        longer arrive (the stream is over) override this to release it.
        """
        return StreamingParseResult()

    def supports_structural_tag(self) -> bool:
        """Return True if this detector supports structural tag format."""
        return True

    def parses_required_natively(self) -> bool:
        """Return True if ``tool_choice="required"`` must skip grammar
        constraints and parse the model's native output format instead."""
        return False

    @classmethod
    def parses_without_tools(cls) -> bool:
        """Return True if tool-call markup is parsed even when a request
        declares no tools, so a call to an undeclared tool is not left in content."""
        return False

    def get_required_tool_parser(self, tool_choice):
        return None

    @abstractmethod
    def structure_info(self) -> _GetInfoFunc:
        """
        Return a function that creates StructureInfo for constrained generation.

        The returned function takes a tool name and returns a StructureInfo object
        containing the begin/end patterns and trigger tokens needed for constrained
        generation of function calls in this format.

        Returns:
            A function that takes a tool name (str) and returns StructureInfo
        """
        raise NotImplementedError()

    def get_structural_tag_name(self) -> Optional[str]:
        """Return the XGrammar model name for native structural tags, if supported."""
        return None

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        """
        Return a model-native XGrammar structural tag when supported.

        Args:
            tools: List of available tools
            tool_choice: The tool choice setting from the request
            thinking_mode: Whether to include the model's reasoning prefix in
                the returned structural tag. Pass False when SGLang's
                ReasonerGrammarBackend will own the <think>...</think> prefix
                (the typical case when --reasoning-parser is configured) so
                only one layer constrains the reasoning section.
            parallel_tool_calls: Whether multiple tool calls may appear in one
                assistant response. Forwarded to XGrammar to constrain the
                number of tool calls in the generated structural tag.

        Returns:
            StructuralTag if this detector supports model-native tags, otherwise None
        """
        structural_tag_name = self.get_structural_tag_name()
        if not structural_tag_name or get_model_structural_tag is None:
            return None

        converted_tools = [tool.model_dump() for tool in tools or []]
        converted_tool_choice = (
            tool_choice.model_dump()
            if isinstance(tool_choice, ToolChoice)
            else tool_choice
        )
        return get_model_structural_tag(
            model=structural_tag_name,
            tools=converted_tools,
            tool_choice=converted_tool_choice,
            reasoning=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )

    def get_auto_tool_call_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        """Return an always-on structural tag for automatic tool choice.

        Most formats leave unconstrained text generation enabled for
        ``tool_choice="auto"`` unless strict mode is requested. Formats with a
        token that unambiguously starts a tool payload can override this hook
        to constrain only the payload after that token.
        """
        return None
