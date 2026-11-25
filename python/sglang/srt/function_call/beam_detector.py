"""Beam response-template fallback detector."""

from typing import Any

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.response_template_structural_tag import (
    supports_tool_call_structural_tag,
    tool_call_structural_tag,
)
from sglang.srt.parser.beam import BEAM_RESPONSE_TEMPLATE
from sglang.srt.parser.response_template import ResponseTemplateToolDetector


class BeamDetector(ResponseTemplateToolDetector):
    response_template = BEAM_RESPONSE_TEMPLATE
    constrains_every_turn = True
    # Unheld, Beam's arguments drift from their schemas.
    strict_arguments = True

    def supports_structural_tag(self) -> bool:
        # Forced calls then keep the native format, which this detector parses
        # instead of the JSON array used without a template-derived tag.
        tools = [tool.model_dump() for tool in self._tools or ()]
        return supports_tool_call_structural_tag(self._tool_field, tools)

    def get_structural_tag(
        self,
        tools: list[Tool] | None = None,
        tool_choice: Any = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
        message_start: bool = False,
    ):
        tool_dicts = [tool.model_dump() for tool in tools or ()]
        if tool_choice == "none":
            tool_dicts, tool_choice = [], "auto"
        named = isinstance(tool_choice, ToolChoice)
        # Named calls still fall back if another tool cannot use this format.
        if named and not supports_tool_call_structural_tag(
            self._tool_field, tool_dicts
        ):
            return None
        if named:
            tool_dicts = [
                tool
                for tool in tool_dicts
                if tool["function"]["name"] == tool_choice.function.name
            ]
        tag = tool_call_structural_tag(
            self._tool_field,
            tool_dicts,
            parallel_tool_calls=parallel_tool_calls and not named,
            thinking=self._thinking_field if thinking_mode else None,
            answer=self._content_field if tool_choice == "auto" else None,
            # The tag then covers the output from its first token.
            message_start=self._message_start if message_start else None,
            grammar_requires_reasoning=True,
            # Beam also writes arguments sorted by name.
            sorted_arguments=True,
            strict_arguments=self.strict_arguments,
        )
        if tag is None:
            return None
        from xgrammar import StructuralTag

        return StructuralTag.model_validate(tag)

    def validate_structure_constraint_request(
        self,
        tool_choice: Any,
        parallel_tool_calls: bool,
        strict_requested: bool,
        *,
        tools: list[Tool],
        thinking_mode: bool = False,
    ):
        if tool_choice == "auto" and parallel_tool_calls and not strict_requested:
            return None
        # Raises for unencodable names before the caller would log and drop
        # the error, leaving the request unconstrained.
        tag = self.get_structural_tag(
            tools=tools,
            tool_choice=tool_choice,
            thinking_mode=thinking_mode,
            parallel_tool_calls=parallel_tool_calls,
        )
        if tag is None and tool_choice == "auto":
            requested = (
                "strict tools" if strict_requested else "parallel_tool_calls=False"
            )
            raise ValueError(
                f"{type(self).__name__} cannot enforce {requested} with "
                "automatic tool choice for this response template"
            )
        return tag
