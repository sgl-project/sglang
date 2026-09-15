from typing import List, Literal, Optional, Union

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.base_format_detector import StructuralTag
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector


class DeepSeekV41Detector(DeepSeekV32Detector):
    """DeepSeek V4.1 DSML detector.

    Tag names have a leading space: " calls", " invoke", and " parameter".
    """

    tool_calls_block_name = " calls"
    invoke_tag_name = " invoke"
    parameter_tag_name = " parameter"

    # The encoder joins an assistant turn's content and its calls block with a
    # blank line, and renders it even when there is no content.
    tool_calls_prefix = "\n\n"
    think_end_token = "</think>"

    def get_structural_tag_name(self) -> Optional[str]:
        # xgrammar's builtin "deepseek_v4" tag hardcodes the unspaced names,
        # so the V4.1 tag is assembled in get_structural_tag instead.
        return None

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        """The builtin "deepseek_v4" shape with the spaced tag names.

        Bodies are JSON: xgrammar's "deepseek_xml" body style also hardcodes
        the unspaced " parameter" name, and the V3.2-lineage parser accepts a
        JSON body inside an invoke.
        """
        try:
            from xgrammar.structural_tag import (
                AnyTextFormat,
                ConstStringFormat,
                JSONSchemaFormat,
                OrFormat,
                SequenceFormat,
                StructuralTag,
                TagFormat,
                TagsWithSeparatorFormat,
                TriggeredTagsFormat,
            )
        except ImportError:
            return None

        tools = list(tools or [])
        if isinstance(tool_choice, ToolChoice):
            tools = [
                tool
                for tool in tools
                if tool.function.name == tool_choice.function.name
            ]
            if len(tools) != 1:
                return None
        if not tools:
            return None

        def invoke_tag(tool: Tool) -> TagFormat:
            function = tool.function
            schema = function.parameters if function.strict else True
            if schema is None:
                schema = True
            return TagFormat(
                begin=f'{self.invoke_start_token} name="{function.name}">',
                content=JSONSchemaFormat(json_schema=schema),
                end=f"{self.invoke_end_token}\n",
            )

        tags = [invoke_tag(tool) for tool in tools]
        if isinstance(tool_choice, ToolChoice):
            calls = tags[0]
        elif parallel_tool_calls:
            calls = TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True)
        else:
            calls = OrFormat(elements=tags)
        block_begin = f"{self.bot_token}\n"

        if tool_choice == "auto":
            body = TriggeredTagsFormat(
                triggers=[self.bot_token],
                tags=[TagFormat(begin=block_begin, content=calls, end=self.eot_token)],
                excludes=["<think>", self.think_end_token],
            )
        else:
            body = SequenceFormat(
                elements=[
                    ConstStringFormat(value=self.tool_calls_prefix + block_begin),
                    calls,
                    ConstStringFormat(value=self.eot_token),
                ]
            )
        if not thinking_mode:
            return StructuralTag(format=body)
        reasoning = TagFormat(
            begin="", content=AnyTextFormat(), end=self.think_end_token
        )
        return StructuralTag(format=SequenceFormat(elements=[reasoning, body]))
