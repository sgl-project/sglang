import logging
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Type, Union

from sglang.srt.entrypoints.openai.protocol import (
    StructuralTagResponseFormat,
    StructuresResponseFormat,
    Tool,
    ToolChoice,
)
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
from sglang.srt.function_call.gpt_oss_detector import GptOssDetector
from sglang.srt.function_call.json_detector import JsonDetector
from sglang.srt.function_call.kimik2_detector import KimiK2Detector
from sglang.srt.function_call.llama32_detector import Llama32Detector
from sglang.srt.function_call.mistral_detector import MistralDetector
from sglang.srt.function_call.pythonic_detector import PythonicDetector
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.srt.function_call.step3_detector import Step3Detector

logger = logging.getLogger(__name__)


class FunctionCallParser:
    """
    Parser for function/tool calls in model outputs.

    This class handles both streaming and non-streaming parsing of function calls using a detector.
    In streaming scenarios, each time new_text is received, it calls detector.parse_streaming_increment
    and returns the resulting normal_text and calls to the upper layer (or SSE).
    """

    ToolCallParserEnum: Dict[str, Type[BaseFormatDetector]] = {
        "llama3": Llama32Detector,
        "qwen25": Qwen25Detector,
        "mistral": MistralDetector,
        "deepseekv3": DeepSeekV3Detector,
        "deepseekv31": DeepSeekV31Detector,
        "pythonic": PythonicDetector,
        "kimi_k2": KimiK2Detector,
        "qwen3_coder": Qwen3CoderDetector,
        "glm45": Glm4MoeDetector,
        "step3": Step3Detector,
        "gpt-oss": GptOssDetector,
    }

    def __init__(self, tools: List[Tool], tool_call_parser: str, tool_choice: Optional[Union[str, ToolChoice]] = None):
        # Use JSON parsing for tool_choice="required" or named tool
        if tool_choice == "required" or isinstance(tool_choice, ToolChoice) or (isinstance(tool_choice, dict) and "function" in tool_choice):
            detector = JsonDetector()
        else:
            detector_class = self.ToolCallParserEnum.get(tool_call_parser)
            if detector_class:
                detector = detector_class()
            else:
                raise ValueError(f"Unsupported tool_call_parser: {tool_call_parser}")

        self.detector = detector
        self.tools = tools

    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains a tool call in the format supported by this parser.
        This delegates to the detector's implementation.

        Args:
            text: The text to check for tool calls

        Returns:
            True if the text contains a tool call, False otherwise
        """
        return self.detector.has_tool_call(text)

    def parse_non_stream(self, full_text: str) -> Tuple[str, list[ToolCallItem]]:
        """
        One-time parsing of the full text to extract tool calls.

        Args:
            full_text: The complete text to parse

        Returns:
            A tuple containing:
            - The remaining text after parsing that was not consumed by the detector (can be treated as normal text)
            - A list of tool calls parsed from the text
        """
        parsed_result = self.detector.detect_and_parse(full_text, self.tools)
        tool_call_list = parsed_result.calls
        if tool_call_list:
            return parsed_result.normal_text, tool_call_list
        else:
            return full_text, []

    def parse_stream_chunk(self, chunk_text: str) -> Tuple[str, list[ToolCallItem]]:
        """
        Streaming incremental parsing of chunks of text as they arrive.

        Args:
            chunk_text: The new chunk of text to parse

        Returns:
            A tuple containing:
            - The normal text that should be displayed to the user
            - A list of tool calls parsed from the chunk
        """
        final_normal_text = ""
        final_calls = []

        sp_result = self.detector.parse_streaming_increment(chunk_text, self.tools)
        if sp_result.normal_text:
            final_normal_text = sp_result.normal_text
        if sp_result.calls:
            final_calls.extend(sp_result.calls)
            final_normal_text = sp_result.normal_text

        return final_normal_text, final_calls

    def get_structure_tag(self) -> StructuralTagResponseFormat:
        """
        Generate a structural tag response format for all available tools.

        This creates the necessary structural tags that guide the model's output format.
        """
        tool_structures: List[StructuresResponseFormat] = list()
        tool_trigger_set: Set[str] = set()

        get_structure_info = self.detector.structure_info()
        for tool in self.tools:
            function = tool.function
            name = function.name
            assert name is not None
            info = get_structure_info(name)

            # accept all if not strict, otherwise only accept the schema
            schema = function.parameters if function.strict else {}

            tool_structures.append(
                StructuresResponseFormat(
                    begin=info.begin,
                    schema=schema,  # type: ignore
                    end=info.end,
                )
            )
            tool_trigger_set.add(info.trigger)

        return StructuralTagResponseFormat(
            type="structural_tag",
            structures=tool_structures,
            triggers=list(tool_trigger_set),
        )

    def get_structure_constraint(
        self, tool_choice: Union[ToolChoice, Literal["auto", "required"]]
    ) -> Optional[Tuple[str, Any]]:
        """
        Returns the appropriate structure constraint for tool calls based on the tool_choice.
        The constraint is used to guide the model's output format.

        Args:
            tool_choice: The tool choice setting from the request

        Returns:
            A tuple of (constraint_type, constraint_value) to be added to sampling parameters,
            or None if no constraint applies.
        """
        # NOTE: structural_tag only supports JSON-compatible content between the begin and end.
        # It cannot parse or validate function call Pythonic or XML-ish syntax.
        if (
            self.detector.supports_structural_tag()
            and tool_choice == "auto"
            and any(tool.function.strict for tool in self.tools)
        ):
            strict_tag = self.get_structure_tag()
            return ("structural_tag", strict_tag)
        elif tool_choice == "required" or isinstance(tool_choice, ToolChoice) or (isinstance(tool_choice, dict) and "function" in tool_choice):
            json_schema = self.get_json_schema_constraint(tool_choice)
            return ("json_schema", json_schema)

    def get_ebnf(
        self, tool_choice: Union[ToolChoice, Literal["required"]]
    ) -> Optional[str]:
        """
        Get the EBNF grammar for the specified tool choice.

        Args:
            tool_choice: The tool choice specification

        Returns:
            EBNF grammar string, or None if no valid tools found

        Note:
            If a specific function is requested but not found in available tools,
            logs a warning and falls back to using all available tools for backward compatibility.
        """
        filtered_tools = []
        if isinstance(tool_choice, ToolChoice):
            fn_name = tool_choice.function.name
            filtered_tools = [t for t in self.tools if t.function.name == fn_name]

            # Check if the requested function exists in available tools
            if not filtered_tools:
                available_functions = [t.function.name for t in self.tools]
                logger.warning(
                    f"Function '{fn_name}' not found in available tools. "
                    f"Available functions: {available_functions}. "
                    f"Skipping tool choice."
                )

                # TODO: Return a 400 error instead of warning when adapter supports proper error handling
                # For now, fall back to return None
                return None
        else:
            filtered_tools = self.tools

        return self.detector.build_ebnf(filtered_tools)

    def get_json_schema_constraint(
        self, tool_choice: Union[ToolChoice, Literal["required"]]
    ) -> Optional[dict]:
        """
        Get the JSON schema constraint for the specified tool choice.

        Args:
            tool_choice: The tool choice specification

        Returns:
            JSON schema dict, or None if no valid tools found
        """
        def get_tool_schema(tool):
            return {
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [tool.function.name]
                    },
                    "parameters": tool.function.parameters
                    if tool.function.parameters else {
                        "type": "object",
                        "properties": {}
                    }
                },
                "required": ["name", "parameters"]
            }
        
        if isinstance(tool_choice, ToolChoice) or (isinstance(tool_choice, dict) and "function" in tool_choice):
            # For specific function choice, return the user's parameters schema directly
            if isinstance(tool_choice, ToolChoice):
                fn_name = tool_choice.function.name
            else:
                fn_name = tool_choice["function"]["name"]
            for tool in self.tools:
                if tool.function.name == fn_name:
                    return {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": get_tool_schema(tool)
                    }
            return None
        elif tool_choice == "required":

            def get_tool_schema_defs(tools):
                all_defs = {}
                for tool in tools:
                    if tool.function.parameters is None:
                        continue
                    # Make a copy to avoid modifying original
                    params = tool.function.parameters.copy()
                    defs = params.pop("$defs", {})
                    for def_name, def_schema in defs.items():
                        if def_name in all_defs and all_defs[def_name] != def_schema:
                            raise ValueError(
                                f"Tool definition '{def_name}' has "
                                "multiple schemas, which is not "
                                "supported.")
                        else:
                            all_defs[def_name] = def_schema
                return all_defs

            json_schema = {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "anyOf": [get_tool_schema(tool) for tool in self.tools]
                }
            }
            json_schema_defs = get_tool_schema_defs(self.tools)
            if json_schema_defs:
                json_schema["$defs"] = json_schema_defs
            return json_schema

        return None
