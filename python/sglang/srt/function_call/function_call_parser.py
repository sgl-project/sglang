import logging
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

from sglang.srt.entrypoints.openai.protocol import Tool, ToolCallConstraint, ToolChoice
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.srt.function_call.gpt_oss_detector import GptOssDetector
from sglang.srt.function_call.internlm_detector import InternlmDetector
from sglang.srt.function_call.kimik2_detector import KimiK2Detector
from sglang.srt.function_call.llama32_detector import Llama32Detector
from sglang.srt.function_call.mimo_detector import MiMoDetector
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.srt.function_call.mistral_detector import MistralDetector
from sglang.srt.function_call.pythonic_detector import PythonicDetector
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.srt.function_call.step3_detector import Step3Detector
from sglang.srt.function_call.utils import get_json_schema_constraint

logger = logging.getLogger(__name__)


class FunctionCallParser:
    """
    Parser for function/tool calls in model outputs.

    This class handles both streaming and non-streaming parsing of function calls using a detector.
    In streaming scenarios, each time new_text is received, it calls detector.parse_streaming_increment
    and returns the resulting normal_text and calls to the upper layer (or SSE).
    """

    ToolCallParserEnum: Dict[str, Type[BaseFormatDetector]] = {
        "deepseekv3": DeepSeekV3Detector,
        "deepseekv31": DeepSeekV31Detector,
        "deepseekv32": DeepSeekV32Detector,
        "glm": Glm4MoeDetector,
        "glm45": Glm4MoeDetector,
        "glm47": Glm47MoeDetector,
        "gpt-oss": GptOssDetector,
        "kimi_k2": KimiK2Detector,
        "llama3": Llama32Detector,
        "mimo": MiMoDetector,
        "mistral": MistralDetector,
        "pythonic": PythonicDetector,
        "qwen": Qwen25Detector,
        "qwen25": Qwen25Detector,
        "qwen3_coder": Qwen3CoderDetector,
        "step3": Step3Detector,
        "minimax-m2": MinimaxM2Detector,
        "interns1": InternlmDetector,
    }

    def __init__(self, tools: List[Tool], tool_call_parser: str):
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
        if not self.tools:
            return False
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
        if not self.tools:
            return full_text, []
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
        if not self.tools:
            return chunk_text, []
        final_normal_text = ""
        final_calls = []

        sp_result = self.detector.parse_streaming_increment(chunk_text, self.tools)
        if sp_result.normal_text:
            final_normal_text = sp_result.normal_text
        if sp_result.calls:
            final_calls.extend(sp_result.calls)
            final_normal_text = sp_result.normal_text

        return final_normal_text, final_calls

    def get_structure_constraint(
        self,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]],
        parallel_tool_calls: bool = True,
    ) -> Optional[ToolCallConstraint]:
        """
        Returns the appropriate structure constraint for tool calls based on the tool_choice.
        The constraint is used to guide the model's output format.

        Args:
            tool_choice: The tool choice setting from the request
            parallel_tool_calls: Whether to allow multiple tool calls (default: True)

        Returns:
            A tuple of (constraint_type, constraint_value) to be added to sampling parameters,
            or None if no constraint applies.
        """
        if tool_choice == "auto":
            if self.detector.supports_structural_tag():
                tag = self.detector.build_structural_tag(
                    tools=self.tools,
                    at_least_one=False,
                    stop_after_first=not parallel_tool_calls,
                )
                return ("structural_tag", tag)
            return None
        elif tool_choice == "required" or isinstance(tool_choice, ToolChoice):
            json_schema = get_json_schema_constraint(
                self.tools, tool_choice, parallel_tool_calls
            )
            return ("json_schema", json_schema)
        return None
