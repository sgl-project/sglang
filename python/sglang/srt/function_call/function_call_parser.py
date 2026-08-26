import inspect
import json
import logging
import re
from typing import Dict, List, Literal, Optional, Set, Tuple, Type, Union

from jsonschema import Draft202012Validator

from sglang.srt.entrypoints.openai.protocol import (
    LegacyStructuralTagResponseFormat,
    StructuralTagResponseFormat,
    StructuresResponseFormat,
    Tool,
    ToolCallConstraint,
    ToolChoice,
)
from sglang.srt.environ import ToolStrictLevel, envs
from sglang.srt.function_call.apertus2509_detector import Apertus2509Detector
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.cohere_command4_detector import CohereCommand4Detector
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.dots_detector import DotsToolDetector
from sglang.srt.function_call.gemma4_detector import Gemma4Detector
from sglang.srt.function_call.gigachat3_detector import GigaChat3Detector
from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.srt.function_call.gpt_oss_detector import GptOssDetector
from sglang.srt.function_call.hermes_detector import HermesDetector
from sglang.srt.function_call.hunyuan_detector import HunyuanDetector
from sglang.srt.function_call.inkling_detector import InklingDetector
from sglang.srt.function_call.internlm_detector import InternlmDetector
from sglang.srt.function_call.kimik2_detector import KimiK2Detector
from sglang.srt.function_call.kimik3_detector import KimiK3Detector
from sglang.srt.function_call.lfm2_detector import Lfm2Detector
from sglang.srt.function_call.llama32_detector import Llama32Detector
from sglang.srt.function_call.mimo_detector import MiMoDetector
from sglang.srt.function_call.minicpm5_detector import MiniCPM5Detector
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.srt.function_call.minimax_m3 import MinimaxM3Detector
from sglang.srt.function_call.mistral_detector import MistralDetector
from sglang.srt.function_call.muse_glimmer_detector import MuseGlimmerDetector
from sglang.srt.function_call.poolside_v1_detector import PoolsideV1Detector
from sglang.srt.function_call.pythonic_detector import PythonicDetector
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.srt.function_call.spark25_detector import Spark25Detector
from sglang.srt.function_call.step3_detector import Step3Detector
from sglang.srt.function_call.trinity_detector import TrinityDetector
from sglang.srt.function_call.utils import (
    _get_tool_schema_defs,
    get_json_schema_constraint,
    get_json_schema_properties,
    infer_type_from_json_schema,
    resolve_local_json_schema_refs,
)

logger = logging.getLogger(__name__)


class FunctionCallParser:
    """
    Parser for function/tool calls in model outputs.

    This class handles both streaming and non-streaming parsing of function calls using a detector.
    In streaming scenarios, each time new_text is received, it calls detector.parse_streaming_increment
    and returns the resulting normal_text and calls to the upper layer (or SSE).
    """

    ToolCallParserEnum: Dict[str, Type[BaseFormatDetector]] = {
        "apertus2509": Apertus2509Detector,
        "cohere_command4": CohereCommand4Detector,
        "deepseekv3": DeepSeekV3Detector,
        "deepseekv31": DeepSeekV31Detector,
        "deepseekv32": DeepSeekV32Detector,
        "deepseekv4": DeepSeekV4Detector,
        "dots": DotsToolDetector,
        "glm": Glm4MoeDetector,
        "glm45": Glm4MoeDetector,
        "glm47": Glm47MoeDetector,
        "gpt-oss": GptOssDetector,
        "kimi_k2": KimiK2Detector,
        "kimi_k3": KimiK3Detector,
        "lfm2": Lfm2Detector,
        "llama3": Llama32Detector,
        "mimo": MiMoDetector,
        "minicpm5": MiniCPM5Detector,
        "mistral": MistralDetector,
        "muse": MuseGlimmerDetector,
        "poolside_v1": PoolsideV1Detector,
        "pythonic": PythonicDetector,
        "qwen": Qwen25Detector,
        "qwen25": Qwen25Detector,
        "qwen3_coder": Qwen3CoderDetector,
        "spark25": Spark25Detector,
        "step3": Step3Detector,
        "step3p5": Qwen3CoderDetector,
        "minimax-m2": MinimaxM2Detector,
        "minimax-m3": MinimaxM3Detector,
        "trinity": TrinityDetector,
        "interns1": InternlmDetector,
        "hermes": HermesDetector,
        "hunyuan": HunyuanDetector,
        "gigachat3": GigaChat3Detector,
        "gemma4": Gemma4Detector,
        "inkling": InklingDetector,
    }

    def __init__(self, tools: List[Tool], tool_call_parser: str, tokenizer=None):
        detector_class = self.ToolCallParserEnum.get(tool_call_parser)
        if detector_class:
            kwargs = {}
            if tokenizer is not None:
                sig = inspect.signature(detector_class)
                if "tokenizer" in sig.parameters:
                    kwargs["tokenizer"] = tokenizer
            detector = detector_class(**kwargs)
        else:
            raise ValueError(f"Unsupported tool_call_parser: {tool_call_parser}")

        self.detector = detector
        self.tools = tools
        self.schema_branches = {}
        self.streaming_calls: Dict[int, Tuple[Optional[str], str]] = {}
        self.detector_tools: List[Tool] = []
        for tool in tools:
            parameters = tool.function.parameters
            if isinstance(parameters, dict):
                properties = get_json_schema_properties(parameters)
                if properties != parameters.get("properties", {}):
                    function = tool.function.model_copy(
                        update={"parameters": parameters | {"properties": properties}}
                    )
                    tool = tool.model_copy(update={"function": function})
                schema = resolve_local_json_schema_refs(parameters, parameters)
                conjuncts = [schema]
                index = 0
                while index < len(conjuncts):
                    conjunct = conjuncts[index]
                    nested = (
                        conjunct.get("allOf") if isinstance(conjunct, dict) else None
                    )
                    if isinstance(nested, list):
                        remainder = {
                            key: value
                            for key, value in conjunct.items()
                            if key != "allOf"
                        }
                        conjuncts[index : index + 1] = [remainder, *nested]
                    else:
                        index += 1

                alternative = None
                for index, conjunct in enumerate(conjuncts):
                    if not isinstance(conjunct, dict):
                        continue
                    for keyword in ("anyOf", "oneOf"):
                        branches = conjunct.get(keyword)
                        if isinstance(branches, list):
                            alternative = index, keyword, branches
                            break
                    if alternative:
                        break

                if alternative:
                    index, keyword, branches = alternative
                    if properties == parameters.get("properties", {}):
                        function = tool.function.model_copy(
                            update={"parameters": parameters.copy()}
                        )
                        tool = tool.model_copy(update={"function": function})
                    conjuncts[index] = {
                        key: value
                        for key, value in conjuncts[index].items()
                        if key != keyword
                    }
                    base_schema = {
                        key: parameters[key]
                        for key in ("$defs", "definitions")
                        if key in parameters
                    } | {"allOf": conjuncts}
                    candidates = []
                    for branch in branches:
                        candidate_schema = base_schema | {
                            "allOf": [*base_schema["allOf"], branch]
                        }
                        candidate_properties = get_json_schema_properties(
                            candidate_schema
                        )
                        candidates.append(
                            (
                                candidate_properties,
                                {
                                    key: infer_type_from_json_schema(value)
                                    for key, value in candidate_properties.items()
                                },
                                Draft202012Validator(candidate_schema),
                            )
                        )
                    self.schema_branches[tool.function.name] = {
                        "parameters": tool.function.parameters,
                        "properties": properties,
                        "candidates": candidates,
                    }
            self.detector_tools.append(tool)
        self.tool_strict_level = envs.SGLANG_TOOL_STRICT_LEVEL.get()

    def _update_tool_schemas(
        self, calls: List[ToolCallItem], streaming: bool = False
    ) -> None:
        """Select a root schema branch without decoding valid string arguments."""
        for call in calls:
            pending_name, pending_parameters = self.streaming_calls.get(
                call.tool_index, (None, "")
            )
            name = call.name or pending_name
            branch_config = self.schema_branches.get(name)
            if branch_config is None:
                continue

            tool_parameters = branch_config["parameters"]
            if streaming and call.name and not pending_name:
                tool_parameters["properties"] = branch_config["properties"]

            parameters = pending_parameters + call.parameters
            complete = False
            try:
                arguments = json.loads(parameters)
                complete = True
            except json.JSONDecodeError:
                try:
                    arguments = json.loads(parameters + "}")
                except json.JSONDecodeError:
                    if streaming:
                        self.streaming_calls[call.tool_index] = (name, parameters)
                    continue

            if not isinstance(arguments, dict):
                continue
            if streaming:
                self.streaming_calls[call.tool_index] = (name, parameters)

            candidates = []
            for candidate_properties, property_types, validator in branch_config[
                "candidates"
            ]:
                # Candidate copies can be coerced for validation while the original
                # string remains byte-for-byte available for a string branch.
                candidate_arguments = arguments.copy()
                for key, value in arguments.items():
                    expected_type = property_types.get(key)
                    if isinstance(value, str) and expected_type not in (None, "string"):
                        try:
                            candidate_arguments[key] = json.loads(value)
                        except json.JSONDecodeError:
                            pass
                errors = validator.iter_errors(candidate_arguments)
                if all(error.validator == "required" for error in errors):
                    candidates.append((candidate_properties, candidate_arguments))

            if len(candidates) == 1:
                candidate_properties, selected_arguments = candidates[0]
                selected_properties = candidate_properties | {
                    key: value
                    for key, value in branch_config["properties"].items()
                    if candidate_properties.get(key) in (None, {}, True)
                }
                if streaming and not complete:
                    tool_parameters["properties"] = selected_properties
                if (
                    complete
                    and (not streaming or not pending_parameters)
                    and selected_arguments != arguments
                ):
                    call.parameters = json.dumps(selected_arguments, ensure_ascii=False)

            if streaming and complete:
                self.streaming_calls.pop(call.tool_index, None)
                tool_parameters["properties"] = branch_config["properties"]

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
        has_tool_call = self.detector.has_tool_call(full_text)
        parsed_result = self.detector.detect_and_parse(full_text, self.detector_tools)
        self._update_tool_schemas(parsed_result.calls)
        tool_call_list = parsed_result.calls
        if tool_call_list or has_tool_call:
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

        chunks = (
            re.findall(r".*?</[^>]+>|.+", chunk_text, re.DOTALL) or [""]
            if self.schema_branches
            else [chunk_text]
        )
        for chunk in chunks:
            sp_result = self.detector.parse_streaming_increment(
                chunk, self.detector_tools
            )
            final_normal_text += sp_result.normal_text
            final_calls.extend(sp_result.calls)
            self._update_tool_schemas(sp_result.calls, streaming=True)

        return final_normal_text, final_calls

    def parse_stream_end(self) -> Tuple[str, list[ToolCallItem]]:
        """Flush detector state once the stream ends.

        Text a detector held back waiting for a marker (which can no longer
        arrive) is released as normal text; see BaseFormatDetector.finish().
        """
        if not self.tools:
            return "", []
        sp_result = self.detector.finish(self.detector_tools)
        self._update_tool_schemas(sp_result.calls, streaming=True)
        self.streaming_calls.clear()
        return sp_result.normal_text, sp_result.calls

    def get_legacy_structural_tag(
        self, at_least_one: bool = False
    ) -> StructuralTagResponseFormat:
        """
        Generate a structural tag response format for all available tools.

        This creates the necessary structural tags that guide the model's output format.

        Args:
            at_least_one: If True, the grammar forces at least one tool call
                (no free text allowed). Used for required/named tool_choice.

        Raises:
            ValueError: If tools have conflicting $defs schemas.
        """
        # Validate $defs consistency before building structural tags
        _get_tool_schema_defs(self.tools)

        tool_structures: List[StructuresResponseFormat] = list()
        tool_trigger_set: Set[str] = set()

        get_structure_info = self.detector.structure_info()
        for tool in self.tools:
            function = tool.function
            name = function.name
            assert name is not None
            info = get_structure_info(name)

            # accept all if not strict, otherwise only accept the schema
            is_strict = (
                function.strict or self.tool_strict_level >= ToolStrictLevel.PARAMETER
            )
            schema = function.parameters if is_strict else {}

            tool_structures.append(
                StructuresResponseFormat(
                    begin=info.begin,
                    schema=schema or {},  # type: ignore
                    end=info.end,
                )
            )
            tool_trigger_set.add(info.trigger)

        # TODO(dark): move this into new structural tag format
        # This requires all grammar backend support the new format
        return LegacyStructuralTagResponseFormat(
            type="structural_tag",
            structures=tool_structures,
            triggers=list(tool_trigger_set),
            at_least_one=at_least_one,
        )

    def get_structure_constraint(
        self,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]],
        parallel_tool_calls: bool = True,
        thinking_mode: bool = False,
    ) -> Optional[ToolCallConstraint]:
        """
        Returns the appropriate structure constraint for tool calls based on the tool_choice.
        The constraint is used to guide the model's output format.

        Args:
            tool_choice: The tool choice setting from the request

        Returns:
            A tuple of (constraint_type, constraint_value) to be added to sampling parameters,
            or None if no constraint applies.
        """
        is_required = tool_choice == "required" or isinstance(tool_choice, ToolChoice)
        should_constrain_auto = tool_choice == "auto" and (
            any(tool.function.strict for tool in self.tools)
            or self.tool_strict_level >= ToolStrictLevel.FUNCTION
        )

        # Highest priority: model-native structural_tag when available.
        try:
            if tool_choice == "auto" and not should_constrain_auto:
                structural_tag = self.detector.get_auto_tool_call_structural_tag(
                    tools=self.tools,
                    thinking_mode=thinking_mode,
                    parallel_tool_calls=parallel_tool_calls,
                )
                if structural_tag is not None:
                    return ("structural_tag", structural_tag)

            if is_required or should_constrain_auto:
                structural_tag_tools = self.tools
                if self.tool_strict_level >= ToolStrictLevel.PARAMETER:
                    structural_tag_tools = [
                        tool.model_copy(
                            update={
                                "function": tool.function.model_copy(
                                    update={"strict": True}
                                )
                            }
                        )
                        for tool in self.tools
                    ]
                structural_tag = self.detector.get_structural_tag(
                    tools=structural_tag_tools,
                    thinking_mode=thinking_mode,
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                )
                if structural_tag is not None:
                    return ("structural_tag", structural_tag)

                # Fallback to legacy structural tag if model-native tag is not supported.
                if self.detector.supports_structural_tag():
                    # For "required"/named: always use structural_tag to preserve the
                    # model's native tool call format. Schema is only included when
                    # strict=True, per OpenAI protocol semantics.
                    # For "auto": only constrain when strict is enabled.
                    tag = self.get_legacy_structural_tag(at_least_one=is_required)
                    return ("structural_tag", tag)

            if (
                tool_choice == "required" or isinstance(tool_choice, ToolChoice)
            ) and not self.detector.parses_required_natively():
                json_schema = get_json_schema_constraint(
                    self.tools, tool_choice, parallel_tool_calls=parallel_tool_calls
                )
                return ("json_schema", json_schema)
        except Exception as e:
            logger.error(f"Error getting structure constraint: {e}")
            return None
