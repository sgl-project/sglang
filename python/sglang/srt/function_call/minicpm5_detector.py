"""Compatibility API for the former MiniCPM5 XML parser."""

import json
from typing import TYPE_CHECKING, Dict, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.utils import get_schema_properties, safe_literal_eval

if TYPE_CHECKING:
    from sglang.srt.function_call.minicpm4_xml_detector import (
        MiniCPM4XmlFormatDetector as MiniCPM5Detector,
    )

__all__ = ["MiniCPM5Detector", "get_argument_type", "parse_arguments"]


def get_argument_type(
    func_name: str, arg_key: str, name_to_tool: Dict[str, Tool]
) -> Optional[str]:
    tool = name_to_tool.get(func_name)
    if not tool:
        return None
    params = tool.function.parameters or {}
    if not isinstance(params, dict):
        return None
    return get_schema_properties(params).get(arg_key, {}).get("type")


def parse_arguments(json_value):
    try:
        try:
            parsed_value = json.loads(json_value)
        except (json.JSONDecodeError, TypeError):
            parsed_value = safe_literal_eval(json_value)
        return parsed_value, True
    except (ValueError, SyntaxError, TypeError):
        return json_value, False


def __getattr__(name: str):
    if name == "MiniCPM5Detector":
        # The team detector imports parse_arguments from this module.
        from sglang.srt.function_call.minicpm4_xml_detector import (
            MiniCPM4XmlFormatDetector,
        )

        return MiniCPM4XmlFormatDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
