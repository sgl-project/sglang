from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.inkling_detector import InklingDetector


class Inkling2Detector(InklingDetector):
    """InklingDetector's format; automatic tool choice stays unconstrained unless
    the request or the strict level asks for it."""

    get_auto_tool_call_structural_tag = (
        BaseFormatDetector.get_auto_tool_call_structural_tag
    )
