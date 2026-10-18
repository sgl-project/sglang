import re
from typing import Optional

from sglang.srt.function_call.core_types import StructureInfo, _GetInfoFunc
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector


class AgensDetector(Qwen3CoderDetector):
    """
    Detector for Agens models (e.g., Blockway/Agens-Volundr-32B).

    Tool calls use the Qwen3-Coder XML body inside Agens sentinels:
        <|call|>
        <function=NAME>
        <parameter=ARG>
        VALUE
        </parameter>
        </function>
        <|/call|>
    """

    def __init__(self):
        super().__init__()
        self.tool_call_start_token = "<|call|>"
        self.tool_call_end_token = "<|/call|>"
        self.tool_call_regex = re.compile(r"<\|call\|>(.*?)<\|/call\|>", re.DOTALL)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin="<|call|>\n<function=" + name + ">\n",
            end="</function>\n<|/call|>",
            trigger="<|call|>",
        )

    def get_structural_tag_name(self) -> Optional[str]:
        # XGrammar has no built-in structural tag for this format; constrained
        # decoding uses structure_info() instead.
        return None
