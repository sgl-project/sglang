from typing import Optional

from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector


class DeepSeekV41Detector(DeepSeekV32Detector):
    """DeepSeek V4.1 DSML detector.

    The leading space in each tag name below is intentional, not a typo.
    """

    tool_calls_block_name = " calls"
    invoke_tag_name = " invoke"
    parameter_tag_name = " parameter"

    def get_structural_tag_name(self) -> Optional[str]:
        return "deepseek_v4_1"
