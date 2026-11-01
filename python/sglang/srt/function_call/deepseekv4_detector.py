import logging

from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector

logger = logging.getLogger(__name__)


class DeepSeekV4Detector(DeepSeekV32Detector):
    """DeepSeek-V4's DSML tool calls. deepseek-recipe parses DeepSeek-V4 output,
    so serving uses this detector for its tool-choice structural tag."""

    tool_calls_block_name = "tool_calls"

    def get_structural_tag_name(self) -> str:
        return "deepseek_v4"
