from typing import Optional

from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector


class DeepSeekV41Detector(DeepSeekV32Detector):
    """
    Detector for DeepSeek V4.1 model function call format.

    Same DSML structure as DeepSeek V4, with leading-space tag names:
    ```
    <｜DSML｜ calls>
    <｜DSML｜ invoke name="function_name">
    <｜DSML｜ parameter name="param_name" string="true">value</｜DSML｜ parameter>
    </｜DSML｜ invoke>
    </｜DSML｜ calls>
    ```
    """

    tool_calls_block_name = " calls"
    invoke_tag_name = " invoke"
    parameter_tag_name = " parameter"

    def get_structural_tag_name(self) -> Optional[str]:
        # xgrammar's builtin "deepseek_v4" tag is the unspaced grammar; fall
        # back to structure_info so constrained decoding uses the spaced tags.
        return None
