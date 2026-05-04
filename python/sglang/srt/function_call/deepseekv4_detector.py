from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector


class DeepSeekV4Detector(DeepSeekV32Detector):
    """
    Detector for DeepSeek V4 DSML tool-call format.

    Identical to V3.2 except the outer block wrapper is
    ``<｜DSML｜tool_calls>...</｜DSML｜tool_calls>`` instead of
    ``<｜DSML｜function_calls>...</｜DSML｜function_calls>``. The inner
    ``<｜DSML｜invoke>`` / ``<｜DSML｜parameter>`` shape is unchanged.

    Example (XML parameters):
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_weather">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
        </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜DSML｜tool_calls>"
        self.eot_token = "</｜DSML｜tool_calls>"
        self.function_calls_regex = r"<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>"
