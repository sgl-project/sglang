import logging
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.ebnf_composer import EBNFComposer
from sglang.srt.function_call.qwen25_detector import Qwen25Detector

logger = logging.getLogger(__name__)


class TngR1T2Detector(Qwen25Detector):
    """
    Detector for tngtech/DeepSeek-TNG-R1T2-Chimera call format.

    Format Structure (identical to Qwen25):
    ```
    <tool_call>\n{"name":"func1", "arguments":{...}}\n</tool_call>\n<tool_call>\n{"name":"func2", "arguments":{...}}\n</tool_call>
    ```

    In contrast to Qwen2.5 models, the TNG-R1T2 model does not generate `<tool_call>` and `</tool_call>`
    as single tokens but in multiple chunks. (e.g., ['<', 'tool', '_call', '>\n'] and ['</', 'tool', '_call', '>']).
    While the Qwen25Detector can handle
    - non-streamed messages (here, there is effectively no difference to Qwen-2.5)
    - streamed messages with tool_choice=auto (here, no structure constraints are effective)
    - streamed messages with strict tool formats

    Qwen25Detector fails, however, for tool_choice=auto requests that make use of EBNF grammar.
    The reason is that the model tends to generate closing tool call tags with trailing newline
    (like ['</', 'tool', '_call', '>\n']), which satisfies the EBNF grammar ONLY IF it is followed
    by another tool call.
    The TngR1T2Detector solves this issue by accepting a trailing newline after the closing </tool_call> tag
    in the EBNF grammar.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>\n"
        self.eot_token = "\n</tool_call>"
        self.tool_call_separator = "\n"

    def build_ebnf(self, tools: List[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            individual_call_start_token=self.bot_token.replace("\n", "\\n"),
            individual_call_end_token=self.eot_token.replace("\n", "\\n") + "\\n",
            tool_call_separator="\\n",
            function_format="json",
        )
