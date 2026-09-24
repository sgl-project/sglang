import json
import logging
import re
from typing import Any, Optional

from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
)
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector

logger = logging.getLogger(__name__)


class DeepSeekV4Detector(DeepSeekV32Detector):
    """
    Detector for DeepSeek V4 model function call format.

    The DeepSeek V4 format uses XML-like DSML tags to delimit function calls.
    Supports two parameter formats:

    Format 1 - XML Parameter Tags:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>
        ...
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Format 2 - Direct JSON:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="function_name">
        {
            "param_name": "value"
        }
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Examples:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        { "city": "San Francisco" }
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜DSML｜tool_calls>` and `</｜DSML｜tool_calls>`
    - Individual Tool Call: Wrapped between `<｜DSML｜invoke name="...">` and `</｜DSML｜invoke>`
    - Parameters: Either XML tags or direct JSON format
    - Supports multiple tool calls

    Reference: DeepSeek V4 format specification
    """

    tool_calls_block_name = "tool_calls"

    # Markdown-style pseudo tool call, e.g.:
    #   ### search
    #   {"query": "..."}
    # Some models occasionally emit a tool call in this shape with full intent
    # and valid arguments instead of the DSML protocol. Parse it so the API
    # faithfully reflects the model's tool-call intent. Guarded by:
    #   1) no DSML block present (normal path untouched)
    #   2) block must span the whole generation
    #   3) tool name must match a declared tool (case-insensitive)
    #   4) arguments must be a valid JSON object
    _MD_TOOL_RE = re.compile(
        r"^\s*###\s*([A-Za-z_][\w-]*)[ \t]*\r?\n\s*(\{[\s\S]*?\})\s*$",
        re.DOTALL,
    )

    def _match_markdown_tool_call(
        self, text: str, tools: list[Any]
    ) -> Optional[ToolCallItem]:
        if not text or self.bot_token in text or "<｜DSML｜invoke" in text:
            return None
        m = self._MD_TOOL_RE.match(text)
        if not m:
            return None
        name, args_raw = m.group(1), m.group(2)
        try:
            args = json.loads(args_raw)
        except Exception:
            return None
        if not isinstance(args, dict):
            return None
        for t in tools or []:
            tname = (
                t.function.name if hasattr(t, "function") else t.get("function", {}).get("name")
            )
            if tname and tname.lower() == name.lower():
                return ToolCallItem(
                    tool_index=0,
                    name=tname,
                    parameters=json.dumps(args, ensure_ascii=False),
                )
        return None

    def has_tool_call(self, text: str) -> bool:
        if super().has_tool_call(text):
            return True
        # cheap pre-check: markdown heading + opening brace at start
        stripped = (text or "").lstrip()
        return stripped.startswith("###") and "{" in stripped

    def detect_and_parse(self, text: str, tools: list[Any]) -> StreamingParseResult:
        result = super().detect_and_parse(text, tools)
        if result.calls:
            return result
        call = self._match_markdown_tool_call(text, tools)
        if call is not None:
            logger.info("Parsed markdown-format tool call (name=%s)", call.name)
            return StreamingParseResult(normal_text="", calls=[call])
        return result

    def get_structural_tag_name(self) -> str:
        return "deepseek_v4"
