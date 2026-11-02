import json
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallParseError,
    _GetInfoFunc,
)


class Qwen25Detector(BaseFormatDetector):
    """
    Detector for Qwen 2.5 and Qwen 3 model function call format.

    Format Structure:
    ```
    <tool_call>\n{"name":"func1", "arguments":{...}}\n</tool_call>\n<tool_call>\n{"name":"func2", "arguments":{...}}\n</tool_call>
    ```

    Key Components:
    - Tool Call Tags: `<tool_call>` and `</tool_call>` wrap each individual call
    - Function Call Object: JSON object with "name" and "arguments" fields

    Reference: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct?chat_template=default
    """

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.bot_token = "<tool_call>\n"
        self.eot_token = "\n</tool_call>"
        self.tool_call_separator = "\n"
        self._normal_text_buffer = ""  # Buffer for handling partial end tokens
        self._validation_tail = ""
        self._validation_parts: list[str] | None = None

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Qwen 2.5 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        # Find all <tool_call>\n...\n</tool_call> blocks
        pattern = rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}"
        match_result_list = re.findall(pattern, text, re.DOTALL)
        calls = []
        for match_result in match_result_list:
            parsed_call = self._parse_complete_json(match_result)
            calls.extend(self.parse_base_json(parsed_call, tools))
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    @staticmethod
    def _parse_complete_json(payload: str):
        try:
            return json.loads(payload.strip())
        except json.JSONDecodeError:
            # Do not include generated arguments in the error or server logs.
            raise ToolCallParseError(
                "Invalid JSON in a complete tool-call block. Check that the "
                "model's tool-call format matches the selected JSON parser."
            ) from None

    def _validate_streaming_json(self, new_text: str) -> None:
        # Keep validation independent of the base parser's buffer: it may
        # consume a JSON object before a trailing syntax error or end tag arrives.
        # Save fragments and join once per complete call, avoiding a second
        # quadratic string accumulation on long streaming arguments.
        text = self._validation_tail + new_text
        self._validation_tail = ""
        while text:
            marker = (
                self.bot_token if self._validation_parts is None else self.eot_token
            )
            index = text.find(marker)
            if index == -1:
                keep = min(len(text), len(marker) - 1)
                if self._validation_parts is not None and len(text) > keep:
                    self._validation_parts.append(text[:-keep])
                self._validation_tail = text[-keep:]
                return

            if self._validation_parts is None:
                self._validation_parts = []
            else:
                self._validation_parts.append(text[:index])
                payload = "".join(self._validation_parts)
                self._validation_parts = None
                self._parse_complete_json(payload)
            text = text[index + len(marker) :]

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing for Qwen 2.5 tool calls.
        Uses base class implementation with buffering to handle partial end tokens.
        """
        # Incomplete blocks may still be repaired by later tokens or cut short
        # by the token limit. Only reject a block once its closing tag arrives.
        self._validate_streaming_json(new_text)

        result = super().parse_streaming_increment(new_text, tools)

        # Handle partial end tokens that are streamed character by character
        if result.normal_text:
            self._normal_text_buffer += result.normal_text

            # Check if buffer contains complete end token (without leading newline)
            end_token_without_newline = self.eot_token[1:]  # "</tool_call>"
            if end_token_without_newline in self._normal_text_buffer:
                cleaned_text = self._normal_text_buffer.replace(
                    end_token_without_newline, ""
                )
                self._normal_text_buffer = ""
                result.normal_text = cleaned_text
            else:
                # Check if buffer might contain partial end token at the end
                partial_match_len = self._ends_with_partial_token(
                    self._normal_text_buffer, end_token_without_newline
                )

                if partial_match_len:
                    # Keep potential partial match in buffer, return the rest
                    result.normal_text = self._normal_text_buffer[:-partial_match_len]
                    self._normal_text_buffer = self._normal_text_buffer[
                        -partial_match_len:
                    ]
                else:
                    # No partial match, return all buffered text
                    result.normal_text = self._normal_text_buffer
                    self._normal_text_buffer = ""

        return result

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<tool_call>\n{"name":"' + name + '", "arguments":',
            end="}\n</tool_call>",
            trigger="<tool_call>",
        )
