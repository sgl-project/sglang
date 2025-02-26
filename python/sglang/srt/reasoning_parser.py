import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Type


class BaseReasoningParser:
    """Base class for reasoning parser."""

    def __init__(self, think_start_token: str, think_end_token: str, force_think: bool):
        self._buffer = ""
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self.pattern = re.compile(
            rf"{self.think_start_token}(.*?){self.think_end_token}", re.DOTALL
        )

        # whether we assume the output must have a `think_start_token`
        self.force_think = force_think
        self.is_reasoning = (
            self.force_think
        )  # assume the output has a `think_start_token` at the beginning

        raise NotImplementedError

    def parse_streaming_increment(
        self, new_text: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Parse the new text incrementally, return reasoning_content and content."""
        # Detect the start token for toggling `is_reasoning` when `force_think` is False
        if not self.force_think and self.think_start_token in new_text:
            self.is_reasoning = True

        # Should parse
        if self.is_reasoning:
            if len(self._buffer) == 0:
                self._buffer += new_text
                new_text = new_text.replace(self.think_start_token, "")
            else:
                self._buffer += new_text

            # Reasoning continues
            if self.think_end_token not in self._buffer:
                return new_text, ""
            # Reasoning ends
            else:
                reasoning_part = new_text.split(self.think_end_token)[0]
                content_part = new_text.split(self.think_end_token)[1]

                self.is_reasoning = False
                self._buffer = ""

                return reasoning_part, content_part

        else:
            return "", new_text

    def detect_and_parse(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Detect and parse the text, return reasoning_content and content."""
        if self.think_end_token not in text:
            if self.force_think:  # all the output are reasoning content
                # Remove "<think>" if exists
                return text.replace(self.think_start_token, ""), ""
            elif self.think_start_token in text:
                return text.replace(self.think_start_token, ""), ""
            else:
                return "", text

        else:
            # Add the start token to the beginning of the text.
            if self.think_start_token not in text:
                text = self.think_start_token + text

            reasoning_content = self.pattern.findall(text)[0]
            content = text[
                len(self.think_start_token)
                + len(reasoning_content)
                + len(self.think_end_token) :
            ]

            return reasoning_content, content


class DeepSeekR1ReasoningParser(BaseReasoningParser):
    """
    DeepSeekR1 reasoning parser, which use "<think>" and "</think>" to detect the reasoning part.
    Referring to https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#usage-recommendations~.
    """

    def __init__(self):
        super().__init__("<think> ", "</think> ", True)


class ReasoningParser:
    """Reasoning parser for different reasoning models."""

    ReasoningParserDict: Dict[str, Type[BaseReasoningParser]] = {
        "deepseek-r1": DeepSeekR1ReasoningParser
    }

    def __init__(self, reasoning_parser: str):
        self.parser = self.ReasoningParserDict[reasoning_parser]()

    def parse_non_stream(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Non-streaming parsing for reasoning models.
        Return: reasoning_content, content
        """
        return self.parser.detect_and_parse(full_text)

    def parse_stream_chunk(self, chunk_text: str):
        """
        Streaming parsing for reasoning models.
        Return: reasoning_content, content
        """
        return self.parser.parse_streaming_increment(chunk_text)
