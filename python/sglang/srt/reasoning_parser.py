import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Type


class BaseReasoningParser:
    """Base class for reasoning parser."""

    def __init__(self):
        self._buffer = ""

    def detect_and_parse(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Detect and parse the text, return reasoning_content and content."""
        raise NotImplementedError

    def parse_streaming_increment(
        self, new_text: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Parse the new text incrementally, return reasoning_content and content."""
        raise NotImplementedError


class DeepSeekR1ReasoningParser(BaseReasoningParser):
    """
    DeepSeekR1 reasoning parser, which use "<think>\n" and "\n</think>" to detect the reasoning part.
    Referring to https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#usage-recommendations~.
    """

    def __init__(self):
        super().__init__()
        self.think_start_token = "<think>\n"
        self.think_end_token = "\n</think>"
        self.pattern = re.compile(
            rf"{self.think_start_token}(.*?){self.think_end_token}", re.DOTALL
        )

        self.is_reasoning = True 

    def detect_and_parse(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        # After DeepSeek update their chat templates in R1 series models, the reasoning models do not output "<think>\n"
        # We assume the output has an "<think>\n", and the reasoning part is the whole text.
        if self.think_end_token not in text:
            # Remove "<think>\n" if exists
            return text.replace(self.think_start_token, ""), None

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

            return reasoning_content, content if len(content) > 0 else None

    def parse_streaming_increment(
        self, new_text: str
    ) -> Tuple[Optional[str], Optional[str]]:

        # Should parse
        if self.is_reasoning:
            # Again, we assume the output has an "<think>\n"
            if len(self._buffer) == 0:
                new_text = new_text.replace(self.think_start_token, "")
            self._buffer += new_text
            # Reasoning continues
            if self.think_end_token not in self._buffer:
                return new_text, None
            # Reasoning ends
            else:
                reasoning_part = new_text.split(self.think_end_token)[0]
                content_part = new_text.split(self.think_end_token)[1]

                self.is_reasoning = False
                self._buffer = ""

                return reasoning_part if len(reasoning_part) > 0 else None, (
                    content_part if len(content_part) > 0 else None
                )

        else:
            return None, new_text


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
