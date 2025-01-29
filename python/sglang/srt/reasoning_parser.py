import re
from typing import Optional, Dict

REASONING_MODELS = ["deepseek-r1"]


def is_reasoning_model(model_name: str) -> bool:
    """Checks if the model is a reasoning model."""
    return model_name.lower() in REASONING_MODELS


class StreamingParseResult:
    """Result of streaming incremental parsing."""
    def __init__(self, normal_text: str = "", reasoning_text: str = ""):
        self.normal_text = normal_text
        self.reasoning_text = reasoning_text

class BaseReasoningFormatDetector:
    """Base class providing two sets of interfaces: one-time and streaming incremental."""
    def __init__(self, accumulate_reasoning: bool = False):
        self._buffer = ""
        self._in_reasoning = False
        self._current_reasoning = ""
        self.accumulate_reasoning = accumulate_reasoning

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        """Parses the text in one go."""
        raise NotImplementedError

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        """Streaming incremental parsing."""
        raise NotImplementedError

class DeepSeekR1Detector(BaseReasoningFormatDetector):
    """
    Detector for DeepSeek-R1 model.
    Assumes reasoning format:
      <think>...</think>
    Returns all the text within the <think> tags as `reasoning_text` 
    and the rest of the text as `normal_text`.

    Args:
        accumulate_reasoning (bool): If True, accumulates reasoning content until the end tag.
            If False, streams reasoning content as it arrives.
    """
    def __init__(self, accumulate_reasoning: bool = False):
        super().__init__(accumulate_reasoning=accumulate_reasoning)
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"
        self.reasoning_regex = re.compile(
            rf"{self.think_start_token}(.*?){self.think_end_token}", 
            re.DOTALL
        )

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses reasoning sections in the provided text.
        Returns both reasoning content and normal text separately.
        """
        if self.think_start_token not in text or self.think_end_token not in text:
            return StreamingParseResult(normal_text=text)

        # Extract reasoning content
        reasoning_matches = self.reasoning_regex.findall(text)
        if not reasoning_matches:
            return StreamingParseResult(normal_text=text)

        reasoning_text = reasoning_matches[0]

        # Remove the reasoning section from the text to get normal_text
        start_idx = text.find(self.think_start_token)
        if start_idx != -1:
            end_idx = start_idx + len(
                f"{self.think_start_token}{reasoning_text}{self.think_end_token}"
            )
            normal_text = text[:start_idx] + text[end_idx:]
            normal_text = normal_text.strip()
            
            return StreamingParseResult(
                normal_text=normal_text if normal_text else "",
                reasoning_text=reasoning_text
            )

        return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        """
        Streaming incremental parsing for reasoning content.
        Handles partial reasoning tags and content.
        
        If accumulate_reasoning is True:
            Accumulates reasoning content until the end tag is found
        If accumulate_reasoning is False:
            Streams reasoning content as it arrives
        """
        self._buffer += new_text
        current_text = self._buffer

        # If we're not in a reasoning block and no think start token,
        # return as normal text
        if not self._in_reasoning and self.think_start_token not in current_text:
            self._buffer = ""
            return StreamingParseResult(normal_text=new_text)

        # Handle start of reasoning block
        if not self._in_reasoning and self.think_start_token in current_text:
            start_idx = current_text.find(self.think_start_token)
            self._in_reasoning = True
            normal_text = current_text[:start_idx]
            
            # Get any reasoning content after the start token
            reasoning_start = start_idx + len(self.think_start_token)
            reasoning_text = current_text[reasoning_start:]
            
            if self.accumulate_reasoning:
                self._current_reasoning = reasoning_text
                reasoning_text = ""
            else:
                self._buffer = ""  # Clear buffer since we're streaming
            
            return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text)

        # Handle end of reasoning block
        if self._in_reasoning and self.think_end_token in current_text:
            end_idx = current_text.find(self.think_end_token)
            
            if self.accumulate_reasoning:
                # Return accumulated reasoning plus final chunk
                reasoning_text = self._current_reasoning + current_text[:end_idx]
            else:
                # Just return the final chunk before the end tag
                reasoning_text = current_text[:end_idx]
            
            self._in_reasoning = False
            self._current_reasoning = ""
            normal_text = current_text[end_idx + len(self.think_end_token):]
            self._buffer = ""
            
            return StreamingParseResult(
                normal_text=normal_text,
                reasoning_text=reasoning_text
            )

        # Continue with reasoning content
        if self._in_reasoning:
            if self.accumulate_reasoning:
                # Accumulate content but don't return it yet
                self._current_reasoning += new_text
                return StreamingParseResult()
            else:
                # Stream the content immediately
                self._buffer = ""
                return StreamingParseResult(reasoning_text=new_text)

        return StreamingParseResult()

class ReasoningParser:
    """
    Parser that handles both streaming and non-streaming scenarios for extracting
    reasoning content from model outputs.

    Args:
        model_type (str): Type of model to parse reasoning from
        accumulate_reasoning (bool): If True, accumulates reasoning content until complete.
            If False, streams reasoning content as it arrives.
    """
    DetectorMap: Dict[str, BaseReasoningFormatDetector] = {
        "deepseek-r1": DeepSeekR1Detector
    }

    def __init__(self, model_type: str = None, accumulate_reasoning: bool = False):
        if not model_type:
            raise ValueError("Model type must be specified")

        detector_class = self.DetectorMap.get(model_type.lower())
        if not detector_class:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.detector = detector_class(accumulate_reasoning=accumulate_reasoning)

    def parse_non_stream(self, full_text: str) -> StreamingParseResult:
        """Non-streaming call: one-time parsing"""
        return self.detector.detect_and_parse(full_text)

    def parse_stream_chunk(self, chunk_text: str) -> StreamingParseResult:
        """Streaming call: incremental parsing"""
        return self.detector.parse_streaming_increment(chunk_text)
