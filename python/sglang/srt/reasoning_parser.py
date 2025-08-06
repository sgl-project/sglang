from typing import Dict, Optional, Tuple, Type


class StreamingParseResult:
    """Result of streaming incremental parsing."""

    def __init__(self, normal_text: str = "", reasoning_text: str = ""):
        self.normal_text = normal_text
        self.reasoning_text = reasoning_text


class BaseReasoningFormatDetector:
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    def __init__(
        self,
        think_start_token: str,
        think_end_token: str,
        force_reasoning: bool = False,
        stream_reasoning: bool = True,
    ):
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self._in_reasoning = force_reasoning
        self.stream_reasoning = stream_reasoning

        self._buffer = ""
        self.stripped_think_start = False

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses reasoning sections in the provided text.
        Returns both reasoning content and normal text separately.
        """
        in_reasoning = self._in_reasoning or self.think_start_token in text

        if not in_reasoning:
            return StreamingParseResult(normal_text=text)

        # The text is considered to be in a reasoning block.
        processed_text = text.replace(self.think_start_token, "").strip()

        if self.think_end_token not in processed_text:
            # Assume reasoning was truncated before `</think>` token
            return StreamingParseResult(reasoning_text=processed_text)

        # Extract reasoning content
        splits = processed_text.split(self.think_end_token, maxsplit=1)
        reasoning_text = splits[0]
        normal_text = splits[1].strip()

        return StreamingParseResult(
            normal_text=normal_text, reasoning_text=reasoning_text
        )

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        """
        Streaming incremental parsing for reasoning content.
        Handles partial reasoning tags and content.

        If stream_reasoning is False:
            Accumulates reasoning content until the end tag is found
        If stream_reasoning is True:
            Streams reasoning content as it arrives
        """
        self._buffer += new_text
        current_text = self._buffer

        # If the current text is a prefix of the think token, keep buffering
        if any(
            token.startswith(current_text) and token != current_text
            for token in [self.think_start_token, self.think_end_token]
        ):
            return StreamingParseResult()

        # Strip `<think>` token if present
        if not self.stripped_think_start and self.think_start_token in current_text:
            current_text = current_text.replace(self.think_start_token, "")
            self.stripped_think_start = True
            self._in_reasoning = True

        # Handle end of reasoning block
        if self._in_reasoning and self.think_end_token in current_text:
            end_idx = current_text.find(self.think_end_token)

            reasoning_text = current_text[:end_idx]

            self._buffer = ""
            self._in_reasoning = False
            normal_text = current_text[end_idx + len(self.think_end_token) :]

            return StreamingParseResult(
                normal_text=normal_text, reasoning_text=reasoning_text.rstrip()
            )

        # Continue with reasoning content
        if self._in_reasoning:
            if self.stream_reasoning:
                # Stream the content immediately
                self._buffer = ""
                return StreamingParseResult(reasoning_text=current_text)
            else:
                return StreamingParseResult()

        # If we're not in a reasoning block return as normal text
        if not self._in_reasoning:
            self._buffer = ""
            return StreamingParseResult(normal_text=current_text)

        return StreamingParseResult()


class DeepSeekR1Detector(BaseReasoningFormatDetector):
    """
    Detector for DeepSeek-R1 model.
    Assumes reasoning format:
      (<think>)*(.*)</think>
    Returns all the text before the </think> tag as `reasoning_text`
    and the rest of the text as `normal_text`.

    Supported models:
      - DeepSeek-R1: Always generates thinking content without <think> start tag
      - DeepSeek-R1-0528: Generates thinking content with <think> start tag

    Format patterns:
      - DeepSeek-R1: "I need to think about this...</think>The answer is 42."
      - DeepSeek-R1-0528: "<think>I need to think about this...</think>The answer is 42."

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = True):
        # DeepSeek-R1 is assumed to be reasoning until `</think>` token
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=True,
            stream_reasoning=stream_reasoning,
        )
        # https://github.com/sgl-project/sglang/pull/3202#discussion_r1950153599


class Qwen3Detector(BaseReasoningFormatDetector):
    """
    Detector for Qwen3 models (e.g., Qwen/Qwen3-235B-A22B).
    Assumes reasoning format:
      (<think>)*(.*)</think>

    Qwen3 models released before 07/2025 supports switching between thinking mode and normal
    mode using `enable_thinking` parameter in the request parameter.
      - enable_thinking=True: "<think>reasoning content</think>The answer is 42."
      - enable_thinking=False: "The answer is 42." (no thinking tokens)

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
        )


class KimiDetector(BaseReasoningFormatDetector):
    """
    Detector for Kimi Thinking model.
    Assumes reasoning format:
      ◁think▷*(.*)◁/think▷
    Returns all the text before the ◁/think▷ tag as `reasoning_text`
    and the rest of the text as `normal_text`.
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False):
        super().__init__(
            "◁think▷",
            "◁/think▷",
            force_reasoning=False,
            stream_reasoning=stream_reasoning,
        )


class GPTOSSDetector(BaseReasoningFormatDetector):
    """
    Detector for T4-style reasoning format.

    Assumes reasoning format with two channels:
      <|channel|>analysis<|message|>...reasoning content...<|end|>
      <|start|>assistant<|channel|>final<|message|>...final answer...<|return|>

    Returns content from 'analysis' channel as reasoning_text
    and content from 'final' channel as normal_text.

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until complete.
            If True, streams reasoning content as it arrives.
    """

    def __init__(self, stream_reasoning: bool = True):
        # TypeScript uses channel tokens instead of simple start/end tokens
        super().__init__(
            "<|channel|>analysis<|message|>",
            "<|end|>",
            force_reasoning=False,
            stream_reasoning=stream_reasoning,
        )
        self.final_channel_start = "<|start|>assistant<|channel|>final<|message|>"
        self.final_channel_end = "<|return|>"
        self._in_final_channel = False
        self._analysis_complete = False

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses both analysis and final channels.
        """
        reasoning_text = ""
        normal_text = ""
        
        # Extract ALL analysis channel content (can be multiple)
        analysis_parts = []
        remaining = text
        while self.think_start_token in remaining:
            analysis_start = remaining.find(self.think_start_token) + len(
                self.think_start_token
            )
            analysis_end = remaining.find(self.think_end_token, analysis_start)
            
            if analysis_end != -1:
                analysis_parts.append(remaining[analysis_start:analysis_end].strip())
                remaining = remaining[analysis_end + len(self.think_end_token):]
            else:
                # Analysis not complete
                analysis_parts.append(remaining[analysis_start:].strip())
                reasoning_text = "".join(analysis_parts)
                return StreamingParseResult(reasoning_text=reasoning_text)
        
        # Combine all analysis parts
        if analysis_parts:
            reasoning_text = "".join(analysis_parts)
        
        # Check for final channel
        if self.final_channel_start in remaining:
            final_start = remaining.find(self.final_channel_start)
            
            # Capture any intermediate content between analysis and final channels
            intermediate_content = ""
            if final_start > 0:
                intermediate_content = remaining[:final_start].strip()
            
            final_content_start = final_start + len(self.final_channel_start)
            final_end = remaining.find(self.final_channel_end, final_content_start)

            if final_end != -1:
                final_text = remaining[final_content_start:final_end].strip()
                # Include intermediate content in normal_text
                normal_text = (intermediate_content + final_text).strip() if intermediate_content else final_text
                # Add any remaining text after final channel
                remaining_after_final = remaining[final_end + len(self.final_channel_end):].strip()
                if remaining_after_final:
                    normal_text = (normal_text + remaining_after_final).strip() if normal_text else remaining_after_final
            else:
                # Final channel not complete
                final_text = remaining[final_content_start:].strip()
                normal_text = (intermediate_content + final_text).strip() if intermediate_content else final_text
        else:
            # No final channel, treat remaining as normal text
            normal_text = remaining.strip()

        return StreamingParseResult(
            normal_text=normal_text, reasoning_text=reasoning_text
        )

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        """
        Streaming incremental parsing for TypeScript two-channel format.
        """
        self._buffer += new_text
        current_text = self._buffer

        tokens = [
            self.think_start_token,
            self.think_end_token,
            self.final_channel_start,
            self.final_channel_end,
        ]
        # Check if we're at a potential token boundary
        if any(
            token.startswith(current_text) and token != current_text for token in tokens
        ):
            return StreamingParseResult()

        # Handle analysis channel
        if not self._analysis_complete:
            if self.think_start_token in current_text:
                # Strip analysis start token
                if not self.stripped_think_start:
                    start_idx = current_text.find(self.think_start_token)
                    current_text = current_text[
                        start_idx + len(self.think_start_token) :
                    ]
                    self.stripped_think_start = True
                    self._in_reasoning = True
                    self._buffer = current_text

            # Check for end of analysis
            if self._in_reasoning and self.think_end_token in current_text:
                end_idx = current_text.find(self.think_end_token)
                reasoning_text = current_text[:end_idx]
                self._buffer = current_text[end_idx + len(self.think_end_token) :]
                self._in_reasoning = False
                self._analysis_complete = True

                return StreamingParseResult(reasoning_text=reasoning_text.rstrip())
            elif self._in_reasoning and self.stream_reasoning:
                # Stream analysis content
                self._buffer = ""
                return StreamingParseResult(reasoning_text=current_text)
            else:
                # Accumulate analysis content
                return StreamingParseResult()

        # Handle final channel
        if self._analysis_complete or not self._in_reasoning:
            if self.final_channel_start in current_text:
                if not self._in_final_channel:
                    start_idx = current_text.find(self.final_channel_start)
                    current_text = current_text[
                        start_idx + len(self.final_channel_start) :
                    ]
                    self._in_final_channel = True
                    self._buffer = current_text

            # Check for end of final channel
            elif self._in_final_channel:
                if self.final_channel_end in current_text:
                    end_idx = current_text.find(self.final_channel_end)
                    normal_text = current_text[:end_idx]
                    self._buffer = ""
                    self._in_final_channel = False

                    return StreamingParseResult(normal_text=normal_text.rstrip())
                else:
                    # Stream final content (still accumulating)
                    self._buffer = ""
                    return StreamingParseResult(normal_text=current_text)
            elif not self._in_reasoning and not self._in_final_channel:
                # Regular text before any channel
                self._buffer = ""
                return StreamingParseResult(normal_text=new_text)

        return StreamingParseResult()
    

class ReasoningParser:
    """
    Parser that handles both streaming and non-streaming scenarios for extracting
    reasoning content from model outputs.

    Args:
        model_type (str): Type of model to parse reasoning from
        stream_reasoning (bool): If False, accumulates reasoning content until complete.
            If True, streams reasoning content as it arrives.
    """

    DetectorMap: Dict[str, Type[BaseReasoningFormatDetector]] = {
        "deepseek-r1": DeepSeekR1Detector,
        "qwen3": Qwen3Detector,
        "qwen3-thinking": Qwen3Detector,
        "glm45": Qwen3Detector,
        "kimi": KimiDetector,
        "step3": DeepSeekR1Detector,
        "gpt-oss": GPTOSSDetector,
    }

    def __init__(
        self,
        model_type: Optional[str] = None,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
    ):
        if not model_type:
            raise ValueError("Model type must be specified")

        detector_class = self.DetectorMap.get(model_type.lower())
        if not detector_class:
            raise ValueError(f"Unsupported model type: {model_type}")

        if model_type.lower() == "qwen3-thinking":
            force_reasoning = True

        self.detector = detector_class(
            stream_reasoning=stream_reasoning, force_reasoning=force_reasoning
        )

    def parse_non_stream(self, full_text: str) -> Tuple[str, str]:
        """Non-streaming call: one-time parsing"""
        ret = self.detector.detect_and_parse(full_text)
        return ret.reasoning_text, ret.normal_text

    def parse_stream_chunk(self, chunk_text: str) -> Tuple[str, str]:
        """Streaming call: incremental parsing"""
        ret = self.detector.parse_streaming_increment(chunk_text)
        return ret.reasoning_text, ret.normal_text
