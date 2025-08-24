import re
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


class GptOssDetector(BaseReasoningFormatDetector):
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

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = True):
        # TypeScript uses channel tokens instead of simple start/end tokens
        super().__init__(
            "<|channel|>analysis<|message|>",
            "<|end|>",
            force_reasoning=True,
            stream_reasoning=stream_reasoning,
        )
        self.final_channel_start = "<|start|>assistant<|channel|>final<|message|>"
        self.final_channel_end = "<|return|>"
        self._in_final_channel = False
        self._analysis_complete = False
        self._in_reasoning = True

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses both analysis and final channels.
        Tool call channels are preserved in normal_text for downstream processing.

        HACK: Also handles simplified format where text starts with "analysis" and transitions
        to "assistantfinal" without full channel markers.
        """
        # HACK: Handle simplified format (analysis...assistantfinal) without channel markers
        if (
            text.startswith("analysis")
            and "assistantfinal" in text
            and "<|channel|>" not in text
        ):
            # Split on "assistantfinal"
            parts = text.split("assistantfinal", 1)
            self._in_reasoning = False
            if len(parts) == 2:
                reasoning_text = parts[0][
                    len("analysis") :
                ].strip()  # Remove "analysis" prefix
                normal_text = parts[1].strip()
                return StreamingParseResult(
                    normal_text=normal_text, reasoning_text=reasoning_text
                )

        reasoning_parts = []
        normal_parts = []
        current_pos = 0

        # Process text sequentially to preserve tool calls between analysis sections
        while current_pos < len(text):
            # Look for next analysis channel
            analysis_start_idx = text.find(self.think_start_token, current_pos)

            if analysis_start_idx == -1:
                # No more analysis channels, rest goes to remaining
                break

            # Preserve any content before this analysis channel (could include tool calls)
            if analysis_start_idx > current_pos:
                between_content = text[current_pos:analysis_start_idx]
                # This content will be added to normal_parts later
                normal_parts.append(between_content)

            # Extract analysis content
            analysis_content_start = analysis_start_idx + len(self.think_start_token)
            analysis_end_idx = text.find(self.think_end_token, analysis_content_start)

            if analysis_end_idx != -1:
                reasoning_parts.append(
                    text[analysis_content_start:analysis_end_idx].strip()
                )
                current_pos = analysis_end_idx + len(self.think_end_token)
            else:
                # Analysis not complete
                reasoning_parts.append(text[analysis_content_start:].strip())
                reasoning_text = "".join(reasoning_parts)
                return StreamingParseResult(reasoning_text=reasoning_text)

        # Add any remaining text after all analysis sections
        if current_pos < len(text):
            remaining = text[current_pos:]
            normal_parts.append(remaining)

        # Process non-analysis content for commentary sections
        full_normal_text = "".join(normal_parts)

        # Extract reasoning from non-tool-call commentary sections
        # Tool calls have "to=" in their header, regular commentary does not
        commentary_pattern = re.compile(
            r"<\|start\|>assistant<\|channel\|>commentary<\|message\|>(.*?)(?:<\|end\|>|<\|call\|>)",
            re.DOTALL,
        )

        cleaned_text = full_normal_text
        for match in reversed(list(commentary_pattern.finditer(full_normal_text))):
            # Check if this commentary is a tool call by looking at the text before <|message|>
            match_start = match.start()
            # Find where "<|channel|>commentary" starts within the matched pattern
            # The pattern starts with "<|start|>assistant<|channel|>commentary"
            # So we look for the text between "commentary" and "<|message|>" in the match
            match_text = full_normal_text[match_start : match.end()]
            commentary_idx = match_text.find("<|channel|>commentary")
            if commentary_idx != -1:
                message_idx = match_text.find("<|message|>", commentary_idx)
                if message_idx != -1:
                    between_text = match_text[commentary_idx:message_idx]
                    # If no "to=" found, this is regular commentary (reasoning content)
                    if " to=" not in between_text:
                        content = match.group(1).strip()
                        reasoning_parts.append(content)
                        # Remove this commentary section from normal text
                        cleaned_text = (
                            cleaned_text[: match.start()] + cleaned_text[match.end() :]
                        )

        full_normal_text = cleaned_text

        # Combine all reasoning parts
        reasoning_text = "".join(reasoning_parts)

        # Process full_normal_text for final output
        normal_text = ""
        if self.final_channel_start in full_normal_text:
            final_start = full_normal_text.find(self.final_channel_start)
            final_content_start = final_start + len(self.final_channel_start)
            final_end = full_normal_text.find(
                self.final_channel_end, final_content_start
            )

            if final_end != -1:
                # Extract content before final channel (includes tool calls)
                before_final = full_normal_text[:final_start].strip()
                # Extract ONLY the final channel content (not the channel markers)
                final_text = full_normal_text[final_content_start:final_end].strip()
                # Extract content after final channel
                after_final = full_normal_text[
                    final_end + len(self.final_channel_end) :
                ].strip()

                # For tool calls + final answer: concatenate tool calls with final text
                parts = []
                if before_final:
                    parts.append(before_final)
                if final_text:
                    parts.append(final_text)
                if after_final:
                    parts.append(after_final)
                normal_text = " ".join(parts)
            else:
                # Final channel not complete - extract what we have
                # Look for just <|channel|>final<|message|> without <|return|>
                alt_final_start = full_normal_text.find("<|channel|>final<|message|>")
                if alt_final_start != -1:
                    before_alt_final = full_normal_text[:alt_final_start].strip()
                    alt_final_content = full_normal_text[
                        alt_final_start + len("<|channel|>final<|message|>") :
                    ].strip()

                    parts = []
                    if before_alt_final:
                        parts.append(before_alt_final)
                    if alt_final_content:
                        parts.append(alt_final_content)
                    normal_text = " ".join(parts)
                else:
                    normal_text = full_normal_text.strip()
        else:
            # No final channel, treat all as normal text (includes tool calls)
            normal_text = full_normal_text.strip()

        return StreamingParseResult(
            normal_text=normal_text, reasoning_text=reasoning_text
        )

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        """
        Streaming incremental parsing for GPT-OSS format.

        This is a simplified streaming implementation that accumulates content
        and delegates to the non-streaming parser for complex multi-channel parsing.
        TODO: Implement proper incremental parsing for better streaming performance.
        """
        self._buffer += new_text

        if not self._in_reasoning:
            return StreamingParseResult(normal_text=new_text)

        # Check if we have complete sections to process
        # For GPT-OSS, we need to wait for complete channel sections
        # HACK: For now, use simplified approach - wait for key markers before processing
        key_markers = ["<|end|>", "<|call|>", "<|return|>", "assistantfinal"]
        has_complete_section = any(marker in self._buffer for marker in key_markers)

        if not has_complete_section:
            # Still accumulating, don't process yet
            return StreamingParseResult()

        # Handle simplified format (analysis...assistantfinal) with true incremental streaming
        if (
            "<|channel|>" not in self._buffer
        ):  # Simplified format without channel markers
            if self._buffer.startswith("analysis"):
                # Check if we have the transition to assistantfinal
                if "assistantfinal" in self._buffer:
                    self._in_reasoning = False
                    # Complete reasoning section - extract and stream it
                    parts = self._buffer.split("assistantfinal", 1)
                    reasoning_text = parts[0][len("analysis") :].strip()
                    final_content = parts[1].strip()

                    # Clear buffer and return both reasoning and final content
                    self._buffer = ""
                    return StreamingParseResult(
                        reasoning_text=reasoning_text if self.stream_reasoning else "",
                        normal_text=final_content,
                    )
                elif self.stream_reasoning:
                    # Stream reasoning content incrementally as it arrives
                    current_reasoning = self._buffer[len("analysis") :].strip()
                    self._buffer = ""
                    return StreamingParseResult(reasoning_text=current_reasoning)
                else:
                    # Wait for assistantfinal
                    return StreamingParseResult()
            elif self._buffer.startswith("assistantfinal"):
                # Direct final content without analysis
                final_content = self._buffer[len("assistantfinal") :].strip()
                self._buffer = ""
                return StreamingParseResult(normal_text=final_content)

        # For full channel format, process sections as they complete
        result = StreamingParseResult()

        # Process complete analysis sections
        while (
            self.think_start_token in self._buffer
            and self.think_end_token in self._buffer
        ):
            start_idx = self._buffer.find(self.think_start_token)
            start_pos = start_idx + len(self.think_start_token)
            end_pos = self._buffer.find(self.think_end_token, start_pos)

            if end_pos != -1:
                reasoning_content = self._buffer[start_pos:end_pos].strip()
                if self.stream_reasoning and reasoning_content:
                    result.reasoning_text += reasoning_content

                # Remove processed analysis section
                self._buffer = (
                    self._buffer[:start_idx]
                    + self._buffer[end_pos + len(self.think_end_token) :]
                )
            else:
                break

        # Process complete commentary sections
        commentary_pattern = re.compile(
            r"<\|start\|>assistant<\|channel\|>commentary<\|message\|>(.*?)(?:<\|end\|>|<\|call\|>)",
            re.DOTALL,
        )

        for match in reversed(list(commentary_pattern.finditer(self._buffer))):
            # Check if this is a tool call
            start_pos = match.start()
            commentary_content = match.group(1).strip()
            if self.stream_reasoning and commentary_content:
                result.reasoning_text += commentary_content

            # Remove this commentary section
            self._buffer = self._buffer[: match.start()] + self._buffer[match.end() :]
            # Clean up any standalone <|start|>assistant
            self._buffer = re.sub(
                r"<\|start\|>assistant(?=<\|start\|>assistant)", "", self._buffer
            )

        # Handle final channel completion
        if self.final_channel_start in self._buffer:
            final_start = self._buffer.find(self.final_channel_start)
            final_content_start = final_start + len(self.final_channel_start)

            # Check if final channel is complete
            final_end = self._buffer.find(self.final_channel_end, final_content_start)
            if final_end != -1:
                # Complete final channel - process everything
                final_result = self.detect_and_parse(self._buffer)
                self._buffer = ""
                return StreamingParseResult(
                    normal_text=final_result.normal_text,
                    reasoning_text=result.reasoning_text + final_result.reasoning_text,
                )
            else:
                # Extract content before final channel (e.g. tool calls)
                before_final = self._buffer[:final_start]
                if before_final:
                    # Output tool calls for processing
                    result.normal_text += before_final
                    # Keep the final channel part in buffer
                    self._buffer = self._buffer[final_start:]

        return result


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
        "deepseek-v3": Qwen3Detector,
        "glm45": Qwen3Detector,
        "gpt-oss": GptOssDetector,
        "kimi": KimiDetector,
        "qwen3": Qwen3Detector,
        "qwen3-thinking": Qwen3Detector,
        "step3": DeepSeekR1Detector,
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
