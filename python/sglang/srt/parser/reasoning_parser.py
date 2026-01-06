from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Type

import regex as re

from sglang.srt.parser.harmony_parser import HarmonyParser


class StreamingParseResult:
    """Result of streaming incremental parsing."""

    def __init__(
        self,
        normal_text: Optional[str] = None,
        reasoning_text: Optional[str] = None,
    ):
        self.normal_text = normal_text or ""
        self.reasoning_text = reasoning_text or ""


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


class Olmo3ReasoningState(Enum):
    REASONING = 1
    CONTENT = 2


@dataclass(frozen=True)
class Indices:
    start: int
    end: int

    def __len__(self):
        return self.end - self.start


def string_overlap(a: str, b: str) -> Tuple[Optional[Indices], Optional[Indices]]:
    """
    Find the longest overlap where the end of one string matches
    the start of the other, or the shorter string is contained
    in the longer one.
    """

    def _build_prefix(pattern: str) -> list[int]:
        """
        Prefix-function (π) for KMP: longest proper prefix == suffix for each prefix.
        """
        pi = [0] * len(pattern)
        for i in range(1, len(pattern)):
            j = pi[i - 1]
            while j > 0 and pattern[i] != pattern[j]:
                j = pi[j - 1]
            if pattern[i] == pattern[j]:
                j += 1
            pi[i] = j
        return pi

    def _overlap_suffix_prefix(text: str, pattern: str) -> int:
        """
        Length of the longest suffix of `text` that equals a prefix of `pattern`.
        Computed in O(len(text) + len(pattern)) via KMP.
        """
        if not text or not pattern:
            return 0

        pi = _build_prefix(pattern)
        j = 0  # current matched prefix length in `pattern`

        for ch in text:
            # standard KMP transition
            while j > 0 and (j == len(pattern) or ch != pattern[j]):
                j = pi[j - 1]
            if ch == pattern[j]:
                j += 1

        # j is the length of the longest prefix of pattern that is a suffix of text
        return j

    # Ensure `a` is the shorter string for simpler handling.
    a, b, swapped = (a, b, False) if len(a) < len(b) else (b, a, True)

    # Substring case: same as `if a in b` + `b.index(a)`
    idx = b.find(a)
    if idx != -1:
        ind_a = Indices(0, len(a))
        ind_b = Indices(idx, idx + len(a))
        return (ind_b, ind_a) if swapped else (ind_a, ind_b)

    # 1) suffix(a) == prefix(b), prefer this direction just like your first loop
    l1 = _overlap_suffix_prefix(a, b)
    # original code only considers overlaps of length 1..len(a)-1, not len(a)
    if l1 >= len(a):
        l1 = len(a) - 1

    if l1 > 0:
        i = l1
        ind_a = Indices(len(a) - i, len(a))
        ind_b = Indices(0, i)
        return (ind_b, ind_a) if swapped else (ind_a, ind_b)

    # 2) suffix(b) == prefix(a), only if first direction failed
    l2 = _overlap_suffix_prefix(b, a)
    if l2 >= len(a):
        l2 = len(a) - 1

    if l2 > 0:
        i = l2
        ind_a = Indices(0, i)
        ind_b = Indices(len(b) - i, len(b))
        return (ind_b, ind_a) if swapped else (ind_a, ind_b)

    return None, None


@dataclass
class Olmo3ReasoningBuffer:
    think_start: str = "<think>"
    think_end: str = "</think>"
    buffer: str = ""
    state: Olmo3ReasoningState = Olmo3ReasoningState.REASONING

    def __len__(self):
        return len(self.buffer)

    def process_buffer(self) -> Optional[Tuple[str, str]]:
        start_idx = self.buffer.find(self.think_start)
        if start_idx >= 0:
            self.state = Olmo3ReasoningState.REASONING
            pretext, self.buffer = (
                self.buffer[:start_idx],
                self.buffer[start_idx + len(self.think_start) :],
            )
            if start_idx > 0:
                return "", pretext

        end_idx = self.buffer.rfind(self.think_end)
        if end_idx >= 0:
            self.state = Olmo3ReasoningState.CONTENT
            pretext, self.buffer = (
                self.buffer[:end_idx],
                self.buffer[end_idx + len(self.think_end) :],
            )
            if end_idx > 0:
                return pretext, ""

        if self.state == Olmo3ReasoningState.REASONING:
            text_buffer, self.buffer = self.buffer, ""
            if text_buffer:
                return text_buffer, ""

        if self.state == Olmo3ReasoningState.CONTENT:
            text_buffer, self.buffer = self.buffer, ""
            if text_buffer:
                return "", text_buffer

        return None

    def add_text(self, delta_text: str) -> Optional[Tuple[str, str]]:
        self.buffer += delta_text

        _, overlap_start = string_overlap(delta_text, self.think_start)
        _, overlap_end = string_overlap(delta_text, self.think_end)

        partial_start = overlap_start is not None and len(overlap_start) < len(
            self.think_start
        )
        partial_end = overlap_end is not None and len(overlap_end) < len(self.think_end)

        if partial_start and self.think_start in self.buffer and not partial_end:
            return self.process_buffer()
        if partial_end and self.think_end in self.buffer:
            return self.process_buffer()
        if partial_start or partial_end:
            return None
        return self.process_buffer()


class Olmo3Detector(BaseReasoningFormatDetector):
    """
    Parser for Olmo 3 models where `<think>` and `</think>` are plain strings
    in the vocabulary. Adapted from vLLM's Olmo3 reasoning parser.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
    ):
        super().__init__(
            think_start_token="<think>",
            think_end_token="</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
        )

        reasoning_expr = (
            rf"^(?:{self.think_start_token})?(?P<reasoning>.*?)"
            + rf"{self.think_end_token}(?P<content>.*)$"
        )
        self.reasoning_regex = re.compile(reasoning_expr, re.DOTALL)

        initial_state = (
            Olmo3ReasoningState.REASONING
            if force_reasoning
            else Olmo3ReasoningState.CONTENT
        )
        self.buffer = Olmo3ReasoningBuffer(
            think_start=self.think_start_token,
            think_end=self.think_end_token,
            state=initial_state,
        )
        self._pending_reasoning = ""

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        match = self.reasoning_regex.match(text)
        if match:
            reasoning = match.group("reasoning") or ""
            content = match.group("content") or ""
            return StreamingParseResult(
                normal_text=content,
                reasoning_text=reasoning,
            )
        # Fallback to the generic detector which handles force_reasoning mode
        # and truncated reasoning blocks that never emit </think>.
        return super().detect_and_parse(text)

    def _handle_streaming_controls(self, reasoning_text: str) -> str:
        if self.stream_reasoning:
            return reasoning_text

        if not reasoning_text:
            return ""

        if self.buffer.state == Olmo3ReasoningState.REASONING:
            self._pending_reasoning += reasoning_text
            return ""

        reasoning = self._pending_reasoning + reasoning_text
        self._pending_reasoning = ""
        return reasoning

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        result = self.buffer.add_text(new_text)
        if result is None and self.buffer.think_end in self.buffer.buffer:
            result = self.buffer.process_buffer()

        if result is None:
            return StreamingParseResult()

        reasoning_text, normal_text = result
        if reasoning_text:
            reasoning_text = self._handle_streaming_controls(reasoning_text)

        if normal_text and not self.stream_reasoning and self._pending_reasoning:
            reasoning_text = self._pending_reasoning
            self._pending_reasoning = ""

        return StreamingParseResult(
            normal_text=normal_text,
            reasoning_text=reasoning_text,
        )


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
    Detector for T4-style reasoning format (GPT-OSS), using the HarmonyParser.
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = True):
        super().__init__(
            "<|channel|>analysis<|message|>",
            "<|end|>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
        )
        self.parser = HarmonyParser()

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        events = self.parser.parse(text)
        # Flush the buffer for one-shot parsing
        events += self.parser.parse("")

        reasoning_text = "".join(
            [e.content for e in events if e.event_type == "reasoning"]
        )
        normal_parts = []
        for e in events:
            if e.event_type == "normal":
                normal_parts.append(e.content)
            elif e.event_type == "tool_call":
                # Use raw_text to preserve structural markers for function call detector
                normal_parts.append(e.raw_text if e.raw_text else e.content)
        normal_text = "".join(normal_parts)
        # Tool call events preserve raw text with structural markers

        return StreamingParseResult(
            normal_text=normal_text,
            reasoning_text=reasoning_text,
        )

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        events = self.parser.parse(new_text)

        reasoning_text = "".join(
            [e.content for e in events if e.event_type == "reasoning"]
        )
        normal_parts = []
        for e in events:
            if e.event_type == "normal":
                normal_parts.append(e.content)
            elif e.event_type == "tool_call":
                # Use raw_text to preserve structural markers for function call detector
                normal_parts.append(e.raw_text if e.raw_text else e.content)
        normal_text = "".join(normal_parts)

        return StreamingParseResult(
            normal_text=normal_text,
            reasoning_text=reasoning_text,
        )


class MiniMaxAppendThinkDetector(BaseReasoningFormatDetector):
    """
    Append `<think>` token to the beginning of the text.
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False):
        # scheduler.py need `reasoning_parser.detector.think_end_token`
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
        )
        self.is_first_chunk = False

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        if not self.is_first_chunk:
            self.is_first_chunk = True
            new_text = self.think_start_token + new_text
        return StreamingParseResult(normal_text=new_text)

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        return StreamingParseResult(normal_text=self.think_start_token + text)


class NanoV3Detector(BaseReasoningFormatDetector):
    """
    Detector for NanoV3 model.
    Uses the same reasoning format as DeepSeek-R1: (<think>)*(.*)</think>

    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
        )


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
        "kimi_k2": DeepSeekR1Detector,
        "olmo3": Olmo3Detector,
        "qwen3": Qwen3Detector,
        "qwen3-thinking": Qwen3Detector,
        "minimax": Qwen3Detector,
        "minimax-append-think": MiniMaxAppendThinkDetector,
        "step3": DeepSeekR1Detector,
        "nano_v3": NanoV3Detector,
        "interns1": Qwen3Detector,
    }

    def __init__(
        self,
        model_type: Optional[str] = None,
        stream_reasoning: bool = True,
        force_reasoning: Optional[bool] = None,
    ):
        if not model_type:
            raise ValueError("Model type must be specified")

        detector_class = self.DetectorMap.get(model_type.lower())
        if not detector_class:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Special cases where we override force_reasoning
        if model_type.lower() in {"qwen3-thinking", "gpt-oss", "minimax"}:
            force_reasoning = True

        # Only pass force_reasoning if explicitly set, let detectors use their defaults
        kwargs = {"stream_reasoning": stream_reasoning}
        if force_reasoning is not None:
            kwargs["force_reasoning"] = force_reasoning

        self.detector = detector_class(**kwargs)

    def parse_non_stream(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Non-streaming call: one-time parsing"""
        ret = self.detector.detect_and_parse(full_text)
        return ret.reasoning_text, ret.normal_text

    def parse_stream_chunk(
        self, chunk_text: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Streaming call: incremental parsing"""
        ret = self.detector.parse_streaming_increment(chunk_text)
        return ret.reasoning_text, ret.normal_text
