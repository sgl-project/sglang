import inspect
import re
from typing import Dict, List, Optional, Tuple, Type

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.function_call.hunyuan_detector import resolve_hunyuan_tokens
from sglang.srt.parser.harmony_parser import HarmonyParser
from sglang.srt.parser.inkling_tokenizer import (
    CONTENT_INVOKE_TOOL_JSON,
    CONTENT_MODEL_END_SAMPLING,
    CONTENT_TEXT,
    CONTENT_THINKING,
    END_MESSAGE,
    INKLING_CONTROL_TOKENS,
    MESSAGE_MODEL,
)


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
        think_excluded_tokens: Optional[List[str]] = None,
        force_reasoning: bool = False,
        stream_reasoning: bool = True,
        tool_start_token: Optional[str] = None,
        continue_final_message: bool = False,
        previous_content: str = "",
        thinks_internally: bool = False,
        reasoning_default: str = "always",
        force_nonempty_content: bool = False,
    ):
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self.think_excluded_tokens = think_excluded_tokens
        self.tool_start_token = tool_start_token
        self.force_reasoning = force_reasoning
        self._in_reasoning = force_reasoning
        self.stream_reasoning = stream_reasoning
        self.thinks_internally = thinks_internally
        self.reasoning_default = reasoning_default

        self._buffer = ""
        self.stripped_think_start = False
        self.think_start_self_label = ""

        self._force_nonempty_content = force_nonempty_content
        self._accumulated_reasoning = ""

        self.continue_final_message = continue_final_message
        if self.continue_final_message:
            self.previous_content = previous_content
            self.previous_count = len(previous_content)
        else:
            self.previous_content = ""
            self.previous_count = 0

        if self.think_start_token in self.previous_content:
            self._in_reasoning = True
        if self.think_end_token in self.previous_content:
            self._in_reasoning = False

    def _maybe_apply_force_nonempty_content(
        self, ret: StreamingParseResult
    ) -> StreamingParseResult:
        if self._force_nonempty_content and not ret.normal_text:
            ret.normal_text, ret.reasoning_text = ret.reasoning_text, ret.normal_text
        return ret

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses reasoning sections in the provided text.
        Returns both reasoning content and normal text separately.
        """
        return self._maybe_apply_force_nonempty_content(
            self._detect_and_parse_impl(text)
        )

    def _detect_and_parse_impl(self, text: str) -> StreamingParseResult:
        in_reasoning = self._in_reasoning or self.think_start_token in text

        if not in_reasoning:
            return StreamingParseResult(normal_text=text)

        # The text is considered to be in a reasoning block.
        think_start_text = self.think_start_token + self.think_start_self_label
        processed_text = text
        while processed_text.startswith(think_start_text):
            processed_text = processed_text[len(think_start_text) :]

        if (
            self.think_end_token not in processed_text
            and self.think_end_token not in self.previous_content
        ):
            # Check for tool_start_token interruption
            if (
                in_reasoning
                and self.tool_start_token is not None
                and self.tool_start_token in processed_text
            ):
                # Find the first occurrence of tool_start_token and split there
                tool_idx = processed_text.find(self.tool_start_token)
                reasoning_text = processed_text[:tool_idx]
                # Preserve tool_start_token in normal text
                normal_text = processed_text[tool_idx:]
                return StreamingParseResult(
                    normal_text=normal_text, reasoning_text=reasoning_text
                )
            # Assume reasoning was truncated before end token
            return StreamingParseResult(reasoning_text=processed_text)

        # Extract reasoning content
        if self.think_end_token in processed_text:
            splits = processed_text.split(self.think_end_token, maxsplit=1)
            reasoning_text = splits[0]
            normal_text = splits[1]

            return StreamingParseResult(
                normal_text=normal_text, reasoning_text=reasoning_text
            )
        else:
            # think_end_token is in self.previous_content for continue_final_message=True case
            return StreamingParseResult(normal_text=processed_text)

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        """
        Streaming incremental parsing for reasoning content.
        Handles partial reasoning tags and content.

        If stream_reasoning is False:
            Accumulates reasoning content until the end tag is found
        If stream_reasoning is True:
            Streams reasoning content as it arrives
        """
        ret = self._parse_streaming_increment_impl(new_text)
        if self._force_nonempty_content:
            if self._in_reasoning:
                self._accumulated_reasoning += ret.reasoning_text
            else:
                self._accumulated_reasoning = ""
        return ret

    def _parse_streaming_increment_impl(self, new_text: str) -> StreamingParseResult:
        self._buffer += new_text
        current_text = self._buffer

        think_start_text = self.think_start_token + self.think_start_self_label

        # If the current text is a prefix of the think token, keep buffering
        tokens_to_check = [think_start_text, self.think_end_token]
        if self.tool_start_token:
            tokens_to_check.append(self.tool_start_token)
        if any(
            token.startswith(current_text) and token != current_text
            for token in tokens_to_check
        ):
            return StreamingParseResult()

        # Strip `<think>` token if present
        if not self.stripped_think_start and think_start_text in current_text:
            current_text = current_text.replace(think_start_text, "", 1)
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
                normal_text=normal_text, reasoning_text=reasoning_text
            )

        # Continue with reasoning content
        if self._in_reasoning:
            # Check for tool_start_token interruption
            if self.tool_start_token and self.tool_start_token in current_text:
                tool_idx = current_text.find(self.tool_start_token)
                reasoning_text = current_text[:tool_idx]
                # Preserve tool_start_token in normal text
                normal_text = current_text[tool_idx:]
                self._buffer = ""
                self._in_reasoning = False
                return StreamingParseResult(
                    normal_text=normal_text, reasoning_text=reasoning_text
                )
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

    def finish(self) -> StreamingParseResult:
        """
        Called once when the stream ends. If force_nonempty_content is set
        and the stream ended mid-reasoning, reclassifies the accumulated
        reasoning (plus any partial token still buffered) as normal text.
        """
        if self._force_nonempty_content and self._in_reasoning:
            # stream_reasoning=False never clears _buffer, so the opening think
            # token (stripped only from the base class's local view) survives here.
            buffer = self._buffer
            think_start_text = self.think_start_token + self.think_start_self_label
            if buffer.startswith(think_start_text):
                buffer = buffer[len(think_start_text) :]
            normal_text = self._accumulated_reasoning + buffer
            self._accumulated_reasoning = ""
            self._buffer = ""
            if normal_text:
                return StreamingParseResult(normal_text=normal_text)
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

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = True,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        # DeepSeek-R1 is assumed to be reasoning until `</think>` token
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=True,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            force_nonempty_content=force_nonempty_content,
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

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        think_excluded_tokens = [
            "<tool_call>",
            "</tool_call>",
            "<|im_end|>",
            "<|endoftext|>",
        ]
        super().__init__(
            "<think>",
            "</think>",
            think_excluded_tokens=think_excluded_tokens,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            # Qwen3.5 sometimes opens ``<tool_call>`` without closing
            # ``</think>``; treat it as an implicit reasoning close.
            tool_start_token="<tool_call>",
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            thinks_internally=True,
            reasoning_default="enable_thinking",
            force_nonempty_content=force_nonempty_content,
        )


class KimiDetector(BaseReasoningFormatDetector):
    """
    Detector for Kimi Thinking model.
    Assumes reasoning format:
      ◁think▷*(.*)◁/think▷
    Returns all the text before the ◁/think▷ tag as `reasoning_text`
    and the rest of the text as `normal_text`.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            "◁think▷",
            "◁/think▷",
            force_reasoning=False,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            force_nonempty_content=force_nonempty_content,
        )


class KimiK2Detector(BaseReasoningFormatDetector):
    """
    Detector for Kimi K2 models.
    Assumes reasoning format:
      (<think>)*(.*)</think>

    Kimi K2 can switch from reasoning to tool-call section with
    `<|tool_calls_section_begin|>` before emitting `</think>`.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        think_excluded_tokens = [
            "<think>",
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>",
            "<|tool_call_argument_begin|>",
            "<|tool_call_section_end|>",
            "<|tool_call_end|>",
            "[EOS]",
            "<|im_end|>",
            "<|end_header_id|>",
            "[EOT]",
        ]
        super().__init__(
            "<think>",
            "</think>",
            think_excluded_tokens=think_excluded_tokens,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<|tool_calls_section_begin|>",
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            reasoning_default="thinking",
            force_nonempty_content=force_nonempty_content,
        )


class Glm45Detector(BaseReasoningFormatDetector):
    """
    Detector for GLM-4.5 models.
    Assumes reasoning format:
      (<think>)*(.*)</think>

    GLM-4.5 uses `<tool_call>` as the tool start token to switch from reasoning mode to normal mode.

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        force_nonempty_content: bool = False,
    ):
        think_excluded_tokens = [
            "<tool_call>",
            "</tool_call>",
            "<eop>",
            "<|user|>",
            "<|endoftext|>",
        ]
        super().__init__(
            "<think>",
            "</think>",
            think_excluded_tokens=think_excluded_tokens,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<tool_call>",
            thinks_internally=True,
            reasoning_default="enable_thinking",
            force_nonempty_content=force_nonempty_content,
        )


class GptOssDetector(BaseReasoningFormatDetector):
    """
    Detector for T4-style reasoning format (GPT-OSS), using the HarmonyParser.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = True,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            "<|channel|>analysis<|message|>",
            "<|end|>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            force_nonempty_content=force_nonempty_content,
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

        return self._maybe_apply_force_nonempty_content(
            StreamingParseResult(
                normal_text=normal_text,
                reasoning_text=reasoning_text,
            )
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

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        # scheduler.py need `reasoning_parser.detector.think_end_token`
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            force_nonempty_content=force_nonempty_content,
        )
        self.is_first_chunk = False

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        if not self.is_first_chunk:
            self.is_first_chunk = True
            new_text = self.think_start_token + new_text
        return StreamingParseResult(normal_text=new_text)

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        return StreamingParseResult(normal_text=self.think_start_token + text)


class Nemotron3Detector(BaseReasoningFormatDetector):
    """
    Detector for Nemotron3 model.
    Uses the same reasoning format as DeepSeek-R1: (<think>)*(.*)</think>

    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<tool_call>",
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            reasoning_default="enable_thinking",
            force_nonempty_content=force_nonempty_content,
        )


class MiniMaxM3Detector(BaseReasoningFormatDetector):
    """MiniMax-M3 detector. Format: (<mm:think>)*(.*)</mm:think>.

    In multi-turn chats M3 prefixes earlier non-thinking turns with a bare
    ``</mm:think>``, so a non-thinking reply may open with one stray closer; drop it unless thinking.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            "<mm:think>",
            "</mm:think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )
        self._lead_buffer = ""
        self._checked_leading_close = False
        self._force_nonempty_content = force_nonempty_content

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        if not self._in_reasoning and text.lstrip().startswith(self.think_end_token):
            text = text.lstrip()[len(self.think_end_token) :]
        ret = super().detect_and_parse(text)
        if self._force_nonempty_content and not ret.normal_text:
            ret.normal_text, ret.reasoning_text = ret.reasoning_text, ret.normal_text
        return ret

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        # ``</mm:think>`` is a single token, so a stray leading closer arrives whole.
        if not self._checked_leading_close and not self._in_reasoning:
            self._lead_buffer += new_text
            stripped = self._lead_buffer.lstrip()
            if not stripped:
                return StreamingParseResult()
            self._checked_leading_close = True
            if stripped.startswith(self.think_end_token):
                new_text = stripped[len(self.think_end_token) :]
            else:
                new_text = self._lead_buffer
            self._lead_buffer = ""
            if not new_text:
                return StreamingParseResult()
        return super().parse_streaming_increment(new_text)


class MistralDetector(BaseReasoningFormatDetector):
    """
    Detector for Mistral models with reasoning (e.g., Mistral-Small-4-119B-2603).
    Assumes reasoning format:
      [THINK]reasoning content[/THINK]answer

    Reasoning is optional — it only appears when reasoning_effort="high" is set.
    When reasoning_effort="none", the model outputs directly without thinking tokens.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            "[THINK]",
            "[/THINK]",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            reasoning_default="mistral",
            force_nonempty_content=force_nonempty_content,
        )


class HunyuanDetector(BaseReasoningFormatDetector):
    """
    Detector for Hunyuan models (e.g., tencent/Hunyuan-A13B-Instruct).

    Like Glm45Detector but uses ``<tool_calls>`` (plural) as the tool start token.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        tokenizer=None,
        force_nonempty_content: bool = False,
    ):
        t = resolve_hunyuan_tokens(tokenizer)
        think_open = t["think"]
        think_close = (
            "</" + think_open[1:] if think_open.startswith("<") else think_open
        )
        super().__init__(
            think_open,
            think_close,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token=t["tool_calls"],
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            force_nonempty_content=force_nonempty_content,
        )


class Gemma4Detector(BaseReasoningFormatDetector):
    """Gemma4 reasoning detector."""

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            "<|channel>",
            "<channel|>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            reasoning_default="explicit_enable_thinking",
            force_nonempty_content=force_nonempty_content,
        )
        self.think_start_self_label = "thought\n"


_INKLING_CONTENT_KINDS = {
    CONTENT_THINKING: "reasoning",
    CONTENT_TEXT: "content",
}
_INKLING_END_TOKENS = {
    CONTENT_MODEL_END_SAMPLING,
    END_MESSAGE,
}
_INKLING_CONTROL_TOKENS = INKLING_CONTROL_TOKENS
_INKLING_CONTROL_RE = re.compile(
    "|".join(re.escape(t) for t in sorted(_INKLING_CONTROL_TOKENS))
)


class InklingDetector(BaseReasoningFormatDetector):
    """Detector for Inkling typed content blocks."""

    # Parse the model's sequence of typed content blocks, for example:
    #   <|message_model|><|content_thinking|>reasoning<|end_message|>
    #   <|message_model|><|content_text|>visible answer<|end_message|>
    #   <|content_model_end_sampling|>
    # Special tokens must decode literally so thinking and visible text can be
    # routed to their respective response fields.
    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        del force_nonempty_content
        super().__init__(
            CONTENT_THINKING,
            END_MESSAGE,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            thinks_internally=False,
            reasoning_default="always",
        )

        self._kind: str | None = None
        self._pending_header = ""
        self._pending_reasoning = ""

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        self._buffer = ""
        self._kind = None
        self._pending_header = ""
        self._pending_reasoning = ""
        ret = self._parse_blocks(text)
        if self._kind == "reasoning" and not self.stream_reasoning:
            ret.reasoning_text += self._pending_reasoning
        self._kind = None
        self._pending_header = ""
        self._pending_reasoning = ""
        return ret

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        text = self._buffer + new_text
        partial_len = self._partial_control_length(text)
        if partial_len:
            self._buffer = text[-partial_len:]
            text = text[:-partial_len]
        else:
            self._buffer = ""
        return self._parse_blocks(text)

    @staticmethod
    def _partial_control_length(text: str) -> int:
        max_token_len = max(map(len, _INKLING_CONTROL_TOKENS))
        for length in range(min(len(text), max_token_len - 1), 0, -1):
            suffix = text[-length:]
            if any(
                len(suffix) < len(token) and token.startswith(suffix)
                for token in _INKLING_CONTROL_TOKENS
            ):
                return length
        return 0

    def _parse_blocks(self, text: str) -> StreamingParseResult:
        reasoning: list[str] = []
        content: list[str] = []
        saw_control = False
        pos = 0

        def emit(text: str) -> None:
            if self._kind == "reasoning":
                if self.stream_reasoning:
                    reasoning.append(text)
                else:
                    self._pending_reasoning += text
            elif self._kind == "content":
                content.append(text)
            elif self._kind == "tool":
                content.append(text)
            elif self._kind == "header":
                self._pending_header += text
            elif text:
                # No open block — e.g. a continue_final_message stream resuming
                # mid text block. Route to visible content, matching the
                # no-control-token path below.
                content.append(text)

        def flush_reasoning() -> None:
            if self._kind == "reasoning" and not self.stream_reasoning:
                reasoning.append(self._pending_reasoning)
                self._pending_reasoning = ""

        for match in _INKLING_CONTROL_RE.finditer(text):
            saw_control = True
            emit(text[pos : match.start()])

            token = match.group(0)
            pos = match.end()
            if token == MESSAGE_MODEL:
                if self._kind in (None, "header"):
                    flush_reasoning()
                    self._pending_header = ""
                    self._kind = "header"
                else:
                    # Inside an open block a decoded <|message_model|> string
                    # is payload the model wrote (e.g. quoting the protocol) —
                    # a real header can only follow an end token. Preserve it
                    # instead of rerouting the rest of the block into a header.
                    emit(token)
            elif token == CONTENT_INVOKE_TOOL_JSON:
                flush_reasoning()
                if self._kind == "header":
                    content.extend(
                        (MESSAGE_MODEL, self._pending_header, CONTENT_INVOKE_TOOL_JSON)
                    )
                    self._pending_header = ""
                else:
                    content.append(token)
                self._kind = "tool"
            elif self._kind == "tool":
                content.append(token)
                if token in _INKLING_END_TOKENS:
                    self._kind = None
            elif token in _INKLING_CONTENT_KINDS:
                flush_reasoning()
                self._pending_header = ""
                self._kind = _INKLING_CONTENT_KINDS[token]
            elif token in _INKLING_END_TOKENS:
                flush_reasoning()
                self._pending_header = ""
                self._kind = None

        tail = text[pos:]
        if saw_control or self._kind is not None:
            emit(tail)
        else:
            content.append(text)

        return StreamingParseResult(
            normal_text="".join(content),
            reasoning_text="".join(reasoning),
        )


class _DeepSeekV3Detector(Qwen3Detector):
    """DeepSeek-V3 reuses Qwen3 tokens but requires explicit thinking=True to enable."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reasoning_default = "explicit_thinking"


class _MimoDetector(Qwen3Detector):
    """MIMO reuses Qwen3 tokens but requires explicit enable_thinking=True to enable."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reasoning_default = "explicit_enable_thinking"


class _PoolsideV1Detector(Qwen3Detector):
    """Poolside v1 (Laguna-XS.2) reuses Qwen3 <think> tokens but the HF chat template
    defaults `enable_thinking=False`; reasoning is opt-in via `enable_thinking=True`."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reasoning_default = "explicit_enable_thinking"


class Apertus2509Detector(BaseReasoningFormatDetector):
    """
    Detector for Apertus 2509 models

    Reasoning blocks are delimited by:
        <|inner_prefix|> ... <|inner_suffix|>
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            "<|inner_prefix|>",
            "<|inner_suffix|>",
            force_reasoning=False,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            force_nonempty_content=force_nonempty_content,
        )
        self._force_reasoning = force_reasoning
        self._tool_start_token = "<|tools_prefix|>["
        self._tool_end_token = "<|tools_suffix|>"
        self._reasoning_acc: str = ""
        self._in_inner_tool: bool = False

    @staticmethod
    def _ends_with_partial_token(buffer: str, token: str) -> int:
        for i in range(1, min(len(buffer) + 1, len(token))):
            if token.startswith(buffer[-i:]):
                return i
        return 0

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        blocks = self.detect_and_parse_block_sequence(text)
        reasoning_parts = [t for k, t in blocks if k == "reasoning"]
        text_parts = [t for k, t in blocks if k == "text"]
        ret = StreamingParseResult(
            normal_text="".join(text_parts),
            reasoning_text="".join(reasoning_parts),
        )
        return self._maybe_apply_force_nonempty_content(ret)

    def detect_and_parse_block_sequence(self, text: str) -> list[tuple[str, str]]:
        """Return an ordered sequence of blocks: [("reasoning"|"text", content), ...]"""
        start_tok = self.think_start_token
        end_tok = self.think_end_token
        blocks: list[tuple[str, str]] = []
        cursor = 0

        # continue_final_message can resume inside an existing inner
        if self._in_reasoning:
            if (e := text.find(end_tok, cursor)) == -1:
                blocks.extend(self._split_inner_reasoning(text[cursor:]))
                blocks.append(("text", ""))
                return blocks
            blocks.extend(self._split_inner_reasoning(text[cursor:e]))
            cursor = e + len(end_tok)

        while True:
            if (s := text.find(start_tok, cursor)) == -1:
                # Always include the trailing text block (may be empty)
                blocks.append(("text", text[cursor:]))
                break
            if s > cursor:
                blocks.append(("text", text[cursor:s]))

            cursor = s + len(start_tok)
            if (e := text.find(end_tok, cursor)) == -1:
                blocks.extend(self._split_inner_reasoning(text[cursor:]))
                blocks.append(("text", ""))
                break
            blocks.extend(self._split_inner_reasoning(text[cursor:e]))
            cursor = e + len(end_tok)

        last_idx = len(blocks) - 1
        blocks = [
            (k, t)
            for i, (k, t) in enumerate(blocks)
            if not (k == "text" and t == "" and i != last_idx)
        ]

        return blocks

    def _split_inner_reasoning(self, inner_text: str) -> list[tuple[str, str]]:
        """
        Split content inside <|inner_prefix|>...<|inner_suffix|> into:
          - ("reasoning", <thoughts text>)
          - ("text", <|tools_prefix|>[...]<|tools_suffix|>) for any tool calls inside reasoning
        """
        tool_start = self._tool_start_token
        tool_end = self._tool_end_token
        out: list[tuple[str, str]] = []
        cursor = 0

        while True:
            if (s := inner_text.find(tool_start, cursor)) == -1:
                if (tail := inner_text[cursor:]) != "":
                    out.append(("reasoning", tail))
                break
            if s > cursor:
                out.append(("reasoning", inner_text[cursor:s]))

            if (e := inner_text.find(tool_end, s)) == -1:
                out.append(("text", inner_text[s:]))
                break

            out.append(("text", inner_text[s : e + len(tool_end)]))
            cursor = e + len(tool_end)

        return out

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        self._buffer += new_text

        out_reasoning = ""
        out_normal = ""

        start_tok = self.think_start_token
        end_tok = self.think_end_token
        tool_start = self._tool_start_token
        tool_end = self._tool_end_token

        while True:
            if not self._in_reasoning:
                if (s := self._buffer.find(start_tok)) == -1:
                    if partial := self._ends_with_partial_token(
                        self._buffer, start_tok
                    ):
                        out_normal += self._buffer[:-partial]
                        self._buffer = self._buffer[-partial:]
                    else:
                        out_normal += self._buffer
                        self._buffer = ""
                    return StreamingParseResult(
                        normal_text=out_normal, reasoning_text=out_reasoning
                    )

                out_normal += self._buffer[:s]
                self._buffer = self._buffer[s + len(start_tok) :]
                self._in_reasoning = True
                self._reasoning_acc = ""
                self._in_inner_tool = False
                continue

            if self._in_inner_tool:
                if (end_pos := self._buffer.find(tool_end)) == -1:
                    if (
                        hold := self._ends_with_partial_token(self._buffer, tool_end)
                    ) != 0:
                        out_normal += self._buffer[:-hold]
                        self._buffer = self._buffer[-hold:]
                    else:
                        out_normal += self._buffer
                        self._buffer = ""
                    return StreamingParseResult(
                        normal_text=out_normal, reasoning_text=out_reasoning
                    )

                out_normal += self._buffer[: end_pos + len(tool_end)]
                self._buffer = self._buffer[end_pos + len(tool_end) :]
                self._in_inner_tool = False
                continue

            pos_tool = self._buffer.find(tool_start)
            pos_end = self._buffer.find(end_tok)

            if pos_tool == -1 and pos_end == -1:
                if self.stream_reasoning:
                    if (
                        hold := max(
                            self._ends_with_partial_token(self._buffer, end_tok),
                            self._ends_with_partial_token(self._buffer, tool_start),
                        )
                    ) != 0:
                        out_reasoning += self._buffer[:-hold]
                        self._buffer = self._buffer[-hold:]
                    else:
                        out_reasoning += self._buffer
                        self._buffer = ""
                return StreamingParseResult(
                    normal_text=out_normal, reasoning_text=out_reasoning
                )

            next_pos = min(p for p in [pos_tool, pos_end] if p != -1)

            if pos_end != -1 and pos_end == next_pos:
                reasoning_chunk = self._buffer[:pos_end]
                if self.stream_reasoning:
                    out_reasoning += reasoning_chunk
                else:
                    self._reasoning_acc += reasoning_chunk
                    out_reasoning += self._reasoning_acc
                    self._reasoning_acc = ""
                self._buffer = self._buffer[pos_end + len(end_tok) :]
                self._in_reasoning = False
                continue

            reasoning_chunk = self._buffer[:pos_tool]
            if self.stream_reasoning:
                out_reasoning += reasoning_chunk
            else:
                self._reasoning_acc += reasoning_chunk
            self._buffer = self._buffer[pos_tool:]
            self._in_inner_tool = True
            continue


class CohereCommand4Detector(BaseReasoningFormatDetector):
    """Detector for Cohere Command4 / Command-A family (incl. cohere2_moe and
    cohere2_vision Command-A-Plus).

    Generated format (the assistant prefix in the chat template already emits
    ``<|START_THINKING|>`` when ``reasoning=True``, so the *generated* text
    typically begins inside the thinking block):

        thinking_content<|END_THINKING|><|START_TEXT|>final_answer<|END_TEXT|>

    When ``reasoning=False`` the chat template emits both START/END_THINKING
    in the prefix and the generated text is just::

        <|START_TEXT|>final_answer<|END_TEXT|>

    This detector returns:
      - ``reasoning_text`` = the thinking block (between START_THINKING and
        END_THINKING, with the START tag stripped if the model echoed it).
      - ``normal_text`` = the content between ``<|START_TEXT|>`` and
        ``<|END_TEXT|>``, with both markers stripped. If no ``<|START_TEXT|>``
        appears (the model exhausted max_new_tokens still inside thinking),
        ``normal_text`` is the empty string.

    Matches the public token names from the model's
    ``special_tokens_map.json`` (``<|START_THINKING|>`` etc.).
    """

    TEXT_START_TOKEN = "<|START_TEXT|>"
    TEXT_END_TOKEN = "<|END_TEXT|>"
    # When the model decides to call tools instead of producing a final text
    # block, it emits an action block instead of a text block. The reasoning
    # parser must leave that block intact so the downstream tool-call parser
    # can pick it up.
    ACTION_START_TOKEN = "<|START_ACTION|>"

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = True,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        # The chat template puts <|START_THINKING|> in the assistant prefix
        # when reasoning is enabled, so the *generated* text usually starts
        # already inside thinking. ``force_reasoning=True`` makes the base
        # detector treat the leading bytes as reasoning even though the
        # generated stream typically does not echo <|START_THINKING|>.
        super().__init__(
            think_start_token="<|START_THINKING|>",
            think_end_token="<|END_THINKING|>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            force_nonempty_content=force_nonempty_content,
        )
        # Streaming state machine. The model emits, in order:
        #   1. reasoning  (between START_THINKING [in prefix] and END_THINKING)
        #   2. either ``<|START_TEXT|>...<|END_TEXT|>`` (final answer) or
        #      ``<|START_ACTION|>...<|END_ACTION|>`` (tool calls) -- never both.
        # When ``reasoning=False`` the chat template emits both START/END
        # thinking in the prefix and step 1 is empty; the generated stream
        # then starts directly with the text or action block.
        self._reasoning_done = False
        self._saw_text_start = False
        self._saw_text_end = False
        self._in_action_mode = False

    @classmethod
    def _strip_text_markers(cls, raw: str) -> str:
        """Extract the substring between ``<|START_TEXT|>`` and
        ``<|END_TEXT|>``. If ``<|START_TEXT|>`` is absent but a
        ``<|START_ACTION|>`` block is present, the model produced a tool
        call instead of a text answer -- return the raw text untouched so
        the downstream tool-call parser can pick up the action block. If
        neither marker is present (ran out of tokens still inside
        thinking) return ``""``. If ``<|END_TEXT|>`` is absent (stop token
        or max_new_tokens cut the stream off inside the text block) return
        everything after ``<|START_TEXT|>``.
        """
        if not raw:
            return ""
        s = raw.find(cls.TEXT_START_TOKEN)
        if s == -1:
            if cls.ACTION_START_TOKEN in raw:
                return raw
            return ""
        s += len(cls.TEXT_START_TOKEN)
        tail = raw[s:]
        e = tail.find(cls.TEXT_END_TOKEN)
        if e == -1:
            return tail
        return tail[:e]

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        # Direct parse: split on the (single) ``<|END_THINKING|>`` token if
        # present. Anything before is reasoning, anything after is the
        # final-text block. If no END_THINKING but a START_TEXT exists,
        # we're in the reasoning=False case (chat template emitted both
        # START/END thinking in the prefix; the model only generated the
        # text block). Otherwise the model exhausted tokens still thinking
        # and ``normal_text`` ends up empty -- matching the convention of
        # the other detectors in this module (DeepSeekR1, Qwen3, ...). The
        # empty content is propagated as ``message.content = None`` by
        # serving_chat, and downstream code is expected to treat that as
        # "no answer" rather than falling back to ``reasoning_content``.
        end_think_idx = text.find(self.think_end_token)
        text_start_idx = text.find(self.TEXT_START_TOKEN)
        action_start_idx = text.find(self.ACTION_START_TOKEN)
        if end_think_idx != -1:
            reasoning = text[:end_think_idx]
            rest = text[end_think_idx + len(self.think_end_token) :]
        elif text_start_idx != -1:
            reasoning = text[:text_start_idx]
            rest = text[text_start_idx:]
        elif action_start_idx != -1:
            # reasoning=False + tool call: chat template emitted both
            # START/END thinking in the prefix, the model only generated
            # an action block. Treat the prefix before the action block as
            # (probably empty) reasoning so the action block reaches the
            # tool-call parser intact.
            reasoning = text[:action_start_idx]
            rest = text[action_start_idx:]
        else:
            reasoning = text
            rest = ""

        # Some checkpoints echo the START_THINKING token even though the
        # chat template put it in the prefix; drop it if so.
        think_start_text = self.think_start_token + self.think_start_self_label
        if reasoning.startswith(think_start_text):
            reasoning = reasoning[len(think_start_text) :]

        return self._maybe_apply_force_nonempty_content(
            StreamingParseResult(
                normal_text=self._strip_text_markers(rest),
                reasoning_text=reasoning,
            )
        )

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        """Streaming parse. Custom state machine -- we don't reuse the base
        class because Cohere's "reasoning=False" path (the model emits no
        ``<|END_THINKING|>``, just goes straight to a text or action block)
        is fundamentally incompatible with the base detector's
        ``force_reasoning`` semantics."""
        self._buffer += new_text
        buf = self._buffer

        if not self._reasoning_done:
            # Look for any marker that ends reasoning: an explicit
            # END_THINKING, or an implicit transition via the start of the
            # final-text or action block (reasoning=False case).
            markers = (
                (self.think_end_token, "think_end"),
                (self.TEXT_START_TOKEN, "text"),
                (self.ACTION_START_TOKEN, "action"),
            )
            first_pos = None
            first_marker = None
            first_kind = None
            for marker_text, kind in markers:
                p = buf.find(marker_text)
                if p != -1 and (first_pos is None or p < first_pos):
                    first_pos, first_marker, first_kind = p, marker_text, kind
            if first_pos is None:
                # No marker seen yet. Stream the reasoning prefix, but keep
                # enough tail in the buffer to recognise a marker split
                # across chunk boundaries.
                if not self.stream_reasoning:
                    return StreamingParseResult()
                max_keep = max(len(m) for m, _ in markers) - 1
                if len(buf) > max_keep:
                    head = buf[:-max_keep]
                    self._buffer = buf[-max_keep:]
                    return StreamingParseResult(reasoning_text=head)
                return StreamingParseResult()

            reasoning_chunk = buf[:first_pos]
            if first_kind == "think_end":
                self._buffer = buf[first_pos + len(first_marker) :]
            else:
                # Implicit reasoning-end: leave the start-of-block marker in
                # the buffer for the post-thinking branch below to consume.
                self._buffer = buf[first_pos:]
            self._reasoning_done = True
            if reasoning_chunk:
                return StreamingParseResult(reasoning_text=reasoning_chunk)
            buf = self._buffer

        # Reasoning is closed. Decide between text-stripping and
        # action-passthrough on first sight of a marker.
        if self._in_action_mode:
            if not buf:
                return StreamingParseResult()
            self._buffer = ""
            return StreamingParseResult(normal_text=buf)

        if not self._saw_text_start:
            s_text = buf.find(self.TEXT_START_TOKEN)
            s_action = buf.find(self.ACTION_START_TOKEN)
            picks = [
                (p, k) for p, k in ((s_text, "text"), (s_action, "action")) if p != -1
            ]
            if not picks:
                max_keep = (
                    max(len(self.TEXT_START_TOKEN), len(self.ACTION_START_TOKEN)) - 1
                )
                if len(buf) > max_keep:
                    self._buffer = buf[-max_keep:]
                return StreamingParseResult()
            picks.sort()
            first_pos, first_kind = picks[0]
            if first_kind == "action":
                self._in_action_mode = True
                out_normal = buf[first_pos:]
                self._buffer = ""
                return StreamingParseResult(normal_text=out_normal)
            # Found <|START_TEXT|>. Drop everything up to and including the
            # marker -- text content streams next.
            self._buffer = buf[first_pos + len(self.TEXT_START_TOKEN) :]
            self._saw_text_start = True
            buf = self._buffer

        if self._saw_text_start and not self._saw_text_end:
            e = buf.find(self.TEXT_END_TOKEN)
            if e == -1:
                # Emit everything except a possible partial END_TEXT tail.
                keep = len(self.TEXT_END_TOKEN) - 1
                if len(buf) > keep:
                    out_normal = buf[:-keep]
                    self._buffer = buf[-keep:]
                    return StreamingParseResult(normal_text=out_normal)
                return StreamingParseResult()
            out_normal = buf[:e]
            self._buffer = buf[e + len(self.TEXT_END_TOKEN) :]
            self._saw_text_end = True
            return StreamingParseResult(normal_text=out_normal)

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
        "apertus2509": Apertus2509Detector,
        "deepseek-r1": DeepSeekR1Detector,
        "deepseek-v3": _DeepSeekV3Detector,
        "deepseek-v4": _DeepSeekV3Detector,
        "glm45": Glm45Detector,
        "hunyuan": HunyuanDetector,
        "gpt-oss": GptOssDetector,
        "kimi": KimiDetector,
        "kimi_k2": KimiK2Detector,
        "mimo": _MimoDetector,
        "poolside_v1": _PoolsideV1Detector,
        "qwen3": Qwen3Detector,
        "qwen3-thinking": Qwen3Detector,
        "minimax": Qwen3Detector,
        "minimax-append-think": MiniMaxAppendThinkDetector,
        "minimax-m3": MiniMaxM3Detector,
        "step3": DeepSeekR1Detector,
        "step3p5": DeepSeekR1Detector,
        "mistral": MistralDetector,
        "nemotron_3": Nemotron3Detector,
        "interns1": Qwen3Detector,
        "gemma4": Gemma4Detector,
        "inkling": InklingDetector,
        "cohere_command4": CohereCommand4Detector,
    }

    def __init__(
        self,
        model_type: Optional[str] = None,
        stream_reasoning: bool = True,
        force_reasoning: Optional[bool] = None,
        request: ChatCompletionRequest = None,
        tokenizer=None,
    ):
        if not model_type:
            raise ValueError("Model type must be specified")

        detector_class = self.DetectorMap.get(model_type.lower())
        if not detector_class:
            raise ValueError(f"Unsupported model type: {model_type}")

        chat_template_kwargs = getattr(request, "chat_template_kwargs", None) or {}

        # Special cases where we override force_reasoning
        if model_type.lower() in {
            "qwen3-thinking",
            "gpt-oss",
            "minimax",
        }:
            force_reasoning = True

        # M3 consumes the <mm:think> start tag only for thinking_mode=enabled
        # (absent from output → must force); mirror serving_chat's M3 branch.
        if model_type.lower() == "minimax-m3" and force_reasoning is None:
            force_reasoning = chat_template_kwargs.get("thinking_mode") == "enabled"

        # Only pass force_reasoning if explicitly set, let detectors use their defaults
        kwargs = {"stream_reasoning": stream_reasoning}
        if force_reasoning is not None:
            kwargs["force_reasoning"] = force_reasoning

        if (
            request is not None
            and isinstance(request, ChatCompletionRequest)
            and request.continue_final_message
            and request.messages[-1].role == "assistant"
        ):
            kwargs["continue_final_message"] = True
            kwargs["previous_content"] = request.messages[-1].content

        if chat_template_kwargs.get("force_nonempty_content") is True:
            kwargs["force_nonempty_content"] = True

        if tokenizer is not None:
            sig = inspect.signature(detector_class)
            if "tokenizer" in sig.parameters:
                kwargs["tokenizer"] = tokenizer

        self.detector = detector_class(**kwargs)

    def parse_non_stream(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Non-streaming call: one-time parsing"""
        ret = self.detector.detect_and_parse(full_text)
        return ret.reasoning_text, ret.normal_text

    def parse_non_stream_blocks(self, full_text: str) -> list[dict]:
        """Non-streaming call: return an ordered sequence of reasoning/text blocks"""
        if hasattr(self.detector, "detect_and_parse_block_sequence"):
            seq = self.detector.detect_and_parse_block_sequence(full_text)
            return [{"type": k, "text": t} for k, t in seq]

        ret = self.detector.detect_and_parse(full_text)
        blocks: list[dict] = []
        if ret.reasoning_text:
            blocks.append({"type": "reasoning", "text": ret.reasoning_text})
        blocks.append({"type": "text", "text": ret.normal_text or ""})
        return blocks

    def parse_stream_chunk(
        self, chunk_text: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Streaming call: incremental parsing"""
        ret = self.detector.parse_streaming_increment(chunk_text)
        return ret.reasoning_text, ret.normal_text

    def parse_stream_end(self) -> Tuple[Optional[str], Optional[str]]:
        """Streaming call: flush any detector-specific buffered state once
        the stream ends."""
        ret = self.detector.finish()
        return ret.reasoning_text, ret.normal_text
