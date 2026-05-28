from typing import Dict, List, Optional, Tuple, Type

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
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
        think_excluded_tokens: Optional[List[str]] = None,
        force_reasoning: bool = False,
        stream_reasoning: bool = True,
        tool_start_token: Optional[str] = None,
        continue_final_message: bool = False,
        previous_content: str = "",
        thinks_internally: bool = False,
        reasoning_default: str = "always",
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

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses reasoning sections in the provided text.
        Returns both reasoning content and normal text separately.
        """
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
    ):
        # DeepSeek-R1 is assumed to be reasoning until `</think>` token
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=True,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
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
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            thinks_internally=True,
            reasoning_default="enable_thinking",
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
    ):
        super().__init__(
            "◁think▷",
            "◁/think▷",
            force_reasoning=False,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
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

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False):
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
    ):
        super().__init__(
            "<|channel|>analysis<|message|>",
            "<|end|>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
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

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        # scheduler.py need `reasoning_parser.detector.think_end_token`
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
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
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            reasoning_default="enable_thinking",
        )
        self._force_nonempty_content = force_nonempty_content

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        ret = super().detect_and_parse(text)
        if self._force_nonempty_content and not ret.normal_text:
            ret.normal_text, ret.reasoning_text = ret.reasoning_text, ret.normal_text
        return ret


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
    ):
        super().__init__(
            "[THINK]",
            "[/THINK]",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            reasoning_default="mistral",
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
    ):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<tool_calls>",
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )


class Gemma4Detector(BaseReasoningFormatDetector):
    """Gemma4 reasoning detector."""

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        super().__init__(
            "<|channel>",
            "<channel|>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            reasoning_default="explicit_enable_thinking",
        )
        self.think_start_self_label = "thought\n"


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
        ``normal_text`` is the empty string -- mirrors what vLLM's
        ``cohere_command4`` reasoning parser produces.

    Matches the public token names from the model's
    ``special_tokens_map.json`` (``<|START_THINKING|>`` etc.).
    """

    TEXT_START_TOKEN = "<|START_TEXT|>"
    TEXT_END_TOKEN = "<|END_TEXT|>"

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = True,
        continue_final_message: bool = False,
        previous_content: str = "",
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
        )
        # Streaming state for the final-text block: once we've stepped past
        # <|END_THINKING|> we still need to strip the surrounding
        # <|START_TEXT|>...<|END_TEXT|> markers from the normal-text stream.
        self._saw_text_start = False
        self._saw_text_end = False
        self._normal_buffer = ""

    @classmethod
    def _strip_text_markers(cls, raw: str) -> str:
        """Extract the substring between ``<|START_TEXT|>`` and
        ``<|END_TEXT|>``. If ``<|START_TEXT|>`` is absent (model never
        produced a final text block, e.g. ran out of tokens still thinking)
        return ``""``. If ``<|END_TEXT|>`` is absent (stop token or
        max_new_tokens cut the stream off inside the text block) return
        everything after ``<|START_TEXT|>``.
        """
        if not raw:
            return ""
        s = raw.find(cls.TEXT_START_TOKEN)
        if s == -1:
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
        if end_think_idx != -1:
            reasoning = text[:end_think_idx]
            rest = text[end_think_idx + len(self.think_end_token) :]
        elif text_start_idx != -1:
            reasoning = text[:text_start_idx]
            rest = text[text_start_idx:]
        else:
            reasoning = text
            rest = ""

        # Some checkpoints echo the START_THINKING token even though the
        # chat template put it in the prefix; drop it if so.
        think_start_text = self.think_start_token + self.think_start_self_label
        if reasoning.startswith(think_start_text):
            reasoning = reasoning[len(think_start_text) :]

        return StreamingParseResult(
            normal_text=self._strip_text_markers(rest),
            reasoning_text=reasoning,
        )

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        # Let the base detector route bytes into reasoning vs normal first.
        result = super().parse_streaming_increment(new_text)
        # While we're still inside the reasoning block (or the base detector
        # decided this chunk is purely reasoning) nothing to strip.
        if result.reasoning_text and not result.normal_text:
            return result

        normal_inc = result.normal_text or ""
        if not normal_inc:
            return result

        out_normal = ""
        # Append the new normal-stream slice to our buffer and walk through
        # the START_TEXT / END_TEXT markers.
        self._normal_buffer += normal_inc

        if not self._saw_text_start:
            s = self._normal_buffer.find(self.TEXT_START_TOKEN)
            if s == -1:
                # The buffer may contain a partial START_TEXT marker at the
                # tail; keep the last (len-1) chars in case the next chunk
                # completes the marker.
                keep = len(self.TEXT_START_TOKEN) - 1
                if len(self._normal_buffer) > keep:
                    self._normal_buffer = self._normal_buffer[-keep:]
                result.normal_text = ""
                return result
            # Drop everything up to and including the START_TEXT marker.
            self._normal_buffer = self._normal_buffer[s + len(self.TEXT_START_TOKEN) :]
            self._saw_text_start = True

        if self._saw_text_start and not self._saw_text_end:
            e = self._normal_buffer.find(self.TEXT_END_TOKEN)
            if e == -1:
                # Emit everything except possible partial END_TEXT tail.
                keep = len(self.TEXT_END_TOKEN) - 1
                if len(self._normal_buffer) > keep:
                    out_normal = self._normal_buffer[:-keep]
                    self._normal_buffer = self._normal_buffer[-keep:]
                else:
                    out_normal = ""
            else:
                out_normal = self._normal_buffer[:e]
                self._normal_buffer = self._normal_buffer[e + len(self.TEXT_END_TOKEN) :]
                self._saw_text_end = True

        result.normal_text = out_normal
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
        "step3": DeepSeekR1Detector,
        "step3p5": DeepSeekR1Detector,
        "mistral": MistralDetector,
        "nemotron_3": Nemotron3Detector,
        "interns1": Qwen3Detector,
        "gemma4": Gemma4Detector,
        "cohere_command4": CohereCommand4Detector,
    }

    def __init__(
        self,
        model_type: Optional[str] = None,
        stream_reasoning: bool = True,
        force_reasoning: Optional[bool] = None,
        request: ChatCompletionRequest = None,
    ):
        if not model_type:
            raise ValueError("Model type must be specified")

        detector_class = self.DetectorMap.get(model_type.lower())
        if not detector_class:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Special cases where we override force_reasoning
        if model_type.lower() in {
            "qwen3-thinking",
            "gpt-oss",
            "minimax",
        }:
            force_reasoning = True

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

        chat_template_kwargs = getattr(request, "chat_template_kwargs", None) or {}
        if chat_template_kwargs.get("force_nonempty_content") is True:
            kwargs["force_nonempty_content"] = True

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
