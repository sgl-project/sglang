from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest


class InklingChatEncodingAdapter:
    """Inkling prompt encoding through ``inkling_renderer``; output stays with
    the registered ``inkling`` reasoning and tool-call parsers."""

    collects_openai_media = True
    output = None

    def __init__(self, *, tokenizer: Any):
        self._tokenizer = tokenizer
        # Resolved at boot: a misconfigured env default fails startup, not requests.
        self._default_reasoning_effort = (
            OpenAIServingChat._get_inkling_default_reasoning_effort()
        )

    def encode_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        request: ChatCompletionRequest,
        tools: Optional[list[dict]],
    ) -> list[int]:
        from sglang.srt.parser.inkling_renderer import render_inkling_messages
        from sglang.srt.parser.inkling_tokenizer import (
            CONTENT_TEXT,
            MESSAGE_MODEL,
            InklingTokenizer,
        )

        # Framing tokens + ONE placeholder per media (encoding/expansion happens later
        # in InklingMultimodalProcessor). The server's tokenizer is the base tiktoken
        # backend; wrap it so encode_special supplies the framing-token overlay.
        inkling_tokenizer = InklingTokenizer(tokenizer=self._tokenizer)
        reasoning_effort = OpenAIServingChat._parse_inkling_reasoning_effort(
            request.reasoning_effort
        )
        if reasoning_effort is None:
            reasoning_effort = self._default_reasoning_effort
        assistant_prefix = OpenAIServingChat._pop_inkling_assistant_prefix(
            messages, request
        )
        prompt_ids = render_inkling_messages(
            messages,
            inkling_tokenizer,
            add_generation_prompt=False,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )
        if assistant_prefix is not None:
            # Continue the final assistant message inside an OPEN model text
            # block: header + payload, no <|end_message|> and no
            # <|content_model_end_sampling|>, so the model resumes the turn.
            prompt_ids += [
                inkling_tokenizer.encode_special(MESSAGE_MODEL),
                inkling_tokenizer.encode_special(CONTENT_TEXT),
                *inkling_tokenizer.encode_text(assistant_prefix),
            ]
        return prompt_ids
