"""Single home for the chat-encoding dispatch.

Which encoder turns chat messages into prompt tokens is a property of the
model, so the serving path and offline tools (benchmarks, evals) must resolve
it here instead of re-deriving it from model architectures themselves.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def resolve_chat_encoding_spec(
    *,
    hf_config: Any,
    tokenizer: Any,
    tool_call_parser: Optional[str] = None,
) -> Optional[str]:
    """Return the chat encoding spec for a model: "dsv4", "dsv32", "inkling", or None.

    None means the default path (HF chat template).
    """
    if tool_call_parser == "deepseekv4":
        return "dsv4"
    if tool_call_parser == "deepseekv32":
        return "dsv32"

    architectures = hf_config.architectures
    arch = architectures[0] if architectures else ""

    if "DeepseekV4" in arch:
        return "dsv4"

    # Inkling has no Jinja chat_template and uses a tiktoken base + a special-token
    # overlay + negative MM placeholders, so it can't go through apply_chat_template;
    # render input_ids directly via the Inkling renderer (serving_chat._encode_messages).
    if "InklingForConditionalGeneration" in arch:
        return "inkling"

    has_chat_template = tokenizer is not None and tokenizer.chat_template is not None
    if "DeepseekV3" in arch and not has_chat_template:
        return "dsv32"
    return None


def encode_simple_chat(
    *,
    tokenizer: Any,
    spec: Optional[str],
    messages: List[Dict[str, Any]],
    thinking_mode: str = "chat",
) -> List[int]:
    """Encode a plain-text chat conversation into prompt token ids.

    Minimal encode for offline tools: no tools, no multimodal content, no
    continue_final_message; the serving path keeps its full request-level
    pipeline in ``serving_chat``. Like
    ``serving_chat``, an empty system message is prepended when the
    conversation does not start with one (for the dsv4/dsv32 encoders this
    currently renders to zero tokens, but keeping the insertion explicit ties
    this helper to the serving semantics rather than to that coincidence).
    """
    if spec == "inkling":
        from sglang.srt.parser.inkling_renderer import render_inkling_messages
        from sglang.srt.parser.inkling_tokenizer import InklingTokenizer

        return render_inkling_messages(
            messages,
            InklingTokenizer(tokenizer=tokenizer),
            add_generation_prompt=False,
        )

    if spec in ("dsv4", "dsv32"):
        if messages and messages[0]["role"] != "system":
            messages = [{"role": "system", "content": ""}] + list(messages)
        if spec == "dsv4":
            from sglang.srt.entrypoints.openai import encoding_dsv4

            real_input = encoding_dsv4.encode_messages(
                messages, thinking_mode=thinking_mode
            )
        else:
            from sglang.srt.entrypoints.openai import encoding_dsv32

            real_input = encoding_dsv32.encode_messages(
                messages, thinking_mode=thinking_mode
            )
        return tokenizer.encode(real_input)

    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "This model has no HF chat template and no custom chat encoder; "
            f"cannot encode chat messages with {getattr(tokenizer, 'name_or_path', tokenizer)!r}."
        )
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
