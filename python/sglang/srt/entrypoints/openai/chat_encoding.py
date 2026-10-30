"""Single home for the chat-encoding dispatch.

Which encoder turns chat messages into prompt tokens is a property of the
model, so the serving path and offline tools (benchmarks, evals) must resolve
it here instead of re-deriving it from model architectures themselves.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def is_deepseek_v41_arch(*, arch: str, model_type: str) -> bool:
    """Check model_type before matching the DeepseekV4 architecture substring;
    V4.1 configs can also use the V4 architecture name.
    """
    return model_type == "deepseek_v41" or "DeepseekV41" in arch


def resolve_chat_encoding_spec(
    *,
    hf_config: Any,
    tokenizer: Any,
    tool_call_parser: Optional[str] = None,
) -> Optional[str]:
    """Return the chat encoding spec for a model.

    None means the default path (HF chat template); any non-None spec also owns
    reasoning-history rendering (:func:`spec_owns_reasoning_history`).
    """
    if tool_call_parser == "deepseekv41":
        return "dsv41"
    if tool_call_parser == "deepseekv4":
        return "dsv4"
    if tool_call_parser == "deepseekv32":
        return "dsv32"
    if tool_call_parser == "kimi_k3":
        return "kimi_k3"

    architectures = hf_config.architectures
    arch = architectures[0] if architectures else ""

    if is_deepseek_v41_arch(arch=arch, model_type=hf_config.model_type):
        return "dsv41"
    if "DeepseekV4" in arch:
        return "dsv4"
    if "KimiK3" in arch:
        return "kimi_k3"

    # Inkling has no Jinja chat_template and uses a tiktoken base + a special-token
    # overlay + negative MM placeholders, so it can't go through apply_chat_template;
    # render input_ids directly via the Inkling renderer (serving_chat._encode_messages).
    if "InklingForConditionalGeneration" in arch:
        return "inkling"

    has_chat_template = tokenizer is not None and tokenizer.chat_template is not None
    if "DeepseekV3" in arch and not has_chat_template:
        return "dsv32"
    return None


def spec_supports_inline_system(spec: Optional[str]) -> bool:
    """Native encoders verified to render mid-conversation system messages in place."""
    # K3's tokenizer renders in Python, so there is no Jinja source to probe.
    return spec == "kimi_k3"


def spec_owns_reasoning_history(spec: Optional[str]) -> bool:
    """Whether the encoder for ``spec`` renders assistant reasoning history itself.

    Custom encoders frame the reasoning and content channels, so history must be
    passed as assistant ``reasoning_content``. Splicing a detector's markers into
    content instead nests a reasoning block inside the content channel and leaves
    the real one empty, teaching the model to emit raw markers as visible text.

    Answered for the whole family rather than a list of specs, so a new spec gets
    the safe default: worst case is dropped history, not a leak.
    """
    return spec is not None


def spec_renders_prompt_ids(spec: Optional[str]) -> bool:
    """Whether the encoder for ``spec`` returns pre-tokenized prompt ids.

    Token-first encoders leave the text prompt empty; the MM processor
    expands their single placeholder ids rather than re-tokenizing text.
    """
    return spec in ("inkling", "kimi_k3", "dsv41")


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
    pipeline in ``serving_chat``. dsv32 gets an empty system message prepended,
    as ``serving_chat`` does; DeepSeek-V4 / V4.1 render through deepseek-recipe.
    """
    if spec == "inkling":
        from sglang.srt.parser.inkling_renderer import render_inkling_messages

        return render_inkling_messages(messages)

    if spec in ("dsv4", "dsv41"):
        from sglang.srt.parser.deepseek_recipe_renderer import DeepSeekRecipe

        deepseek_recipe = DeepSeekRecipe.from_hf_tokenizer(spec, tokenizer)
        return deepseek_recipe.encode(
            {"messages": list(messages), "thinking": thinking_mode == "thinking"}
        )

    if spec == "dsv32":
        from sglang.srt.entrypoints.openai import encoding_dsv32

        if messages and messages[0]["role"] != "system":
            messages = [{"role": "system", "content": ""}] + list(messages)
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
        messages, add_generation_prompt=True, tokenize=True, return_dict=False
    )
