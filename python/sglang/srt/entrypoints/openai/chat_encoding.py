"""Single home for the chat-encoding dispatch.

Which encoder turns chat messages into prompt tokens is a property of the
model, so the serving path and offline tools (benchmarks, evals) must resolve
it here instead of re-deriving it from model architectures themselves.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from sglang.srt.entrypoints.openai import encoding_dsv4, encoding_dsv41

logger = logging.getLogger(__name__)

DSV4_REASONING_EFFORT_PROFILE_OVERRIDE = "dsv4_reasoning_effort_profile"
_DSV4_REASONING_EFFORT_ENCODER = "encoding/encoding_dsv4.py"
_MAX_DSV4_ENCODER_BYTES = 1 << 20


def _detect_dsv4_reasoning_effort_profile(
    model_path: str, revision: Optional[str] = None
) -> Optional[str]:
    encoder_path = Path(model_path) / _DSV4_REASONING_EFFORT_ENCODER
    try:
        if not encoder_path.is_file():
            from huggingface_hub import hf_hub_download

            encoder_path = Path(
                hf_hub_download(
                    model_path,
                    _DSV4_REASONING_EFFORT_ENCODER,
                    revision=revision,
                )
            )
        if encoder_path.stat().st_size > _MAX_DSV4_ENCODER_BYTES:
            return None
        tree = ast.parse(encoder_path.read_text(encoding="utf-8"))
    except Exception as error:
        logger.debug(
            "Could not inspect DeepSeek-V4 checkpoint encoder at %s: %s",
            encoder_path,
            error,
        )
        return None

    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue

        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                assignments[target.id] = ast.literal_eval(value)
            except (TypeError, ValueError):
                continue

    prompts = assignments.get("REASONING_EFFORT_PROMPTS")
    if (
        assignments.get("DEFAULT_REASONING_EFFORT") == "low"
        and isinstance(prompts, dict)
        and {"low", "high", "max"} <= prompts.keys()
    ):
        return "official"
    if "REASONING_EFFORT_MAX" in assignments:
        return "preview"
    return None


def _validate_dsv4_reasoning_effort_profile(profile: str) -> str:
    if profile not in encoding_dsv4.REASONING_EFFORT_PROFILES:
        raise ValueError(
            f"Invalid {DSV4_REASONING_EFFORT_PROFILE_OVERRIDE}: {profile!r}; "
            f"expected one of {list(encoding_dsv4.REASONING_EFFORT_PROFILES)}"
        )
    return profile


def resolve_dsv4_reasoning_effort_profile(
    *,
    model_path: str,
    revision: Optional[str] = None,
    override: Optional[str] = None,
) -> str:
    if override is not None:
        return _validate_dsv4_reasoning_effort_profile(override)

    return (
        _detect_dsv4_reasoning_effort_profile(
            model_path=model_path,
            revision=revision,
        )
        or "preview"
    )


def is_deepseek_v41_config(*, arch: str, model_type: str) -> bool:
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

    if is_deepseek_v41_config(arch=arch, model_type=hf_config.model_type):
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


def resolve_dsv41_reasoning_effort(value: Any) -> Union[str, int, None]:
    """Map an API ``reasoning_effort`` onto what the V4.1 encoder accepts.

    Tiers pass through; an OpenAI float in [0, 0.99] becomes a 1-100 budget;
    an int budget (only reachable via ``chat_template_kwargs``) passes through
    when in range. None means unsupported and the caller applies its default.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1 <= value <= 100 else None
    if isinstance(value, float):
        return min(100, max(1, round(value * 100)))
    if value in encoding_dsv41.REASONING_EFFORT_MAPPINGS:
        return value
    return None


_OPENAI_FUNCTION_FIELD_ORDER = ("name", "description", "parameters")


def dsv41_tool_payload(tool: Any) -> Dict[str, Any]:
    """The tool dict the V4.1 encoder renders verbatim into the prompt.

    Only fields the client sent, in the OpenAI field order; pydantic would
    otherwise add defaults (strict=false) and reorder keys by declaration.
    """
    payload = tool.model_dump(exclude_unset=True, exclude_none=True)
    function = dict(payload.get("function") or {})
    ordered = {
        k: function.pop(k) for k in _OPENAI_FUNCTION_FIELD_ORDER if k in function
    }
    ordered.update(function)
    payload["function"] = ordered
    return payload


def default_dsv41_reasoning_effort_from_env(raw: str) -> Union[str, int]:
    """Parse ``SGLANG_DSV41_REASONING_EFFORT``; raises so a bad value fails at boot."""
    value: Any = int(raw) if raw.strip().isdigit() else raw.strip()
    effort = resolve_dsv41_reasoning_effort(value)
    if effort is None:
        raise ValueError(
            f"Invalid SGLANG_DSV41_REASONING_EFFORT={raw!r}; expected one of "
            f"{list(encoding_dsv41.REASONING_EFFORT_MAPPINGS)} or an integer in [1, 100]"
        )
    return effort


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
    pipeline in ``serving_chat``. System-message handling matches
    ``serving_chat``: dsv4/dsv32 get an empty one prepended, dsv41 does not
    (it renders a system token even for empty content).
    """
    if spec == "inkling":
        from sglang.srt.parser.inkling_renderer import render_inkling_messages
        from sglang.srt.parser.inkling_tokenizer import InklingTokenizer

        return render_inkling_messages(
            messages,
            InklingTokenizer(tokenizer=tokenizer),
            add_generation_prompt=False,
        )

    if spec in ("dsv4", "dsv32", "dsv41"):
        if spec != "dsv41" and messages and messages[0]["role"] != "system":
            messages = [{"role": "system", "content": ""}] + list(messages)
        if spec == "dsv4":
            from sglang.srt.entrypoints.openai import encoding_dsv4

            real_input = encoding_dsv4.encode_messages(
                messages, thinking_mode=thinking_mode
            )
        elif spec == "dsv41":
            real_input = encoding_dsv41.encode_messages(
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
