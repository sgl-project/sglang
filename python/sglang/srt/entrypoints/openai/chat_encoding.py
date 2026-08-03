"""Single home for the chat-encoding dispatch.

Which encoder turns chat messages into prompt tokens is a property of the
model, so the serving path and offline tools (benchmarks, evals) must resolve
it here instead of re-deriving it from model architectures themselves.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from sglang.srt.entrypoints.openai import encoding_dsv4

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
            # Remote inspection needs Hugging Face Hub; local profiles do not.
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

    prompt_keys = set()
    default_effort = None
    has_legacy_max = False
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
            if target.id == "REASONING_EFFORT_PROMPTS" and isinstance(value, ast.Dict):
                prompt_keys = {
                    key.value
                    for key in value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
            elif (
                target.id == "DEFAULT_REASONING_EFFORT"
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                default_effort = value.value
            elif target.id == "REASONING_EFFORT_MAX":
                has_legacy_max = True

    if default_effort == "low" and {"low", "high", "max"} <= prompt_keys:
        return "0731"
    if has_legacy_max:
        return "legacy"
    return None


def resolve_dsv4_reasoning_effort_profile(
    *,
    model_path: str,
    revision: Optional[str] = None,
    override: Optional[str] = None,
) -> str:
    if override is not None:
        if override not in encoding_dsv4.REASONING_EFFORT_PROFILES:
            raise ValueError(
                f"Invalid {DSV4_REASONING_EFFORT_PROFILE_OVERRIDE}: {override!r}; "
                f"expected one of {list(encoding_dsv4.REASONING_EFFORT_PROFILES)}"
            )
        return override

    normalized_path = model_path.rstrip("/").lower()
    is_local = Path(model_path).is_dir()
    if is_local:
        detected_profile = _detect_dsv4_reasoning_effort_profile(
            model_path=model_path,
            revision=revision,
        )
        if detected_profile is not None:
            return detected_profile
    if "deepseek-v4-flash-0731" in normalized_path:
        return "0731"
    if (
        normalized_path.rsplit("/", 1)[-1]
        in {
            "deepseek-v4-flash",
            "deepseek-v4-flash-dspark",
        }
        and not is_local
    ):
        return "legacy"
    if is_local:
        return "legacy"

    return (
        _detect_dsv4_reasoning_effort_profile(
            model_path=model_path,
            revision=revision,
        )
        or "legacy"
    )


class Dsv4ReasoningEffortProfileResolver:
    def __init__(
        self,
        *,
        model_path: str,
        revision: Optional[str] = None,
        override: Optional[str] = None,
    ):
        self._override = override
        self._model_path = None
        self._profile = None
        self.resolve(model_path=model_path, revision=revision)

    def resolve(self, *, model_path: str, revision: Optional[str] = None) -> str:
        if model_path != self._model_path:
            self._profile = resolve_dsv4_reasoning_effort_profile(
                model_path=model_path,
                revision=revision,
                override=self._override,
            )
            self._model_path = model_path
            logger.info(
                "Resolved DeepSeek-V4 reasoning effort profile %r for %s",
                self._profile,
                model_path,
            )
        if self._profile is None:
            raise RuntimeError("DeepSeek-V4 reasoning effort profile was not resolved")
        return self._profile


def resolve_chat_encoding_spec(
    *,
    hf_config: Any,
    tokenizer: Any,
    tool_call_parser: Optional[str] = None,
) -> Optional[str]:
    """Return the chat encoding spec for a model.

    None means the default path (HF chat template).
    """
    if tool_call_parser == "deepseekv4":
        return "dsv4"
    if tool_call_parser == "deepseekv32":
        return "dsv32"
    if tool_call_parser == "kimi_k3":
        return "kimi_k3"

    architectures = hf_config.architectures
    arch = architectures[0] if architectures else ""

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
