from __future__ import annotations

import logging
import math

from sglang.srt.entrypoints.chat_input.types import ChatModelConfig
from sglang.srt.entrypoints.openai import chat_encoding
from sglang.srt.parser.reasoning_parser import ReasoningParser

logger = logging.getLogger(__name__)


def resolve_chat_model_config(
    *,
    tokenizer,
    template_manager,
    hf_config,
    model_path,
    revision,
    tool_call_parser,
    reasoning_parser,
    default_chat_template_kwargs,
    is_multimodal,
) -> ChatModelConfig:
    spec = chat_encoding.resolve_chat_encoding_spec(
        hf_config=hf_config, tokenizer=tokenizer, tool_call_parser=tool_call_parser
    )
    detector = None
    if reasoning_parser:
        try:
            detector = ReasoningParser(
                model_type=reasoning_parser, stream_reasoning=True, tokenizer=tokenizer
            ).detector
        except ValueError as error:
            logger.warning(
                "Failed to initialize reasoning detector for parser '%s': %s",
                reasoning_parser,
                error,
            )
    profile = (
        chat_encoding.resolve_dsv4_reasoning_effort_profile(
            model_path=model_path,
            revision=revision,
            override=hf_config.to_dict().get(
                chat_encoding.DSV4_REASONING_EFFORT_PROFILE_OVERRIDE
            ),
        )
        if spec == "dsv4"
        else None
    )
    inkling_effort = (
        get_inkling_default_reasoning_effort() if spec == "inkling" else None
    )
    try:
        auto_adds_specials = len(tokenizer.encode("")) > 0
    except Exception:
        auto_adds_specials = True
    return ChatModelConfig(
        tokenizer=tokenizer,
        template_manager=template_manager,
        is_multimodal=is_multimodal,
        model_name=model_path,
        chat_encoding_spec=spec,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        reasoning_detector=detector,
        default_chat_template_kwargs=default_chat_template_kwargs or {},
        dsv4_reasoning_effort_profile=profile,
        inkling_default_reasoning_effort=inkling_effort,
        tokenizer_auto_adds_specials=auto_adds_specials,
        is_gpt_oss=getattr(hf_config, "model_type", None) == "gpt_oss",
        is_gemma4=getattr(hf_config, "model_type", None)
        in ("gemma4", "gemma4_unified"),
    )


def get_inkling_default_reasoning_effort() -> float:
    """Read the default Inkling reasoning effort from the environment."""
    from sglang.srt.environ import envs

    val = envs.SGLANG_INKLING_DEFAULT_REASONING_EFFORT.get()
    if not val:
        return 0.9
    try:
        parsed = float(val)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "SGLANG_INKLING_DEFAULT_REASONING_EFFORT must be numeric"
        ) from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 0.99:
        raise ValueError(
            "SGLANG_INKLING_DEFAULT_REASONING_EFFORT must be in [0.0, 0.99]"
        )
    return parsed
