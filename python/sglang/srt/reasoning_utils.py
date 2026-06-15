# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)
_REASONING_DEFAULT_UNSET = object()


def _resolve_by_mode(
    mode: Optional[str],
    chat_template_kwargs: Optional[dict[str, Any]],
    reasoning_effort: Optional[str],
) -> bool:
    if mode is None:
        return False
    if mode == "always":
        return True
    if mode == "mistral":
        return reasoning_effort is not None and reasoning_effort != "none"
    if mode in ("thinking", "enable_thinking"):
        return not chat_template_kwargs or chat_template_kwargs.get(mode) is not False
    if mode in ("explicit_thinking", "explicit_enable_thinking"):
        toggle = mode.replace("explicit_", "")
        return (
            chat_template_kwargs is not None
            and chat_template_kwargs.get(toggle) is True
        )
    logger.warning(
        "Unknown reasoning_default mode '%s', defaulting to reasoning disabled",
        mode,
    )
    return False


def resolve_require_reasoning(
    reasoning_parser: Optional[str],
    chat_template_kwargs: Optional[dict[str, Any]] = None,
    reasoning_effort: Optional[str] = None,
    reasoning_config: Optional[Any] = None,
    reasoning_default: Optional[str] = _REASONING_DEFAULT_UNSET,
) -> bool:
    """Return whether this request should use reasoning-aware generation.

    This is the single place that maps model/parser-specific chat-template
    switches to GenerateReqInput.require_reasoning. Callers should pass through
    the original request semantics rather than duplicating these rules.
    """
    if not reasoning_parser:
        return False

    if reasoning_parser == "hunyuan":
        return reasoning_effort not in (None, "none", "no_think")

    if reasoning_config is not None:
        special_case = getattr(reasoning_config, "special_case", None)
        if special_case == "always":
            return True
        if special_case == "mistral":
            return reasoning_effort is not None and reasoning_effort != "none"

        toggle_param = getattr(reasoning_config, "toggle_param", None)
        default_enabled = getattr(reasoning_config, "default_enabled", None)
        if toggle_param is None or default_enabled is None:
            return False
        if default_enabled:
            return (
                not chat_template_kwargs
                or chat_template_kwargs.get(toggle_param) is not False
            )
        return (
            chat_template_kwargs is not None
            and chat_template_kwargs.get(toggle_param) is True
        )

    if reasoning_default is not _REASONING_DEFAULT_UNSET:
        return _resolve_by_mode(
            reasoning_default, chat_template_kwargs, reasoning_effort
        )

    if reasoning_parser == "deepseek-v3":
        return (
            chat_template_kwargs is not None
            and chat_template_kwargs.get("thinking") is True
        )
    if reasoning_parser == "gemma4":
        return (
            chat_template_kwargs is not None
            and chat_template_kwargs.get("enable_thinking") is True
        )
    if reasoning_parser in ["kimi_k2"]:
        return (
            not chat_template_kwargs
            or chat_template_kwargs.get("thinking") is not False
        )
    if reasoning_parser in ["qwen3", "glm45", "nemotron_3", "interns1"]:
        return (
            not chat_template_kwargs
            or chat_template_kwargs.get("enable_thinking") is not False
        )
    if reasoning_parser in ["mimo"]:
        return (
            chat_template_kwargs is not None
            and chat_template_kwargs.get("enable_thinking") is True
        )
    if reasoning_parser == "mistral":
        return reasoning_effort is not None and reasoning_effort != "none"
    return True
