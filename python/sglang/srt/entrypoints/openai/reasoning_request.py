"""Standalone helpers for interpreting reasoning controls on a chat request.

OpenAI-compatible frontends that pre-process requests outside the serving path
(e.g. token-in-token-out routers that render ``input_ids`` themselves) must
interpret the accepted reasoning spellings exactly as the server does.  These
helpers hold that logic unbound from ``ChatCompletionRequest`` and
``serving_chat`` so the server and external callers share one implementation.
"""

from typing import Dict, Optional

# The V4 official encoder only accepts these efforts; anything else (including
# OpenAI's default "medium") renders as None.
V4_ENCODER_REASONING_EFFORTS = ("max", "high")


def normalize_reasoning_inputs(values: Dict) -> Dict:
    """Fold the OpenAI ``reasoning`` object and ``reasoning_effort`` field into
    the ``chat_template_kwargs`` toggles the chat templates read.

    Mutates *values* (including its nested ``chat_template_kwargs``) in place
    and returns it.  ``ChatCompletionRequest.normalize_reasoning_inputs``
    delegates here, so this runs for every chat completion request.
    """
    r = values.get("reasoning")

    if r is not None and isinstance(r, dict):
        effort = r.get("effort") or r.get("reasoning_effort")
        if effort in {"none", "low", "medium", "high"}:
            values["reasoning_effort"] = effort

        enabled = (
            r.get("enabled") if r.get("enabled") is not None else r.get("enable", False)
        )
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in {"1", "true", "yes", "y", "on"}
        if enabled:
            ctk = values.get("chat_template_kwargs")
            if not isinstance(ctk, dict):
                ctk = {}
            # different models check different keys:
            # - "thinking" for deepseek-v3, kimi_k2
            # - "enable_thinking" for qwen3, glm45, nemotron_3, interns1, mimo
            ctk.setdefault("thinking", True)
            ctk.setdefault("enable_thinking", True)
            values["chat_template_kwargs"] = ctk

    if values.get("reasoning_effort") == "none":
        ctk = values.get("chat_template_kwargs")
        if not isinstance(ctk, dict):
            ctk = {}
        # different models check different keys:
        # - "thinking" for deepseek-v3, kimi_k2
        # - "enable_thinking" for qwen3, glm45, nemotron_3, interns1
        ctk.setdefault("thinking", False)
        ctk.setdefault("enable_thinking", False)
        values["chat_template_kwargs"] = ctk

    return values


def pop_reasoning_effort_kwarg(chat_template_kwargs: Optional[Dict]) -> Optional[str]:
    """Pop the ``chat_template_kwargs`` spelling of ``reasoning_effort``.

    The serving path promotes this spelling to the top-level field after model
    validation; the key must leave the kwargs so chat-template renderers never
    see it.
    """
    if not chat_template_kwargs:
        return None
    return chat_template_kwargs.pop("reasoning_effort", None)


def normalize_reasoning_request(values: Dict) -> Dict:
    """Apply the server's complete reasoning normalization to a raw request dict.

    Runs :func:`normalize_reasoning_inputs` and then the serving-path promotion
    of ``chat_template_kwargs["reasoning_effort"]``, in the server's order.
    Mutates *values* in place and returns it; callers that need the original
    request unchanged should pass a deep copy.
    """
    values = normalize_reasoning_inputs(values)
    effort = pop_reasoning_effort_kwarg(values.get("chat_template_kwargs"))
    if effort is not None:
        values["reasoning_effort"] = effort
    return values


def resolve_v4_reasoning_effort(effort: Optional[str]) -> Optional[str]:
    """The effort value the V4 official encoder receives: only ``max`` and
    ``high`` render; anything else renders as ``None``."""
    return effort if effort in V4_ENCODER_REASONING_EFFORTS else None
