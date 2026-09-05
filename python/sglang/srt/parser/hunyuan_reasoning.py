from typing import Optional

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.parser.template_detection import ReasoningToggleConfig


def uses_hunyuan_reasoning_effort(
    reasoning_parser: Optional[str], reasoning_config: Optional[ReasoningToggleConfig]
) -> bool:
    return (
        reasoning_parser == "hunyuan"
        and reasoning_config is not None
        and reasoning_config.special_case == "hunyuan_effort"
    )


def normalize_hunyuan_reasoning_effort(
    request: ChatCompletionRequest,
    reasoning_parser: Optional[str],
    reasoning_config: Optional[ReasoningToggleConfig],
) -> None:
    if not uses_hunyuan_reasoning_effort(reasoning_parser, reasoning_config):
        return

    effort = request.reasoning_effort
    if effort is None and request.chat_template_kwargs is not None:
        effort = request.chat_template_kwargs.get("reasoning_effort")
    if effort is None:
        normalized_effort = "high"
    elif effort in ("none", "no_think"):
        normalized_effort = "no_think"
    elif effort in ("minimal", "low"):
        normalized_effort = "low"
    elif effort in ("medium", "high", "xhigh", "max"):
        normalized_effort = "high"
    else:
        raise ValueError(
            "Hunyuan reasoning_effort must be one of none, minimal, low, "
            "medium, high, xhigh, or max"
        )

    request.reasoning_effort = normalized_effort
    if request.chat_template_kwargs is not None:
        request.chat_template_kwargs.pop("reasoning_effort", None)
