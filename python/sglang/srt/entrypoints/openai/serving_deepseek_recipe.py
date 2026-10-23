"""DeepSeek-V4 / V4.1 serving glue for the ``dsv4`` / ``dsv41`` chat encoding
specs: prompt encoding through deepseek-recipe under SGLang's request semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from sglang.srt.environ import envs
from sglang.srt.parser.deepseek_recipe_renderer import (
    RecipeEncoder,
    parse_effort_budget,
)
from sglang.srt.utils import ImageData

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

DEFAULT_EFFORT_ENVS = {
    "dsv4": envs.SGLANG_DSV4_REASONING_EFFORT,
    "dsv41": envs.SGLANG_DSV41_REASONING_EFFORT,
}


class DeepSeekRecipeChatEncodingAdapter:
    # recipe sorts tool results by call order, so images come from its render.
    collects_openai_media = False
    output = None
    strict_tool_call_arguments = False  # recipe serializes object arguments itself.

    def __init__(
        self,
        *,
        chat_encoding_spec: str,
        tokenizer: Any,
        default_chat_template_kwargs: Optional[dict[str, Any]],
    ):
        self._spec = chat_encoding_spec
        self.encoder = RecipeEncoder.from_hf_tokenizer(chat_encoding_spec, tokenizer)
        raw = (DEFAULT_EFFORT_ENVS[chat_encoding_spec].get() or "").strip()
        self._default_effort = int(raw) if raw.isdigit() else raw or None
        # Encoded at boot, so a bad server default fails startup, not requests.
        defaults = default_chat_template_kwargs or {}
        self._encode(
            {"messages": [{"role": "user", "content": ""}], "thinking": True},
            effort=defaults.get("reasoning_effort"),
        )

    def encode_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        request: ChatCompletionRequest,
        tools: Optional[list[dict]],
        thinking: bool,
        image_data: list,
    ) -> list[int]:
        # The wire request: the handler's message dumps carry None for unset fields.
        wire = request.model_dump(exclude_unset=True, by_alias=True, mode="json")
        wire_messages = wire["messages"]
        continue_final = request.continue_final_message
        task_prefix = None
        # SGLang's trailing-assistant rule, as `_handle_last_assistant_message`.
        last = wire_messages[-1] if wire_messages else None
        if (
            last
            and last["role"] == "assistant"
            and isinstance(last.get("content"), str)
        ):
            if not continue_final:
                wire_messages[-1] = {"role": "user", "content": last["content"]}
            elif request.task is not None:
                # A task renders its own tail, so the prefix follows it.
                task_prefix = wire_messages.pop()["content"]
                continue_final = False
        body = {
            "messages": wire_messages,
            "thinking": thinking,
            "continue_final_message": continue_final,
            "response_format": wire.get("response_format"),
        }
        if tools is not None:
            # `model_dump` adds None for unset fields, which recipe renders as values.
            body["tools"] = [
                {k: v for k, v in tool.items() if v is not None}
                | {
                    "function": {
                        k: v for k, v in tool["function"].items() if v is not None
                    }
                }
                for tool in tools
            ]
            body["tool_choice"] = wire.get("tool_choice")
        body = {key: value for key, value in body.items() if value is not None}
        image_urls = []
        prompt_ids = self._encode(
            body,
            effort=request.reasoning_effort,
            task=request.task,
            image_urls=image_urls,
        )
        image_data.extend(ImageData(url=url) for url in image_urls)
        if task_prefix:
            prompt_ids += self.encoder.encode_text(task_prefix)
        return prompt_ids

    def _encode(
        self,
        body: dict[str, Any],
        *,
        effort: Union[str, int, float, None],
        task: Optional[str] = None,
        image_urls: Optional[list[str]] = None,
    ) -> list[int]:
        if effort is None or effort == "none":
            effort = self._default_effort
        budget = None
        if effort is not None and not isinstance(effort, str):
            if self._spec != "dsv41":
                raise ValueError(
                    f"DeepSeek-V4 reasoning_effort must be a named tier, got {effort!r}"
                )
            budget = parse_effort_budget(effort)
        elif effort is not None:
            body["reasoning_effort"] = effort
        return self.encoder.encode(
            body, task=task, effort_budget=budget, image_urls=image_urls
        )
