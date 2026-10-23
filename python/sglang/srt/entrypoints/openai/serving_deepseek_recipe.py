"""DeepSeek-V4 / V4.1 serving glue for the ``recipe`` backend of the ``dsv4`` /
``dsv41`` chat encoding specs (``--chat-encoding-backend recipe``): prompt encoding,
the thinking decision and the sampling overlay through deepseek-recipe."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.parser.deepseek_recipe_renderer import (
    REASONING_EFFORTS,
    RecipeEncoder,
    RecipeRender,
    overlay_sampling,
)

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

logger = logging.getLogger(__name__)

MODEL_VERSIONS = {"dsv4": "v4", "dsv41": "v41"}


class DeepSeekRecipeChatEncodingAdapter:
    """recipe renders and tokenizes the prompt, decides thinking and resolves the
    API's sampling defaults; output stays with the registered DeepSeek parsers."""

    collects_openai_media = True
    output = None
    strict_tool_call_arguments = False  # recipe serializes object arguments itself.

    def __init__(
        self,
        *,
        chat_encoding_spec: str,
        tokenizer: Any,
        default_chat_template_kwargs: Optional[dict[str, Any]],
    ):
        # Resolved at boot: a bad server default fails startup, not requests.
        defaults = default_chat_template_kwargs or {}
        effort = defaults.get("reasoning_effort")
        if effort is not None and effort not in REASONING_EFFORTS:
            raise ValueError(
                f"--default-chat-template-kwargs reasoning_effort={effort!r} is not "
                f"a deepseek-recipe effort ({', '.join(REASONING_EFFORTS)})"
            )
        self.encoder = RecipeEncoder.from_hf_tokenizer(
            MODEL_VERSIONS[chat_encoding_spec], tokenizer
        )
        self._default_kwargs = dict(defaults)

    def render(
        self,
        *,
        request: ChatCompletionRequest,
        tools: Optional[list[dict]],
        encode: bool = True,
    ) -> RecipeRender:
        """recipe's view of the request; `tools` is the handler's effective list."""
        # The wire request, not the handler's normalized messages.
        body = request.model_dump(exclude_unset=True, by_alias=True, mode="json")
        if tools is None:
            body.pop("tools", None)
            body.pop("tool_choice", None)
        else:
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
        return self.encoder.convert(body, self._default_kwargs, encode=encode)

    def encode_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        request: ChatCompletionRequest,
        tools: Optional[list[dict]],
    ) -> list[int]:
        return self.render(request=request, tools=tools).prompt_ids

    def thinking_enabled(self, *, request: ChatCompletionRequest) -> Optional[bool]:
        # The API defaults thinking on and reads its own `thinking` field, which
        # the handler's rule does not; tools do not change the decision.
        try:
            return self.render(request=request, tools=None, encode=False).thinking
        except ValueError as error:
            logger.debug("deepseek-recipe thinking decision skipped: %s", error)
            return None

    def apply_sampling(
        self,
        *,
        request: ChatCompletionRequest,
        sampling_params: dict[str, Any],
        prompt_tokens: int,
        context_length: int,
    ) -> None:
        # Resolved from the request again: the adapter is shared, and a request
        # forwarded with input_ids never went through encode_messages.
        tools = (
            [tool.model_dump() for tool in request.tools]
            if request.tools and request.tool_choice != "none"
            else None
        )
        try:
            render = self.render(request=request, tools=tools, encode=False)
        except ValueError as error:
            logger.debug("deepseek-recipe sampling overlay skipped: %s", error)
            return
        overlay_sampling(
            sampling_params,
            render,
            prompt_tokens=prompt_tokens,
            context_length=context_length,
        )
