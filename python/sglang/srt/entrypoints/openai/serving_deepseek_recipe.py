"""DeepSeek-V4 / V4.1 chat prompts rendered by deepseek-recipe, the encoder behind
the official API: the ``recipe`` backend of the ``dsv4`` / ``dsv41`` chat encoding
specs (``--chat-encoding-backend recipe``), registered in ``chat_encoding_adapters``.
``deepseek_recipe_bridge`` holds the request rules."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.entrypoints.openai import deepseek_recipe_bridge as bridge

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.chat_encoding_adapters import (
        ChatEncodingAdapterContext,
    )
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

logger = logging.getLogger(__name__)


class DeepSeekRecipeChatEncodingAdapter:
    """recipe renders and tokenizes the prompt (the checkpoint's tokenizer, special
    tokens by id, plain BPE between them) and resolves the sampling defaults the
    official API applies. Output parsing stays with the registered DeepSeek
    reasoning and tool-call parsers, so ``tool_choice`` is enforced by SGLang's
    grammar rather than by recipe's pre-opened calls block (see
    ``deepseek_recipe_bridge._normalize_tool_choice``)."""

    collects_openai_media = True
    output = None
    # recipe serializes object-typed arguments itself and passes strings through.
    strict_tool_call_arguments = False

    def __init__(
        self,
        *,
        encoder: bridge.RecipeEncoder,
        default_kwargs: Optional[dict[str, Any]] = None,
    ):
        self._encoder = encoder
        self._default_kwargs = dict(default_kwargs or {})

    @classmethod
    def from_context(
        cls, ctx: ChatEncodingAdapterContext
    ) -> DeepSeekRecipeChatEncodingAdapter:
        if not bridge.is_available():
            raise ValueError(
                "--chat-encoding-backend recipe requires the deepseek-recipe package"
            )
        model_version = (
            bridge.model_version_for(ctx.hf_config)
            or {"dsv4": "v4", "dsv41": "v41"}[ctx.chat_encoding_spec]
        )
        # The served tokenizer, so ids come from the checkpoint's vocabulary.
        encoder = bridge.RecipeEncoder.from_hf_tokenizer(
            model_version, ctx.require("tokenizer")
        )
        return cls(encoder=encoder, default_kwargs=ctx.default_chat_template_kwargs)

    def render(
        self,
        *,
        request: ChatCompletionRequest,
        tools: Optional[list[dict]],
        encode: bool = True,
    ) -> bridge.RecipeRender:
        """recipe's view of the request. `tools` is the handler's effective list,
        None when tools are off for this request; it replaces the request's."""
        # The wire request rather than the handler's normalized messages: recipe
        # applies its own message rules.
        body = request.model_dump(exclude_unset=True, by_alias=True, mode="json")
        if tools is None:
            body.pop("tools", None)
            body.pop("tool_choice", None)
        else:
            body["tools"] = [_without_unset(tool) for tool in tools]
        try:
            return self._encoder.convert(body, self._default_kwargs, encode=encode)
        except bridge.RecipeRequestError as error:
            raise ValueError(str(error)) from error

    def encode_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        request: ChatCompletionRequest,
        tools: Optional[list[dict]],
    ) -> list[int]:
        return self.render(request=request, tools=tools).prompt_ids

    def apply_sampling(
        self,
        *,
        request: ChatCompletionRequest,
        sampling_params: dict[str, Any],
        prompt_tokens: int,
        context_length: int,
    ) -> None:
        # Resolved from the request again rather than kept from encode_messages:
        # the adapter is shared by concurrent requests, and a request that arrives
        # with input_ids (router-forwarded) never passed through encode_messages.
        tools = (
            [
                tool.model_dump(exclude_unset=True, by_alias=True, mode="json")
                for tool in request.tools
            ]
            if request.tools and request.tool_choice != "none"
            else None
        )
        try:
            render = self.render(request=request, tools=tools, encode=False)
        except ValueError as error:
            logger.debug("deepseek-recipe sampling overlay skipped: %s", error)
            return
        bridge.apply_sampling(
            sampling_params,
            render,
            prompt_tokens=prompt_tokens,
            context_length=context_length,
        )


def _without_unset(tool: dict[str, Any]) -> dict[str, Any]:
    """The handler dumps tool models with None for unset fields; a client omits
    those keys, and recipe renders a null description or parameters differently."""
    cleaned = {key: value for key, value in tool.items() if value is not None}
    function = cleaned.get("function")
    if isinstance(function, dict):
        cleaned["function"] = {
            key: value for key, value in function.items() if value is not None
        }
    return cleaned
