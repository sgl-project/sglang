"""DeepSeek-V4 / V4.1 chat prompts rendered by deepseek-recipe, the encoder behind
the official API: the ``recipe`` backend of the ``dsv4`` / ``dsv41`` chat encoding
specs (``--chat-encoding-backend recipe``), registered in ``chat_encoding_adapters``.

`pre_normalize` translates SGLang's own request extensions into recipe's Chat
Completions request; everything else, validation included, is recipe's.
`rust/sglang-processor`'s `render/models/deepseek_recipe.rs` mirrors it, and the
``*-recipe.json`` fixtures in ``rust/sglang-processor/tests/fixtures/parity/`` pin
both to this serving path's output.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import TYPE_CHECKING, Any, Optional

import msgspec

try:
    import deepseek_recipe
except ImportError:  # The builtin encoders need no wheel.
    deepseek_recipe = None

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.chat_encoding_adapters import (
        ChatEncodingAdapterContext,
    )
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

logger = logging.getLogger(__name__)

# recipe resolves the model version from the request `model` and rejects any
# other name; SGLang serves whatever name the client sent, so recipe's own name
# for the served checkpoint replaces it.
MODEL_NAMES = {"v4": "DeepSeek-V4-Pro", "v41": "DeepSeek-V4.1-Flash"}
# The checkpoint tokenizer's image token; recipe's bundled tokenizer spells it <｜image｜>.
IMAGE_TOKEN = "<｜deepseek_image｜>"
# The efforts recipe accepts; numbers are rejected as the API rejects them. It
# folds them onto four: minimal and low -> Low, medium and high -> High,
# xhigh -> Xhigh, ultra and max -> Max.
REASONING_EFFORTS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "ultra",
    "max",
)


class RecipeRender(msgspec.Struct, kw_only=True):
    """A rendered request plus the sampling recipe resolved for it."""

    prompt: str
    # None when the encoder has no tokenizer or the caller skipped tokenization.
    prompt_ids: Optional[list[int]]
    temperature: Optional[float]
    top_p: Optional[float]
    # recipe's ConversationRequest, for `resolve_max_tokens`; None in tests.
    request: Any = None


def is_available() -> bool:
    return deepseek_recipe is not None


def pre_normalize(
    body: dict[str, Any], defaults: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Rewrite an SGLang chat request body for recipe's Chat Completions adapter.

    Returns a new body; `defaults` are the server's --default-chat-template-kwargs.
    Only SGLang's own request extensions are translated: the chat_template_kwargs
    thinking flag and effort, `reasoning`, `developer` turns, object-typed tool
    call arguments, `continue_final_message` and the structural_tag response
    format. recipe validates and renders the rest as the official API does; the
    caller sets `model`.

    The Python handler hands over a validated request, where protocol.py has
    already folded `reasoning` and serving_chat has applied the kwargs effort and
    the defaults; the Rust hosts hand over the wire body, so those steps repeat
    here. Mirrored in `deepseek_recipe.rs`.
    """
    defaults = defaults or {}
    request = copy.deepcopy(body)
    if request.get("task") is not None:
        raise ValueError(
            "task is not supported by the deepseek-recipe chat encoding backend"
        )

    # protocol.normalize_reasoning_inputs
    kwargs = dict(request.pop("chat_template_kwargs", None) or {})
    reasoning = request.pop("reasoning", None)
    thinking = None
    if reasoning is not None and isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if effort is None:
            effort = reasoning.get("reasoning_effort")
        if effort is not None:
            request["reasoning_effort"] = effort
        enabled = (
            reasoning.get("enabled")
            if reasoning.get("enabled") is not None
            else reasoning.get("enable", False)
        )
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in {"1", "true", "yes", "y", "on"}
        if enabled:
            thinking = True
    effort = request.get("reasoning_effort")
    if effort is not None:
        thinking = effort != "none"
    if thinking is not None:
        kwargs.setdefault("thinking", thinking)
    # serving_chat._convert_to_internal_request: the kwargs effort replaces the
    # request's; _process_messages: the server defaults fill in.
    effort = kwargs.pop("reasoning_effort", None)
    if effort is not None:
        request["reasoning_effort"] = effort
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    if request.get("reasoning_effort") is None and "reasoning_effort" in kwargs:
        request["reasoning_effort"] = kwargs["reasoning_effort"]
    # The dsv41 branch reads the kwargs flag; the API's own `thinking` object wins.
    if request.get("thinking") is None and "thinking" in kwargs:
        request["thinking"] = {"type": "enabled" if kwargs["thinking"] else "disabled"}

    # The host enforces structural_tag with its grammar; recipe does not know it.
    response_format = request.get("response_format")
    if response_format and response_format.get("type") == "structural_tag":
        del request["response_format"]
    _normalize_tool_choice(request)
    continue_final = bool(request.pop("continue_final_message", False))
    _normalize_history(request["messages"], continue_final)
    return request


def _normalize_tool_choice(request: dict[str, Any]) -> None:
    """`required` and a named tool render as `auto`: recipe would pre-open the
    DSML calls block in the prompt, but SGLang enforces the choice with its own
    grammar and its parsers expect the model to emit the block. A named tool
    keeps only that tool, and `none` or no tools drops them, as
    `serving_chat._process_messages` hands them to the encoder."""
    choice = request.get("tool_choice")
    if not request.get("tools") or choice == "none":
        request.pop("tools", None)
        request.pop("tool_choice", None)
    elif isinstance(choice, dict):
        name = choice["function"]["name"]
        chosen = [
            tool
            for tool in request.get("tools") or []
            if tool["function"].get("name") == name
        ]
        if chosen:
            request["tools"] = chosen
        request["tool_choice"] = "auto"
    elif choice == "required":
        request["tool_choice"] = "auto"


def _normalize_history(messages: list[dict[str, Any]], continue_final: bool) -> None:
    """In place: `developer` turns become `system` (recipe has no developer role),
    object-typed tool call arguments become the JSON string recipe takes, and
    `continue_final_message` marks the last assistant turn as recipe's `prefix`."""
    for message in messages:
        if message["role"] == "developer":
            message["role"] = "system"
        for call in message.get("tool_calls") or []:
            function = call["function"]
            if "arguments" in function and not isinstance(function["arguments"], str):
                function["arguments"] = json.dumps(
                    function["arguments"], ensure_ascii=False, separators=(",", ":")
                )
    if continue_final and messages and messages[-1]["role"] == "assistant":
        messages[-1]["prefix"] = True


def _error_message(body: Any) -> str:
    """The `error.message` of recipe's API-style error body, else the body."""
    try:
        return json.loads(body)["error"]["message"]
    except (TypeError, ValueError, KeyError):
        return str(body)


class RecipeEncoder:
    """One recipe encoding per served model: conversion options, the prompt
    encoding, and the tokenizer that turns rendered prompts into ids."""

    def __init__(self, model_version: str, tokenizer_json: Optional[str] = None):
        self.model = MODEL_NAMES[model_version]
        special = deepseek_recipe.SpecialTokens(image=IMAGE_TOKEN)
        self._options = deepseek_recipe.ConversionOptions().with_special_tokens(special)
        encoding = (
            deepseek_recipe.DeepseekV4Encoding()
            if model_version == "v4"
            else deepseek_recipe.DeepseekV41Encoding()
        )
        encoding = encoding.with_special_tokens(special)
        self._tokenizer = (
            deepseek_recipe.Tokenizer.from_str(tokenizer_json)
            if tokenizer_json
            else None
        )
        if self._tokenizer is not None:
            encoding = encoding.with_tokenizer(self._tokenizer)
        self._encoding = encoding

    @classmethod
    def from_hf_tokenizer(cls, model_version: str, tokenizer: Any) -> RecipeEncoder:
        """Build from the served Hugging Face fast tokenizer, so ids come from the
        checkpoint's own vocabulary rather than recipe's bundled copy."""
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None:
            raise ValueError(
                "deepseek-recipe needs the checkpoint's fast tokenizer (tokenizer.json)"
            )
        return cls(model_version, backend.to_str())

    def convert(
        self,
        body: dict[str, Any],
        defaults: Optional[dict[str, Any]] = None,
        *,
        encode: bool = True,
    ) -> RecipeRender:
        """Normalize, convert and render one request; `encode=False` skips the
        tokenization (`prompt_ids` is None). Raises ValueError for a request
        recipe rejects."""
        normalized = pre_normalize(body, defaults)
        normalized["model"] = self.model
        try:
            converted = deepseek_recipe.ChatCompletionRequest(normalized).convert(
                self._options
            )
        except deepseek_recipe.ConversionError as error:
            raise ValueError(_error_message(error.body)) from error
        conversation = converted.conversation
        # The binding tokenizes from the conversation (rendering it again); it
        # exposes no prompt-plus-ranges entry point.
        return RecipeRender(
            prompt=self._encoding.render_conversation(conversation).prompt,
            prompt_ids=(
                self._encoding.encode(conversation)
                if encode and self._tokenizer is not None
                else None
            ),
            temperature=converted.inference_options.temperature,
            top_p=converted.inference_options.top_p,
            request=converted,
        )


def overlay_sampling(
    sampling_params: dict[str, Any],
    render: RecipeRender,
    *,
    prompt_tokens: int,
    context_length: int,
) -> None:
    """Overlay the sampling the official API resolves: recipe's top_p rule (on
    V4.1, 1.0 outside thinking mode and at least 0.95 inside it), its temperature
    when it sets one, and its default max_tokens for a request without one.
    Grammar stays SGLang's: `to_sampling_params` derives it from response_format
    and tool_choice."""
    if render.temperature is not None:
        sampling_params["temperature"] = render.temperature
    if render.top_p is not None:
        sampling_params["top_p"] = render.top_p
    if sampling_params.get("max_new_tokens") is None and render.request is not None:
        try:
            sampling_params["max_new_tokens"] = render.request.resolve_max_tokens(
                prompt_tokens, context_length
            )
        except deepseek_recipe.ConversionError:
            pass  # The prompt alone exceeds the context; SGLang's own check reports it.


class DeepSeekRecipeChatEncodingAdapter:
    """recipe renders and tokenizes the prompt (the checkpoint's tokenizer, special
    tokens by id, plain BPE between them) and resolves the sampling defaults the
    official API applies. Output parsing stays with the registered DeepSeek
    reasoning and tool-call parsers, so ``tool_choice`` is enforced by SGLang's
    grammar rather than by recipe's pre-opened calls block (see
    `_normalize_tool_choice`)."""

    collects_openai_media = True
    output = None
    # recipe serializes object-typed arguments itself and passes strings through.
    strict_tool_call_arguments = False

    def __init__(
        self,
        *,
        encoder: RecipeEncoder,
        default_kwargs: Optional[dict[str, Any]] = None,
    ):
        self._encoder = encoder
        self._default_kwargs = dict(default_kwargs or {})

    @classmethod
    def from_context(
        cls, ctx: ChatEncodingAdapterContext
    ) -> DeepSeekRecipeChatEncodingAdapter:
        defaults = ctx.default_chat_template_kwargs or {}
        effort = defaults.get("reasoning_effort")
        if effort is not None and effort not in REASONING_EFFORTS:
            raise ValueError(
                f"--default-chat-template-kwargs reasoning_effort={effort!r} is not "
                f"a deepseek-recipe effort ({', '.join(REASONING_EFFORTS)})"
            )
        if not is_available():
            raise ValueError(
                "--chat-encoding-backend recipe requires the deepseek-recipe package"
            )
        # The spec already names the checkpoint family (resolve_chat_encoding_spec);
        # the served tokenizer makes the ids come from the checkpoint's vocabulary.
        encoder = RecipeEncoder.from_hf_tokenizer(
            {"dsv4": "v4", "dsv41": "v41"}[ctx.chat_encoding_spec],
            ctx.require("tokenizer"),
        )
        return cls(encoder=encoder, default_kwargs=defaults)

    def render(
        self,
        *,
        request: ChatCompletionRequest,
        tools: Optional[list[dict]],
        encode: bool = True,
    ) -> RecipeRender:
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
        return self._encoder.convert(body, self._default_kwargs, encode=encode)

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
        overlay_sampling(
            sampling_params,
            render,
            prompt_tokens=prompt_tokens,
            context_length=context_length,
        )


def _without_unset(tool: dict[str, Any]) -> dict[str, Any]:
    """The handler dumps tool models with None for unset fields; a client omits
    those keys, and recipe renders a null description or parameters differently."""
    cleaned = {key: value for key, value in tool.items() if value is not None}
    cleaned["function"] = {
        key: value for key, value in tool["function"].items() if value is not None
    }
    return cleaned
