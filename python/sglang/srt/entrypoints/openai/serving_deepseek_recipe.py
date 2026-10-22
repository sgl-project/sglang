"""DeepSeek-V4 / V4.1 chat encoding through deepseek-recipe, the encoder behind the
official API: the ``recipe`` backend of the ``dsv4`` / ``dsv41`` chat encoding specs
(``--chat-encoding-backend recipe``). ``render/models/deepseek_recipe.rs`` mirrors
`pre_normalize`; the ``*-recipe.json`` parity fixtures pin both to this serving path.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import TYPE_CHECKING, Any, Optional

import msgspec

try:
    import deepseek_recipe
except ImportError:
    deepseek_recipe = None

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.chat_encoding_adapters import (
        ChatEncodingAdapterContext,
    )
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

logger = logging.getLogger(__name__)

MODEL_NAMES = {"v4": "DeepSeek-V4-Pro", "v41": "DeepSeek-V4.1-Flash"}
# The checkpoint tokenizer's image token; recipe's bundled tokenizer spells it <｜image｜>.
IMAGE_TOKEN = "<｜deepseek_image｜>"
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
    prompt_ids: Optional[list[int]]  # None when the caller skipped tokenization.
    temperature: Optional[float]
    top_p: Optional[float]
    request: Any = None  # recipe's ConversationRequest, for `resolve_max_tokens`.


def is_deepseek_recipe_available() -> bool:
    return deepseek_recipe is not None


def pre_normalize(
    body: dict[str, Any], defaults: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Rewrite an SGLang chat request body for recipe's Chat Completions adapter.

    Only SGLang's own extensions are translated; recipe validates and renders the
    rest. The Rust hosts pass the wire body, so the `reasoning` fold and the
    kwargs-effort / default-kwargs steps the Python handler already ran repeat here.
    """
    defaults = defaults or {}
    request = copy.deepcopy(body)
    if request.get("task") is not None:
        raise ValueError(
            "task is not supported by the deepseek-recipe chat encoding backend"
        )

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
    effort = kwargs.pop("reasoning_effort", None)
    if effort is not None:
        request["reasoning_effort"] = effort
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    if request.get("reasoning_effort") is None and "reasoning_effort" in kwargs:
        request["reasoning_effort"] = kwargs["reasoning_effort"]
    if request.get("thinking") is None and "thinking" in kwargs:
        request["thinking"] = {"type": "enabled" if kwargs["thinking"] else "disabled"}

    # The host enforces structural_tag with its grammar.
    response_format = request.get("response_format")
    if response_format and response_format.get("type") == "structural_tag":
        del request["response_format"]
    _normalize_tool_choice(request)
    continue_final = bool(request.pop("continue_final_message", False))
    _normalize_history(request["messages"], continue_final)
    return request


def _normalize_tool_choice(request: dict[str, Any]) -> None:
    """`required` and a named tool render as `auto`: SGLang enforces the choice
    with its grammar, and its parsers expect the model to open the calls block."""
    choice = request.get("tool_choice")
    if not request.get("tools") or choice == "none":
        request.pop("tools", None)
        request.pop("tool_choice", None)
    elif choice == "required" or isinstance(choice, dict):
        request["tool_choice"] = "auto"


def _normalize_history(messages: list[dict[str, Any]], continue_final: bool) -> None:
    """developer -> system, object arguments -> JSON, continue_final_message -> prefix."""
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
    """recipe's conversion options, prompt encoding and tokenizer for one served model."""

    def __init__(self, model_version: str, tokenizer_json: str):
        self.model = MODEL_NAMES[model_version]
        special = deepseek_recipe.SpecialTokens(image=IMAGE_TOKEN)
        self._options = deepseek_recipe.ConversionOptions().with_special_tokens(special)
        encoding = (
            deepseek_recipe.DeepseekV4Encoding()
            if model_version == "v4"
            else deepseek_recipe.DeepseekV41Encoding()
        )
        self._tokenizer = deepseek_recipe.Tokenizer.from_str(tokenizer_json)
        self._encoding = encoding.with_special_tokens(special).with_tokenizer(
            self._tokenizer
        )

    @classmethod
    def from_hf_tokenizer(cls, model_version: str, tokenizer: Any) -> RecipeEncoder:
        """From the served fast tokenizer, so ids come from the checkpoint's vocabulary."""
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
        """Render one request; `encode=False` skips tokenization. Raises ValueError."""
        normalized = pre_normalize(body, defaults)
        normalized["model"] = self.model
        try:
            converted = deepseek_recipe.ChatCompletionRequest(normalized).convert(
                self._options
            )
        except deepseek_recipe.ConversionError as error:
            raise ValueError(_error_message(error.body)) from error
        conversation = converted.conversation
        return RecipeRender(
            prompt=self._encoding.render_conversation(conversation).prompt,
            prompt_ids=self._encoding.encode(conversation) if encode else None,
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
    """Overlay the temperature, top_p and default max_tokens recipe resolved;
    grammar stays SGLang's (`to_sampling_params`)."""
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
    """recipe renders and tokenizes the prompt and resolves the API's sampling
    defaults; output stays with the registered DeepSeek parsers."""

    collects_openai_media = True
    output = None
    strict_tool_call_arguments = False  # recipe serializes object arguments itself.

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
        # Resolved at boot: a bad server default fails startup, not requests.
        defaults = ctx.default_chat_template_kwargs or {}
        effort = defaults.get("reasoning_effort")
        if effort is not None and effort not in REASONING_EFFORTS:
            raise ValueError(
                f"--default-chat-template-kwargs reasoning_effort={effort!r} is not "
                f"a deepseek-recipe effort ({', '.join(REASONING_EFFORTS)})"
            )
        if not is_deepseek_recipe_available():
            raise ValueError(
                "--chat-encoding-backend recipe requires the deepseek-recipe package"
            )
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
        """recipe's view of the request; `tools` is the handler's effective list."""
        # The wire request, not the handler's normalized messages.
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
        # Resolved from the request again: the adapter is shared, and a request
        # forwarded with input_ids never went through encode_messages.
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
    """Drop the None fields `model_dump` adds; recipe renders them as values."""
    cleaned = {key: value for key, value in tool.items() if value is not None}
    cleaned["function"] = {
        key: value for key, value in tool["function"].items() if value is not None
    }
    return cleaned
