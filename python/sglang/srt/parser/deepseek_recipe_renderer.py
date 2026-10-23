"""DeepSeek-V4 / V4.1 prompts through deepseek-recipe, the library behind the official
API: request translation, conversion, rendering and tokenization.
``rust/sglang-processor/src/render/models/deepseek_recipe.rs`` mirrors `pre_normalize`."""

from __future__ import annotations

import copy
import functools
import json
from types import ModuleType
from typing import Any, Optional

import msgspec

DEEPSEEK_RECIPE_INSTALL_HINT = (
    "Serving DeepSeek-V4 / V4.1 with --chat-encoding-backend recipe requires the "
    "`deepseek-recipe` package (DeepSeek's reference encoder and parser; wheels for "
    "Linux and macOS). Install it with `pip install deepseek-recipe`."
)

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


@functools.cache
def load_deepseek_recipe() -> ModuleType:
    try:
        import deepseek_recipe
    except ImportError as exc:
        raise ImportError(DEEPSEEK_RECIPE_INSTALL_HINT) from exc
    return deepseek_recipe


class RecipeRender(msgspec.Struct, kw_only=True):
    """A rendered request plus the sampling recipe resolved for it."""

    prompt: str
    prompt_ids: Optional[list[int]]  # None when the caller skipped tokenization.
    temperature: Optional[float]
    top_p: Optional[float]
    thinking: bool
    request: Any = None  # recipe's ConversationRequest: max_tokens, parsing, chunks.


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
    with its grammar, which expects the model to open the calls block itself."""
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
    """recipe's conversion options, prompt encoding and tokenizer for one served model.

    `tokenizer` is the checkpoint's `tokenizers.Tokenizer`; recipe encodes prompts with
    its own copy of it and the output parser decodes with the original."""

    def __init__(self, model_version: str, tokenizer: Any):
        recipe = load_deepseek_recipe()
        self.model = MODEL_NAMES[model_version]
        self.hf_tokenizer = tokenizer
        special = recipe.SpecialTokens(image=IMAGE_TOKEN)
        self._options = recipe.ConversionOptions().with_special_tokens(special)
        encoding = (
            recipe.DeepseekV4Encoding()
            if model_version == "v4"
            else recipe.DeepseekV41Encoding()
        )
        self.tokenizer = recipe.Tokenizer.from_str(tokenizer.to_str())
        self._encoding = encoding.with_special_tokens(special).with_tokenizer(
            self.tokenizer
        )

    @classmethod
    def from_hf_tokenizer(cls, model_version: str, tokenizer: Any) -> RecipeEncoder:
        """From the served fast tokenizer, so ids come from the checkpoint's vocabulary."""
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None:
            raise ValueError(
                "deepseek-recipe needs the checkpoint's fast tokenizer (tokenizer.json)"
            )
        return cls(model_version, backend)

    def convert(
        self,
        body: dict[str, Any],
        defaults: Optional[dict[str, Any]] = None,
        *,
        encode: bool = True,
    ) -> RecipeRender:
        """Render one request; `encode=False` skips tokenization. Raises ValueError."""
        recipe = load_deepseek_recipe()
        normalized = pre_normalize(body, defaults)
        normalized["model"] = self.model
        try:
            converted = recipe.ChatCompletionRequest(normalized).convert(self._options)
        except recipe.ConversionError as error:
            raise ValueError(_error_message(error.body)) from error
        conversation = converted.conversation
        return RecipeRender(
            prompt=self._encoding.render_conversation(conversation).prompt,
            prompt_ids=self._encoding.encode(conversation) if encode else None,
            temperature=converted.inference_options.temperature,
            top_p=converted.inference_options.top_p,
            thinking=conversation.thinking_mode,
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
        except load_deepseek_recipe().ConversionError:
            pass  # The prompt alone exceeds the context; SGLang's own check reports it.
