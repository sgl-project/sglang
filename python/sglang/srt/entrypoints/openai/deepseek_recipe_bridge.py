"""DeepSeek-V4 / V4.1 chat encoding through deepseek-recipe, the encoder behind the
official API. `pre_normalize` rewrites an SGLang Chat Completions body into what
recipe's adapter accepts; `RecipeEncoder.convert` renders and tokenizes it and
returns the settings recipe resolved. `rust/sglang-processor`'s
`render/models/deepseek_recipe.rs` mirrors `pre_normalize` rule for rule, and
`tests/fixtures/recipe_parity/` pins both to the same outputs.

`deepseek_recipe` is imported lazily: the builtin encoders need no wheel, and
`is_available()` says whether this backend can be selected.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Optional

import msgspec

# The request `model` recipe sees. recipe resolves the model version by looking
# the name up in its alias table and rejects any other name, while SGLang serves
# whatever name the client sent; `RecipeEncoder` binds this alias to the served
# checkpoint's version.
RECIPE_MODEL_ALIAS = "sglang-deepseek"
# The checkpoint tokenizer's image token; recipe's bundled tokenizer spells it <｜image｜>.
IMAGE_TOKEN = "<｜deepseek_image｜>"
# recipe's request-level efforts. It folds them onto four: minimal and low ->
# Low, medium and high -> High, xhigh -> Xhigh, ultra and max -> Max. V4.1
# renders Low as 50, High and Xhigh as 75, Max as 100; V4 renders Xhigh like High.
_EFFORT_TIERS = ("none", "minimal", "low", "medium", "high", "xhigh", "ultra", "max")


class RecipeRequestError(ValueError):
    """A request recipe rejects; `status_code` follows the official API."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class RecipeRender(msgspec.Struct, kw_only=True):
    """A rendered request plus the settings recipe resolved for it."""

    prompt: str
    # Code-point ranges of `prompt` holding explicit special tokens.
    special_ranges: list[tuple[int, int]]
    # None when the encoder has no tokenizer or the caller skipped tokenization.
    prompt_ids: Optional[list[int]]
    thinking_mode: bool
    reasoning_effort: Optional[str]
    temperature: Optional[float]
    top_p: Optional[float]
    max_tokens: Optional[int]
    # (kind, schema) of recipe's OutputConstraint: json / regex / tool_call / tool_call_required.
    output_constraint: Optional[tuple[str, str]]
    starts_in_tool_calls: bool
    image_count: int = 0
    # recipe's ConversationRequest, for `resolve_max_tokens`; None in tests.
    request: Any = None


def is_available() -> bool:
    try:
        import deepseek_recipe  # noqa: F401
    except ImportError:
        return False
    return True


def recipe_version() -> Optional[str]:
    try:
        import deepseek_recipe
    except ImportError:
        return None
    return deepseek_recipe.__version__


def _conversion_errors() -> tuple[type[BaseException], ...]:
    """recipe's ConversionError when the package is installed; without it nothing
    can raise one, so there is nothing to catch."""
    try:
        import deepseek_recipe
    except ImportError:
        return ()
    return (deepseek_recipe.ConversionError,)


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def effort_tier(effort: Any) -> Optional[str]:
    """Map an SGLang effort onto recipe's tiers; None for an unusable value, which
    recipe treats as unset (high).

    Tier names pass through. SGLang's numeric budgets, an int in [1, 100] from
    chat_template_kwargs or a float in [0, 0.99] from the request field
    (`chat_encoding.parse_dsv41_reasoning_effort`), collapse onto low / high /
    max at the thresholds of `encoding_dsv41.REASONING_EFFORT_MAPPINGS`
    (50 / 75 / 100). recipe 0.2.0 takes no number, and those three tiers are all
    the distinct renderings it has, so only the exact budget is lost: 60 renders
    as 75 where the builtin encoder writes 60.
    """
    if isinstance(effort, bool):
        return None
    if isinstance(effort, str):
        effort = effort.strip()
        if effort in _EFFORT_TIERS:
            return effort
        try:
            effort = float(effort)
        except ValueError:
            return None
    elif not isinstance(effort, (int, float)):
        return None
    if 0.0 <= effort < 1.0:
        effort = max(1, round(effort * 100))
    if not 1 <= effort <= 100:
        return None
    if effort <= 50:
        return "low"
    if effort <= 75:
        return "high"
    return "max"


def pre_normalize(
    body: dict[str, Any], defaults: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Rewrite an SGLang chat request body for recipe's Chat Completions adapter.

    Returns a new body; `defaults` are the server's --default-chat-template-kwargs.
    The Python handler passes a validated request, where protocol.py has already
    folded `reasoning` and serving_chat has applied the kwargs effort and the
    defaults; the Rust hosts pass the wire body, so the thinking and effort rules
    below repeat those steps. Mirrored in `deepseek_recipe.rs`.
    """
    defaults = defaults or {}
    request = copy.deepcopy(body)
    if request.get("task") is not None:
        raise RecipeRequestError(
            "task is not supported by the deepseek-recipe chat encoding backend"
        )
    request["model"] = RECIPE_MODEL_ALIAS

    kwargs = request.pop("chat_template_kwargs", None) or {}
    reasoning = request.pop("reasoning", None) or {}
    # protocol.normalize_reasoning_inputs: reasoning.effort, then
    # reasoning.reasoning_effort, then the request field.
    effort = _first(
        reasoning.get("effort"),
        reasoning.get("reasoning_effort"),
        request.get("reasoning_effort"),
    )
    # The API's own `thinking` object wins. Otherwise the kwargs flag, which
    # serving_chat reads for the DeepSeek encoders and normalize_reasoning_inputs
    # only fills with setdefault; then what the effort or `reasoning.enabled`
    # implies, as that validator derives it; then the server defaults.
    if request.get("thinking") is None:
        thinking = _first(kwargs.get("thinking"), kwargs.get("enable_thinking"))
        if thinking is None and effort is not None:
            thinking = effort != "none"
        if thinking is None:
            enabled = _first(reasoning.get("enabled"), reasoning.get("enable"))
            if isinstance(enabled, str):
                enabled = enabled.strip().lower() in ("1", "true", "yes", "y", "on")
            if enabled:
                thinking = True
        if thinking is None:
            thinking = defaults.get("thinking")
        if thinking is not None:
            request["thinking"] = {"type": "enabled" if thinking else "disabled"}
    # serving_chat._convert_to_internal_request: the kwargs effort replaces the
    # request's; _process_messages: the default fills in last.
    tier = effort_tier(
        _first(kwargs.get("reasoning_effort"), effort, defaults.get("reasoning_effort"))
    )
    if tier is None:
        request.pop("reasoning_effort", None)
    else:
        request["reasoning_effort"] = tier

    if (
        request.get("max_tokens") is None
        and request.get("max_completion_tokens") is not None
    ):
        request["max_tokens"] = request["max_completion_tokens"]
    # The host samples `n` completions from one prompt and enforces
    # structural_tag, which recipe does not know, with its grammar.
    request.pop("n", None)
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
    keeps only that tool, as `serving_chat._process_messages` does."""
    choice = request.get("tool_choice")
    if isinstance(choice, dict):
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
    """In place: developer -> system, ids pairing historical tool calls with
    their tool messages, string arguments, and the continuation `prefix` mark."""
    pending_call_ids: list[str] = []
    synthesized = 0
    for message in messages:
        if message["role"] == "developer":
            message["role"] = "system"
        if message["role"] == "assistant":
            # recipe requires every call to carry an id answered by a tool message
            # with the same tool_call_id. The builtin encoders never did (the
            # DSML prompt prints no ids), so unlabeled ones pair by position, the
            # order `sort_tool_results_by_call_order` assumes as well.
            pending_call_ids = []
            for call in message.get("tool_calls") or []:
                if call.get("id") is None:
                    call["id"] = f"call_{synthesized:02d}_sglang"
                    synthesized += 1
                pending_call_ids.append(call["id"])
                # recipe takes the OpenAI JSON string; SGLang also accepts an object.
                function = call["function"]
                if "arguments" in function and not isinstance(
                    function["arguments"], str
                ):
                    function["arguments"] = json.dumps(
                        function["arguments"], ensure_ascii=False, separators=(",", ":")
                    )
        elif message["role"] == "tool":
            if message.get("tool_call_id") is None and pending_call_ids:
                message["tool_call_id"] = pending_call_ids.pop(0)
            elif message.get("tool_call_id") in pending_call_ids:
                pending_call_ids.remove(message["tool_call_id"])
    if continue_final and messages and messages[-1]["role"] == "assistant":
        messages[-1]["prefix"] = True


def _error_message(body: Any) -> str:
    try:
        parsed = json.loads(body) if isinstance(body, str) else body
    except ValueError:
        return str(body)
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict) and "message" in error:
            return str(error["message"])
    return str(body)


class RecipeEncoder:
    """One recipe encoding per served model: conversion options, the prompt
    encoding, and the tokenizer that turns rendered prompts into ids."""

    def __init__(self, model_version: str, tokenizer_json: Optional[str] = None):
        import deepseek_recipe as dr

        if model_version not in ("v4", "v41"):
            raise ValueError(f"unknown DeepSeek recipe model version {model_version!r}")
        self.model_version = model_version
        version = (
            dr.ModelVersion.DeepseekV4Pro
            if model_version == "v4"
            else dr.ModelVersion.DeepseekV41Flash
        )
        special = dr.SpecialTokens(image=IMAGE_TOKEN)
        self._options = (
            dr.ConversionOptions()
            .with_model_alias(RECIPE_MODEL_ALIAS, version)
            .with_special_tokens(special)
        )
        encoding = (
            dr.DeepseekV4Encoding()
            if model_version == "v4"
            else dr.DeepseekV41Encoding()
        )
        encoding = encoding.with_special_tokens(special)
        self._tokenizer = (
            dr.Tokenizer.from_str(tokenizer_json) if tokenizer_json else None
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
        tokenization (`prompt_ids` is None). Raises RecipeRequestError."""
        import deepseek_recipe as dr

        normalized = pre_normalize(body, defaults)
        try:
            request = dr.ChatCompletionRequest(normalized)
        except Exception as exc:  # the binding raises ValueError for serde failures
            raise RecipeRequestError(str(exc), status_code=422) from exc
        try:
            converted = request.convert(self._options)
        except dr.ConversionError as exc:
            raise RecipeRequestError(
                _error_message(exc.body), status_code=exc.status_code
            ) from exc
        conversation = converted.conversation
        rendered = self._encoding.render_conversation(conversation)
        # The binding tokenizes from the conversation (rendering it again); it
        # exposes no prompt-plus-ranges entry point.
        prompt_ids = (
            self._encoding.encode(conversation)
            if encode and self._tokenizer is not None
            else None
        )
        inference = converted.inference_options
        constraint = inference.output_constraint
        return RecipeRender(
            prompt=rendered.prompt,
            special_ranges=[tuple(r) for r in rendered.special_ranges],
            prompt_ids=prompt_ids,
            thinking_mode=conversation.thinking_mode,
            reasoning_effort=conversation.reasoning_effort,
            temperature=inference.temperature,
            top_p=inference.top_p,
            max_tokens=inference.max_tokens,
            output_constraint=(constraint.kind, constraint.schema)
            if constraint is not None
            else None,
            starts_in_tool_calls=converted.parsing_options.starts_in_tool_calls,
            image_count=len(rendered.image_sources),
            request=converted,
        )


def apply_sampling(
    sampling_params: dict[str, Any],
    render: RecipeRender,
    *,
    prompt_tokens: int,
    context_length: int,
) -> None:
    """Overlay the sampling the official API resolves for the request.

    top_p follows recipe's rule (on V4.1: 1.0 outside thinking mode, at least
    0.95 inside it), temperature follows recipe when it sets one, and a request
    without max_tokens gets recipe's default. Grammar stays SGLang's:
    `to_sampling_params` derives it from response_format and tool_choice.
    """
    if render.temperature is not None:
        sampling_params["temperature"] = render.temperature
    if render.top_p is not None:
        sampling_params["top_p"] = render.top_p
    if sampling_params.get("max_new_tokens") is None and render.request is not None:
        try:
            sampling_params["max_new_tokens"] = render.request.resolve_max_tokens(
                prompt_tokens, context_length
            )
        except _conversion_errors():
            pass  # The prompt alone exceeds the context; SGLang's own check reports it.
