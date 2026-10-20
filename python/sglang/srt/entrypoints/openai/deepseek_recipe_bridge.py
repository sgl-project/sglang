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

# The request `model` recipe sees; SGLang serves whatever name the client sent.
RECIPE_MODEL_ALIAS = "sglang-deepseek"
# The checkpoint tokenizer's image token; recipe's bundled tokenizer spells it <｜image｜>.
IMAGE_TOKEN = "<｜deepseek_image｜>"
# Chat Completions response formats recipe understands; others are the host's.
_RECIPE_RESPONSE_FORMATS = ("text", "json_object", "json_schema", "regex")
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


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def effort_tier(effort: Any) -> Optional[str]:
    """Map an effort value onto recipe's tiers.

    Integers 1-100 and floats 0-0.99 are SGLang's numeric budgets; they collapse
    onto low / high / max (the inverse of `encoding_dsv41.REASONING_EFFORT_MAPPINGS`)
    until recipe accepts 1-100 directly. None for an unusable value, which recipe
    then treats as unset (its default is high).
    """
    if isinstance(effort, bool) or effort is None:
        return None
    if isinstance(effort, str):
        text = effort.strip()
        if text in _EFFORT_TIERS:
            return text
        try:
            budget = float(text)
        except ValueError:
            return None
    elif isinstance(effort, (int, float)):
        budget = float(effort)
    else:
        return None
    if 0.0 <= budget < 1.0:
        budget = max(1.0, round(budget * 100))
    if not 1.0 <= budget <= 100.0:
        return None
    if budget <= 50:
        return "low"
    if budget <= 75:
        return "high"
    return "max"


def pre_normalize(
    body: dict[str, Any], defaults: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Rewrite an SGLang chat request body for recipe's Chat Completions adapter.

    The result is a new body; `defaults` are the server's
    `--default-chat-template-kwargs`. Mirrored in `deepseek_recipe.rs`.
    """
    defaults = defaults or {}
    request = copy.deepcopy(body)
    if not isinstance(request, dict):
        raise RecipeRequestError("request body must be a JSON object")
    if request.get("task") is not None:
        raise RecipeRequestError(
            "task is not supported by the deepseek-recipe chat encoding backend"
        )
    # SGLang serves whatever name the client sent.
    request["model"] = RECIPE_MODEL_ALIAS

    kwargs = request.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}
    reasoning = request.get("reasoning")
    if not isinstance(reasoning, dict):
        reasoning = {}
    # An explicit `thinking` object (the API's own control) wins.
    if not isinstance(request.get("thinking"), dict):
        toggle = _resolve_thinking(request, kwargs, reasoning, defaults)
        if toggle is not None:
            request["thinking"] = {"type": "enabled" if toggle else "disabled"}
    tier = effort_tier(_resolve_effort(request, kwargs, reasoning, defaults))
    if tier is None:
        request.pop("reasoning_effort", None)
    else:
        request["reasoning_effort"] = tier
    request.pop("reasoning", None)
    request.pop("chat_template_kwargs", None)

    if (
        request.get("max_tokens") is None
        and request.get("max_completion_tokens") is not None
    ):
        request["max_tokens"] = request["max_completion_tokens"]
    # The host samples `n` completions from one prompt and enforces its own
    # response formats (structural_tag) with its grammar.
    request.pop("n", None)
    response_format = request.get("response_format")
    if (
        isinstance(response_format, dict)
        and response_format.get("type") not in _RECIPE_RESPONSE_FORMATS
    ):
        request.pop("response_format")
    _normalize_tool_choice(request)

    continue_final = _truthy(request.pop("continue_final_message", False))
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise RecipeRequestError("messages must be an array")
    _normalize_history(messages, continue_final)
    return request


def _resolve_thinking(
    request: dict[str, Any],
    kwargs: dict[str, Any],
    reasoning: dict[str, Any],
    defaults: dict[str, Any],
) -> Optional[bool]:
    """The kwargs toggle, else what the effort or `reasoning.enabled` implies,
    else the server default; None leaves recipe's own default (on)."""
    for key in ("thinking", "enable_thinking"):
        if kwargs.get(key) is not None:
            return _truthy(kwargs[key])
    effort = _requested_effort(request, reasoning)
    if effort is not None:
        return effort != "none"
    enabled = reasoning.get("enabled")
    if enabled is None:
        enabled = reasoning.get("enable")
    if enabled is not None and _truthy(enabled):
        return True
    if defaults.get("thinking") is not None:
        return _truthy(defaults["thinking"])
    return None


def _resolve_effort(
    request: dict[str, Any],
    kwargs: dict[str, Any],
    reasoning: dict[str, Any],
    defaults: dict[str, Any],
) -> Any:
    """The kwargs effort replaces the request's, as serving_chat does; the server
    default fills in last."""
    effort = kwargs.get("reasoning_effort")
    if effort is None:
        effort = _requested_effort(request, reasoning)
    if effort is None:
        effort = defaults.get("reasoning_effort")
    return effort


def _normalize_tool_choice(request: dict[str, Any]) -> None:
    """`required` and a named tool render as `auto`: recipe would pre-open the
    DSML calls block in the prompt, but SGLang enforces the choice with its own
    grammar and its parsers expect the model to emit the block. A named tool
    keeps only that tool, as `serving_chat._process_messages` does."""
    choice = request.get("tool_choice")
    if isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name")
        tools = request.get("tools")
        if isinstance(tools, list) and name is not None:
            chosen = [
                tool
                for tool in tools
                if isinstance(tool, dict)
                and (tool.get("function") or {}).get("name") == name
            ]
            if chosen:
                request["tools"] = chosen
        request["tool_choice"] = "auto"
    elif choice == "required":
        request["tool_choice"] = "auto"


def _normalize_history(messages: list[Any], continue_final: bool) -> None:
    """In place: developer -> system, ids for historical tool calls and their tool
    messages, serialized arguments, and the continuation `prefix` mark."""
    pending_call_ids: list[str] = []
    synthesized = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "developer":
            message["role"] = "system"
        role = message.get("role")
        if role == "assistant":
            pending_call_ids = []
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                if call.get("id") is None:
                    call["id"] = f"call_{synthesized:02d}_sglang"
                    synthesized += 1
                pending_call_ids.append(call["id"])
                function = call.get("function")
                if (
                    isinstance(function, dict)
                    and "arguments" in function
                    and not isinstance(function["arguments"], str)
                ):
                    function["arguments"] = json.dumps(
                        function["arguments"], ensure_ascii=False, separators=(",", ":")
                    )
        elif role == "tool":
            if message.get("tool_call_id") is None and pending_call_ids:
                message["tool_call_id"] = pending_call_ids.pop(0)
            elif message.get("tool_call_id") in pending_call_ids:
                pending_call_ids.remove(message["tool_call_id"])
    if (
        continue_final
        and messages
        and isinstance(messages[-1], dict)
        and messages[-1].get("role") == "assistant"
    ):
        messages[-1]["prefix"] = True


def _requested_effort(request: dict[str, Any], reasoning: dict[str, Any]) -> Any:
    """`protocol.normalize_reasoning_inputs`: reasoning.effort, reasoning.reasoning_effort, then the top-level field."""
    for value in (
        reasoning.get("effort"),
        reasoning.get("reasoning_effort"),
        request.get("reasoning_effort"),
    ):
        if value is not None:
            return value
    return None


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


def model_version_for(hf_config: Any) -> Optional[str]:
    """`v41` / `v4` for a DeepSeek-V4 family config, else None."""
    model_type = getattr(hf_config, "model_type", None)
    architectures = getattr(hf_config, "architectures", None) or []
    arch = architectures[0] if architectures else ""
    if model_type == "deepseek_v41" or "DeepseekV41" in arch:
        return "v41"
    if model_type == "deepseek_v4" or "DeepseekV4" in arch:
        return "v4"
    return None


def apply_sampling(
    sampling_params: dict[str, Any],
    render: RecipeRender,
    *,
    prompt_tokens: int,
    context_length: int,
) -> None:
    """Overlay recipe's resolved settings on SGLang's sampling params.

    temperature and top_p follow recipe (1.0 in thinking mode; top_p clamped per
    model version). max_new_tokens gets recipe's default when the request set
    none. A json / regex output constraint fills the grammar fields SGLang did
    not already derive from response_format; tool calls stay with SGLang's own
    grammar (see `_normalize_tool_choice`).
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
    if render.output_constraint is not None:
        kind, schema = render.output_constraint
        if kind == "json" and not sampling_params.get("json_schema"):
            sampling_params["json_schema"] = schema or '{"type": "object"}'
        elif kind == "regex" and not sampling_params.get("regex"):
            sampling_params["regex"] = schema
