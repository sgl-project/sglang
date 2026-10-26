"""DeepSeek-V4 / V4.1 prompts through deepseek-recipe, plus the SGLang extensions
recipe does not model (quick instruction tasks, V4.1 integer effort)."""

from __future__ import annotations

import copy
import functools
import json
from types import ModuleType
from typing import Any, Optional, Union

DEEPSEEK_RECIPE_INSTALL_HINT = (
    "Serving DeepSeek-V4 / V4.1 requires the `deepseek-recipe` package (DeepSeek's "
    "reference encoder and parser; wheels for Linux and macOS). Install it with "
    "`pip install deepseek-recipe`."
)

MODEL_NAMES = {"dsv4": "DeepSeek-V4-Pro", "dsv41": "DeepSeek-V4.1-Flash"}
# The checkpoint tokenizer's image token; recipe's bundled tokenizer spells it <｜image｜>.
IMAGE_TOKEN = "<｜deepseek_image｜>"
ASSISTANT_TOKEN = "<｜Assistant｜>"
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"
TASK_TOKENS = {
    "action": "<｜action｜>",
    "query": "<｜query｜>",
    "authority": "<｜authority｜>",
    "domain": "<｜domain｜>",
    "title": "<｜title｜>",
    "read_url": "<｜read_url｜>",
}
# `action` follows the generation header; the other tasks replace it.
TASKS_AFTER_HEADER = ("action",)
# V4.1 renders an integer budget as this line; recipe's "high" tier is budget 75.
BUDGET_TIER, BUDGET_TIER_VALUE = "high", 75
BUDGET_LINE = "Reasoning Effort: {budget}"


@functools.cache
def load_deepseek_recipe() -> ModuleType:
    try:
        import deepseek_recipe
    except ImportError as exc:
        raise ImportError(DEEPSEEK_RECIPE_INSTALL_HINT) from exc
    return deepseek_recipe


def parse_effort_budget(value: Union[int, float]) -> int:
    if isinstance(value, bool):
        raise ValueError("reasoning_effort must not be a boolean")
    if isinstance(value, int) and 1 <= value <= 100:
        return value
    if isinstance(value, float) and 0.0 <= value <= 0.99:
        return max(1, round(value * 100))
    raise ValueError(
        f"reasoning_effort={value!r} is outside the integer budget [1, 100] "
        "and the float range [0, 0.99]"
    )


def pre_normalize(body: dict[str, Any]) -> dict[str, Any]:
    """Rewrite an SGLang chat request body for recipe's Chat Completions adapter.

    `body` carries only what the prompt depends on: messages, thinking and, when
    set, tools, tool_choice, response_format, continue_final_message and a named
    reasoning_effort.
    """
    request = copy.deepcopy(body)
    request["thinking"] = {"type": "enabled" if request["thinking"] else "disabled"}
    # The host enforces structural_tag with its grammar.
    response_format = request.get("response_format")
    if response_format and response_format.get("type") == "structural_tag":
        del request["response_format"]
    choice = request.get("tool_choice")
    if not request.get("tools") or choice == "none":
        request.pop("tools", None)
        request.pop("tool_choice", None)
    elif choice == "required" or isinstance(choice, dict):
        # SGLang enforces the choice with its grammar, which expects the model
        # to open the calls block itself.
        request["tool_choice"] = "auto"
    messages = request["messages"]
    for message in messages:
        if message["role"] == "developer":
            message["role"] = "system"
        for call in message.get("tool_calls") or []:
            function = call["function"]
            if "arguments" in function and not isinstance(function["arguments"], str):
                function["arguments"] = json.dumps(
                    function["arguments"], ensure_ascii=False, separators=(",", ":")
                )
    continue_final = request.pop("continue_final_message", False)
    if continue_final and messages and messages[-1]["role"] == "assistant":
        messages[-1]["prefix"] = True
    return request


def _error_message(body: Any) -> str:
    try:
        return json.loads(body)["error"]["message"]
    except (TypeError, ValueError, KeyError):
        return str(body)


class DeepSeekRecipe:
    """`tokenizer` is the checkpoint's `tokenizers.Tokenizer`; recipe encodes prompts
    with its own copy of it, and the SGLang extensions splice ids from the original."""

    def __init__(self, chat_encoding_spec: str, tokenizer: Any):
        recipe = load_deepseek_recipe()
        self.model = MODEL_NAMES[chat_encoding_spec]
        self.hf_tokenizer = tokenizer
        self.special_tokens = recipe.SpecialTokens(image=IMAGE_TOKEN)
        self._options = recipe.ConversionOptions().with_special_tokens(
            self.special_tokens
        )
        if chat_encoding_spec == "dsv4":
            encoding = recipe.DeepseekV4Encoding()
            self.model_version = recipe.ModelVersion.DeepseekV4Pro
        else:
            encoding = recipe.DeepseekV41Encoding()
            self.model_version = recipe.ModelVersion.DeepseekV41Flash
        self.tokenizer = recipe.Tokenizer.from_str(tokenizer.to_str())
        self._encoding = encoding.with_special_tokens(
            self.special_tokens
        ).with_tokenizer(self.tokenizer)

    @classmethod
    def from_hf_tokenizer(
        cls, chat_encoding_spec: str, tokenizer: Any
    ) -> DeepSeekRecipe:
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None:
            raise ValueError(
                "deepseek-recipe needs the checkpoint's fast tokenizer (tokenizer.json)"
            )
        return cls(chat_encoding_spec, backend)

    def encode(
        self,
        body: dict[str, Any],
        *,
        task: Optional[str] = None,
        effort_budget: Optional[int] = None,
        image_urls: Optional[list[str]] = None,
    ) -> list[int]:
        """Prompt ids for `body` (see `pre_normalize`); a request recipe rejects
        raises ValueError. `image_urls` receives the images in placeholder order."""
        recipe = load_deepseek_recipe()
        normalized = pre_normalize(body)
        normalized["model"] = self.model
        if effort_budget is not None:
            normalized["reasoning_effort"] = BUDGET_TIER
        try:
            converted = recipe.ChatCompletionRequest(normalized).convert(self._options)
        except recipe.ConversionError as error:
            raise ValueError(_error_message(error.body)) from error
        prompt_ids = self._encoding.encode(converted.conversation)
        if image_urls is not None:
            rendered = self._encoding.render_conversation(converted.conversation)
            image_urls.extend(
                source.url or source.data_url for source in rendered.image_sources
            )
        if effort_budget is not None and body["thinking"]:
            prompt_ids = self._replace_effort_budget(prompt_ids, effort_budget)
        if task is not None:
            prompt_ids = self._attach_task(prompt_ids, task, body["thinking"])
        return prompt_ids

    def parse_start(
        self,
        *,
        thinking: bool,
        task: Optional[str] = None,
        continued: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Where output parsing begins after the prompt `encode` renders for these
        settings, as recipe's ThinkingInitialStage; None when the output is answer
        only. `continued` is the final assistant message the prompt continues."""
        if not thinking or (task is not None and task not in TASKS_AFTER_HEADER):
            return None
        recipe = load_deepseek_recipe()
        if continued is None:
            return recipe.ThinkingInitialStage.ReasoningStart
        if task is not None:
            # The continued prefix follows the task token.
            return recipe.ThinkingInitialStage.ContentContinuation
        body = pre_normalize(
            {
                "messages": [{"role": "user", "content": ""}, continued],
                "thinking": True,
                "continue_final_message": True,
            }
        )
        body["model"] = self.model
        try:
            converted = recipe.ChatCompletionRequest(body).convert(self._options)
        except recipe.ConversionError as error:
            raise ValueError(_error_message(error.body)) from error
        return converted.parsing_options.thinking_initial_stage

    def encode_text(self, text: str) -> list[int]:
        return self.hf_tokenizer.encode(text, add_special_tokens=False).ids

    def _replace_effort_budget(self, prompt_ids: list[int], budget: int) -> list[int]:
        # The effort line opens the prompt, so its first match is the line itself.
        rendered = self.encode_text(BUDGET_LINE.format(budget=BUDGET_TIER_VALUE))
        for start in range(len(prompt_ids) - len(rendered) + 1):
            if prompt_ids[start : start + len(rendered)] == rendered:
                end = start + len(rendered)
                return (
                    prompt_ids[:start]
                    + self.encode_text(BUDGET_LINE.format(budget=budget))
                    + prompt_ids[end:]
                )
        raise ValueError("deepseek-recipe rendered no reasoning effort line")

    def _attach_task(
        self, prompt_ids: list[int], task: str, thinking: bool
    ) -> list[int]:
        """A quick instruction task on the last user turn, as DeepSeek's checkpoint
        encoder renders it: `action` follows the generation header, other tasks
        replace it."""
        header = [
            self.hf_tokenizer.token_to_id(ASSISTANT_TOKEN),
            self.hf_tokenizer.token_to_id(
                THINK_START_TOKEN if thinking else THINK_END_TOKEN
            ),
        ]
        if prompt_ids[-len(header) :] != header:
            raise ValueError(
                "task requires the conversation to end with a user message"
            )
        task_id = self.hf_tokenizer.token_to_id(TASK_TOKENS[task])
        if task in TASKS_AFTER_HEADER:
            return prompt_ids + [task_id]
        return prompt_ids[: -len(header)] + [task_id]
