"""DeepSeek-V4 / V4.1 prompts through deepseek-recipe, plus the SGLang extensions
recipe does not model (quick instruction tasks, V4.1 integer effort)."""

from __future__ import annotations

import copy
import functools
import json
import uuid
from types import ModuleType
from typing import Any, Optional, Union

import msgspec

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
DYNAMIC_TOOLS_SPEC = "dsv41"


class NativeRequest(msgspec.Struct, frozen=True):
    """A Responses or Messages payload that recipe's own adapter converts, so tools
    it discovers or references mid-history render where recipe places them.
    `tool_name` is a named tool choice: only that tool stays visible."""

    protocol: str
    body: dict[str, Any]
    tool_name: Optional[str] = None


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


def _convert(recipe: ModuleType, request: Any, options: Any) -> Any:
    try:
        return request.convert(options)
    except recipe.ConversionError as error:
        raise ValueError(_error_message(error.body)) from error


def _splice_declarations(
    messages: list[Any], declarations: list[tuple[str, str, list[str]]]
) -> list[tuple[Any, Optional[list[str]]]]:
    """Pair each converted message with the names it declares. A declaring message
    is what conversion left between its markers: itself, or nothing when empty."""
    spliced = [(message, None) for message in messages]
    for opening, closing, names in declarations:
        start, end = (_marker_index(spliced, marker) for marker in (opening, closing))
        declared = [message for message, _ in spliced[start + 1 : end]]
        spliced[start : end + 1] = [(declared[0] if declared else None, names)]
    return spliced


def _marker_index(spliced: list[tuple[Any, Any]], marker: str) -> int:
    for index, (message, _) in enumerate(spliced):
        if message is not None and message.role == "system":
            if message.content == marker:
                return index
    raise ValueError("deepseek-recipe dropped a message that declares tools")


def _system_message(message: Any, tools: list[Any]) -> Any:
    """`message` (None for empty content) declaring `tools` after its content."""
    recipe = load_deepseek_recipe()
    content = message.content if message is not None else ""
    image_sources = message.image_sources if message is not None else []
    # The constructor re-derives image placeholders from the token's spellings.
    if image_sources and content.count(IMAGE_TOKEN) != len(image_sources):
        raise ValueError(
            f"a message that declares tools cannot spell {IMAGE_TOKEN} as text"
        )
    return recipe.SystemMessage(
        content=content,
        image_sources=image_sources,
        tools=tools or None,
        image_token=IMAGE_TOKEN,
    )


def _deferred(tool: Any) -> Any:
    recipe = load_deepseek_recipe()
    return recipe.ToolDefinition(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        strict=tool.strict,
        defer_loading=True,
    )


def _keep_tool(
    messages: list[Any], tools: list[Any], name: str
) -> tuple[list[Any], list[Any]]:
    """A named tool choice keeps only that tool, in the tool list and in earlier
    declarations, dropping declarations left with no content, as recipe does."""
    recipe = load_deepseek_recipe()
    kept = []
    for message in messages:
        if isinstance(message, recipe.SystemMessage) and message.tools:
            selected = [tool for tool in message.tools if tool.name == name]
            if not selected and not message.content:
                continue
            if len(selected) != len(message.tools):
                message = _system_message(message, tools=selected)
        kept.append(message)
    return kept, [tool for tool in tools if tool.name == name]


def _rebuild(
    recipe: ModuleType,
    messages: list[Any],
    tools: list[Any],
    *,
    choice: str,
    settings: Any,
) -> Any:
    """A conversation of `messages` and `tools` with the thinking mode, effort and
    response format of the converted `settings`."""
    response_format = settings.response_format
    formats = {
        "json_schema": {"response_schema": settings.response_schema},
        "regex": {"response_regex": settings.response_regex},
    }
    return recipe.Conversation(
        messages,
        thinking_mode=settings.thinking_mode,
        tools=tools,
        tool_choice=choice,
        reasoning_effort=settings.reasoning_effort,
        **formats.get(response_format, {"response_format": response_format}),
    )


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
        native: Optional[NativeRequest] = None,
    ) -> list[int]:
        """Prompt ids for `body` (see `pre_normalize`); a request recipe rejects
        raises ValueError. `image_urls` receives the images in placeholder order."""
        conversation = self.convert(body, effort_budget=effort_budget, native=native)
        prompt_ids = self.encode_conversation(conversation, effort_budget=effort_budget)
        if image_urls is not None:
            rendered = self._encoding.render_conversation(conversation)
            image_urls.extend(
                source.url or source.data_url for source in rendered.image_sources
            )
        if task is not None:
            prompt_ids = self._attach_task(prompt_ids, task, body["thinking"])
        return prompt_ids

    def convert(
        self,
        body: dict[str, Any],
        *,
        effort_budget: Optional[int] = None,
        native: Optional[NativeRequest] = None,
    ) -> Any:
        """Convert `body` to recipe's `Conversation`; invalid requests raise ValueError.
        Pass the same `effort_budget` to `encode_conversation`. With `native`, `body`
        supplies only the thinking, effort and response format."""
        recipe = load_deepseek_recipe()
        normalized = pre_normalize(body)
        normalized["model"] = self.model
        if effort_budget is not None:
            normalized["reasoning_effort"] = BUDGET_TIER
        choice = body.get("tool_choice")
        tool_name = choice["function"]["name"] if isinstance(choice, dict) else None
        messages, declared, declarations = self._take_message_tools(
            normalized["messages"], disabled=choice == "none"
        )
        normalized["messages"] = messages
        if declared:
            normalized["tools"] = normalized.get("tools", []) + declared
        settings = _convert(
            recipe,
            request=recipe.ChatCompletionRequest(normalized),
            options=self._options,
        ).conversation
        if native is not None:
            conversation = self._convert_native(native, thinking=settings.thinking_mode)
            messages, tools = list(conversation.messages), list(conversation.tools)
            tool_name = native.tool_name
        elif declarations:
            # Message-declared tools are deferred: they render where declared.
            conversation = settings
            deferred = {name for _, _, names in declarations for name in names}
            tools = [
                _deferred(tool) if tool.name in deferred else tool
                for tool in settings.tools
            ]
            by_name = {tool.name: tool for tool in tools}
            messages = [
                (
                    message
                    if names is None
                    else _system_message(
                        message, tools=[by_name[name] for name in names]
                    )
                )
                for message, names in _splice_declarations(
                    list(settings.messages), declarations=declarations
                )
            ]
        else:
            return settings
        if tool_name is not None:
            messages, tools = _keep_tool(messages, tools=tools, name=tool_name)
        return _rebuild(
            recipe,
            messages=messages,
            tools=tools,
            choice=conversation.tool_choice,
            settings=settings,
        )

    def native_tools(self, native: NativeRequest) -> list[Any]:
        """The tools the model can call after `native`'s history: those loaded from
        the start and the deferred ones a search result, reference or addition has
        loaded. Thinking does not change them."""
        recipe = load_deepseek_recipe()
        conversation = self._convert_native(native, thinking=False)
        loaded = {
            tool.name
            for message in conversation.messages
            if isinstance(message, recipe.SystemMessage) and message.tools
            for tool in message.tools
        }
        return [
            tool
            for tool in conversation.tools
            if not tool.defer_loading or tool.name in loaded
        ]

    def namespace_tool_map(
        self, tools: list[dict[str, Any]]
    ) -> dict[str, tuple[str, str]]:
        """Responses call names of namespace members mapped to (namespace, name).
        Pass the request tools, then the tools of each tool_search_output in input
        order: conversion registers them in that order."""
        recipe = load_deepseek_recipe()
        request = recipe.ResponsesRequest({"model": self.model, "tools": tools})
        try:
            return request.namespace_tool_map(self._options)
        except recipe.ConversionError as error:
            raise ValueError(_error_message(error.body)) from error

    def _convert_native(self, native: NativeRequest, *, thinking: bool) -> Any:
        recipe = load_deepseek_recipe()
        if self.model != MODEL_NAMES[DYNAMIC_TOOLS_SPEC]:
            raise ValueError(
                "DeepSeek-V4 cannot load tools mid-conversation (tool_search, "
                "tool_reference, tool_addition); serve DeepSeek-V4.1 for them"
            )
        request_type = {
            "responses": recipe.ResponsesRequest,
            "messages": recipe.MessagesRequest,
        }[native.protocol]
        # The handler's thinking decision, so history reasoning is kept or dropped
        # as the prompt renders it.
        options = self._options.with_default_thinking_mode(thinking)
        request = request_type({**native.body, "model": self.model})
        return _convert(recipe, request=request, options=options).conversation

    def _take_message_tools(
        self, messages: list[dict[str, Any]], *, disabled: bool
    ) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], list[tuple[str, str, list[str]]]
    ]:
        """`messages` without their tool declarations, each declaring message
        bracketed by marker system messages that conversion keeps even when its
        content is empty; the declared tools, for recipe to validate and normalize;
        and the markers with the names they declare. `disabled` drops declarations."""
        if not any(message.get("tools") for message in messages):
            return messages, [], []
        if self.model != MODEL_NAMES[DYNAMIC_TOOLS_SPEC]:
            raise ValueError(
                "DeepSeek-V4 does not render message-level tools; serve DeepSeek-V4.1"
            )
        token = uuid.uuid4().hex
        rewritten, declared, declarations = [], [], []
        for index, message in enumerate(messages):
            tools = message.get("tools")
            message = {key: value for key, value in message.items() if key != "tools"}
            if not tools or disabled:
                rewritten.append(message)
                continue
            if message["role"] != "system":
                raise ValueError(
                    f"messages[{index}]: only system and developer messages can "
                    "declare tools"
                )
            opening = f"<sglang message tools {token} {index}>"
            closing = f"</sglang message tools {token} {index}>"
            rewritten += [
                {"role": "system", "content": opening},
                message,
                {"role": "system", "content": closing},
            ]
            declared += tools
            declarations.append(
                (opening, closing, [tool["function"]["name"] for tool in tools])
            )
        return rewritten, declared, declarations

    def encode_conversation(
        self, conversation: Any, *, effort_budget: Optional[int] = None
    ) -> list[int]:
        """Encode a recipe `Conversation`, including a prefix of its messages."""
        prompt_ids = self._encoding.encode(conversation)
        if effort_budget is not None and conversation.thinking_mode:
            prompt_ids = self._replace_effort_budget(prompt_ids, effort_budget)
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
