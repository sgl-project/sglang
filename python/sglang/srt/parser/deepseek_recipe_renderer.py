"""DeepSeek-V4 / V4.1 prompts through deepseek-recipe, plus the SGLang extensions
recipe does not model (quick instruction tasks, V4.1 integer effort)."""

from __future__ import annotations

import copy
import functools
import itertools
import json
import logging
import re
import uuid
from types import ModuleType
from typing import Any, Optional, Union

import msgspec

DEEPSEEK_RECIPE_INSTALL_HINT = (
    "Serving DeepSeek-V4 / V4.1 requires the `deepseek-recipe` package (DeepSeek's "
    "reference encoder and parser; wheels for Linux and macOS). Install it with "
    "`pip install deepseek-recipe`."
)

logger = logging.getLogger(__name__)

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
TOOL_SEARCH = "tool_search"
TOOL_SEARCH_OUTPUT = "tool_search_output"
# recipe takes one definition per tool name, so a search result's redefinition of a
# callable function converts under such a name until restoration gives its own back.
REDEFINED_NAME_PREFIX = "sglang_redefined_"
# Converted in the search tool's place beside strict tools, keeping its position and
# name check: recipe reads its search tool as non-strict and rejects mixed strictness.
STRICT_SEARCH_STAND_IN = {
    "type": "function",
    "name": TOOL_SEARCH,
    "strict": True,
    "parameters": {"type": "object", "properties": {}},
}
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


def loaded_tools(conversation: Any) -> list[Any]:
    """The tools the model can call after `conversation`'s history: those it does
    not defer and the deferred ones a system message has loaded, each as the last
    system message to load it defines it."""
    recipe = load_deepseek_recipe()
    loaded = {
        tool.name: tool
        for message in conversation.messages
        if isinstance(message, recipe.SystemMessage)
        for tool in message.tools or []
    }
    return [
        loaded.get(tool.name, tool)
        for tool in conversation.tools
        if not tool.defer_loading or tool.name in loaded
    ]


def _replaced(conversation: Any, **fields: Any) -> Any:
    """`conversation` with `fields` (messages, tools or response_schema) replaced."""
    recipe = load_deepseek_recipe()
    return recipe.Conversation(
        fields.get("messages", conversation.messages),
        thinking_mode=conversation.thinking_mode,
        tools=fields.get("tools", conversation.tools),
        tool_choice=conversation.tool_choice,
        reasoning_effort=conversation.reasoning_effort,
        response_schema=fields.get("response_schema", conversation.response_schema),
    )


def _declares_strict_tool(tools: list[dict[str, Any]]) -> bool:
    members = [
        member
        for tool in tools
        if tool["type"] == "namespace"
        for member in tool["tools"]
    ]
    return any(tool.get("strict") for tool in [*tools, *members])


def _discovered_tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        tool
        for item in body["input"]
        if isinstance(item, dict) and item.get("type") == TOOL_SEARCH_OUTPUT
        for tool in item["tools"]
    ]


def _responses_namespaces(body: dict[str, Any]) -> list[dict[str, Any]]:
    """The namespace tools a Responses body declares and its history discovers, in
    the order recipe's conversion names their members."""
    return [
        tool
        for tool in [*body["tools"], *_discovered_tools(body)]
        if tool["type"] == "namespace"
    ]


def _definition(
    tool: dict[str, Any], namespace: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """The fields that render of `tool`, a function or custom tool or a member of
    `namespace`."""
    fields = {k: v for k, v in tool.items() if k != "defer_loading" and v is not None}
    if namespace is not None:
        fields["namespace"] = _definition(
            {k: v for k, v in namespace.items() if k != "tools"}
        )
    return fields


class _ResultEntry(msgspec.Struct, frozen=True):
    """One entry of a search result, in the order its JSON array lists them: the
    converted result's next entry when `converted`, which `alone`, if given,
    replaces; else `alone`'s rendering as a search result's only entry."""

    converted: bool
    alone: Optional[dict[str, Any]] = None


class _Relistings(msgspec.Struct, frozen=True, kw_only=True):
    """A Responses body as converted when its search results list tools that are
    already callable: a repeated definition is left out of its result, and a
    redefined function converts under a name of its own."""

    body: dict[str, Any]
    redefined: dict[str, str]
    """The name each redefinition converts under, mapped to the tool's own name."""
    results: dict[str, list[_ResultEntry]]
    """By call_id, the entries of each result that relists a tool."""


def _relistings(body: dict[str, Any]) -> _Relistings:
    callable_tools = {}
    for tool in body["tools"]:
        if tool["type"] == "namespace":
            for member in tool["tools"]:
                key = (tool["name"], member["name"])
                callable_tools[key] = _definition(member, tool)
        elif tool["type"] in ("function", "custom"):
            callable_tools[(None, tool["name"])] = _definition(tool)
    # A namespace member's call name always has its namespace's prefix, so only
    # top-level names can collide with a converted redefinition's.
    in_use = {tool.get("name") for tool in [*body["tools"], *_discovered_tools(body)]}
    unused = (
        name
        for name in map(f"{REDEFINED_NAME_PREFIX}{{}}".format, itertools.count())
        if name not in in_use
    )
    redefined = {}

    def listed(tool: dict[str, Any]) -> tuple[list[_ResultEntry], Optional[dict]]:
        """The entries `tool` renders as, and `tool` as converted; None when no part
        of it is."""
        if tool["type"] == "namespace":
            entries, members = [], []
            for member in tool["tools"]:
                key, definition = (
                    (tool["name"], member["name"]),
                    _definition(member, tool),
                )
                if callable_tools.get(key) == definition:
                    entries.append(_ResultEntry(False, {**tool, "tools": [member]}))
                    continue
                callable_tools[key] = definition
                entries.append(_ResultEntry(True))
                members.append(member)
            converted = {**tool, "tools": members}
            return entries, converted if members or not tool["tools"] else None
        key, definition = (None, tool["name"]), _definition(tool)
        earlier = callable_tools.get(key)
        callable_tools[key] = definition
        if earlier is None:
            return [_ResultEntry(True)], tool
        # recipe renders one apply_patch, whatever fields a listing gives it.
        if earlier == definition or tool["type"] == "custom":
            return [_ResultEntry(False, tool)], None
        name = next(unused)
        redefined[name] = tool["name"]
        return [_ResultEntry(True, tool)], {**tool, "name": name}

    items = []
    results = {}
    for item in body["input"]:
        if isinstance(item, dict) and item.get("type") == TOOL_SEARCH_OUTPUT:
            entries, tools = [], []
            for tool in item["tools"]:
                tool_entries, converted = listed(tool)
                entries += tool_entries
                if converted is not None:
                    tools.append(converted)
            if any(not entry.converted or entry.alone for entry in entries):
                results[item.get("call_id")] = entries
            item = {**item, "tools": tools}
        items.append(item)
    return _Relistings(
        body={**body, "input": items}, redefined=redefined, results=results
    )


def _json_array_items(array: str) -> list[str]:
    """The items of `array`, a compact JSON array, as it spells them."""
    decoder = json.JSONDecoder()
    items, index = [], 1
    while array[index] != "]":
        _, end = decoder.raw_decode(array, index)
        items.append(array[index:end])
        index = end + (array[end] == ",")
    return items


def _renamed(tool: Any, name: str) -> Any:
    recipe = load_deepseek_recipe()
    return recipe.ToolDefinition(
        name, tool.description, tool.parameters, tool.strict, tool.defer_loading
    )


def _error_message(body: Any) -> str:
    try:
        return json.loads(body)["error"]["message"]
    except (TypeError, ValueError, KeyError):
        return str(body)


def _take_message_tools(
    messages: list[dict[str, Any]], tool_choice: Any
) -> tuple[
    list[dict[str, Any]], list[tuple[str, str, list[dict[str, Any]]]], list[int]
]:
    """Brackets each declaring message with marker system messages, which conversion
    keeps even when it drops an empty declaring message. The last list holds the
    request index of each rewritten message."""
    token = uuid.uuid4().hex
    rewritten, declarations, origins = [], [], []
    for index, message in enumerate(messages):
        tools = message.get("tools") or []
        message = {key: value for key, value in message.items() if key != "tools"}
        if tool_choice == "none" or message["role"] != "system":
            tools = []
        elif isinstance(tool_choice, dict):
            name = tool_choice["function"]["name"]
            tools = [tool for tool in tools if tool["function"]["name"] == name]
        if not tools:
            rewritten.append(message)
            origins.append(index)
            continue
        opening = f"<sglang message tools {token} {index}>"
        closing = f"</sglang message tools {token} {index}>"
        rewritten += [
            {"role": "system", "content": opening},
            message,
            {"role": "system", "content": closing},
        ]
        origins += [index] * 3
        declarations.append((opening, closing, tools))
    return rewritten, declarations, origins


def _declare_message_tools(
    conversation: Any,
    declarations: list[tuple[str, str, list[dict[str, Any]]]],
    encoding: Any,
) -> Any:
    """Deferred tools render only where a system message declares them; the opening
    tools block keeps the call format even when every tool is deferred."""
    recipe = load_deepseek_recipe()
    declared = {
        tool["function"]["name"] for _, _, tools in declarations for tool in tools
    }
    tools = [
        (
            recipe.ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                strict=tool.strict,
                defer_loading=True,
            )
            if tool.name in declared
            else tool
        )
        for tool in conversation.tools
    ]
    by_name = {tool.name: tool for tool in tools}

    def special_ranges(message):
        alone = recipe.Conversation([message], thinking_mode=False)
        return encoding.render_conversation(alone).special_ranges

    messages = list(conversation.messages)
    for opening, closing, declaration in declarations:
        bracket = [
            index
            for index, message in enumerate(messages)
            if message.role == "system" and message.content in (opening, closing)
        ]
        if len(bracket) != 2:
            raise ValueError("deepseek-recipe dropped a message that declares tools")
        start, end = bracket
        # Conversion drops a declaring message whose content is empty.
        message = messages[start + 1] if end > start + 1 else recipe.SystemMessage("")
        # The constructor takes the token's first spellings as the image
        # placeholders, so one spelled as text ahead of an image would take its place.
        if message.image_sources and special_ranges(message) != special_ranges(
            recipe.SystemMessage(
                message.content,
                image_sources=message.image_sources,
                image_token=IMAGE_TOKEN,
            )
        ):
            raise ValueError(
                f"a message that declares tools cannot spell {IMAGE_TOKEN} as text "
                "before one of its images"
            )
        messages[start : end + 1] = [
            recipe.SystemMessage(
                message.content,
                image_sources=message.image_sources,
                tools=[by_name[tool["function"]["name"]] for tool in declaration],
                image_token=IMAGE_TOKEN,
            )
        ]
    response_format = conversation.response_format
    return recipe.Conversation(
        messages,
        thinking_mode=conversation.thinking_mode,
        tools=tools,
        tool_choice=conversation.tool_choice,
        reasoning_effort=conversation.reasoning_effort,
        response_format="json_object" if response_format == "json_object" else "text",
        response_schema=conversation.response_schema,
        response_regex=conversation.response_regex,
    )


class DeepSeekRecipe:
    """`tokenizer` is the checkpoint's `tokenizers.Tokenizer`; recipe encodes prompts
    with its own copy of it, and the SGLang extensions splice ids from the original."""

    def __init__(self, chat_encoding_spec: str, tokenizer: Any):
        recipe = load_deepseek_recipe()
        self.model = MODEL_NAMES[chat_encoding_spec]
        self.hf_tokenizer = tokenizer
        self._fast_prompt_tokenizer = None
        if type(tokenizer).__module__.partition(".")[0] == "fastokens":
            from fastokens import Tokenizer

            if hasattr(Tokenizer, "encode_segments"):
                self._fast_prompt_tokenizer = Tokenizer.from_json_str(
                    tokenizer.to_str()
                )
                logger.info("DeepSeek recipe prompt encoder: fastokens segmented")
            else:
                logger.warning(
                    "deepseek-recipe prompt acceleration requires fastokens.encode_segments; "
                    "using the reference prompt tokenizer"
                )
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
        conversation = self.convert(body, effort_budget=effort_budget)
        prompt_ids = self.encode_conversation(conversation, effort_budget=effort_budget)
        if image_urls is not None:
            image_urls.extend(self.image_urls(conversation))
        if task is not None:
            prompt_ids = self._attach_task(prompt_ids, task, body["thinking"])
        return prompt_ids

    def convert(
        self, body: dict[str, Any], *, effort_budget: Optional[int] = None
    ) -> Any:
        """Convert `body` to recipe's `Conversation`; invalid requests raise ValueError.
        Pass the same `effort_budget` to `encode_conversation`."""
        recipe = load_deepseek_recipe()
        normalized = pre_normalize(body)
        normalized["model"] = self.model
        if effort_budget is not None:
            normalized["reasoning_effort"] = BUDGET_TIER
        normalized["messages"], declarations, origins = _take_message_tools(
            normalized["messages"], tool_choice=body.get("tool_choice")
        )
        if declarations:
            if self.model == MODEL_NAMES["dsv4"]:
                raise ValueError(
                    "DeepSeek-V4 cannot render tools declared in messages; declare "
                    "them in the request's tools"
                )
            # Conversion validates and normalizes the declared tools; like the
            # request's tools, they drop None fields and take the type recipe requires.
            normalized["tools"] = normalized.get("tools", []) + [
                {"type": "function"}
                | {k: v for k, v in tool.items() if v is not None}
                | {
                    "function": {
                        k: v for k, v in tool["function"].items() if v is not None
                    }
                }
                for _, _, tools in declarations
                for tool in tools
            ]
        try:
            converted = recipe.ChatCompletionRequest(normalized).convert(self._options)
        except recipe.ConversionError as error:
            # Recipe locates an error by the index of the rewritten message.
            message = _error_message(error.body)
            located = re.match(r"\.messages\[(\d+)\]", message)
            if located and int(located[1]) < len(origins):
                index = origins[int(located[1])]
                message = f".messages[{index}]{message[located.end() :]}"
            raise ValueError(message) from error
        if not declarations:
            return converted.conversation
        return _declare_message_tools(
            converted.conversation, declarations, self._encoding
        )

    def convert_native(
        self, protocol: str, body: dict[str, Any], *, thinking: bool
    ) -> tuple[Any, dict[str, tuple[str, str]]]:
        """Convert a Responses or Messages request to recipe's `Conversation` in the
        host's thinking mode, with the namespace and name each namespace member's
        call name stands for; invalid requests raise ValueError."""
        recipe = load_deepseek_recipe()
        options = self._options.with_default_thinking_mode(thinking)
        if protocol == "messages":
            return self._converted(recipe.MessagesRequest, body, options), {}
        return self._convert_responses(body, options)

    def _converted(self, request_type: Any, body: dict[str, Any], options: Any) -> Any:
        recipe = load_deepseek_recipe()
        try:
            converted = request_type({**body, "model": self.model}).convert(options)
        except recipe.ConversionError as error:
            raise ValueError(_error_message(error.body)) from error
        return converted.conversation

    def _convert_responses(
        self, body: dict[str, Any], options: Any
    ) -> tuple[Any, dict[str, tuple[str, str]]]:
        """recipe's conversion of a Responses `body`, where strict tools may stand
        beside the search tool and a search result may list a callable tool again,
        and the namespace and name each namespace member's call name stands for."""
        recipe = load_deepseek_recipe()
        search = next(
            (tool for tool in body["tools"] if tool["type"] == TOOL_SEARCH), None
        )
        strict_search = search is not None and _declares_strict_tool(
            [*body["tools"], *_discovered_tools(body)]
        )
        relistings = _relistings(body)
        converted = relistings.body
        if strict_search:
            converted = {
                **converted,
                "tools": [
                    STRICT_SEARCH_STAND_IN if tool["type"] == TOOL_SEARCH else tool
                    for tool in converted["tools"]
                ],
            }
        conversation = self._converted(recipe.ResponsesRequest, converted, options)
        if strict_search:
            (definition,) = self._converted(
                recipe.ResponsesRequest, {"input": "", "tools": [search]}, options
            ).tools
            conversation = _replaced(
                conversation,
                tools=[
                    definition if tool.name == TOOL_SEARCH else tool
                    for tool in conversation.tools
                ],
            )
        if relistings.results:
            conversation = self._relistings_restored(
                conversation, relistings=relistings, options=options
            )
        names = self.namespace_tool_names(_responses_namespaces(converted))
        return conversation, names

    def _relistings_restored(
        self, conversation: Any, *, relistings: _Relistings, options: Any
    ) -> Any:
        """`conversation` whose search results list all the entries they were given,
        and whose redefinitions load under the tools' own names."""
        recipe = load_deepseek_recipe()
        rendered_alone: dict[str, str] = {}

        def alone(tool: dict[str, Any]) -> str:
            key = json.dumps(tool, sort_keys=True)
            if key not in rendered_alone:
                rendered_alone[key] = self._rendered_alone(tool, options)
            return rendered_alone[key]

        messages = []
        for message in conversation.messages:
            entries = isinstance(
                message, recipe.ToolMessage
            ) and relistings.results.get(message.tool_call_id)
            if entries:
                converted = iter(_json_array_items(message.content))
                rendered = []
                for entry in entries:
                    item = next(converted) if entry.converted else None
                    rendered.append(item if entry.alone is None else alone(entry.alone))
                message = recipe.ToolMessage(
                    "[" + ",".join(rendered) + "]", message.tool_call_id
                )
            elif isinstance(message, recipe.SystemMessage) and any(
                tool.name in relistings.redefined for tool in message.tools or []
            ):
                tools = [
                    (
                        _renamed(tool, relistings.redefined[tool.name])
                        if tool.name in relistings.redefined
                        else tool
                    )
                    for tool in message.tools
                ]
                message = recipe.SystemMessage(
                    message.content, message.image_sources, tools
                )
            messages.append(message)
        return _replaced(
            conversation,
            messages=messages,
            tools=[
                tool
                for tool in conversation.tools
                if tool.name not in relistings.redefined
            ],
        )

    def _rendered_alone(self, tool: dict[str, Any], options: Any) -> str:
        """recipe's rendering of `tool` as a search result's only entry; a namespace
        lists one member."""
        recipe = load_deepseek_recipe()
        call_id = "rendered_alone"
        searched = [
            {"type": "tool_search_call", "call_id": call_id, "arguments": {}},
            {"type": TOOL_SEARCH_OUTPUT, "call_id": call_id, "tools": [tool]},
        ]
        conversation = self._converted(
            recipe.ResponsesRequest, {"input": searched}, options
        )
        (message,) = (
            message
            for message in conversation.messages
            if isinstance(message, recipe.ToolMessage)
        )
        (entry,) = _json_array_items(message.content)
        return entry

    def with_host_settings(
        self, conversation: Any, *, response_schema: Optional[dict[str, Any]]
    ) -> Any:
        """`conversation` with what recipe's Responses and Messages conversions
        leave out: a JSON response schema, and SGLang's reading of a trailing
        assistant turn, re-sent as a user turn."""
        recipe = load_deepseek_recipe()
        messages = conversation.messages
        last = messages[-1] if messages else None
        if last is not None and last.role == "assistant":
            # recipe closes the turn and opens another; SGLang, as
            # `_handle_last_assistant_message`, re-sends it as a user turn.
            messages = [*messages[:-1], recipe.UserMessage(last.content)]
        elif response_schema is None:
            return conversation
        return _replaced(
            conversation,
            messages=messages,
            response_schema=(
                response_schema
                if response_schema is not None
                else conversation.response_schema
            ),
        )

    def namespace_tool_names(
        self, namespaces: list[dict[str, Any]]
    ) -> dict[str, tuple[str, str]]:
        """The name recipe's Responses conversion calls each member of `namespaces`
        by, mapped to its namespace and its name there. `namespaces` lists the
        request's namespace tools in conversion order: declared, then discovered."""
        recipe = load_deepseek_recipe()
        body = {"model": self.model, "input": "", "tools": namespaces}
        try:
            return recipe.ResponsesRequest(body).namespace_tool_map(self._options)
        except recipe.ConversionError as error:
            raise ValueError(_error_message(error.body)) from error

    def image_urls(self, conversation: Any) -> list[str]:
        """The images of `conversation` in the order its prompt places them."""
        rendered = self._encoding.render_conversation(conversation)
        return [source.url or source.data_url for source in rendered.image_sources]

    def encode_conversation(
        self, conversation: Any, *, effort_budget: Optional[int] = None
    ) -> list[int]:
        """Encode a recipe `Conversation`, including a prefix of its messages."""
        if self._fast_prompt_tokenizer is None:
            prompt_ids = self._encoding.encode(conversation)
        else:
            rendered = self._encoding.render_conversation(conversation)
            prompt = rendered.prompt
            segments = []
            cursor = 0
            # Recipe distinguishes control tokens from identical spellings in
            # user text. Its Python ranges use code-point offsets, not bytes.
            for start, end in rendered.special_ranges:
                if start > cursor:
                    segments.append((prompt[cursor:start], False))
                segments.append((prompt[start:end], True))
                cursor = end
            if cursor < len(prompt):
                segments.append((prompt[cursor:], False))
            prompt_ids = self._fast_prompt_tokenizer.encode_segments(segments).ids
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
