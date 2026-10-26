"""DeepSeek-V4 / V4.1 serving glue for the ``dsv4`` / ``dsv41`` chat encoding
specs: prompt encoding and output parsing through deepseek-recipe under SGLang's
request semantics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, Union

from sglang.srt.entrypoints.openai.chat_encoding_adapters import (
    ADAPTER_OUTPUT_PARSERS,
    select_new_output_ids,
    token_output_sse_chunks,
)
from sglang.srt.entrypoints.openai.protocol import (
    FunctionResponse,
    ToolCall,
)
from sglang.srt.environ import envs
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.parser.deepseek_recipe_output import (
    DeepSeekRecipeOutputParser,
    RecipeToolCall,
    RecipeToolCallDelta,
    parse_output_text,
    whole_tool_calls,
)
from sglang.srt.parser.deepseek_recipe_renderer import (
    DeepSeekRecipe,
    parse_effort_budget,
)
from sglang.srt.runtime_context import get_serving
from sglang.srt.utils import ImageData

if TYPE_CHECKING:
    from sglang.srt.entrypoints.openai.chat_encoding_adapters import ChatOutputHooks
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        ResponsesRequest,
        Tool,
    )

DEFAULT_EFFORT_ENVS = {
    "dsv4": envs.SGLANG_DSV4_REASONING_EFFORT,
    "dsv41": envs.SGLANG_DSV41_REASONING_EFFORT,
}
JSON_RESPONSE_FORMATS = ("json_object", "json_schema")


class DeepSeekRecipeChatEncodingAdapter:
    # recipe sorts tool results by call order, so images come from its render.
    collects_openai_media = False
    strict_tool_call_arguments = False  # recipe serializes object arguments itself.

    def __init__(
        self,
        *,
        chat_encoding_spec: str,
        tokenizer: Any,
        reasoning_parser: Optional[str],
        tool_call_parser: Optional[str],
        chat_hooks: ChatOutputHooks,
        default_chat_template_kwargs: Optional[dict[str, Any]],
    ):
        self._spec = chat_encoding_spec
        self.deepseek_recipe = DeepSeekRecipe.from_hf_tokenizer(
            chat_encoding_spec, tokenizer
        )
        raw = (DEFAULT_EFFORT_ENVS[chat_encoding_spec].get() or "").strip()
        self._default_effort = int(raw) if raw.isdigit() else raw or None
        # Encoded at boot, so a bad server default fails startup, not requests.
        defaults = default_chat_template_kwargs or {}
        self._encode(
            {"messages": [{"role": "user", "content": ""}], "thinking": True},
            effort=defaults.get("reasoning_effort"),
        )
        # recipe parses the output once a DeepSeek parser is configured; other
        # parsers keep the text path.
        reasoning_name, tool_name = ADAPTER_OUTPUT_PARSERS[chat_encoding_spec]
        self._token_output = (
            reasoning_parser == reasoning_name or tool_call_parser == tool_name
        )
        if self._token_output and (
            reasoning_parser not in (None, reasoning_name)
            or tool_call_parser not in (None, tool_name)
        ):
            raise ValueError(
                f"--reasoning-parser {reasoning_name} and --tool-call-parser "
                f"{tool_name} cannot be mixed with other parsers, got "
                f"{reasoning_parser!r} and {tool_call_parser!r}"
            )
        self._separate_reasoning = reasoning_parser is not None
        self._hooks = chat_hooks

    @property
    def output(self) -> Optional[DeepSeekRecipeChatEncodingAdapter]:
        return self if self._token_output else None

    def encode_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        request: ChatCompletionRequest,
        tools: Optional[list[dict]],
        thinking: bool,
        image_data: list,
    ) -> list[int]:
        # The wire request: the handler's message dumps carry None for unset fields.
        wire = request.model_dump(exclude_unset=True, by_alias=True, mode="json")
        wire_messages = wire["messages"]
        continue_final = request.continue_final_message
        task_prefix = None
        last = _trailing_assistant_turn(wire_messages)
        if last is not None:
            if not continue_final:
                wire_messages[-1] = {"role": "user", "content": last["content"]}
            elif request.task is not None:
                # A task renders its own tail, so the prefix follows it.
                task_prefix = wire_messages.pop()["content"]
                continue_final = False
        body = {
            "messages": wire_messages,
            "thinking": thinking,
            "continue_final_message": continue_final,
            "response_format": wire.get("response_format"),
        }
        if tools is not None:
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
            body["tool_choice"] = wire.get("tool_choice")
        body = {key: value for key, value in body.items() if value is not None}
        image_urls = []
        prompt_ids = self._encode(
            body,
            effort=request.reasoning_effort,
            task=request.task,
            image_urls=image_urls,
        )
        image_data.extend(ImageData(url=url) for url in image_urls)
        if task_prefix:
            prompt_ids += self.deepseek_recipe.encode_text(task_prefix)
        return prompt_ids

    def _encode(
        self,
        body: dict[str, Any],
        *,
        effort: Union[str, int, float, None],
        task: Optional[str] = None,
        image_urls: Optional[list[str]] = None,
    ) -> list[int]:
        if effort is None or effort == "none":
            effort = self._default_effort
        budget = None
        if effort is not None and not isinstance(effort, str):
            if self._spec != "dsv41":
                raise ValueError(
                    f"DeepSeek-V4 reasoning_effort must be a named tier, got {effort!r}"
                )
            budget = parse_effort_budget(effort)
        elif effort is not None:
            body["reasoning_effort"] = effort
        return self.deepseek_recipe.encode(
            body, task=task, effort_budget=budget, image_urls=image_urls
        )

    def parse_chat_response(
        self,
        *,
        request: ChatCompletionRequest,
        output_ids: list[int],
        finish_reason: dict[str, Any],
    ) -> tuple[Optional[str], str, Optional[list[ToolCall]], dict[str, Any]]:
        delta = self._new_output_parser(request).finish(
            output_ids, finish_reason, keep_matched_stop=request.no_stop_trim
        )
        calls = whole_tool_calls(delta)
        if not calls:
            return delta.reasoning or None, delta.content, None, finish_reason
        history_tool_calls_cnt = self._hooks.history_tool_calls_count(request)
        tool_calls = [
            ToolCall(
                id=self._tool_call_id(call.index, call.name, history_tool_calls_cnt),
                index=call.index,
                function=FunctionResponse(name=call.name, arguments=call.arguments),
            )
            for call in calls
        ]
        if finish_reason["type"] == "stop":
            finish_reason = {**finish_reason, "type": "tool_calls", "matched": None}
        return delta.reasoning or None, delta.content, tool_calls, finish_reason

    def stream_chat_chunks(
        self,
        *,
        content: dict[str, Any],
        index: int,
        request: ChatCompletionRequest,
        created: int,
        parser_dict: dict[int, Any],
        has_tool_calls: dict[int, bool],
        choice_logprobs: Optional[dict],
        finish_reason_type: Optional[str],
        usage: Optional[dict[str, Any]],
    ) -> list[str]:
        if index not in parser_dict:
            parser_dict[index] = self._new_output_parser(request)
        parser = parser_dict[index]
        new_output_ids = select_new_output_ids(
            output_ids=content["output_ids"],
            num_consumed_tokens=parser.num_consumed_tokens,
            completion_tokens=content["meta_info"]["completion_tokens"],
            finish_reason_type=finish_reason_type,
            incremental=get_serving().incremental_streaming_output,
        )
        if finish_reason_type is None:
            delta = parser.feed(new_output_ids)
        else:
            delta = parser.finish(
                new_output_ids,
                content["meta_info"]["finish_reason"],
                keep_matched_stop=request.no_stop_trim,
            )

        history_tool_calls_cnt = self._hooks.history_tool_calls_count(request)
        if delta.tool_calls:
            has_tool_calls[index] = True
        return token_output_sse_chunks(
            chunk_id=content["meta_info"]["id"],
            created=created,
            model=request.model,
            index=index,
            usage=usage,
            logprobs=choice_logprobs,
            reasoning=delta.reasoning,
            content=delta.content,
            tool_calls=[
                ToolCall(
                    # The first piece of a call names it; later pieces extend its
                    # arguments.
                    id=(
                        self._tool_call_id(
                            piece.index, piece.name, history_tool_calls_cnt
                        )
                        if piece.name is not None
                        else None
                    ),
                    index=piece.index,
                    function=FunctionResponse(
                        name=piece.name, arguments=piece.arguments
                    ),
                )
                for piece in delta.tool_calls
            ],
        )

    def parse_responses_output(
        self,
        *,
        request: ResponsesRequest,
        require_reasoning: bool,
        output_ids: list[int],
        finish_reason: Optional[dict[str, Any]],
        separate_reasoning: bool,
        parse_tool_calls: bool,
    ) -> tuple[str, str, list[ToolCallItem]]:
        parser = self._new_responses_parser(
            request, require_reasoning, separate_reasoning, parse_tool_calls
        )
        delta = parser.finish(output_ids, finish_reason or {"type": "stop"})
        return delta.reasoning, delta.content, _tool_call_items(whole_tool_calls(delta))

    def new_responses_stream(
        self,
        *,
        request: ResponsesRequest,
        require_reasoning: bool,
        separate_reasoning: bool,
        parse_tool_calls: bool,
    ) -> DeepSeekRecipeResponsesStream:
        return DeepSeekRecipeResponsesStream(
            self._new_responses_parser(
                request, require_reasoning, separate_reasoning, parse_tool_calls
            )
        )

    def separate_reasoning_text(self, text: str) -> tuple[str, str]:
        delta = parse_output_text(self.deepseek_recipe, text, parse_tool_calls=False)
        return delta.reasoning, delta.content

    def parse_function_call_text(
        self, text: str, tools: list[Tool]
    ) -> tuple[str, list[ToolCallItem]]:
        if not tools:
            return text, []  # as the text parsers: nothing to call
        delta = parse_output_text(
            self.deepseek_recipe, text, parse_tool_calls=True, thinking=False
        )
        # As in chat, a call to a tool outside `tools` is returned, at index -1.
        indices = {tool.function.name: i for i, tool in enumerate(tools)}
        calls = [
            ToolCallItem(
                tool_index=indices.get(call.name, -1),
                name=call.name,
                parameters=call.arguments,
            )
            for call in whole_tool_calls(delta)
        ]
        return delta.content, calls

    def _new_output_parser(
        self, request: ChatCompletionRequest
    ) -> DeepSeekRecipeOutputParser:
        response_format = request.response_format
        continued = (
            _trailing_assistant_turn(
                [
                    message.model_dump(exclude_unset=True, by_alias=True, mode="json")
                    for message in request.messages[-1:]
                ]
            )
            if request.continue_final_message
            else None
        )
        return DeepSeekRecipeOutputParser(
            self.deepseek_recipe,
            thinking_stage=self.deepseek_recipe.parse_start(
                thinking=self._hooks.thinking_requested(request),
                task=request.task,
                continued=continued,
            ),
            parse_tool_calls=request.task is None
            and self._hooks.tool_call_parsing_active(request),
            parse_json_output=response_format is not None
            and response_format.type in JSON_RESPONSE_FORMATS,
            separate_reasoning=self._separate_reasoning and request.separate_reasoning,
            stream_reasoning=request.stream_reasoning,
            stop=_stop_strings(request.stop),
        )

    def _new_responses_parser(
        self,
        request: ResponsesRequest,
        require_reasoning: bool,
        separate_reasoning: bool,
        parse_tool_calls: bool,
    ) -> DeepSeekRecipeOutputParser:
        return DeepSeekRecipeOutputParser(
            self.deepseek_recipe,
            thinking_stage=self.deepseek_recipe.parse_start(thinking=require_reasoning),
            parse_tool_calls=parse_tool_calls,
            separate_reasoning=separate_reasoning,
            stop=_stop_strings(request.stop),
        )

    def _tool_call_id(self, index: int, name: str, history_tool_calls_cnt: int) -> str:
        return self._hooks.tool_call_id(
            ToolCallItem(tool_index=index, name=name, parameters=""),
            history_tool_calls_cnt,
        )


class DeepSeekRecipeResponsesStream:
    def __init__(self, parser: DeepSeekRecipeOutputParser):
        self._parser = parser
        self._incremental = get_serving().incremental_streaming_output

    def step(
        self, chunk: dict[str, Any], finish_reason: Optional[dict[str, Any]]
    ) -> tuple[str, str, list[ToolCallItem]]:
        new_output_ids = select_new_output_ids(
            output_ids=chunk.get("output_ids") or [],
            num_consumed_tokens=self._parser.num_consumed_tokens,
            completion_tokens=chunk["meta_info"]["completion_tokens"],
            finish_reason_type=finish_reason["type"] if finish_reason else None,
            incremental=self._incremental,
        )
        if finish_reason is None:
            delta = self._parser.feed(new_output_ids)
        else:
            delta = self._parser.finish(new_output_ids, finish_reason)
        return delta.reasoning, delta.content, _tool_call_items(delta.tool_calls)


def _trailing_assistant_turn(
    messages: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """The trailing text turn SGLang continues or, without continue_final_message,
    re-sends as a user turn (as `_handle_last_assistant_message`)."""
    last = messages[-1] if messages else None
    if last and last["role"] == "assistant" and isinstance(last.get("content"), str):
        return last
    return None


def _stop_strings(stop: Union[str, list[str], None]) -> list[str]:
    return [stop] if isinstance(stop, str) else stop or []


def _tool_call_items(
    calls: Sequence[Union[RecipeToolCall, RecipeToolCallDelta]],
) -> list[ToolCallItem]:
    return [
        ToolCallItem(tool_index=call.index, name=call.name, parameters=call.arguments)
        for call in calls
    ]
