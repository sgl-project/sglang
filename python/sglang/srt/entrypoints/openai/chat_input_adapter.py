import copy
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from sglang.srt.entrypoints.chat_input.config import resolve_chat_model_config
from sglang.srt.entrypoints.chat_input.normalization import (
    normalize_reasoning_inputs,
    set_json_schema,
    set_tool_choice_default,
    validate_reasoning_effort_type,
)
from sglang.srt.entrypoints.chat_input.processor import ChatInputProcessor
from sglang.srt.entrypoints.chat_input.schema import Function, Tool
from sglang.srt.entrypoints.chat_input.types import ChatInput, PreparedChat
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ResponsesRequest,
    TokenizeRequest,
)
from sglang.srt.runtime_context import get_model, get_serving


def create_chat_input_processor(tokenizer_manager, template_manager):
    config = resolve_chat_model_config(
        tokenizer=tokenizer_manager.tokenizer,
        template_manager=template_manager,
        hf_config=tokenizer_manager.model_config.hf_config,
        model_path=tokenizer_manager.model_path,
        revision=get_model().revision,
        tool_call_parser=tokenizer_manager.config_value("tool_call_parser"),
        reasoning_parser=tokenizer_manager.config_value("reasoning_parser"),
        default_chat_template_kwargs=get_serving().default_chat_template_kwargs,
        is_multimodal=tokenizer_manager.model_config.is_multimodal,
    )
    return ChatInputProcessor(config)


def from_chat_request(request: ChatCompletionRequest) -> ChatInput:
    return ChatInput.model_validate(
        {name: copy.deepcopy(getattr(request, name)) for name in ChatInput.model_fields}
    )


def _validate_input(data: Dict[str, Any]) -> ChatInput:
    data = copy.deepcopy(data)
    data = set_tool_choice_default(data)
    data = normalize_reasoning_inputs(data)
    data = set_json_schema(data)
    validate_reasoning_effort_type(data.get("reasoning_effort"))
    return ChatInput.model_validate(data)


def from_responses_request(
    request: ResponsesRequest, messages: List[Dict[str, Any]]
) -> ChatInput:
    tools = [
        Tool(
            type="function",
            function=Function(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                strict=tool.strict,
            ),
        )
        for tool in request.tools or []
        if tool.type == "function"
    ]
    choice = request.effective_tool_choice() if tools else "none"
    if isinstance(choice, dict):
        choice = {"type": "function", "function": {"name": choice["name"]}}
    return _validate_input(
        {
            "messages": messages,
            "tools": tools or None,
            "tool_choice": choice,
            "parallel_tool_calls": (
                request.parallel_tool_calls
                if request.parallel_tool_calls is not None
                else True
            ),
            "stop": request.stop,
            "reasoning_effort": request.reasoning.effort if request.reasoning else None,
            "chat_template_kwargs": request.chat_template_kwargs,
        }
    )


def from_tokenize_request(request: TokenizeRequest) -> ChatInput:
    return _validate_input(
        request.model_dump(exclude={"prompt", "add_special_tokens"}, exclude_none=True)
    )


def with_prepared_options(request, prepared: PreparedChat):
    updates = {
        "chat_template_kwargs": copy.deepcopy(prepared.chat_template_kwargs),
        "skip_special_tokens": prepared.skip_special_tokens,
        "reasoning_effort": prepared.reasoning_effort,
    }
    fields = type(request).model_fields
    return request.model_copy(
        deep=True,
        update={key: value for key, value in updates.items() if key in fields},
    )


class _GenerationValidationOptions(BaseModel):
    """Keep coercion for legacy generation fields accepted by TokenizeRequest."""

    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    return_sampling_mask: bool = False
    return_meta_info: bool = False


def validate_chat_generation_options(request):
    options = _GenerationValidationOptions.model_validate(
        {
            name: getattr(request, name)
            for name in _GenerationValidationOptions.model_fields
            if hasattr(request, name)
        }
    )
    if options.return_sampling_mask and not options.return_meta_info:
        return "return_sampling_mask requires return_meta_info=true."
    max_output_tokens = options.max_completion_tokens or options.max_tokens
    server_context_length = get_model().context_length
    if (
        max_output_tokens
        and server_context_length
        and max_output_tokens > server_context_length
    ) and not get_serving().allow_auto_truncate:
        return (
            f"max_completion_tokens is too large: {max_output_tokens}."
            f"This model supports at most {server_context_length} completion tokens."
        )
    return None
