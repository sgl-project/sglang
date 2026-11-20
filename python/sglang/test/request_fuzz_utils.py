"""Malformed request bodies for the HTTP request-validation fuzz tests.

Type cases are derived from a pydantic request model: each field gets every JSON
kind its annotation rejects, with values lax-mode coercion cannot turn valid.
Value cases are listed by hand: the right JSON kind with a value a validation
rule rejects.
"""

import functools
import random
import types
import typing

import msgspec
from pydantic import BaseModel

from sglang.srt.entrypoints.openai import protocol

SEED = 50045

# null is left out on purpose: whether an explicit null on a defaulted field is
# rejected or treated as omitted is open in #50366.
_KINDS = ("object", "list", "str", "float", "int", "bool")


class FuzzCase(msgspec.Struct, frozen=True):
    field: str
    case_id: str
    # Merged over a valid request body.
    overrides: dict
    # Rejected only after SamplingParams.verify (in TokenizerManager or the
    # scheduler), so only a real server shows the status.
    needs_server: bool = False


def _accepted_kinds(annotation) -> set[str]:
    """JSON kinds that pydantic (lax mode) may accept for this annotation."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        return set().union(*(_accepted_kinds(a) for a in typing.get_args(annotation)))
    if origin is typing.Annotated:
        return _accepted_kinds(typing.get_args(annotation)[0])
    if origin is typing.Literal:
        # A non-matching value of a literal's own type is still malformed.
        return set()
    if annotation in (typing.Any, object):
        # protocol.py falls back to Any for StructuralTag when xgrammar is missing.
        return set(_KINDS)
    if origin in (list, typing.List):
        return {"list"}
    if origin in (dict, typing.Dict) or annotation in (dict, typing.Dict):
        return {"object"}
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return {"object"}
    if annotation is type(None):
        return set()
    if annotation is bool:
        # Lax mode coerces 0/1 and strings like "true" to bool.
        return {"bool"}
    if annotation is int:
        # Lax mode accepts integral floats and numeric strings; bool is accepted as 0/1.
        return {"int", "bool"}
    if annotation is float:
        return {"int", "float", "bool"}
    if annotation is str:
        return {"str"}
    raise TypeError(f"no fuzz rule for annotation {annotation!r}")


def _malformed_value(kind: str, rng: random.Random):
    # Non-numeric strings, non-integral floats and ints other than 0/1, so
    # lax-mode coercion cannot turn any of them into a valid value.
    suffix = rng.randrange(1_000_000)
    if kind == "object":
        return {f"fuzz_{suffix}": [suffix]}
    if kind == "list":
        return [{"fuzz": suffix}]
    if kind == "str":
        return f"fuzz-{suffix}-not-a-number"
    if kind == "float":
        return suffix + 0.5
    if kind == "int":
        return 2 + suffix
    if kind == "bool":
        return rng.choice([True, False])
    raise ValueError(kind)


def type_cases(model: type[BaseModel], seed: int = SEED) -> list[FuzzCase]:
    """One case per (field, JSON kind the field's annotation rejects)."""
    rng = random.Random(seed)
    cases = []
    for name, info in model.model_fields.items():
        # The wire key, e.g. "schema" for JsonSchemaResponseFormat.schema_.
        field = info.alias or name
        accepted = _accepted_kinds(info.annotation)
        for kind in _KINDS:
            if kind not in accepted:
                value = _malformed_value(kind, rng)
                cases.append(
                    FuzzCase(
                        field=field, case_id=f"type_{kind}", overrides={field: value}
                    )
                )
    return cases


def nested_type_cases(
    model: type[BaseModel],
    *,
    path: str,
    valid: dict,
    wrap: typing.Callable[[dict], dict],
    skip: tuple[str, ...] = (),
    seed: int = SEED,
) -> list[FuzzCase]:
    """type_cases for a nested object: each case puts one malformed field into
    `valid` (a valid instance of `model`), and `wrap` builds the request overrides."""
    return [
        FuzzCase(
            field=f"{path}.{case.field}",
            case_id=case.case_id,
            overrides=wrap({**valid, **case.overrides}),
        )
        for case in type_cases(model, seed)
        if case.field not in skip
    ]


def _value(field, case_id, value, needs_server=False, **extra) -> FuzzCase:
    return FuzzCase(
        field=field,
        case_id=case_id,
        overrides={field: value, **extra},
        needs_server=needs_server,
    )


# Ids far past any test model's vocabulary, so the cases hold for every model.
_HUGE = 10**9

# Each case is rejected by a distinct rule; see the stage noted per group.
COMPLETIONS_VALUE_CASES = [
    # OpenAIServingCompletion._validate_request
    _value("prompt", "empty_str", ""),
    _value("prompt", "empty_list", []),
    _value("prompt", "empty_nested_list", [[]]),
    _value("logprobs", "negative", -1),
    _value("logprobs", "above_vocab", _HUGE),
    # CompletionRequest field constraints
    _value("n", "zero", 0),
    _value("n", "above_max", 129),
    _value("max_tokens", "zero", 0),
    _value("max_tokens", "negative", -1),
    _value("return_hidden_states", "unknown_literal", "first"),
    # JSON parsing rejects an integer outside int64 before any validator runs.
    _value("priority", "above_int64", 2**70),
    _value("response_format", "unknown_type", {"type": "bogus"}),
    _value("stream_options", "include_usage_not_bool", {"include_usage": "maybe"}),
    # _convert_to_internal_request
    _value("response_format", "json_schema_without_schema", {"type": "json_schema"}),
    # GenerateReqInput.normalize_batch_and_arguments
    _value("cache_salt", "list_for_single_prompt", ["a", "b"]),
    _value("extra_key", "list_for_single_prompt", ["a", "b"]),
    # SamplingParams.normalize / verify
    _value("min_tokens", "negative", -1),
    _value("min_tokens", "above_max_tokens", 100),
    _value("temperature", "negative", -1.0),
    _value("top_p", "zero", 0.0),
    _value("top_p", "above_one", 1.5),
    # temperature=0 forces greedy top_k=1, which would hide a bad top_k.
    _value("top_k", "zero", 0, temperature=1.0),
    _value("top_k", "below_minus_one", -2, temperature=1.0),
    _value("min_p", "negative", -0.1),
    _value("min_p", "above_one", 1.5),
    _value("frequency_penalty", "above_two", 3.0),
    _value("presence_penalty", "below_minus_two", -3.0),
    _value("repetition_penalty", "zero", 0.0),
    _value("repetition_penalty", "above_two", 3.0),
    _value("seed", "above_uint64", 2**64),
    _value("logit_bias", "key_not_integer", {"abc": 1.0}),
    _value("logit_bias", "key_out_of_vocab", {str(_HUGE): 1.0}),
    _value("logit_bias", "value_above_max", {"1": 1e31}),
    _value("watermark", "unknown_field", {"bogus": 1}),
    _value("json_schema", "with_regex", "{}", regex="a"),
    _value("stop_regex", "unbalanced_paren", "("),
    # TokenizerManager.generate_request
    _value("prompt", "token_id_negative", [-1], needs_server=True),
    _value("prompt", "token_id_out_of_vocab", [_HUGE], needs_server=True),
    _value("routed_dp_rank", "negative", -5, needs_server=True),
    _value("custom_logit_processor", "not_enabled", "x", needs_server=True),
    _value("rid", "list_for_single_prompt", ["a", "b"], needs_server=True),
    # No session_params cases: /v1/completions does not forward the field (#28140).
]

# (field, case_id) -> the open PR that makes it a 4xx; drop the entry once it merges.
# A real server is never sent these: out-of-vocab prompt ids may take it down.
COMPLETIONS_KNOWN_FAILURES: dict[tuple[str, str], str] = {
    ("stop_regex", "unbalanced_paren"): "#40985",
    ("prompt", "token_id_negative"): "#31908",
    ("prompt", "token_id_out_of_vocab"): "#31908",
    ("rid", "list_for_single_prompt"): "#50709",
}


_USER = {"role": "user", "content": "Hello"}
_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}
_TOOL_CALL = {
    "id": "call_0",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
}
_ASSISTANT = {"role": "assistant", "content": "Hi", "tool_calls": [_TOOL_CALL]}
_NAMED_TOOL_CHOICE = {"type": "function", "function": {"name": "get_weather"}}


def _with_tool_call(**fields) -> dict:
    return {"messages": [_USER, {**_ASSISTANT, "tool_calls": [fields]}]}


def chat_nested_type_cases(seed: int = SEED) -> list[FuzzCase]:
    """type_cases for the objects nested in messages, tools and tool_choice."""
    nested = functools.partial(nested_type_cases, seed=seed)
    return [
        *nested(
            protocol.ChatCompletionMessageUserParam,
            path="messages[0]",
            valid=_USER,
            wrap=lambda message: {"messages": [message]},
        ),
        *nested(
            protocol.ChatCompletionMessageContentTextPart,
            path="messages[0].content[0]",
            valid={"type": "text", "text": "Hello"},
            wrap=lambda part: {"messages": [{"role": "user", "content": [part]}]},
        ),
        *nested(
            protocol.ChatCompletionMessageGenericParam,
            path="messages[1]",
            valid=_ASSISTANT,
            wrap=lambda message: {"messages": [_USER, message]},
        ),
        *nested(
            protocol.ToolCall,
            path="messages[1].tool_calls[0]",
            valid=_TOOL_CALL,
            wrap=lambda call: _with_tool_call(**call),
        ),
        *nested(
            protocol.FunctionResponse,
            path="messages[1].tool_calls[0].function",
            valid=_TOOL_CALL["function"],
            wrap=lambda fn: _with_tool_call(**{**_TOOL_CALL, "function": fn}),
        ),
        *nested(
            protocol.Tool,
            path="tools[0]",
            valid=_TOOL,
            wrap=lambda tool: {"tools": [tool]},
        ),
        *nested(
            protocol.Function,
            path="tools[0].function",
            valid=_TOOL["function"],
            wrap=lambda fn: {"tools": [{"type": "function", "function": fn}]},
            # Internal: a before-validator overwrites it from the keys sent.
            skip=("key_order",),
        ),
        *nested(
            protocol.ToolChoice,
            path="tool_choice",
            valid=_NAMED_TOOL_CHOICE,
            wrap=lambda choice: {"tool_choice": choice, "tools": [_TOOL]},
        ),
        *nested(
            protocol.ToolChoiceFuncName,
            path="tool_choice.function",
            valid={"name": "get_weather"},
            wrap=lambda fn: {
                "tool_choice": {"type": "function", "function": fn},
                "tools": [_TOOL],
            },
        ),
        *nested(
            protocol.JsonSchemaResponseFormat,
            path="response_format.json_schema",
            valid={"name": "answer", "schema": {"type": "object"}},
            wrap=lambda js: {
                "response_format": {"type": "json_schema", "json_schema": js}
            },
        ),
        *nested(
            protocol.StreamOptions,
            path="stream_options",
            valid={"include_usage": True},
            wrap=lambda options: {"stream_options": options},
        ),
    ]


# Each case is rejected by a distinct rule; see the stage noted per group.
CHAT_VALUE_CASES = [
    # OpenAIServingChat._validate_request
    _value("messages", "empty_list", []),
    _value(
        "messages",
        "image_on_text_model",
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "http://x/a.png"}}
                ],
            }
        ],
    ),
    _value(
        "messages",
        "duplicate_tool_names",
        [{"role": "system", "content": "s", "tools": [_TOOL]}, _USER],
        tools=[_TOOL],
    ),
    _value("chat_template_kwargs", "chat_template_not_trusted", {"chat_template": "x"}),
    _value("return_sampling_mask", "without_return_meta_info", True),
    _value("return_flat_raw_output_top_logprobs", "without_return_meta_info", True),
    _value("top_logprobs", "above_vocab", _HUGE, logprobs=True),
    _value("tool_choice", "required_without_tools", "required"),
    _value("tool_choice", "named_without_tools", _NAMED_TOOL_CHOICE),
    _value(
        "tool_choice",
        "named_not_in_tools",
        {"type": "function", "function": {"name": "nope"}},
        tools=[_TOOL],
    ),
    _value(
        "tools",
        "invalid_parameters_schema",
        [
            {
                "type": "function",
                "function": {"name": "f", "parameters": {"type": "bogus"}},
            }
        ],
    ),
    _value(
        "response_format",
        "json_schema_without_schema",
        {"type": "json_schema", "json_schema": {"name": "answer"}},
    ),
    # ChatCompletionRequest field constraints and validators
    _value("messages", "unknown_role", [{"role": "wizard", "content": "hi"}]),
    _value("messages", "user_without_content", [{"role": "user"}]),
    _value(
        "messages",
        "unknown_content_part",
        [{"role": "user", "content": [{"type": "bogus"}]}],
    ),
    _value(
        "messages",
        "tool_call_without_name",
        [
            _USER,
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"arguments": "{}"},
                    }
                ],
            },
        ],
    ),
    _value("n", "zero", 0),
    _value("n", "above_max", 129),
    _value("max_tokens", "negative", -1),
    _value("max_completion_tokens", "negative", -1),
    _value("max_completion_tokens", "bool", True),
    _value("top_logprobs", "negative", -1, logprobs=True),
    _value("temperature", "negative", -1.0),
    _value("reasoning_effort", "unknown_tier", "bogus"),
    _value("reasoning_effort", "above_max", 1.5),
    _value("reasoning_effort", "bool", True),
    _value("thinking", "unknown_effort", {"effort": "bogus"}),
    _value("response_format", "unknown_type", {"type": "bogus"}),
    _value(
        "response_format",
        "json_schema_strict_not_bool",
        {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object"},
                "strict": "yes",
            },
        },
    ),
    _value("doc_split_options", "zero_pages", {"max_batch_pages": 0}),
    _value("seed", "above_uint64", 2**64),
    # JSON parsing rejects an integer outside int64 before any validator runs.
    _value("priority", "above_int64", 2**70),
    # OpenAIServingChat._convert_to_internal_request
    _value("return_meta_info", "with_stream", True, stream=True),
    _value("return_prompt_token_ids", "with_stream", True, stream=True),
    _value("return_token_ids", "with_stream", True, stream=True),
    _value(
        "messages",
        "tool_call_arguments_not_json",
        [
            _USER,
            {
                **_ASSISTANT,
                "tool_calls": [
                    {
                        **_TOOL_CALL,
                        "function": {"name": "get_weather", "arguments": "{x"},
                    }
                ],
            },
        ],
    ),
    _value(
        "tool_choice",
        "required_with_response_format",
        "required",
        tools=[_TOOL],
        response_format={"type": "json_object"},
    ),
    # GenerateReqInput.normalize_batch_and_arguments
    _value("cache_salt", "list_for_single_prompt", ["a", "b"]),
    _value("extra_key", "list_for_single_prompt", ["a", "b"]),
    _value("input_ids", "empty", []),
    # SamplingParams.normalize / verify
    _value("min_tokens", "negative", -1),
    _value("min_tokens", "above_max_tokens", 100),
    _value("top_p", "zero", 0.0),
    _value("top_p", "above_one", 1.5),
    # temperature=0 forces greedy top_k=1, which would hide a bad top_k.
    _value("top_k", "zero", 0, temperature=1.0),
    _value("top_k", "below_minus_one", -2, temperature=1.0),
    _value("min_p", "negative", -0.1),
    _value("min_p", "above_one", 1.5),
    _value("frequency_penalty", "above_two", 3.0),
    _value("presence_penalty", "below_minus_two", -3.0),
    _value("repetition_penalty", "zero", 0.0),
    _value("repetition_penalty", "above_two", 3.0),
    _value("logit_bias", "key_not_integer", {"abc": 1.0}),
    _value("logit_bias", "key_out_of_vocab", {str(_HUGE): 1.0}),
    _value("logit_bias", "value_above_max", {"1": 1e31}),
    _value("regex", "with_ebnf", "a", ebnf='root ::= "a"'),
    _value("watermark", "unknown_field", {"bogus": 1}),
    _value("stop_regex", "unbalanced_paren", "("),
    # Rejected against the served model's context length or TokenizerManager state.
    _value("max_completion_tokens", "above_context", _HUGE, needs_server=True),
    _value("lora_path", "not_enabled", "does-not-exist", needs_server=True),
    _value("routed_dp_rank", "negative", -5, needs_server=True),
    _value("custom_logit_processor", "not_enabled", "x", needs_server=True),
    _value("rid", "list_for_single_prompt", ["a", "b"], needs_server=True),
    _value("input_ids", "token_id_negative", [-1], needs_server=True),
    _value("input_ids", "token_id_out_of_vocab", [_HUGE], needs_server=True),
    # No session_params cases: /v1/chat/completions does not forward the field.
]

# Values ChatCompletionRequest accepts on purpose (see the NOTE on its OpenAI
# limits); a validator that starts rejecting them breaks existing clients.
CHAT_ACCEPTED_CASES = [
    _value("temperature", "above_openai_max", 5.0),
    _value("top_logprobs", "without_logprobs", 5),
    _value("logit_bias", "beyond_openai_range", {"1": 1000.0}),
]

# (field, case_id) -> the open PR that makes it a 4xx; drop the entry once it merges.
# A real server is never sent these: out-of-vocab input ids may take it down.
CHAT_KNOWN_FAILURES: dict[tuple[str, str], str] = {
    # set_json_schema calls .get on a non-object response_format: 500.
    **{
        ("response_format", f"type_{kind}"): "#36011"
        for kind in ("list", "str", "float", "int", "bool")
    },
    ("stop_regex", "unbalanced_paren"): "#40985",
    ("rid", "list_for_single_prompt"): "#50709",
    ("input_ids", "token_id_negative"): "#31908",
    ("input_ids", "token_id_out_of_vocab"): "#31908",
}
