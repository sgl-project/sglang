"""Malformed requests to /v1/completions, /v1/chat/completions and /generate get a
4xx from a real server, which keeps serving: a valid request to the same route
succeeds after every case, and /health passes at the end.

Type cases are derived from each route's request schema: every field gets each
JSON kind its annotation rejects. Value cases are listed by hand: the right JSON
kind with a value a validation rule rejects.
"""

import types
import typing
import unittest
from typing import Callable, NamedTuple

import msgspec
import requests
from pydantic import BaseModel

from sglang.api.v1 import api_types
from sglang.srt.entrypoints.openai import protocol
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-small")

COMPLETIONS = "/v1/completions"
CHAT = "/v1/chat/completions"
GENERATE = "/generate"

MODEL = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
_GENERATE_SAMPLING = {"max_new_tokens": 1, "temperature": 0}
VALID_BODIES = {
    COMPLETIONS: {
        "model": MODEL,
        "prompt": "The capital of France is",
        "max_tokens": 1,
        "temperature": 0,
    },
    CHAT: {
        "model": MODEL,
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "max_tokens": 1,
        "temperature": 0,
    },
    GENERATE: {
        "text": "The capital of France is",
        "sampling_params": _GENERATE_SAMPLING,
    },
}


class Case(NamedTuple):
    field: str
    case_id: str
    # Merged over the route's valid body.
    overrides: dict


# One value per JSON kind that lax-mode coercion cannot make valid: no numeric
# string, no integral float, no int that reads as a bool. null is left out on
# purpose: whether an explicit null on a defaulted field is rejected or treated
# as omitted is open in #50366.
_MALFORMED = {
    "object": {"malformed": [1]},
    "list": [{"malformed": 1}],
    "str": "malformed-not-a-number",
    "float": 1.5,
    "int": 2,
    "bool": True,
}


def _accepted_kinds(annotation) -> set[str]:
    """JSON kinds that pydantic (lax mode) or msgspec may accept for this annotation."""
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
        return set(_MALFORMED)
    if origin in (list, typing.List):
        return {"list"}
    if origin in (dict, typing.Dict) or annotation in (dict, typing.Dict):
        return {"object"}
    if isinstance(annotation, type) and issubclass(
        annotation, (BaseModel, msgspec.Struct)
    ):
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
    raise TypeError(f"no type-case rule for annotation {annotation!r}")


def _wire_fields(model) -> list[tuple[str, typing.Any]]:
    """(JSON key, annotation) for a pydantic model or a msgspec struct."""
    if issubclass(model, msgspec.Struct):
        return [(f.encode_name, f.type) for f in msgspec.structs.fields(model)]
    # The alias is the wire key, e.g. "schema" for JsonSchemaResponseFormat.schema_.
    return [
        (info.alias or name, info.annotation)
        for name, info in model.model_fields.items()
    ]


def type_cases(model) -> list[Case]:
    """One case per (field, JSON kind the field's annotation rejects)."""
    return [
        Case(field, f"type_{kind}", {field: value})
        for field, annotation in _wire_fields(model)
        for kind, value in _MALFORMED.items()
        if kind not in _accepted_kinds(annotation)
    ]


def nested_type_cases(
    model,
    *,
    path: str,
    valid: dict,
    wrap: Callable[[dict], dict],
    skip: tuple[str, ...] = (),
) -> list[Case]:
    """type_cases for a nested object: each case puts one malformed field into
    `valid` (a valid instance of `model`), and `wrap` builds the request overrides."""
    return [
        Case(f"{path}.{case.field}", case.case_id, wrap({**valid, **case.overrides}))
        for case in type_cases(model)
        if case.field not in skip
    ]


def _value(field, case_id, value, **extra) -> Case:
    return Case(field, case_id, {field: value, **extra})


def _sampling(case_id, **params) -> Case:
    """A /generate case on sampling_params; the first keyword is the field under
    test, any others support it."""
    field = next(iter(params))
    return Case(
        f"sampling_params.{field}",
        case_id,
        {"sampling_params": {**_GENERATE_SAMPLING, **params}},
    )


# Far past any test model's vocabulary and context length, so the cases hold for
# every model.
_HUGE = 10**9

# Value cases are grouped by the stage that rejects them.
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
    _value("response_format", "unknown_type", {"type": "bogus"}),
    _value("stream_options", "include_usage_not_bool", {"include_usage": "maybe"}),
    # orjson reads an integer beyond 64 bits as a float, which an int field rejects.
    _value("priority", "above_int64", 2**70),
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
    _value("seed", "above_int64", 2**63),
    _value("logit_bias", "key_not_integer", {"abc": 1.0}),
    _value("logit_bias", "key_out_of_vocab", {str(_HUGE): 1.0}),
    _value("logit_bias", "value_above_max", {"1": 1e31}),
    _value("watermark", "unknown_field", {"bogus": 1}),
    _value("json_schema", "with_regex", "{}", regex="a"),
    _value("stop_regex", "unbalanced_paren", "("),
    # TokenizerManager.generate_request
    _value("prompt", "token_id_negative", [-1]),
    _value("prompt", "token_id_out_of_vocab", [_HUGE]),
    _value("routed_dp_rank", "negative", -5),
    _value("custom_logit_processor", "not_enabled", "x"),
    _value("rid", "list_for_single_prompt", ["a", "b"]),
    # No session_params cases: /v1/completions does not forward the field (#28140).
]

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


# type_cases for the objects nested in messages, tools and tool_choice.
CHAT_NESTED_TYPE_CASES = [
    *nested_type_cases(
        protocol.ChatCompletionMessageUserParam,
        path="messages[0]",
        valid=_USER,
        wrap=lambda message: {"messages": [message]},
    ),
    *nested_type_cases(
        protocol.ChatCompletionMessageContentTextPart,
        path="messages[0].content[0]",
        valid={"type": "text", "text": "Hello"},
        wrap=lambda part: {"messages": [{"role": "user", "content": [part]}]},
    ),
    *nested_type_cases(
        protocol.ChatCompletionMessageGenericParam,
        path="messages[1]",
        valid=_ASSISTANT,
        wrap=lambda message: {"messages": [_USER, message]},
    ),
    *nested_type_cases(
        protocol.ToolCall,
        path="messages[1].tool_calls[0]",
        valid=_TOOL_CALL,
        wrap=lambda call: _with_tool_call(**call),
    ),
    *nested_type_cases(
        protocol.FunctionResponse,
        path="messages[1].tool_calls[0].function",
        valid=_TOOL_CALL["function"],
        wrap=lambda fn: _with_tool_call(**{**_TOOL_CALL, "function": fn}),
    ),
    *nested_type_cases(
        protocol.Tool,
        path="tools[0]",
        valid=_TOOL,
        wrap=lambda tool: {"tools": [tool]},
    ),
    *nested_type_cases(
        protocol.Function,
        path="tools[0].function",
        valid=_TOOL["function"],
        wrap=lambda fn: {"tools": [{"type": "function", "function": fn}]},
        # Internal: a before-validator overwrites it from the keys sent.
        skip=("key_order",),
    ),
    *nested_type_cases(
        protocol.ToolChoice,
        path="tool_choice",
        valid=_NAMED_TOOL_CHOICE,
        wrap=lambda choice: {"tool_choice": choice, "tools": [_TOOL]},
    ),
    *nested_type_cases(
        protocol.ToolChoiceFuncName,
        path="tool_choice.function",
        valid={"name": "get_weather"},
        wrap=lambda fn: {
            "tool_choice": {"type": "function", "function": fn},
            "tools": [_TOOL],
        },
    ),
    *nested_type_cases(
        protocol.JsonSchemaResponseFormat,
        path="response_format.json_schema",
        valid={"name": "answer", "schema": {"type": "object"}},
        wrap=lambda js: {"response_format": {"type": "json_schema", "json_schema": js}},
    ),
    *nested_type_cases(
        protocol.StreamOptions,
        path="stream_options",
        valid={"include_usage": True},
        wrap=lambda options: {"stream_options": options},
    ),
]

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
    # orjson reads an integer beyond 64 bits as a float, which an int field rejects.
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
    _value("seed", "above_int64", 2**63),
    _value("logit_bias", "key_not_integer", {"abc": 1.0}),
    _value("logit_bias", "key_out_of_vocab", {str(_HUGE): 1.0}),
    _value("logit_bias", "value_above_max", {"1": 1e31}),
    _value("regex", "with_ebnf", "a", ebnf='root ::= "a"'),
    _value("watermark", "unknown_field", {"bogus": 1}),
    _value("stop_regex", "unbalanced_paren", "("),
    # Served model's context length and TokenizerManager state
    _value("max_completion_tokens", "above_context", _HUGE),
    _value("lora_path", "not_enabled", "does-not-exist"),
    _value("routed_dp_rank", "negative", -5),
    _value("custom_logit_processor", "not_enabled", "x"),
    _value("rid", "list_for_single_prompt", ["a", "b"]),
    _value("input_ids", "token_id_negative", [-1]),
    _value("input_ids", "token_id_out_of_vocab", [_HUGE]),
    # No session_params cases: /v1/chat/completions does not forward the field.
]

# Values ChatCompletionRequest accepts on purpose (see the NOTE on its OpenAI
# limits); a later stage that starts rejecting them breaks existing clients.
CHAT_ACCEPTED_CASES = [
    _value("temperature", "above_openai_max", 5.0),
    _value("top_logprobs", "without_logprobs", 5),
    _value("logit_bias", "beyond_openai_range", {"1": 1000.0}),
]

_BATCH = {"text": ["a", "b"]}

GENERATE_VALUE_CASES = [
    # The api.v1 JSON contract (generate_contract_error)
    _sampling("unknown_field", bogus=1),
    _sampling("unknown_field", watermark={"bogus": 1}),
    _value("max_thinking_tokens", "negative", -1),
    # GenerateReqInput dataclass parse; orjson reads an int beyond 64 bits as a float.
    _value("text", "list_of_int", [1, 2]),
    _value("priority", "above_int64", 2**70),
    _value("custom_labels", "value_not_string", {"a": 1}),
    # GenerateReqInput.normalize_batch_and_arguments
    _value("text", "null_without_input_ids", None),
    _value("input_ids", "empty", [], text=None),
    _value("session_id", "with_session_params", "s", session_params={"id": "s"}),
    _value("return_flat_raw_top_logprobs_b64", "without_flat_raw", True),
    _value("kv_hints", "list_for_single_prompt", [{}]),
    _value("cache_salt", "list_for_single_prompt", ["a", "b"]),
    _value("extra_key", "list_for_single_prompt", ["a", "b"]),
    _value("rid", "batch_length_mismatch", ["x"], **_BATCH),
    _value("rid", "batch_duplicate", ["x", "x"], **_BATCH),
    _value("cache_salt", "batch_length_mismatch", ["a"], **_BATCH),
    _value("extra_key", "batch_length_mismatch", ["a"], **_BATCH),
    _value("return_hidden_states", "batch_length_mismatch", [True], **_BATCH),
    _value("image_data", "batch_length_mismatch", [["x"]], **_BATCH),
    # SamplingParams.normalize / verify
    _sampling("negative", max_new_tokens=-1),
    _sampling("negative", min_new_tokens=-1),
    _sampling("above_max_new_tokens", min_new_tokens=100),
    _sampling("negative", temperature=-1.0),
    _sampling("zero", top_p=0.0),
    _sampling("above_one", top_p=1.5),
    # temperature=0 forces greedy top_k=1, which would hide a bad top_k.
    _sampling("zero", top_k=0, temperature=1.0),
    _sampling("below_minus_one", top_k=-2, temperature=1.0),
    _sampling("negative", min_p=-0.1),
    _sampling("above_one", min_p=1.5),
    _sampling("above_two", frequency_penalty=3.0),
    _sampling("below_minus_two", presence_penalty=-3.0),
    _sampling("zero", repetition_penalty=0.0),
    _sampling("above_two", repetition_penalty=3.0),
    _sampling("above_int64", sampling_seed=2**63),
    _sampling("zero", n=0),
    _sampling("zero", beam_width=0),
    _sampling("key_not_integer", logit_bias={"abc": 1.0}),
    _sampling("key_out_of_vocab", logit_bias={str(_HUGE): 1.0}),
    _sampling("value_above_max", logit_bias={"1": 1e31}),
    _sampling("with_ebnf", regex="a", ebnf='root ::= "a"'),
    _sampling("unbalanced_paren", stop_regex="("),
    # TokenizerManager._validate_one_request and request state
    _value("top_logprobs_num", "above_vocab", _HUGE, return_logprob=True),
    _value("top_logprobs_num", "negative", -1, return_logprob=True),
    _value("logprob_start_len", "below_minus_one", -2, return_logprob=True),
    _value("token_ids_logprob", "negative", [-1]),
    _value("token_ids_logprob", "out_of_vocab", [_HUGE]),
    _value("routed_dp_rank", "negative", -5),
    _value("custom_logit_processor", "not_enabled", "x"),
    _value("lora_path", "not_enabled", "does-not-exist"),
    _value("rid", "list_for_single_prompt", ["a", "b"]),
    _value("return_logprob", "batch_length_mismatch", [True], **_BATCH),
    _value("input_ids", "token_id_negative", [-1], text=None),
    _value("input_ids", "token_id_out_of_vocab", [_HUGE], text=None),
    # A prompt that tokenizes to zero tokens (no BOS) reaches the model forward.
    _value("text", "empty_str", ""),
]

CASES = {
    COMPLETIONS: type_cases(protocol.CompletionRequest) + COMPLETIONS_VALUE_CASES,
    CHAT: (
        type_cases(protocol.ChatCompletionRequest)
        + CHAT_NESTED_TYPE_CASES
        + CHAT_VALUE_CASES
    ),
    GENERATE: (
        type_cases(api_types.GenerateRequest)
        + nested_type_cases(
            api_types.SamplingParams,
            path="sampling_params",
            valid=_GENERATE_SAMPLING,
            wrap=lambda params: {"sampling_params": params},
        )
        + GENERATE_VALUE_CASES
    ),
}

# (route, field, case_id) -> the open fix. These still get a 5xx, and the test
# asserts they do: once a fix changes that, the test fails until its entry
# here is dropped.
KNOWN_FAILURES: dict[tuple[str, str, str], str] = {
    (COMPLETIONS, "stop_regex", "unbalanced_paren"): "#40985",
    (CHAT, "stop_regex", "unbalanced_paren"): "#40985",
    (GENERATE, "sampling_params.stop_regex", "unbalanced_paren"): "#40985",
    # set_json_schema calls .get on a non-object response_format.
    **{
        (CHAT, "response_format", f"type_{kind}"): "#36011"
        for kind in ("list", "str", "float", "int", "bool")
    },
    (GENERATE, "return_logprob", "batch_length_mismatch"): "#44333",
}

# (route, field, case_id) -> the open fix. These take the server down, so they
# are never sent; drop the entry when its fix lands.
KNOWN_CRASHES: dict[tuple[str, str, str], str] = {
    # Out-of-vocabulary ids index the embedding out of bounds.
    (COMPLETIONS, "prompt", "token_id_negative"): "#31908",
    (COMPLETIONS, "prompt", "token_id_out_of_vocab"): "#31908",
    (CHAT, "input_ids", "token_id_negative"): "#31908",
    (CHAT, "input_ids", "token_id_out_of_vocab"): "#31908",
    (GENERATE, "input_ids", "token_id_negative"): "#31908",
    (GENERATE, "input_ids", "token_id_out_of_vocab"): "#31908",
    (GENERATE, "text", "empty_str"): "#44330",
}


class TestMalformedRequests(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def _post(self, route, overrides):
        return requests.post(
            f"{self.base_url}{route}",
            json={**VALID_BODIES[route], **overrides},
            timeout=30,
        )

    def test_malformed_requests_get_4xx_and_server_keeps_serving(self):
        for route, cases in CASES.items():
            for case in cases:
                key = (route, case.field, case.case_id)
                if key in KNOWN_CRASHES:
                    continue
                with self.subTest(route=route, field=case.field, case=case.case_id):
                    response = self._post(route, case.overrides)
                    rejected = 400 <= response.status_code < 500
                    detail = f"{case.overrides!r}: {response.status_code} {response.text[:300]}"
                    fix = KNOWN_FAILURES.get(key)
                    if fix is None:
                        self.assertTrue(rejected, detail)
                    else:
                        self.assertGreaterEqual(
                            response.status_code,
                            500,
                            f"no longer a 5xx, drop its KNOWN_FAILURES entry ({fix}): {detail}",
                        )
                    valid = self._post(route, {})
                    self.assertEqual(valid.status_code, 200, valid.text[:300])
        health = requests.get(f"{self.base_url}/health", timeout=30)
        self.assertEqual(health.status_code, 200, health.text[:300])

    def test_values_accepted_on_purpose_still_succeed(self):
        for case in CHAT_ACCEPTED_CASES:
            with self.subTest(field=case.field, case=case.case_id):
                response = self._post(CHAT, case.overrides)
                self.assertEqual(response.status_code, 200, response.text[:300])


if __name__ == "__main__":
    unittest.main(verbosity=3)
