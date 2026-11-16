"""Malformed request bodies for the HTTP request-validation fuzz tests.

Type cases are derived from a pydantic request model: each field gets every JSON
kind its annotation rejects, with values lax-mode coercion cannot turn valid.
Value cases are listed by hand: the right JSON kind with a value a validation
rule rejects.
"""

import random
import types
import typing

import msgspec
from pydantic import BaseModel

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
    if origin is typing.Literal:
        # A non-matching value of a literal's own type is still malformed.
        return set()
    if annotation is typing.Any:
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
    for field, info in model.model_fields.items():
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
