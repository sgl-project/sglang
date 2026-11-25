"""Malformed requests to /v1/completions, /v1/chat/completions and /generate get a
4xx from a real server, which keeps serving: a valid request to the same route
succeeds after every case, and /health passes at the end.

Every case comes from the route's request schema, nested objects included: each
field gets each JSON kind its annotation rejects (for a Literal, a value outside
it), and a value just outside each numeric bound it declares. Rules that live
only in code, such as sampling-parameter checks and batch normalization, are
not covered here.
"""

import collections
import types
import typing
import unittest
from typing import Callable, NamedTuple

import annotated_types
import msgspec
import requests
from pydantic import AfterValidator, BaseModel, Strict
from pydantic.fields import FieldInfo

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

register_cuda_ci(est_time=360, stage="base-b", runner_config="1-gpu-small")

COMPLETIONS = "/v1/completions"
CHAT = "/v1/chat/completions"
GENERATE = "/generate"

MODEL = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
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
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
    },
}

REQUEST_MODELS = {
    COMPLETIONS: protocol.CompletionRequest,
    CHAT: protocol.ChatCompletionRequest,
    GENERATE: api_types.GenerateRequest,
}


class Case(NamedTuple):
    # JSON path from the request root; a nested object names its model, e.g.
    # "messages[0]<ChatCompletionMessageUserParam>.role".
    field: str
    case_id: str
    body: dict


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


def _is_model(annotation) -> bool:
    return isinstance(annotation, type) and issubclass(
        annotation, (BaseModel, msgspec.Struct)
    )


def _accepted_kinds(annotation) -> set[str]:
    """JSON kinds that pydantic (lax mode) or msgspec may accept for this annotation."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        return set().union(*(_accepted_kinds(a) for a in typing.get_args(annotation)))
    if origin is typing.Annotated:
        return _accepted_kinds(typing.get_args(annotation)[0])
    if origin is typing.Literal:
        # Only the malformed values that happen to be members; any other value,
        # of the literal's own type included, is malformed.
        members = typing.get_args(annotation)
        return {kind for kind, value in _MALFORMED.items() if value in members}
    if annotation in (typing.Any, object):
        # Any accepts every kind, so the field gets no type case.
        return set(_MALFORMED)
    if origin in (list, typing.List):
        return {"list"}
    if origin in (dict, typing.Dict) or annotation in (dict, typing.Dict):
        return {"object"}
    if _is_model(annotation):
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


def _metadata(annotated) -> list:
    """The metadata of an Annotated[...], with a FieldInfo's constraints unpacked."""
    extras = typing.get_args(annotated)[1:]
    return [
        c
        for extra in extras
        for c in (extra.metadata if isinstance(extra, FieldInfo) else [extra])
    ]


def _annotated_constraints(annotation) -> list:
    """Constraints an Annotated[...] inside the annotation declares, through unions."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        return [c for a in args for c in _annotated_constraints(a)]
    if origin is typing.Annotated:
        return _metadata(annotation) + _annotated_constraints(args[0])
    if any(_declares_values(a) for a in args):
        raise TypeError(f"no rule for a bound or Literal inside {annotation!r}")
    return []


def _declares_values(annotation) -> bool:
    """Whether the annotation holds a Literal or a numeric bound at any depth."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Literal:
        return True
    if origin is typing.Annotated and _bound_cases(_metadata(annotation)):
        return True
    return any(_declares_values(a) for a in args)


def _bound_cases(constraints) -> list[tuple[str, object]]:
    """(case id, value) just outside each numeric bound."""
    cases = []
    for c in constraints:
        if isinstance(c, annotated_types.Ge):
            cases.append(("below_ge", c.ge - 1))
        elif isinstance(c, annotated_types.Gt):
            cases.append(("at_gt", c.gt))
        elif isinstance(c, annotated_types.Le):
            cases.append(("above_le", c.le + 1))
        elif isinstance(c, annotated_types.Lt):
            cases.append(("at_lt", c.lt))
        elif isinstance(c, (Strict, AfterValidator)):
            # Both only reject more: Strict narrows the kinds lax mode accepts,
            # and an after-validator runs once the type is valid.
            continue
        elif getattr(c, "__dict__", None) == {"allow_inf_nan": False}:
            # JSON has no inf or nan.
            continue
        else:
            raise TypeError(f"no constraint-case rule for {c!r}")
    return cases


class WireField(NamedTuple):
    key: str
    annotation: typing.Any
    # pydantic Field(...) constraints; Annotated ones stay in the annotation.
    constraints: list
    required: bool


def _wire_fields(model) -> list[WireField]:
    """The fields of a pydantic model or a msgspec struct, by JSON key."""
    if issubclass(model, msgspec.Struct):
        return [
            WireField(f.encode_name, f.type, [], f.required)
            for f in msgspec.structs.fields(model)
        ]
    # The alias is the wire key, e.g. "schema" for JsonSchemaResponseFormat.schema_.
    return [
        WireField(
            info.alias or name, info.annotation, info.metadata, info.is_required()
        )
        for name, info in model.model_fields.items()
    ]


def _bad_values(field: WireField) -> list[tuple[str, object]]:
    """(case id, value) for each value the field's schema rejects."""
    type_cases = [
        (f"type_{kind}", value)
        for kind, value in _MALFORMED.items()
        if kind not in _accepted_kinds(field.annotation)
    ]
    constraints = field.constraints + _annotated_constraints(field.annotation)
    return type_cases + _bound_cases(constraints)


def _valid_value(annotation):
    """A minimal value the annotation accepts, for a required field."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        return _valid_value(next(a for a in args if a is not type(None)))
    if origin is typing.Annotated:
        return _valid_value(args[0])
    if origin is typing.Literal:
        return args[0]
    if origin in (list, typing.List):
        return []
    if _is_model(annotation):
        return _valid_instance(annotation)
    if annotation is str:
        return "x"
    raise TypeError(f"no valid-value rule for annotation {annotation!r}")


def _valid_instance(model) -> dict:
    """A valid instance of a nested model: its required fields, nothing else."""
    return {
        field.key: _valid_value(field.annotation)
        for field in _wire_fields(model)
        if field.required
    }


def _nested_models(annotation) -> list[tuple[str, Callable, type]]:
    """(path step, embed, model) for each model the annotation holds; `embed`
    turns an instance of the model into a value of the annotation."""
    if isinstance(annotation, typing.ForwardRef):
        raise TypeError(f"unresolved forward reference {annotation!r}")
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        return [nested for a in args for nested in _nested_models(a)]
    if origin is typing.Annotated:
        return _nested_models(args[0])
    if origin in (list, typing.List):
        return [
            (f"[0]{step}", lambda instance, embed=embed: [embed(instance)], model)
            for step, embed, model in _nested_models(args[0])
        ]
    if _is_model(annotation):
        return [("", lambda instance: instance, annotation)]
    if any(_nested_models(a) for a in args):
        raise TypeError(f"no rule to place the model nested in {annotation!r}")
    return []


def _placer(place, valid, key, embed) -> Callable[[dict], dict]:
    """Turns an instance of a model nested at `valid[key]` into a request body.
    A function, so the caller's loop variables are bound now, not when it runs."""
    return lambda instance: place({**valid, key: embed(instance)})


def schema_cases(request_model, valid_body) -> list[Case]:
    """Each bad value of each field of `request_model` and of every model nested
    in it, placed in an otherwise valid body. Breadth first, so a model nested
    in several places is tried once, where it sits least deep."""
    cases = []
    seen = {request_model}
    queue = collections.deque([(request_model, "", valid_body, lambda body: body)])
    while queue:
        model, path, valid, place = queue.popleft()
        for field in _wire_fields(model):
            if (model, field.key) in NOT_FUZZED:
                continue
            for case_id, value in _bad_values(field):
                if (model, field.key, case_id) in NOT_FUZZED:
                    continue
                body = place({**valid, field.key: value})
                cases.append(Case(path + field.key, case_id, body))
            for step, embed, nested in _nested_models(field.annotation):
                if nested in seen or (nested,) in NOT_FUZZED:
                    continue
                seen.add(nested)
                queue.append(
                    (
                        nested,
                        f"{path}{field.key}{step}<{nested.__name__}>.",
                        _valid_instance(nested),
                        _placer(place, valid, field.key, embed),
                    )
                )
    return cases


# What the scan leaves out, by (model,), (model, field), or (model, field,
# case_id) -> the reason.
NOT_FUZZED: dict[tuple, str] = {
    (protocol.StructuralTag,): (
        "xgrammar's format tree, not SGLang's schema: it changes with the xgrammar "
        "version, and OrFormat.elements is an unresolved forward reference"
    ),
    (protocol.Function, "key_order"): (
        "internal: a before-validator overwrites it from the keys sent"
    ),
    (protocol.ChatCompletionMessageContentImagePart, "image_url", "type_str"): (
        "a before-validator accepts a bare URL string"
    ),
}


# Values ChatCompletionRequest accepts on purpose (see the NOTE on its OpenAI
# limits); a later stage that starts rejecting them breaks existing clients.
CHAT_ACCEPTED = {
    "temperature_above_openai_max": {"temperature": 5.0},
    "top_logprobs_without_logprobs": {"top_logprobs": 5},
    "logit_bias_beyond_openai_range": {"logit_bias": {"1": 1000.0}},
}

# (route, field, case_id) -> the open fix. These still get a 5xx, and the test
# asserts they do: once a fix changes that, the test fails until its entry
# here is dropped.
KNOWN_FAILURES: dict[tuple[str, str, str], str] = {
    # set_json_schema calls .get on a non-object response_format.
    **{
        (CHAT, "response_format", f"type_{kind}"): "#36011"
        for kind in ("list", "str", "float", "int", "bool")
    },
}

# (route, field, case_id) -> the open fix. These take the server down, so they
# are never sent; drop the entry when its fix lands.
KNOWN_CRASHES: dict[tuple[str, str, str], str] = {}


CASES = {
    route: schema_cases(REQUEST_MODELS[route], VALID_BODIES[route])
    for route in VALID_BODIES
}
_CASE_KEYS = [
    (route, case.field, case.case_id)
    for route, cases in CASES.items()
    for case in cases
]
# KNOWN_FAILURES and KNOWN_CRASHES name cases by key: a shared key matches two
# cases, and a stale one leaves its renamed case unlisted, so a crash is sent.
_shared = [key for key, n in collections.Counter(_CASE_KEYS).items() if n > 1]
assert not _shared, f"cases share a key: {_shared}"
_stale = (set(KNOWN_FAILURES) | set(KNOWN_CRASHES)) - set(_CASE_KEYS)
assert not _stale, f"known entries name no case: {sorted(_stale)}"


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

    def _post(self, route, body):
        return requests.post(f"{self.base_url}{route}", json=body, timeout=30)

    def test_malformed_requests_get_4xx_and_server_keeps_serving(self):
        for route, cases in CASES.items():
            for case in cases:
                key = (route, case.field, case.case_id)
                if key in KNOWN_CRASHES:
                    continue
                with self.subTest(route=route, field=case.field, case=case.case_id):
                    response = self._post(route, case.body)
                    rejected = 400 <= response.status_code < 500
                    detail = (
                        f"{case.body!r}: {response.status_code} {response.text[:300]}"
                    )
                    fix = KNOWN_FAILURES.get(key)
                    if fix is None:
                        self.assertTrue(rejected, detail)
                    else:
                        self.assertGreaterEqual(
                            response.status_code,
                            500,
                            f"no longer a 5xx, drop its KNOWN_FAILURES entry ({fix}): {detail}",
                        )
                    valid = self._post(route, VALID_BODIES[route])
                    self.assertEqual(valid.status_code, 200, valid.text[:300])
        health = requests.get(f"{self.base_url}/health", timeout=30)
        self.assertEqual(health.status_code, 200, health.text[:300])

    def test_values_accepted_on_purpose_still_succeed(self):
        for name, overrides in CHAT_ACCEPTED.items():
            with self.subTest(case=name):
                response = self._post(CHAT, {**VALID_BODIES[CHAT], **overrides})
                self.assertEqual(response.status_code, 200, response.text[:300])


if __name__ == "__main__":
    unittest.main(verbosity=3)
