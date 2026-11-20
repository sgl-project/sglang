"""The Rust tests (rust/sglang-server/src/api_server/core/request_log/tests.rs)
render the cases below from rust/sglang-server/testdata/request_log.json: the
settings Python hands the Rust server and the lines Python's `RequestLogger`
logs. The fixture keeps only the fields in `_GENERATE_FIELDS` and
`_OPENAI_REQUESTS`, so it pins the log format, not the request classes' fields;
the Rust server gets the full field list from Python at runtime."""

import dataclasses
import datetime
import json
import os
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.protocol import (  # noqa: E402
    ChatCompletionRequest,
    CompletionRequest,
)
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.observability import request_logger  # noqa: E402
from sglang.srt.observability.request_logger import RequestLogger  # noqa: E402
from sglang.srt.runtime_context import (  # noqa: E402
    ParallelContext,
    RuntimeContext,
    use_context,
)
from sglang.srt.rust_server import config as rust_config  # noqa: E402
from sglang.srt.rust_server.config import request_log_settings  # noqa: E402
from sglang.srt.utils import log_utils  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

RUST_REQUEST_LOG_FIXTURE = os.path.join(
    os.path.dirname(__file__),
    *[os.pardir] * 4,
    "rust/sglang-server/testdata/request_log.json",
)

# The `GenerateReqInput` fields the fixture keeps, in the order its lines print.
_GENERATE_FIELDS = (
    "rid",
    "session_id",  # shared by a batch's items
    "text",
    "input_ids",
    "input_embeds",  # a skip name
    "image_data",
    "video_data",
    "audio_data",
    "sampling_params",
    "watermark",
    "return_logprob",
    "logprob_start_len",
    "top_logprobs_num",
    "token_ids_logprob",
    "return_sampling_mask",
    "return_text_in_logprobs",  # a `False` default
    "stream",
    "log_metrics",
    "routed_experts_start_len",  # a `0` default
    "modalities",  # set with a batch item's images
    "lora_path",  # a skip name
    "bootstrap_host",
    "bootstrap_port",
    "bootstrap_room",
    "routed_dp_rank",
    "routing_key",  # neither per item nor shared: an item keeps the default
    "priority",
    "extra_key",
    "custom_labels",
    "cache_salt",
)

# Per OpenAI request: its model, its settings key, and its kept fields in order.
_OPENAI_REQUESTS = {
    "chat": (
        ChatCompletionRequest,
        "chat_completion_fields",
        (
            "routed_dp_rank",
            "messages",
            "model",
            "frequency_penalty",
            "logprobs",
            "n",
            "stop",
            "stream",
            "temperature",
            "tools",
            "tool_choice",
            "skip_special_tokens",
        ),
    ),
    "completion": (
        CompletionRequest,
        "completion_fields",
        (
            "routed_dp_rank",
            "model",
            "prompt",
            "echo",
            "logit_bias",
            "max_tokens",
            "n",
            "stop",
            "stream",
            "temperature",
            "top_k",
        ),
    ),
}

_ProjectedGenerateReqInput = dataclasses.make_dataclass(
    GenerateReqInput.__name__, _GENERATE_FIELDS
)

_META_INFO = {
    "id": "rid-1",
    "finish_reason": {"type": "length", "length": 16},
    "prompt_tokens": 7,
    "weight_version": "default",
    "completion_tokens": 16,
    "cached_tokens": 0,
    "weight_versions": [{"version": "default", "start": 0, "end": 16}],
    "e2e_latency": 0.123456789,
    "request_received_ts": 1234567890.1234567,
    "queue_time": 1e-05,
    "big": 1e16,
    "neg_zero": -0.0,
}

# Strings that exercise `repr`: both quote kinds, escapes, control characters
# and printable non-ASCII text.
_TRICKY = [
    "\n",
    "it's",
    'say "hi"',
    "both ' and \"",
    "tab\tback\\slash\x01\x7f",
    "é中🙂",
]

# Each case: the `/generate` body, the request headers, the frame the request
# finished with and the levels it runs at (default: all). Every body names its rid, so both servers log it.
_CASES = [
    {
        "name": "text_with_sampling_params",
        "body": {
            "rid": "rid-1",
            "text": "Hello",
            "sampling_params": {
                "temperature": 0.7,
                "top_p": 1.0,
                "max_new_tokens": 16,
                "stop": _TRICKY,
                "frequency_penalty": -0.0,
                "min_p": 1e-05,
                "logit_bias": {"42": 1e16, "7": -100},
                "json_schema": None,
            },
            "stream": True,
        },
        "headers": {"x-sgl-routing-key": "conv'1"},
        "out": {"text": " world", "output_ids": [1, 2], "meta_info": _META_INFO},
    },
    {
        "name": "input_ids_with_normalized_fields",
        "body": {
            "rid": "rid-2",
            "input_ids": [1, 2, 3],
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": 2,
            "token_ids_logprob": [],
            "extra_key": "",
            "cache_salt": "salt",
            "priority": 3,
            "bootstrap_host": "10.0.0.1",
            "bootstrap_port": 8998,
            "bootstrap_room": 12345678901234,
            "routing_key": "rk",
            "custom_labels": {"tenant": "a'b"},
            "routed_dp_rank": 1,
        },
        "headers": {},
        "out": {
            "text": "",
            "output_ids": [5],
            "meta_info": {
                "id": "rid-2",
                "finish_reason": {
                    "type": "abort",
                    "message": "Abort in waiting queue",
                    "status_code": 503,
                },
                "prompt_tokens": 3,
                "completion_tokens": 0,
            },
        },
    },
    {
        "name": "health_probe",
        "body": {
            "rid": "HEALTH_CHECK_0123",
            "input_ids": [0],
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "log_metrics": False,
        },
        "headers": {},
        "out": {"text": "", "output_ids": [9], "meta_info": {"id": "x"}},
    },
    {
        # A batch item's `Finish` object is Python's `obj[i]`: its own entry
        # of the per-item fields, the shared ones, defaults for the rest
        # (`routing_key`). The batch's `Receive` is best effort, not compared.
        "name": "batch_item",
        "batch_item": 1,
        "body": {
            "rid": ["b0", "b1"],
            "text": ["x", "y"],
            "sampling_params": [{"max_new_tokens": 1}, {"max_new_tokens": 2}],
            "return_logprob": [True, False],
            "extra_key": "shared-key",
            "priority": 2,
            "custom_labels": {"tenant": "t"},
            "routing_key": "rk",
            # One room for the batch counts up per item.
            "bootstrap_host": "10.0.0.2",
            "bootstrap_room": 100,
            "routed_dp_rank": 0,
        },
        "headers": {"x-sgl-routing-key": "k"},
        "out": {"text": "y", "output_ids": [3], "meta_info": {"id": "b1"}},
    },
    {
        # A batch item's images are a list, with their modality, whichever
        # form the batch sent them in.
        "name": "batch_images",
        "batch_item": 1,
        "body": {
            "rid": ["i0", "i1"],
            "text": ["x", "y"],
            "image_data": ["a.png", "b.png"],
        },
        "headers": {},
        "out": {"text": "", "output_ids": [1], "meta_info": {"id": "i1"}},
    },
    {
        "name": "batch_image_lists",
        "batch_item": 1,
        "body": {
            "rid": ["l0", "l1"],
            "text": ["x", "y"],
            "image_data": [["a.png"], ["b.png", "c.png"]],
        },
        "headers": {},
        "out": {"text": "", "output_ids": [1], "meta_info": {"id": "l1"}},
    },
    {
        "name": "batch_broadcast_image",
        "batch_item": 1,
        "body": {"rid": ["s0", "s1"], "text": ["x", "y"], "image_data": "a.png"},
        "headers": {},
        "out": {"text": "", "output_ids": [1], "meta_info": {"id": "s1"}},
    },
    {
        # Media is summarized per modality at every level; the items
        # themselves show from level 2.
        "name": "single_media",
        "body": {
            "rid": "rid-7",
            "text": "What is this?",
            "image_data": "data:image/png;base64,iVBORw0KGgo=",
            "video_data": {"url": "file:///tmp/clip.mp4", "fps": 2},
            "audio_data": ["a.wav", "é中🙂.wav"],
        },
        "headers": {},
        "out": {"text": "", "output_ids": [1], "meta_info": {"id": "rid-7"}},
    },
    {
        # Watermark keys are redacted, nested ones included.
        "name": "watermark_keys",
        "body": {
            "rid": "rid-6",
            "text": "Hi",
            "sampling_params": {
                "max_new_tokens": 4,
                "watermark": {"enabled": True, "key": "secret-a", "context_window": 4},
            },
            "watermark": {"key": "secret-b", "key_b": "secret-c"},
            "custom_labels": {"watermark_key": "secret-d", "tenant": "t"},
        },
        "headers": {},
        "out": {"text": "", "output_ids": [1], "meta_info": {"id": "rid-6"}},
    },
    {
        # `out` exactly as the Rust server builds it from this output
        # (`core_output`, read by the Rust test only).
        "name": "frame_out",
        "body": {"rid": "rid-5", "input_ids": [1, 2, 3, 4]},
        "headers": {},
        "core_output": {
            "text": "Hi",
            "token_ids": [7, 8],
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "cached_tokens": 1,
            "finish_reason": {"type": "stop", "matched": 2},
            "e2e_latency": 0.25,
        },
        "out": {
            "text": "Hi",
            "output_ids": [7, 8],
            "meta_info": {
                "id": "rid-5",
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "finish_reason": {"type": "stop", "matched": 2},
                "e2e_latency": 0.25,
                "cached_tokens": 1,
                "reasoning_tokens": 0,
                "weight_version": "default",
                "cached_tokens_details": {"device": 1, "host": 0},
                "dp_rank": None,
            },
        },
    },
    {
        # Only level 2 cuts long values; the others print them whole.
        "name": "long_values",
        "levels": [2],
        "body": {
            "rid": "rid-3",
            "text": "ab'" * 690 + "é" * 50,
            "sampling_params": {"stop_token_ids": list(range(2050))},
        },
        "headers": {"x-sgl-routing-key": "k"},
        "out": {
            "text": "x" * 2049,
            "output_ids": list(range(2050)),
            "meta_info": {"id": "rid-3", "output_token_logprobs": [[-0.5, 1, None]]},
        },
    },
]

# OpenAI requests, logged as received from level 2: each the request model,
# the body and the request headers. Their bodies keep to what the Rust server
# prints as Python does: top-level fields, floats where the model coerces to
# float, and nested objects pydantic dumps as sent.
_OPENAI_CASES = [
    {
        "name": "chat_defaults_and_validators",
        "request": "chat",
        "body": {
            "model": "m",
            "messages": [{"role": "user", "content": "hi 'there'"}],
            "temperature": 0.5,
            "stop": _TRICKY,
            "routed_dp_rank": 1,
            # Not a field: dropped.
            "not_a_field": 1,
        },
        "headers": {"x-sgl-routing-key": "conv'1"},
    },
    {
        # Level 2 cuts the long prompt list and, in the JSON format, the
        # long string inside a list.
        "name": "completion_long_prompt",
        "request": "completion",
        "body": {
            "model": "m",
            "prompt": list(range(2050)),
            "stop": ["x" * 2049],
            "max_tokens": 4,
            "logit_bias": {"42": 1e16},
        },
        "headers": {},
    },
]

# Chat requests with media content parts. Their lines print nested objects as
# Python's model filled them in, which the Rust server does not, so only
# their `multimodal` summary is compared.
_OPENAI_MULTIMODAL_CASES = [
    {
        "name": "chat_media_parts",
        "body": {
            "model": "m",
            "messages": [
                {"role": "system", "content": "be brief"},
                {
                    "role": "user",
                    "content": [
                        # Inline audio first: the summary lists it first.
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "UklGRg==", "format": "wav"},
                        },
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,iVBORw0KGgo=",
                                "detail": "high",
                            },
                        },
                        {"type": "video_url", "video_url": {"url": "file:///é.mp4"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "a.png"}},
                        {"type": "audio_url", "audio_url": {"url": "b.mp3"}},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "SUQz", "format": "mp3"},
                        },
                    ],
                },
            ],
        },
    },
    {
        "name": "chat_text_parts_only",
        "body": {
            "model": "m",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ],
        },
    },
]

# What `datetime.now()` reads while the fixture is built, for the JSON lines'
# `timestamp`.
_NOW = datetime.datetime(2026, 11, 4, 1, 2, 3, 456789)


class _Request:
    """The `fastapi.Request` the logger reads headers from."""

    def __init__(self, headers):
        self.headers = headers


class _FixedDatetime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return _NOW


def build_rust_request_log_fixture() -> dict:
    """Every case at every level in both formats, on the kept fields only."""
    # The defaults, whatever this shell or an earlier test in the process sets.
    with (
        use_context(RuntimeContext(parallel=ParallelContext())),
        patch.object(log_utils, "datetime", _FixedDatetime),
        patch.object(
            rust_config,
            "WHITELISTED_HEADERS",
            request_logger._DEFAULT_WHITELISTED_HEADERS,
        ),
        patch.object(
            request_logger,
            "WHITELISTED_HEADERS",
            request_logger._DEFAULT_WHITELISTED_HEADERS,
        ),
        envs.SGLANG_LOG_REQUEST_EXCEEDED_MS.override(-1),
    ):
        return _build_rust_request_log_fixture()


_FORMATS = ("text", "json")


def _logger(level: int, log_format: str, lines: list) -> RequestLogger:
    """A `RequestLogger` whose lines (either format) land in `lines`."""
    logger = RequestLogger(
        log_requests=True,
        log_requests_level=level,
        log_requests_format=log_format,
        log_requests_target=None,
    )
    logger.targets = [SimpleNamespace(info=lines.append)]
    return logger


def _kept_settings(level: int, log_format: str) -> dict:
    settings = request_log_settings(level, GenerateReqInput, log_format)
    defaults = dict(settings["fields"])
    settings["fields"] = [(name, defaults[name]) for name in _GENERATE_FIELDS]
    for key in ("batch_item_fields", "batch_shared_fields"):
        names = set(settings[key])
        settings[key] = [name for name in _GENERATE_FIELDS if name in names]
    # The fixture holds these once, in `openai_fields`.
    for _, key, _ in _OPENAI_REQUESTS.values():
        del settings[key]
    return settings


def _project(obj: GenerateReqInput) -> _ProjectedGenerateReqInput:
    projected = _ProjectedGenerateReqInput(
        **{name: getattr(obj, name) for name in _GENERATE_FIELDS}
    )
    # Not a field, but the multimodal summary reads it.
    projected.parallel_sample_num = getattr(obj, "parallel_sample_num", 1)
    return projected


def _project_openai(kind: str, body: dict) -> dict:
    model, _, names = _OPENAI_REQUESTS[kind]
    # What `log_openai_received_request` logs for a request model.
    dump = model(**json.loads(json.dumps(body))).model_dump(exclude_none=True)
    return {name: dump[name] for name in names if name in dump}


def _build_rust_request_log_fixture() -> dict:
    fixture = {
        "levels": {
            fmt: {str(level): _kept_settings(level, fmt) for level in range(4)}
            for fmt in _FORMATS
        },
        "timestamp": _NOW.isoformat(),
        "cases": [],
        "openai_cases": [],
        "openai_multimodal_cases": [],
    }
    fixture["openai_fields"] = {}
    for model, key, names in _OPENAI_REQUESTS.values():
        defaults = dict(rust_config._openai_model_fields(model))
        fixture["openai_fields"][key] = [(name, defaults[name]) for name in names]
    for case in _CASES:
        expected = {fmt: {} for fmt in _FORMATS}
        for fmt in _FORMATS:
            for level in case.get("levels", range(4)):
                lines = []
                logger = _logger(level, fmt, lines)
                obj = GenerateReqInput(**json.loads(json.dumps(case["body"])))
                obj.normalize_batch_and_arguments()
                request = _Request(case["headers"])
                logger.log_received_request(_project(obj), request=request)
                item = case.get("batch_item")
                logger.log_finished_request(
                    _project(obj if item is None else obj[item]),
                    case["out"],
                    request=request,
                )
                receive, finish = lines
                expected[fmt][str(level)] = {
                    "receive": receive if item is None else None,
                    "finish": finish,
                }
        fixture["cases"].append({**case, "expected": expected})
    for case in _OPENAI_CASES:
        expected = {fmt: {} for fmt in _FORMATS}
        for fmt in _FORMATS:
            for level in (2, 3):
                lines = []
                _logger(level, fmt, lines).log_openai_received_request(
                    _project_openai(case["request"], case["body"]),
                    request=_Request(case["headers"]),
                )
                (expected[fmt][str(level)],) = lines
        fixture["openai_cases"].append({**case, "expected": expected})
    for case in _OPENAI_MULTIMODAL_CASES:
        expected = {}
        for fmt in _FORMATS:
            lines = []
            _logger(2, fmt, lines).log_openai_received_request(
                _project_openai("chat", case["body"]), request=_Request({})
            )
            (line,) = lines
            if fmt == "json":
                expected[fmt] = json.loads(line).get("multimodal")
            else:
                # The summary ends a line without headers.
                start = line.find(", multimodal=")
                expected[fmt] = line[start:] if start >= 0 else ""
        fixture["openai_multimodal_cases"].append({**case, "expected": expected})
    return fixture


def write_rust_request_log_fixture() -> None:
    text = json.dumps(build_rust_request_log_fixture(), indent=1, ensure_ascii=False)
    # One line per list of numbers keeps the long cases readable.
    text = re.sub(
        r"\[\s*(-?\d[\d.e+-]*(?:,\s*-?\d[\d.e+-]*)*)\s*\]",
        lambda m: "[" + ", ".join(re.split(r",\s*", m.group(1))) + "]",
        text,
    )
    # And one line per `(field, default)` pair.
    string = r'"(?:[^"\\]|\\.)*"'
    text = re.sub(
        rf"\[\s*({string}),\s*({string}|null)\s*\]",
        lambda m: f"[{m.group(1)}, {m.group(2)}]",
        text,
    )
    with open(RUST_REQUEST_LOG_FIXTURE, "w", encoding="utf-8") as f:
        f.write(text + "\n")


_WRITE_FIXTURE = (
    "python test/registered/unit/utils/test_rust_request_log.py --write-fixture"
)


def _first_difference(old, new, path: str = "fixture"):
    """`(path, old value, new value)` where `new` first differs, or None."""
    if isinstance(old, dict) and isinstance(new, dict):
        for key in [*old, *(key for key in new if key not in old)]:
            at = f"{path}[{key!r}]"
            if key not in old or key not in new:
                return at, old.get(key, "<missing>"), new.get(key, "<missing>")
            if found := _first_difference(old[key], new[key], at):
                return found
        return None
    if isinstance(old, list) and isinstance(new, list):
        for i, (a, b) in enumerate(zip(old, new)):
            at = f"{path}[{i}]"
            if isinstance(a, dict) and "name" in a:
                at += f" ({a['name']})"
            if found := _first_difference(a, b, at):
                return found
        if len(old) != len(new):
            return f"len({path})", len(old), len(new)
        return None
    if type(old) is type(new) and old == new:
        return None
    return path, old, new


def _excerpt(value, other) -> str:
    """`value` around where it stops matching `other`."""
    if isinstance(value, str) and isinstance(other, str):
        at = len(os.path.commonprefix([value, other]))
        return f"from character {at}: {value[max(at - 40, 0) : at + 80]!r}"
    text = repr(value)
    return text if len(text) <= 200 else text[:200] + "..."


class TestRustRequestLogFixture(CustomTestCase):
    def test_fixture_matches_python(self):
        with open(RUST_REQUEST_LOG_FIXTURE, encoding="utf-8") as f:
            checked_in = json.load(f)
        # Through JSON, as the file holds it (tuples become lists).
        built = json.loads(json.dumps(build_rust_request_log_fixture()))
        difference = _first_difference(checked_in, built)
        if difference is None:
            return
        path, old, new = difference
        self.fail(
            "Python's request log format changed, so the Rust server's must "
            f"follow. rust/sglang-server/testdata/request_log.json differs at "
            f"{path}:\n"
            f"  checked in: {_excerpt(old, new)}\n"
            f"  Python now: {_excerpt(new, old)}\n"
            f"Regenerate it with `{_WRITE_FIXTURE}`, then make "
            "rust/sglang-server/src/api_server/core/request_log.rs pass "
            "`cargo test -p sglang-server request_log`. A field added to a "
            "request class never needs this: the fixture keeps only the "
            "fields named in this test."
        )

    def test_kept_fields_exist(self):
        fields = {field.name for field in dataclasses.fields(GenerateReqInput)}
        self.assertEqual(
            [name for name in _GENERATE_FIELDS if name not in fields],
            [],
            "_GENERATE_FIELDS names fields GenerateReqInput no longer has",
        )
        skip_names = {
            name
            for level in range(4)
            for name in request_log_settings(level, GenerateReqInput)["skip_names"]
        }
        self.assertEqual(
            sorted(skip_names - set(_GENERATE_FIELDS)),
            [],
            "skip names missing from _GENERATE_FIELDS",
        )
        for case in _CASES:
            self.assertEqual(
                [name for name in case["body"] if name not in _GENERATE_FIELDS],
                [],
                f"{case['name']} sets fields missing from _GENERATE_FIELDS",
            )

        openai_fields = {
            kind: dict(rust_config._openai_model_fields(model))
            for kind, (model, _, _) in _OPENAI_REQUESTS.items()
        }
        for kind, (model, _, names) in _OPENAI_REQUESTS.items():
            self.assertEqual(
                [name for name in names if name not in openai_fields[kind]],
                [],
                f"_OPENAI_REQUESTS names fields {model.__name__} no longer has",
            )
        cases = [(case["request"], case) for case in _OPENAI_CASES]
        cases += [("chat", case) for case in _OPENAI_MULTIMODAL_CASES]
        for kind, case in cases:
            _, _, names = _OPENAI_REQUESTS[kind]
            self.assertEqual(
                [
                    name
                    for name in case["body"]
                    if name in openai_fields[kind] and name not in names
                ],
                [],
                f"{case['name']} sets fields missing from _OPENAI_REQUESTS",
            )

    def test_settings_render_the_class_defaults(self):
        """The defaults Rust prints for absent fields are what Python prints
        for a request that leaves every field at its default."""
        for fmt in _FORMATS:
            for level in range(4):
                settings = request_log_settings(level, GenerateReqInput, fmt)
                lines = []
                _logger(level, fmt, lines).log_received_request(GenerateReqInput())
                defaults = [
                    (name, default)
                    for name, default in settings["fields"]
                    if name not in settings["skip_names"]
                ]
                if fmt == "json":
                    self.assertEqual(
                        json.loads(lines[0])["obj"],
                        {name: json.loads(default) for name, default in defaults},
                    )
                else:
                    rendered = ", ".join(f"{name}={value}" for name, value in defaults)
                    self.assertEqual(
                        lines[0], f"Receive: obj=GenerateReqInput({rendered})"
                    )

    def test_targets_name_the_listener_files(self):
        """Stdout and a directory target's `{hostname}_{rank}.log`, the
        directory created as `create_log_targets` does; one file for a
        directory listed twice."""
        with tempfile.TemporaryDirectory() as root:
            directory = os.path.join(root, "logs")
            targets = rust_config._request_log_targets(
                ["STDOUT", directory, directory + os.sep], 3
            )
            self.assertTrue(os.path.isdir(directory))
        self.assertEqual(
            targets,
            dict(
                stdout=True,
                files=[os.path.join(directory, f"{os.uname().nodename}_3.log")],
            ),
        )


if __name__ == "__main__":
    if "--write-fixture" in sys.argv:
        write_rust_request_log_fixture()
    else:
        unittest.main()
