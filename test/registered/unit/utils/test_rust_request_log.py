"""The Rust server renders `--log-requests` lines from what Python hands it
(`rust_server.config.request_log_settings`); the expected lines here come from
Python's own `RequestLogger`, in both formats, and the Rust tests
(rust/sglang-server/src/api_server/core/request_log/tests.rs) render the same
cases from the same fixture."""

import datetime
import json
import os
import re
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
        "headers": {"x-smg-routing-key": "conv'1"},
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
            # Deprecated: Python moves it to `routed_dp_rank`.
            "data_parallel_rank": 1,
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
            # Deprecated: the batch's normalization moves it to `routed_dp_rank`.
            "data_parallel_rank": 0,
        },
        "headers": {"x-smg-routing-key": "k"},
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
        "headers": {"x-smg-routing-key": "k"},
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
            # Deprecated: the validator copies it to `routed_dp_rank`.
            "data_parallel_rank": 1,
            # Not a field: dropped.
            "not_a_field": 1,
        },
        "headers": {"x-smg-routing-key": "conv'1"},
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
    """Every case at every level in both formats: the settings Python hands
    the Rust server and the lines Python's `RequestLogger` logs. Regenerate
    the fixture with `write_rust_request_log_fixture()` after changing either
    side."""
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


def _build_rust_request_log_fixture() -> dict:
    fixture = {
        "levels": {
            fmt: {
                str(level): request_log_settings(level, GenerateReqInput, fmt)
                for level in range(4)
            }
            for fmt in _FORMATS
        },
        "timestamp": _NOW.isoformat(),
        "cases": [],
        "openai_cases": [],
        "openai_multimodal_cases": [],
    }
    # The OpenAI models' fields, once rather than in every level's settings.
    fixture["openai_fields"] = {
        key: fixture["levels"]["text"]["2"][key]
        for key in ("chat_completion_fields", "completion_fields")
    }
    for settings in fixture["levels"].values():
        for level_settings in settings.values():
            del level_settings["chat_completion_fields"]
            del level_settings["completion_fields"]
    for case in _CASES:
        expected = {fmt: {} for fmt in _FORMATS}
        for fmt in _FORMATS:
            for level in case.get("levels", range(4)):
                lines = []
                logger = _logger(level, fmt, lines)
                obj = GenerateReqInput(**json.loads(json.dumps(case["body"])))
                obj.normalize_batch_and_arguments()
                request = _Request(case["headers"])
                logger.log_received_request(obj, request=request)
                item = case.get("batch_item")
                logger.log_finished_request(
                    obj if item is None else obj[item], case["out"], request=request
                )
                receive, finish = lines
                expected[fmt][str(level)] = {
                    "receive": receive if item is None else None,
                    "finish": finish,
                }
        fixture["cases"].append({**case, "expected": expected})
    models = {"chat": ChatCompletionRequest, "completion": CompletionRequest}
    for case in _OPENAI_CASES:
        expected = {fmt: {} for fmt in _FORMATS}
        for fmt in _FORMATS:
            for level in (2, 3):
                lines = []
                request = models[case["request"]](
                    **json.loads(json.dumps(case["body"]))
                )
                _logger(level, fmt, lines).log_openai_received_request(
                    request, request=_Request(case["headers"])
                )
                (expected[fmt][str(level)],) = lines
        fixture["openai_cases"].append({**case, "expected": expected})
    for case in _OPENAI_MULTIMODAL_CASES:
        expected = {}
        for fmt in _FORMATS:
            lines = []
            request = ChatCompletionRequest(**json.loads(json.dumps(case["body"])))
            _logger(2, fmt, lines).log_openai_received_request(
                request, request=_Request({})
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


class TestRustRequestLogFixture(CustomTestCase):
    def test_fixture_matches_python(self):
        with open(RUST_REQUEST_LOG_FIXTURE, encoding="utf-8") as f:
            checked_in = json.load(f)
        self.assertEqual(
            checked_in,
            # Through JSON, as the file holds it (tuples become lists).
            json.loads(json.dumps(build_rust_request_log_fixture())),
            "Python's request log changed: regenerate "
            "rust/sglang-server/testdata/request_log.json (see "
            "build_rust_request_log_fixture) and make the Rust renderer match",
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
    unittest.main()
