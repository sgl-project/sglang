import json
import math
import os
import sys
import tempfile

import pytest
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=420, stage="base-b", runner_config="1-gpu-small")

_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_KEY_A = "0123456789abcdef"
_KEY_B = "fedcba9876543210"
_MASK32 = 0xFFFFFFFF
_UINT32_SCALE = float(1 << 32)
_OMITTED = object()


def _rotl32(value, shift):
    return ((value << shift) | (value >> (32 - shift))) & _MASK32


def _mix(state, value):
    value = (value * 0xCC9E2D51) & _MASK32
    value = _rotl32(value, 15)
    value = (value * 0x1B873593) & _MASK32
    state = _rotl32(state ^ value, 13)
    return (state * 5 + 0xE6546B64) & _MASK32


def _fmix32(value):
    value ^= value >> 16
    value = (value * 0x85EBCA6B) & _MASK32
    value ^= value >> 13
    value = (value * 0xC2B2AE35) & _MASK32
    return value ^ (value >> 16)


def _context_hash(token_ids):
    state = 0
    for token_id in token_ids:
        state = _mix(state, token_id & _MASK32)
    return _fmix32(state ^ (len(token_ids) * 4))


def _token_uniform(key, context, token_id):
    state = _mix(0, key & _MASK32)
    state = _mix(state, (key >> 32) & _MASK32)
    state = _mix(state, _context_hash(context))
    state = _mix(state, token_id & _MASK32)
    return (_fmix32(state ^ 16) + 0.5) / _UINT32_SCALE


def _watermark_z_score(prompt_token_ids, response_token_ids, key, context_window=4):
    token_ids = prompt_token_ids + response_token_ids
    start = len(prompt_token_ids)
    seen = set()
    score = 0.0
    for position in range(start, len(token_ids)):
        context = tuple(token_ids[max(0, position - context_window) : position])
        if not context or context in seen:
            continue
        seen.add(context)
        uniform = _token_uniform(key, context, token_ids[position])
        score -= math.log1p(-uniform)
    return len(seen), (score - len(seen)) / math.sqrt(len(seen))


def _chat_payload(watermark=_OMITTED, *, max_tokens):
    payload = {
        "model": _MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a long, varied field guide to urban trees. Include many "
                    "species, observations, and practical examples."
                ),
            }
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    if watermark is not _OMITTED:
        payload["watermark"] = watermark
    return payload


def _assert_detected(response, key, *, other_key=None):
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    prompt_token_ids = choice["prompt_token_ids"]
    response_token_ids = choice["response_token_ids"]
    count, z_score = _watermark_z_score(
        prompt_token_ids,
        response_token_ids,
        int(key, 16),
    )
    assert count >= 100
    assert z_score >= 5.0
    if other_key is not None:
        _, other_z = _watermark_z_score(
            prompt_token_ids,
            response_token_ids,
            int(other_key, 16),
        )
        assert z_score - other_z >= 4.0


class TestWatermarkDisabledEndpoint(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            _MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    def test_request_requires_server_enablement(self):
        disabled = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
            json=_chat_payload({"enabled": False}, max_tokens=1),
            timeout=60,
        )
        assert disabled.status_code == 200, disabled.text

        for watermark in ({"enabled": True}, {"key": _KEY_A}):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
                json=_chat_payload(watermark, max_tokens=1),
                timeout=60,
            )
            assert response.status_code == 400
            assert _KEY_A not in response.text


class WatermarkServerTest(CustomTestCase):
    mode_args = []

    @classmethod
    def setUpClass(cls):
        cls.config_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump({"key": _KEY_A, "context_window": 4}, cls.config_file)
        cls.config_file.close()
        os.chmod(cls.config_file.name, 0o600)
        cls.process = popen_launch_server(
            _MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-watermark",
                "--watermark-config",
                cls.config_file.name,
                *cls.mode_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)
        if hasattr(cls, "config_file"):
            os.unlink(cls.config_file.name)


class TestWatermarkRequestEndpoint(WatermarkServerTest):
    def test_omitted_and_disabled_requests_are_not_rejected(self):
        for watermark in (_OMITTED, {"enabled": False}):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
                json=_chat_payload(watermark, max_tokens=1),
                timeout=60,
            )
            assert response.status_code == 200, response.text

    def test_bad_request_key_is_rejected_without_echo(self):
        bad_key = "not-a-hex-key"
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
            json=_chat_payload({"key": bad_key}, max_tokens=1),
            timeout=60,
        )
        assert response.status_code == 400
        assert bad_key not in response.text

    def test_per_request_keys_are_isolated(self):
        for key, watermark in (
            (_KEY_A, {"enabled": True}),
            (_KEY_B, {"key": _KEY_B}),
        ):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
                json=_chat_payload(watermark, max_tokens=512),
                timeout=180,
            )
            _assert_detected(
                response,
                key,
                other_key=_KEY_B if key == _KEY_A else _KEY_A,
            )


class TestWatermarkDefaultEnabledEndpoint(WatermarkServerTest):
    mode_args = ["--watermark-default-enabled"]

    def test_omitted_request_uses_server_key_and_opt_out_is_allowed(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
            json=_chat_payload(max_tokens=512),
            timeout=180,
        )
        _assert_detected(response, _KEY_A, other_key=_KEY_B)

        disabled = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
            json=_chat_payload({"enabled": False}, max_tokens=1),
            timeout=60,
        )
        assert disabled.status_code == 200, disabled.text

    def test_structured_output_stays_valid_and_detectable(self):
        schema = {
            "type": "object",
            "properties": {
                "entries": {
                    "type": "array",
                    "minItems": 8,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "minLength": 4,
                                "maxLength": 24,
                            },
                            "habitat": {
                                "type": "string",
                                "minLength": 4,
                                "maxLength": 32,
                            },
                            "observation": {
                                "type": "string",
                                "minLength": 8,
                                "maxLength": 48,
                            },
                            "evergreen": {"type": "boolean"},
                        },
                        "required": [
                            "name",
                            "habitat",
                            "observation",
                            "evergreen",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entries"],
            "additionalProperties": False,
        }
        payload = _chat_payload(max_tokens=768)
        payload["messages"][0]["content"] = (
            "Return a field guide to eight distinct trees as the requested JSON."
        )
        payload["ignore_eos"] = False
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "tree_guide",
                "strict": True,
                "schema": schema,
            },
        }
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
            json=payload,
            timeout=180,
        )

        assert response.status_code == 200, response.text
        choice = response.json()["choices"][0]
        assert len(json.loads(choice["message"]["content"])["entries"]) == 8
        count, z_score = _watermark_z_score(
            choice["prompt_token_ids"],
            choice["response_token_ids"],
            int(_KEY_A, 16),
        )
        assert count >= 100
        assert z_score >= 3.0


class TestWatermarkEnforceAllEndpoint(WatermarkServerTest):
    mode_args = ["--watermark-enforce-all"]

    def test_omitted_request_uses_server_key_and_opt_out_is_rejected(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
            json=_chat_payload(max_tokens=512),
            timeout=180,
        )
        _assert_detected(response, _KEY_A, other_key=_KEY_B)

        disabled = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
            json=_chat_payload({"enabled": False}, max_tokens=1),
            timeout=60,
        )
        assert disabled.status_code == 400


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
