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

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-small")

_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_KEY_A = "0123456789abcdef"
_KEY_B = "fedcba9876543210"
_MASK32 = 0xFFFFFFFF
_UINT32_SCALE = float(1 << 32)


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


def _chat_payload(key, *, max_tokens):
    return {
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
        "watermark": {"key": key, "context_window": 4},
    }


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
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
            json=_chat_payload(_KEY_A, max_tokens=1),
            timeout=60,
        )
        assert response.status_code == 400
        assert _KEY_A not in response.text


class TestWatermarkRequestEndpoint(CustomTestCase):
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
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)
        if hasattr(cls, "config_file"):
            os.unlink(cls.config_file.name)

    def test_bad_request_key_is_rejected_without_echo(self):
        bad_key = "not-a-hex-key"
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
            json=_chat_payload(bad_key, max_tokens=1),
            timeout=60,
        )
        assert response.status_code == 400
        assert bad_key not in response.text

    def test_per_request_keys_are_isolated(self):
        keys = [_KEY_A, _KEY_B]
        generations = []
        for key in keys:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
                json=_chat_payload(key, max_tokens=512),
                timeout=180,
            )
            assert response.status_code == 200, response.text
            choice = response.json()["choices"][0]
            generations.append(
                (choice["prompt_token_ids"], choice["response_token_ids"])
            )

        for index, (prompt_token_ids, response_token_ids) in enumerate(generations):
            own_count, own_z = _watermark_z_score(
                prompt_token_ids, response_token_ids, int(keys[index], 16)
            )
            _, other_z = _watermark_z_score(
                prompt_token_ids, response_token_ids, int(keys[1 - index], 16)
            )
            assert own_count >= 100
            assert own_z >= 5.0
            assert own_z - other_z >= 4.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
