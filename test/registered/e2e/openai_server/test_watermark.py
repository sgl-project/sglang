import json
import os
import sys
import tempfile

import pytest
import requests

from sglang.srt.sampling.watermarking import WatermarkDetector
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=320, stage="base-b", runner_config="1-gpu-small")

_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_KEY_A = "0123456789abcdef"
_KEY_B = "fedcba9876543210"
_OMITTED = object()


def _watermark_statistics(choice, key):
    return (
        WatermarkDetector(key)
        .detect_tokens(
            choice["response_token_ids"],
            prompt_token_ids=choice["prompt_token_ids"],
        )
        .combined
    )


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
    statistics = _watermark_statistics(choice, key)
    assert statistics.num_contexts >= 100
    assert statistics.z_score >= 5.0
    if other_key is not None:
        other = _watermark_statistics(choice, other_key)
        assert statistics.z_score - other.z_score >= 4.0


class WatermarkServerTest(CustomTestCase):
    mode_args = []
    process = None
    config_file = None

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
        if cls.process is not None:
            kill_process_tree(cls.process.pid)
        if cls.config_file is not None:
            os.unlink(cls.config_file.name)


class TestWatermarkRequestEndpoint(WatermarkServerTest):
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
        statistics = _watermark_statistics(choice, _KEY_A)
        assert statistics.num_contexts >= 100
        assert statistics.z_score >= 3.0


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
