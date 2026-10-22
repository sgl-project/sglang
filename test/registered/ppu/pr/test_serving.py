"""One FA3 server, bounded requests; GSM8K remains a separate unchanged gate."""

import json
import sys
import time

import jsonschema
import pytest

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.ppu_pr_utils import generate, prompt_ids, request, server, small_model

register_ppu_ci(est_time=180, suite="stage-b-test-1-gpu-ppu")


@pytest.fixture(scope="module")
def base():
    # Keep the bounded grammar smoke test focused on compact schema output.
    with server(compact_json=True) as url:
        yield url


def test_models_health(base):
    assert request(base, "GET", "/health").status_code == 200
    response = request(base, "GET", "/v1/models")
    response.raise_for_status()
    assert small_model() in [m["id"] for m in response.json()["data"]]


def test_completion_chat_and_stream(base):
    payload = {
        "model": small_model(),
        "prompt": prompt_ids(128),
        "temperature": 0,
        "max_tokens": 16,
    }
    response = request(base, "POST", "/v1/completions", json=payload)
    response.raise_for_status()
    completion = response.json()
    assert 0 < completion["usage"]["completion_tokens"] <= 16
    pieces, finished, done = [], False, False
    deadline = time.monotonic() + 45
    with request(
        base, "POST", "/v1/completions", json={**payload, "stream": True}, stream=True
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            assert time.monotonic() < deadline, "stream deadline exceeded"
            if not line.startswith(b"data: "):
                continue
            if line == b"data: [DONE]":
                done = True
                break
            chunk = json.loads(line[6:])
            for choice in chunk["choices"]:
                pieces.append(choice["text"])
                finished |= choice.get("finish_reason") is not None
    assert done and finished
    assert "".join(pieces) == completion["choices"][0]["text"]
    response = request(
        base,
        "POST",
        "/v1/chat/completions",
        json={
            "model": small_model(),
            "messages": [{"role": "user", "content": "Say hello."}],
            "temperature": 0,
            "max_tokens": 16,
        },
    )
    response.raise_for_status()
    result = response.json()
    assert result["choices"][0]["message"]["content"]
    assert 0 < result["usage"]["completion_tokens"] <= 16


@pytest.mark.parametrize("length", [128, 512])
def test_prefix_cache_and_batches(base, length):
    ids = prompt_ids(length)
    request(base, "GET", "/flush_cache").raise_for_status()
    cold = generate(base, ids)
    warm = generate(base, ids)
    assert cold["text"] == warm["text"]
    assert (
        cold["meta_info"]["completion_tokens"] == warm["meta_info"]["completion_tokens"]
    )
    # This checkout exposes cached_tokens in /generate; absence is a regression.
    assert warm["meta_info"]["cached_tokens"] > 0
    assert warm["meta_info"]["cached_tokens"] > cold["meta_info"]["cached_tokens"]
    batch = generate(base, [ids] * 4)
    assert len(batch) == 4
    for item in [cold, warm, *batch]:
        assert item["text"] == cold["text"]
        assert 0 < item["meta_info"]["completion_tokens"] <= 16


def test_invalid_request_and_token_limit(base):
    payload = {
        "model": small_model(),
        "prompt": prompt_ids(128),
        "temperature": 0,
        "max_tokens": -1,
    }
    response = request(base, "POST", "/v1/completions", json=payload)
    assert response.status_code in (400, 422)
    response = request(
        base, "POST", "/v1/completions", json={**payload, "max_tokens": 1}
    )
    response.raise_for_status()
    assert response.json()["usage"]["completion_tokens"] == 1


def test_xgrammar_schema(base):
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer", "enum": [7]}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    result = generate(
        base,
        prompt_ids(128),
        sampling_params={
            "temperature": 0,
            "max_new_tokens": 32,
            "json_schema": json.dumps(schema),
        },
    )
    jsonschema.validate(json.loads(result["text"]), schema)
    assert 0 < result["meta_info"]["completion_tokens"] <= 32


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
