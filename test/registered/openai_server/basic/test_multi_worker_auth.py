"""Real-model authentication through native single/multi-worker HTTP serving."""

import concurrent.futures
import json
import os

import pytest
import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.network import get_open_port
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    auto_config_device,
    popen_launch_server,
)

register_cuda_ci(est_time=150, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=180, suite="stage-b-test-1-gpu-small-amd")


@pytest.fixture(params=[1, 4], ids=["single-worker", "four-workers"])
def authenticated_server(request):
    model = os.environ.get("SGLANG_TEST_AUTH_MODEL", "Qwen/Qwen3-0.6B")
    base_url = f"http://127.0.0.1:{get_open_port()}"
    cpu_args = (
        [
            "--attention-backend",
            "torch_native",
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
        ]
        if auto_config_device() == "cpu"
        else []
    )
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        api_key="test-user",
        env={"SGLANG_RUST_SERVER": "0"},
        other_args=[
            "--admin-api-key",
            "test-admin",
            "--tokenizer-worker-num",
            str(request.param),
            "--mem-fraction-static",
            "0.5",
            "--context-length",
            "2048",
            "--max-running-requests",
            "8",
            "--max-total-tokens",
            "8192",
            "--cuda-graph-max-bs-decode",
            "8",
            *cpu_args,
        ],
    )
    try:
        yield base_url, model
    finally:
        kill_process_tree(process.pid)


def test_real_model_authentication_and_streaming(authenticated_server):
    base_url, model = authenticated_server
    user = {"Authorization": "Bearer test-user"}
    admin = {"Authorization": "Bearer test-admin"}
    messages = [{"role": "user", "content": "Reply with the word hello."}]
    chat = {
        "model": model,
        "messages": messages,
        "max_tokens": 16,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    responses = {
        "model": model,
        "input": "Say hello.",
        "max_output_tokens": 16,
        "store": False,
    }
    with requests.Session() as client:
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            for path, body in (
                ("/v1/chat/completions", chat),
                ("/v1/responses", responses),
            ):
                result = client.post(
                    base_url + path, json=body, headers=headers, timeout=30
                )
                assert result.status_code == 401, result.text
            assert (
                client.get(
                    base_url + "/v1/models", headers=headers, timeout=30
                ).status_code
                == 401
            )
        assert client.get(base_url + "/health", timeout=30).status_code == 200
        assert (
            client.get(base_url + "/v1/models", headers=user, timeout=30).status_code
            == 200
        )
        assert (
            client.get(base_url + "/flush_cache", headers=user, timeout=30).status_code
            == 401
        )
        assert (
            client.get(base_url + "/flush_cache", headers=admin, timeout=30).status_code
            == 200
        )
        result = client.post(
            base_url + "/v1/chat/completions", json=chat, headers=user, timeout=90
        )
        assert result.status_code == 200, result.text
        assert result.json()["usage"]["completion_tokens"] > 0
        for path, body in (
            ("/v1/chat/completions", chat),
            ("/v1/responses", responses),
        ):
            with client.post(
                base_url + path,
                json={**body, "stream": True},
                headers=user,
                stream=True,
                timeout=90,
            ) as stream:
                assert stream.status_code == 200, stream.text
                assert stream.headers["content-type"].startswith("text/event-stream")
                events = [
                    line
                    for line in stream.iter_lines(decode_unicode=True)
                    if line.startswith("data: ")
                ]
                assert events
                payloads = [
                    json.loads(line[6:]) for line in events if line[6:] != "[DONE]"
                ]
                if path.endswith("completions"):
                    assert events[-1] == "data: [DONE]"
                    assert any(
                        p["choices"][0].get("delta", {}).get("content")
                        or p["choices"][0].get("delta", {}).get("reasoning_content")
                        for p in payloads
                        if p.get("choices")
                    )
                else:
                    assert any(
                        p.get("type") in ("response.completed", "response.incomplete")
                        for p in payloads
                    )

    def generate(_):
        result = requests.post(
            base_url + "/v1/chat/completions", json=chat, headers=user, timeout=90
        )
        assert result.status_code == 200, result.text
        assert result.json()["usage"]["completion_tokens"] > 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(generate, range(8)))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
