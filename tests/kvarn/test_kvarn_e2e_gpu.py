# SPDX-License-Identifier: Apache-2.0
"""End-to-end GPU test: launch sglang server with KVarN and verify it works.

Requires a GPU and a small model.  Set KVARN_MODEL env var to override
the default model.

Run:
    KVARN_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
    python -m pytest tests/kvarn/test_kvarn_e2e_gpu.py -v -s --timeout 180
"""

import os
import subprocess
import time

import pytest

if not os.environ.get("KVARN_E2E_GPU", "0") == "1":
    pytest.skip("Set KVARN_E2E_GPU=1 to run E2E GPU tests", allow_module_level=True)


MODEL = os.environ.get("KVARN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
KVARN_DTYPE = os.environ.get("KVARN_DTYPE", "kvarn_k4v4_g128")
PORT = int(os.environ.get("KVARN_PORT", "30000"))


@pytest.fixture(scope="module")
def sglang_server():
    """Launch and manage a sglang server with KVarN enabled."""
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--kv-cache-dtype",
        KVARN_DTYPE,
        "--port",
        str(PORT),
        "--log-level",
        "info",
        "--disable-radix-cache",
        "--context-length",
        "2048",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        # Wait for server
        import requests

        for _ in range(120):
            if proc.poll() is not None:
                # Server died
                output = proc.stdout.read().decode()
                pytest.fail(f"Server died:\n{output[-2000:]}")
            try:
                r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            output = proc.stdout.read().decode()
            pytest.fail(f"Server failed to start:\n{output[-2000:]}")

        yield f"http://localhost:{PORT}"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_server_health(sglang_server):
    import requests

    r = requests.get(f"{sglang_server}/health")
    assert r.status_code == 200


def test_basic_completion(sglang_server):
    import requests

    r = requests.post(
        f"{sglang_server}/generate",
        json={
            "text": "The capital of France is",
            "sampling_params": {"max_new_tokens": 16, "temperature": 0},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "text" in data
    assert len(data["text"]) > 0
    assert "Paris" in data["text"]


def test_long_context(sglang_server):
    import requests

    text = "Once upon a time, there was a little robot named Pi. " * 10
    r = requests.post(
        f"{sglang_server}/generate",
        json={
            "text": text,
            "sampling_params": {"max_new_tokens": 32, "temperature": 0},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["text"]) > 0


def test_concurrent_requests(sglang_server):
    """Test that multiple concurrent requests work correctly."""
    import threading

    import requests

    results = [None] * 4

    def make_request(i):
        r = requests.post(
            f"{sglang_server}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"max_new_tokens": 16, "temperature": 0},
            },
        )
        results[i] = r.json()

    threads = [threading.Thread(target=make_request, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for r in results:
        assert r is not None
        assert "Paris" in r["text"]


def test_repeated_requests_consistency(sglang_server):
    """Test that the same prompt produces consistent output (deterministic)."""
    import requests

    r1 = requests.post(
        f"{sglang_server}/generate",
        json={
            "text": "The capital of France is",
            "sampling_params": {"max_new_tokens": 16, "temperature": 0},
        },
    )
    r2 = requests.post(
        f"{sglang_server}/generate",
        json={
            "text": "The capital of France is",
            "sampling_params": {"max_new_tokens": 16, "temperature": 0},
        },
    )
    assert r1.json()["text"] == r2.json()["text"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
