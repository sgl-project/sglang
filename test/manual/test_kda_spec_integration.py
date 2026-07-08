"""Engine-level integration test for KDA speculative decoding path.

Launches server with KDA model + no_buffer, sends requests,
and verifies the forward path works correctly. Without a draft model,
this verifies no regression in normal decode/extend paths after adding
target_verify code.

Usage:
    # Start server first:
    python -m sglang.launch_server \
        --model-path /path/to/Kimi-Linear-48B-A3B-Instruct \
        --tp 2 --trust-remote-code \
        --mamba-scheduler-strategy no_buffer

    # Then run this test:
    python test/manual/test_kda_spec_integration.py
"""

import concurrent.futures
import time

import requests

BASE_URL = "http://localhost:30000"
SHARED_PREFIX = "You are a helpful assistant. " * 20


def test_normal_inference_no_regression():
    """Normal inference still works after code changes."""
    resp = requests.post(
        f"{BASE_URL}/generate",
        json={
            "text": "What is 2+2?",
            "sampling_params": {"max_new_tokens": 32, "temperature": 0.0},
        },
    )
    assert resp.status_code == 200, f"Status {resp.status_code}: {resp.text}"
    data = resp.json()
    print(f"Normal inference: {data['text'][:80]}")
    assert len(data["text"]) > 0


def test_prefix_caching_still_works():
    """Prefix caching (radix cache) still works."""
    resp1 = requests.post(
        f"{BASE_URL}/generate",
        json={
            "text": SHARED_PREFIX + "What is 1+1?",
            "sampling_params": {"max_new_tokens": 32, "temperature": 0.0},
        },
    )
    time.sleep(0.5)
    resp2 = requests.post(
        f"{BASE_URL}/generate",
        json={
            "text": SHARED_PREFIX + "What is 3+3?",
            "sampling_params": {"max_new_tokens": 32, "temperature": 0.0},
        },
    )
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    cached = resp2.json().get("meta_info", {}).get("cached_tokens", 0)
    print(f"Cached tokens: {cached}")
    assert cached > 0, "Prefix caching should work"


def test_batch_inference():
    """Multiple concurrent requests work."""
    prompts = [f"Count from 1 to {i + 3}" for i in range(8)]

    def send(p):
        return requests.post(
            f"{BASE_URL}/generate",
            json={
                "text": p,
                "sampling_params": {"max_new_tokens": 64, "temperature": 0.0},
            },
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(send, p) for p in prompts]
        results = [f.result() for f in futures]
    for r in results:
        assert r.status_code == 200
    print(f"Batch test passed: {len(results)} requests OK")


if __name__ == "__main__":
    test_normal_inference_no_regression()
    test_prefix_caching_still_works()
    test_batch_inference()
    print("\nAll tests PASSED!")
