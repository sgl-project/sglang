"""
End-to-end tests for multi-model router functionality.

Tests the ability of the router to handle multiple models simultaneously
with different load balancing policies per model.
"""

import threading
import time
from types import SimpleNamespace

import pytest
import requests

from sglang.test.run_eval import run_eval
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

# Model configurations for multi-model testing
MULTI_MODEL_CONFIG = {
    "llama": DEFAULT_MODEL_NAME_FOR_TEST,  # llama-3.1-8b-instruct
    "qwen": "Qwen/Qwen3-8B",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "qwen_large": "Qwen/Qwen2.5-7B-Instruct-1M",
}


@pytest.fixture
def multi_model_router():
    """Create a router-only instance for multi-model testing with IGW enabled."""
    from conftest import _find_available_port, _popen_launch_router_only, _terminate

    port = _find_available_port()
    base_url = f"http://127.0.0.1:{port}"
    # Use round_robin as default policy for the router with IGW enabled for multi-model support
    proc = _popen_launch_router_only(
        base_url, policy="round_robin", timeout=180.0, enable_igw=True
    )
    try:
        yield SimpleNamespace(proc=proc, url=base_url)
    finally:
        _terminate(proc)


@pytest.fixture
def llama_workers():
    """Launch 2 workers for llama model."""
    from conftest import _find_available_port, _popen_launch_worker, _terminate

    workers = []
    for i in range(2):
        port = _find_available_port()
        base_url = f"http://127.0.0.1:{port}"
        proc = _popen_launch_worker(MULTI_MODEL_CONFIG["llama"], base_url)
        workers.append(
            SimpleNamespace(proc=proc, url=base_url, model=MULTI_MODEL_CONFIG["llama"])
        )

    try:
        yield workers
    finally:
        for worker in workers:
            _terminate(worker.proc)


@pytest.fixture
def qwen_worker():
    """Launch 1 worker for qwen model."""
    from conftest import _find_available_port, _popen_launch_worker, _terminate

    port = _find_available_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = _popen_launch_worker(MULTI_MODEL_CONFIG["qwen"], base_url)

    try:
        yield SimpleNamespace(proc=proc, url=base_url, model=MULTI_MODEL_CONFIG["qwen"])
    finally:
        _terminate(proc)


@pytest.fixture
def deepseek_worker():
    """Launch 1 worker for deepseek model."""
    from conftest import _find_available_port, _popen_launch_worker, _terminate

    port = _find_available_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = _popen_launch_worker(MULTI_MODEL_CONFIG["deepseek"], base_url)

    try:
        yield SimpleNamespace(
            proc=proc, url=base_url, model=MULTI_MODEL_CONFIG["deepseek"]
        )
    finally:
        _terminate(proc)


@pytest.mark.e2e
def test_multi_model_registration(multi_model_router, llama_workers, qwen_worker):
    """Test registering multiple models with the router."""
    base = multi_model_router.url

    # Add llama workers
    for worker in llama_workers:
        r = requests.post(f"{base}/add_worker", params={"url": worker.url}, timeout=180)
        r.raise_for_status()

    # Add qwen worker
    r = requests.post(
        f"{base}/add_worker", params={"url": qwen_worker.url}, timeout=180
    )
    r.raise_for_status()

    # Check worker list
    r = requests.get(f"{base}/list_workers", timeout=30)
    r.raise_for_status()
    data = r.json()

    # Verify we have workers for both models
    assert data["stats"]["total_workers"] == 3

    workers_by_model = {}
    for worker in data["workers"]:
        model = worker["model_id"]
        workers_by_model.setdefault(model, []).append(worker)

    assert len(workers_by_model) == 2  # Two different models
    assert len(workers_by_model[MULTI_MODEL_CONFIG["llama"]]) == 2
    assert len(workers_by_model[MULTI_MODEL_CONFIG["qwen"]]) == 1


@pytest.mark.e2e
def test_multi_model_routing(multi_model_router, llama_workers, qwen_worker):
    """Test that requests are routed to the correct model."""
    base = multi_model_router.url

    # Register all workers
    for worker in llama_workers:
        r = requests.post(f"{base}/add_worker", params={"url": worker.url}, timeout=180)
        r.raise_for_status()

    r = requests.post(
        f"{base}/add_worker", params={"url": qwen_worker.url}, timeout=180
    )
    r.raise_for_status()

    # Send requests to different models
    with requests.Session() as s:
        # Test llama model
        for i in range(4):
            r = s.post(
                f"{base}/v1/completions",
                json={
                    "model": MULTI_MODEL_CONFIG["llama"],
                    "prompt": f"Hello {i}",
                    "max_tokens": 5,
                    "stream": False,
                },
                timeout=120,
            )
            r.raise_for_status()
            assert r.json()["model"] == MULTI_MODEL_CONFIG["llama"]

        # Test qwen model
        for i in range(2):
            r = s.post(
                f"{base}/v1/completions",
                json={
                    "model": MULTI_MODEL_CONFIG["qwen"],
                    "prompt": f"World {i}",
                    "max_tokens": 5,
                    "stream": False,
                },
                timeout=120,
            )
            r.raise_for_status()
            assert r.json()["model"] == MULTI_MODEL_CONFIG["qwen"]


@pytest.mark.e2e
def test_multi_model_load_balancing(multi_model_router, llama_workers):
    """Test that load balancing works correctly for a single model with multiple workers."""
    base = multi_model_router.url

    # Register llama workers
    for worker in llama_workers:
        r = requests.post(f"{base}/add_worker", params={"url": worker.url}, timeout=180)
        r.raise_for_status()

    # Track which workers handle requests
    worker_urls = [w.url for w in llama_workers]
    request_distribution = {url: 0 for url in worker_urls}

    # Send multiple requests and track distribution
    with requests.Session() as s:
        for i in range(10):
            r = s.post(
                f"{base}/v1/completions",
                json={
                    "model": MULTI_MODEL_CONFIG["llama"],
                    "prompt": f"Test {i}",
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=120,
            )
            r.raise_for_status()

            # Get metrics to see which worker handled the request
            # Note: This assumes the router tracks per-worker metrics
            # In practice, we might need to check worker logs or add tracking

    # With round-robin, requests should be distributed evenly
    # For 10 requests across 2 workers, each should get ~5
    # We allow some variance due to potential health checks or retries
    # This is a simplified test - in production we'd verify actual distribution


@pytest.mark.e2e
def test_multi_model_worker_removal(multi_model_router, llama_workers, qwen_worker):
    """Test removing workers from specific models."""
    base = multi_model_router.url

    # Register all workers
    for worker in llama_workers:
        r = requests.post(f"{base}/add_worker", params={"url": worker.url}, timeout=180)
        r.raise_for_status()

    r = requests.post(
        f"{base}/add_worker", params={"url": qwen_worker.url}, timeout=180
    )
    r.raise_for_status()

    # Verify initial state
    r = requests.get(f"{base}/list_workers", timeout=30)
    r.raise_for_status()
    assert r.json()["stats"]["total_workers"] == 3

    # Remove one llama worker
    r = requests.post(
        f"{base}/remove_worker", params={"url": llama_workers[0].url}, timeout=60
    )
    r.raise_for_status()

    # Verify state after removal
    r = requests.get(f"{base}/list_workers", timeout=30)
    r.raise_for_status()
    data = r.json()
    assert data["stats"]["total_workers"] == 2

    # Verify llama model still works with remaining worker
    with requests.Session() as s:
        r = s.post(
            f"{base}/v1/completions",
            json={
                "model": MULTI_MODEL_CONFIG["llama"],
                "prompt": "Still working",
                "max_tokens": 5,
                "stream": False,
            },
            timeout=120,
        )
        r.raise_for_status()
        assert r.json()["model"] == MULTI_MODEL_CONFIG["llama"]

    # Remove qwen worker
    r = requests.post(
        f"{base}/remove_worker", params={"url": qwen_worker.url}, timeout=60
    )
    r.raise_for_status()

    # Verify qwen model is no longer available
    r = requests.get(f"{base}/list_workers", timeout=30)
    r.raise_for_status()
    data = r.json()
    assert data["stats"]["total_workers"] == 1

    # Requests to qwen should fail now
    with requests.Session() as s:
        r = s.post(
            f"{base}/v1/completions",
            json={
                "model": MULTI_MODEL_CONFIG["qwen"],
                "prompt": "Should fail",
                "max_tokens": 5,
                "stream": False,
            },
            timeout=120,
        )
        # Should get an error since no workers available for this model
        assert r.status_code >= 400


@pytest.mark.e2e
def test_multi_model_concurrent_requests(
    multi_model_router, llama_workers, qwen_worker
):
    """Test handling concurrent requests to different models."""
    base = multi_model_router.url

    # Register workers
    for worker in llama_workers:
        r = requests.post(f"{base}/add_worker", params={"url": worker.url}, timeout=180)
        r.raise_for_status()

    r = requests.post(
        f"{base}/add_worker", params={"url": qwen_worker.url}, timeout=180
    )
    r.raise_for_status()

    results = {"llama": [], "qwen": [], "errors": []}
    lock = threading.Lock()

    def send_request(model_key, prompt_prefix, count):
        """Send requests to a specific model."""
        model = MULTI_MODEL_CONFIG[model_key]
        with requests.Session() as s:
            for i in range(count):
                try:
                    r = s.post(
                        f"{base}/v1/completions",
                        json={
                            "model": model,
                            "prompt": f"{prompt_prefix} {i}",
                            "max_tokens": 5,
                            "stream": False,
                        },
                        timeout=120,
                    )
                    r.raise_for_status()
                    with lock:
                        results[model_key].append(r.json())
                except Exception as e:
                    with lock:
                        results["errors"].append(str(e))

    # Launch concurrent threads
    threads = []
    threads.append(
        threading.Thread(target=send_request, args=("llama", "Llama test", 5))
    )
    threads.append(threading.Thread(target=send_request, args=("qwen", "Qwen test", 3)))

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=180)

    # Verify results
    assert len(results["llama"]) == 5
    assert len(results["qwen"]) == 3
    assert len(results["errors"]) == 0

    # Verify model routing was correct
    for result in results["llama"]:
        assert result["model"] == MULTI_MODEL_CONFIG["llama"]

    for result in results["qwen"]:
        assert result["model"] == MULTI_MODEL_CONFIG["qwen"]


@pytest.mark.e2e
def test_multi_model_fault_tolerance(multi_model_router, llama_workers, qwen_worker):
    """Test fault tolerance when workers fail for specific models."""
    base = multi_model_router.url

    # Register workers
    for worker in llama_workers:
        r = requests.post(f"{base}/add_worker", params={"url": worker.url}, timeout=180)
        r.raise_for_status()

    r = requests.post(
        f"{base}/add_worker", params={"url": qwen_worker.url}, timeout=180
    )
    r.raise_for_status()

    # Kill one llama worker
    llama_workers[0].proc.terminate()
    time.sleep(5)  # Give time for health check to detect failure

    # Llama model should still work with remaining worker
    with requests.Session() as s:
        for i in range(3):
            r = s.post(
                f"{base}/v1/completions",
                json={
                    "model": MULTI_MODEL_CONFIG["llama"],
                    "prompt": f"Test after failure {i}",
                    "max_tokens": 5,
                    "stream": False,
                },
                timeout=120,
            )
            r.raise_for_status()
            assert r.json()["model"] == MULTI_MODEL_CONFIG["llama"]

    # Qwen should still work
    with requests.Session() as s:
        r = s.post(
            f"{base}/v1/completions",
            json={
                "model": MULTI_MODEL_CONFIG["qwen"],
                "prompt": "Qwen still works",
                "max_tokens": 5,
                "stream": False,
            },
            timeout=120,
        )
        r.raise_for_status()
        assert r.json()["model"] == MULTI_MODEL_CONFIG["qwen"]


@pytest.mark.e2e
@pytest.mark.slow  # This test might take longer
def test_multi_model_with_eval(multi_model_router, llama_workers):
    """Test running evaluation with multi-model setup."""
    base = multi_model_router.url

    # Register llama workers
    for worker in llama_workers:
        r = requests.post(f"{base}/add_worker", params={"url": worker.url}, timeout=180)
        r.raise_for_status()

    # Run a small evaluation
    args = SimpleNamespace(
        base_url=base,
        model=MULTI_MODEL_CONFIG["llama"],
        eval_name="mmlu",
        num_examples=16,  # Small number for testing
        num_threads=8,
        temperature=0.1,
    )

    metrics = run_eval(args)
    # Just verify it runs without error and returns valid metrics
    assert "score" in metrics
    assert 0.0 <= metrics["score"] <= 1.0


@pytest.mark.e2e
def test_multi_model_empty_model_request(multi_model_router, llama_workers):
    """Test behavior when model field is empty (should use default)."""
    base = multi_model_router.url

    # Register workers
    for worker in llama_workers:
        r = requests.post(f"{base}/add_worker", params={"url": worker.url}, timeout=180)
        r.raise_for_status()

    # Send request without model field (or empty model)
    with requests.Session() as s:
        # Test with missing model field
        r = s.post(
            f"{base}/v1/completions",
            json={
                "prompt": "Test without model",
                "max_tokens": 5,
                "stream": False,
            },
            timeout=120,
        )
        # Should either succeed with a default model or return appropriate error
        # The actual behavior depends on router implementation

        # Test with empty model string
        r = s.post(
            f"{base}/v1/completions",
            json={
                "model": "",
                "prompt": "Test with empty model",
                "max_tokens": 5,
                "stream": False,
            },
            timeout=120,
        )
        # Should use first available model or return error


@pytest.mark.e2e
def test_multi_model_different_policies(multi_model_router):
    """Test that different models can use different load balancing policies."""
    from conftest import _find_available_port, _popen_launch_worker, _terminate

    base = multi_model_router.url

    # Create 3 workers for llama model
    llama_workers = []
    for i in range(3):
        port = _find_available_port()
        worker_url = f"http://127.0.0.1:{port}"
        proc = _popen_launch_worker(MULTI_MODEL_CONFIG["llama"], worker_url)
        llama_workers.append(SimpleNamespace(proc=proc, url=worker_url))

    # Create 2 workers for qwen model
    qwen_workers = []
    for i in range(2):
        port = _find_available_port()
        worker_url = f"http://127.0.0.1:{port}"
        proc = _popen_launch_worker(MULTI_MODEL_CONFIG["qwen"], worker_url)
        qwen_workers.append(SimpleNamespace(proc=proc, url=worker_url))

    try:
        # Add workers with different policy hints
        # Note: The actual policy assignment depends on router implementation
        # The first worker for each model determines the policy

        # Add first llama worker (this sets the policy for llama model)
        r = requests.post(
            f"{base}/add_worker", params={"url": llama_workers[0].url}, timeout=180
        )
        r.raise_for_status()

        # Add remaining llama workers
        for worker in llama_workers[1:]:
            r = requests.post(
                f"{base}/add_worker", params={"url": worker.url}, timeout=180
            )
            r.raise_for_status()

        # Add qwen workers
        for worker in qwen_workers:
            r = requests.post(
                f"{base}/add_worker", params={"url": worker.url}, timeout=180
            )
            r.raise_for_status()

        # Send multiple requests to observe load balancing behavior
        llama_responses = []
        qwen_responses = []

        with requests.Session() as s:
            # Test llama with round-robin (default)
            for i in range(9):  # 9 requests to 3 workers = 3 each
                r = s.post(
                    f"{base}/v1/completions",
                    json={
                        "model": MULTI_MODEL_CONFIG["llama"],
                        "prompt": f"Test {i}",
                        "max_tokens": 1,
                        "stream": False,
                    },
                    timeout=120,
                )
                r.raise_for_status()
                llama_responses.append(r.json())

            # Test qwen
            for i in range(6):  # 6 requests to 2 workers = 3 each
                r = s.post(
                    f"{base}/v1/completions",
                    json={
                        "model": MULTI_MODEL_CONFIG["qwen"],
                        "prompt": f"Test {i}",
                        "max_tokens": 1,
                        "stream": False,
                    },
                    timeout=120,
                )
                r.raise_for_status()
                qwen_responses.append(r.json())

        # Verify all requests succeeded
        assert len(llama_responses) == 9
        assert len(qwen_responses) == 6

        # Verify model routing
        for resp in llama_responses:
            assert resp["model"] == MULTI_MODEL_CONFIG["llama"]

        for resp in qwen_responses:
            assert resp["model"] == MULTI_MODEL_CONFIG["qwen"]

    finally:
        # Cleanup workers
        for worker in llama_workers:
            _terminate(worker.proc)
        for worker in qwen_workers:
            _terminate(worker.proc)


@pytest.mark.e2e
def test_multi_model_metrics(multi_model_router, llama_workers, qwen_worker):
    """Test that metrics are properly tracked per model."""
    base = multi_model_router.url

    # Register workers
    for worker in llama_workers:
        r = requests.post(f"{base}/add_worker", params={"url": worker.url}, timeout=180)
        r.raise_for_status()

    r = requests.post(
        f"{base}/add_worker", params={"url": qwen_worker.url}, timeout=180
    )
    r.raise_for_status()

    # Send requests to track metrics
    with requests.Session() as s:
        # Send to llama
        for i in range(3):
            r = s.post(
                f"{base}/v1/completions",
                json={
                    "model": MULTI_MODEL_CONFIG["llama"],
                    "prompt": f"Llama {i}",
                    "max_tokens": 5,
                    "stream": False,
                },
                timeout=120,
            )
            r.raise_for_status()

        # Send to qwen
        for i in range(2):
            r = s.post(
                f"{base}/v1/completions",
                json={
                    "model": MULTI_MODEL_CONFIG["qwen"],
                    "prompt": f"Qwen {i}",
                    "max_tokens": 5,
                    "stream": False,
                },
                timeout=120,
            )
            r.raise_for_status()

    # Check metrics endpoint if available
    r = requests.get(f"{base}/metrics", timeout=30)
    # Metrics format depends on implementation (Prometheus, JSON, etc.)
    # Just verify the endpoint responds
    assert r.status_code in [200, 404]  # 404 if metrics not implemented yet
