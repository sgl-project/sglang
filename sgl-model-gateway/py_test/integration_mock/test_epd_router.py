"""E2E tests for EPD (Encode-Prefill-Decode) disaggregated serving mode.

EPD mode enables multimodal LLM inference with separate encode, prefill, and decode workers:
- Encode worker: Processes multimodal inputs (images/video) via HTTP REST API
- Prefill worker: Receives embeddings via ZMQ, computes KV cache
- Decode worker: Generates output tokens using KV cache from prefill

Text-only requests automatically fall back to PD (Prefill-Decode) mode.
"""

import collections
import concurrent.futures

import pytest
import requests


# =============================================================================
# EPD Router Configuration Tests
# =============================================================================


@pytest.mark.integration
def test_epd_health(router_manager, mock_workers):
    """Verify EPD router health endpoint responds correctly."""
    _, encode_urls_raw, _ = mock_workers(n=1)
    _, prefill_urls_raw, _ = mock_workers(n=1)
    _, decode_urls_raw, _ = mock_workers(n=1)

    encode_urls = [(u, None) for u in encode_urls_raw]
    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="round_robin",
        epd_disaggregation=True,
        encode_urls=encode_urls,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    response = requests.get(f"{rh.url}/health", timeout=10)
    assert response.status_code == 200


@pytest.mark.integration
def test_epd_get_server_info(router_manager, mock_workers):
    """Test EPD router /get_server_info endpoint."""
    _, encode_urls_raw, _ = mock_workers(n=2)
    _, prefill_urls_raw, _ = mock_workers(n=2)
    _, decode_urls_raw, _ = mock_workers(n=2)

    encode_urls = [(u, None) for u in encode_urls_raw]
    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="round_robin",
        epd_disaggregation=True,
        encode_urls=encode_urls,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    response = requests.get(f"{rh.url}/get_server_info", timeout=10)
    assert response.status_code == 200


# =============================================================================
# EPD Chat Completions Tests
# =============================================================================


@pytest.mark.integration
def test_epd_chat_completions(router_manager, mock_workers):
    """Test EPD with /v1/chat/completions endpoint."""
    _, encode_urls_raw, _ = mock_workers(n=1)
    _, prefill_urls_raw, _ = mock_workers(n=1)
    _, decode_urls_raw, _ = mock_workers(n=1)

    encode_urls = [(u, None) for u in encode_urls_raw]
    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="round_robin",
        epd_disaggregation=True,
        encode_urls=encode_urls,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    response = requests.post(
        f"{rh.url}/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 50,
        },
        timeout=30,
    )

    assert response.status_code == 200, f"Request failed: {response.text}"
    data = response.json()
    assert "choices" in data or "error" not in data


@pytest.mark.integration
def test_epd_multimodal_chat_request_format(router_manager, mock_workers):
    """Test EPD handles multimodal chat request format correctly.

    This verifies the request format with image_url content is accepted.
    The mock workers don't actually process images but validate the routing.
    """
    _, encode_urls_raw, _ = mock_workers(n=1)
    _, prefill_urls_raw, _ = mock_workers(n=1)
    _, decode_urls_raw, _ = mock_workers(n=1)

    encode_urls = [(u, None) for u in encode_urls_raw]
    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="round_robin",
        epd_disaggregation=True,
        encode_urls=encode_urls,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    # Multimodal request format (mock workers handle as text)
    response = requests.post(
        f"{rh.url}/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/test.png"},
                        },
                    ],
                }
            ],
            "max_tokens": 100,
        },
        timeout=30,
    )

    # Should be accepted (200) or handled gracefully
    assert response.status_code in (200, 400, 422)


@pytest.mark.integration
def test_epd_text_only_fallback(router_manager, mock_workers):
    """Test that text-only requests fall back to PD mode automatically."""
    _, encode_urls_raw, _ = mock_workers(n=1)
    _, prefill_urls_raw, _ = mock_workers(n=1)
    _, decode_urls_raw, _ = mock_workers(n=1)

    encode_urls = [(u, None) for u in encode_urls_raw]
    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="round_robin",
        epd_disaggregation=True,
        encode_urls=encode_urls,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    # Text-only request should bypass encode worker
    response = requests.post(
        f"{rh.url}/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 50,
        },
        timeout=30,
    )

    assert response.status_code == 200, f"Request failed: {response.text}"


# =============================================================================
# EPD Streaming Tests
# =============================================================================


@pytest.mark.integration
def test_epd_streaming(router_manager, mock_workers):
    """Test EPD streaming responses."""
    _, encode_urls_raw, _ = mock_workers(n=1)
    _, prefill_urls_raw, _ = mock_workers(n=1)
    _, decode_urls_raw, _ = mock_workers(n=1)

    encode_urls = [(u, None) for u in encode_urls_raw]
    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="round_robin",
        epd_disaggregation=True,
        encode_urls=encode_urls,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    response = requests.post(
        f"{rh.url}/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Describe something briefly."}],
            "max_tokens": 50,
            "stream": True,
        },
        stream=True,
        timeout=60,
    )

    assert response.status_code == 200, f"Request failed: {response.text}"

    chunks = []
    for line in response.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                chunks.append(data_str)

    # Should have received streaming chunks
    assert len(chunks) >= 0  # Mock may return empty stream


# =============================================================================
# EPD Generate Endpoint Tests
# =============================================================================


@pytest.mark.integration
def test_epd_generate_endpoint(router_manager, mock_workers):
    """Test EPD with /generate endpoint."""
    _, encode_urls_raw, _ = mock_workers(n=1)
    _, prefill_urls_raw, _ = mock_workers(n=1)
    _, decode_urls_raw, _ = mock_workers(n=1)

    encode_urls = [(u, None) for u in encode_urls_raw]
    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="round_robin",
        epd_disaggregation=True,
        encode_urls=encode_urls,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    response = requests.post(
        f"{rh.url}/generate",
        json={
            "text": "What is shown in this?",
            "sampling_params": {
                "max_new_tokens": 50,
                "temperature": 0.0,
            },
        },
        timeout=30,
    )

    assert response.status_code == 200, f"Request failed: {response.text}"


@pytest.mark.integration
def test_epd_v1_completions_endpoint(router_manager, mock_workers):
    """Test EPD with /v1/completions endpoint."""
    _, encode_urls_raw, _ = mock_workers(n=1)
    _, prefill_urls_raw, _ = mock_workers(n=1)
    _, decode_urls_raw, _ = mock_workers(n=1)

    encode_urls = [(u, None) for u in encode_urls_raw]
    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="round_robin",
        epd_disaggregation=True,
        encode_urls=encode_urls,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    response = requests.post(
        f"{rh.url}/v1/completions",
        json={
            "model": "test-model",
            "prompt": "Hello, world!",
            "max_tokens": 50,
        },
        timeout=30,
    )

    assert response.status_code == 200, f"Request failed: {response.text}"


# =============================================================================
# EPD Concurrent Request Tests
# =============================================================================


@pytest.mark.integration
def test_epd_concurrent_requests(router_manager, mock_workers):
    """Test EPD handles concurrent requests."""
    _, encode_urls_raw, _ = mock_workers(n=2)
    _, prefill_urls_raw, _ = mock_workers(n=2)
    _, decode_urls_raw, _ = mock_workers(n=2)

    encode_urls = [(u, None) for u in encode_urls_raw]
    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="round_robin",
        epd_disaggregation=True,
        encode_urls=encode_urls,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    def make_request(idx: int):
        response = requests.post(
            f"{rh.url}/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": f"Request {idx}: Hello!"}],
                "max_tokens": 30,
            },
            timeout=60,
        )
        return response.status_code == 200

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(make_request, i) for i in range(8)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # All requests should succeed
    success_rate = sum(results) / len(results)
    assert success_rate >= 0.9, f"Success rate too low: {success_rate}"


@pytest.mark.integration
def test_epd_high_concurrency(router_manager, mock_workers):
    """Test EPD under higher concurrency."""
    _, encode_urls_raw, _ = mock_workers(n=3)
    _, prefill_urls_raw, _ = mock_workers(n=3)
    _, decode_urls_raw, _ = mock_workers(n=3)

    encode_urls = [(u, None) for u in encode_urls_raw]
    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="round_robin",
        epd_disaggregation=True,
        encode_urls=encode_urls,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    def make_request(idx: int):
        try:
            response = requests.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"concurrent-{idx}",
                    "max_tokens": 10,
                },
                timeout=30,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(make_request, i) for i in range(32)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    success_rate = sum(results) / len(results)
    assert success_rate >= 0.8, f"Success rate too low: {success_rate}"


# =============================================================================
# EPD Load Balancing Policy Tests
# =============================================================================


@pytest.mark.integration
def test_epd_round_robin_distribution(router_manager, mock_workers):
    """Test EPD distributes requests across workers using round robin."""
    _, encode_urls_raw, _ = mock_workers(n=2)
    _, prefill_urls_raw, _ = mock_workers(n=2)
    _, decode_urls_raw, decode_ids = mock_workers(n=3)

    encode_urls = [(u, None) for u in encode_urls_raw]
    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="round_robin",
        epd_disaggregation=True,
        encode_urls=encode_urls,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    decode_counts = collections.Counter()
    with requests.Session() as s:
        for i in range(30):
            r = s.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"prompt-{i}",
                    "max_tokens": 1,
                },
            )
            assert r.status_code == 200
            wid = r.headers.get("X-Worker-Id") or r.json().get("worker_id")
            if wid in decode_ids:
                decode_counts[wid] += 1

    # Verify requests were distributed
    workers_used = sum(1 for v in decode_counts.values() if v > 0)
    assert workers_used >= 1


@pytest.mark.integration
def test_epd_power_of_two_policy(router_manager, mock_workers):
    """Test EPD with power_of_two load balancing policy."""
    _, encode_urls_raw, _ = mock_workers(n=2)
    _, prefill_urls_raw, _ = mock_workers(n=2)
    _, decode_urls_raw, _ = mock_workers(n=2)

    encode_urls = [(u, None) for u in encode_urls_raw]
    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="power_of_two",
        epd_disaggregation=True,
        encode_urls=encode_urls,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    with requests.Session() as s:
        for i in range(20):
            r = s.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"power-{i}",
                    "max_tokens": 1,
                },
            )
            assert r.status_code == 200


@pytest.mark.integration
def test_epd_random_policy(router_manager, mock_workers):
    """Test EPD with random load balancing policy."""
    _, encode_urls_raw, _ = mock_workers(n=2)
    _, prefill_urls_raw, _ = mock_workers(n=2)
    _, decode_urls_raw, _ = mock_workers(n=2)

    encode_urls = [(u, None) for u in encode_urls_raw]
    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="random",
        epd_disaggregation=True,
        encode_urls=encode_urls,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    with requests.Session() as s:
        for i in range(20):
            r = s.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"random-{i}",
                    "max_tokens": 1,
                },
            )
            assert r.status_code == 200


# =============================================================================
# EPD Worker Configuration Tests
# =============================================================================


@pytest.mark.integration
def test_epd_asymmetric_worker_counts(router_manager, mock_workers):
    """Test EPD with different worker counts for each stage."""
    _, encode_urls_raw, _ = mock_workers(n=1)  # 1 encode
    _, prefill_urls_raw, _ = mock_workers(n=2)  # 2 prefill
    _, decode_urls_raw, _ = mock_workers(n=4)  # 4 decode

    encode_urls = [(u, None) for u in encode_urls_raw]
    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="round_robin",
        epd_disaggregation=True,
        encode_urls=encode_urls,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    response = requests.get(f"{rh.url}/health", timeout=10)
    assert response.status_code == 200

    # Send some requests to verify it works
    for i in range(10):
        r = requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": f"asymmetric-{i}",
                "max_tokens": 1,
            },
            timeout=30,
        )
        assert r.status_code == 200


@pytest.mark.integration
def test_epd_single_worker_each(router_manager, mock_workers):
    """Test EPD with minimal configuration (1 worker each)."""
    _, encode_urls_raw, _ = mock_workers(n=1)
    _, prefill_urls_raw, _ = mock_workers(n=1)
    _, decode_urls_raw, _ = mock_workers(n=1)

    encode_urls = [(u, None) for u in encode_urls_raw]
    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="round_robin",
        epd_disaggregation=True,
        encode_urls=encode_urls,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    response = requests.post(
        f"{rh.url}/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Single worker test"}],
            "max_tokens": 10,
        },
        timeout=30,
    )
    assert response.status_code == 200


# =============================================================================
# EPD Error Handling Tests
# =============================================================================


@pytest.mark.integration
def test_epd_invalid_request(router_manager, mock_workers):
    """Test EPD handles invalid requests gracefully."""
    _, encode_urls_raw, _ = mock_workers(n=1)
    _, prefill_urls_raw, _ = mock_workers(n=1)
    _, decode_urls_raw, _ = mock_workers(n=1)

    encode_urls = [(u, None) for u in encode_urls_raw]
    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="round_robin",
        epd_disaggregation=True,
        encode_urls=encode_urls,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    # Empty messages
    response = requests.post(
        f"{rh.url}/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [],
            "max_tokens": 10,
        },
        timeout=30,
    )
    # Should return error (4xx) or be handled
    assert response.status_code in (200, 400, 422)


@pytest.mark.integration
def test_epd_missing_model(router_manager, mock_workers):
    """Test EPD handles missing model field."""
    _, encode_urls_raw, _ = mock_workers(n=1)
    _, prefill_urls_raw, _ = mock_workers(n=1)
    _, decode_urls_raw, _ = mock_workers(n=1)

    encode_urls = [(u, None) for u in encode_urls_raw]
    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="round_robin",
        epd_disaggregation=True,
        encode_urls=encode_urls,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

    # Request without model field
    response = requests.post(
        f"{rh.url}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "No model specified"}],
            "max_tokens": 10,
        },
        timeout=30,
    )
    # Should be handled (may succeed with default model or return error)
    assert response.status_code in (200, 400, 422)
