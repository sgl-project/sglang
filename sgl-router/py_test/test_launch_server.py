import threading
import time
from types import SimpleNamespace

import pytest
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
)


def terminate_and_wait(process, timeout=300):
    """Terminate a process and wait until it is terminated."""
    if process is None:
        return

    process.terminate()
    start_time = time.perf_counter()

    while process.poll() is None:
        if time.perf_counter() - start_time > timeout:
            raise TimeoutError(
                f"Process {process.pid} failed to terminate within {timeout}s"
            )
        time.sleep(1)


def test_mmlu(router_manager):
    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = router_manager.ensure_router(
        model=model,
        dp_size=2,
        policy="cache_aware",
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    )

    args = SimpleNamespace(
        base_url=base_url,
        model=model,
        eval_name="mmlu",
        num_examples=64,
        num_threads=32,
        temperature=0.1,
    )

    metrics = run_eval(args)
    score = metrics["score"]
    THRESHOLD = 0.635
    assert score >= THRESHOLD, (
        f"MMLU test failed with score {score:.3f} (threshold: {THRESHOLD})"
    )


def test_add_and_remove_worker(router_manager):
    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = router_manager.ensure_router(
        model=model,
        dp_size=1,
        policy="round_robin",
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    )

    # 1. start a worker
    worker_url = router_manager.start_worker(model=model)

    # 2. add worker to the router (it will be used after healthy)
    with requests.Session() as session:
        response = session.post(f"{base_url}/add_worker?url={worker_url}")
        assert response.status_code == 200

    # 3. run mmlu
    args = SimpleNamespace(
        base_url=base_url,
        model=model,
        eval_name="mmlu",
        num_examples=64,
        num_threads=32,
        temperature=0.1,
    )
    metrics = run_eval(args)
    score = metrics["score"]
    THRESHOLD = 0.635
    assert score >= THRESHOLD, (
        f"MMLU test failed with score {score:.3f} (threshold: {THRESHOLD})"
    )

    # 4. remove worker
    with requests.Session() as session:
        response = session.post(f"{base_url}/remove_worker?url={worker_url}")
        assert response.status_code == 200

    # 5. run mmlu again
    metrics = run_eval(args)
    score = metrics["score"]
    assert score >= THRESHOLD, (
        f"MMLU test failed with score {score:.3f} (threshold: {THRESHOLD})"
    )


def test_lazy_fault_tolerance(router_manager):
    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = router_manager.ensure_router(
        model=model,
        dp_size=1,
        policy="round_robin",
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    )

    # 1. start a worker
    worker_url = router_manager.start_worker(model=model)

    # 2. add worker
    with requests.Session() as session:
        response = session.post(f"{base_url}/add_worker?url={worker_url}")
        assert response.status_code == 200

    # Kill the worker after 10 seconds to mimic abrupt failure
    proc = router_manager._workers[worker_url]

    def kill_worker():
        time.sleep(10)
        kill_process_tree(proc.pid)

    threading.Thread(target=kill_worker, daemon=True).start()

    # 3. run mmlu
    args = SimpleNamespace(
        base_url=base_url,
        model=model,
        eval_name="mmlu",
        num_examples=256,
        num_threads=32,
        temperature=0.1,
    )
    metrics = run_eval(args)
    score = metrics["score"]
    THRESHOLD = 0.635
    assert score >= THRESHOLD, (
        f"MMLU test failed with score {score:.3f} (threshold: {THRESHOLD})"
    )


def test_payload_size(router_manager):
    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = router_manager.ensure_router(
        model=model,
        dp_size=1,
        policy="round_robin",
        max_payload_size=1 * 1024 * 1024,  # 1MB limit
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    )

    # Test case 1: Payload just under 1MB should succeed
    payload_0_5_mb = {
        "text": "x" * int(0.5 * 1024 * 1024),  # 0.5MB of text
        "temperature": 0.0,
    }
    with requests.Session() as session:
        response = session.post(
            f"{base_url}/generate",
            json=payload_0_5_mb,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200

    # Test case 2: Payload over 1MB should fail
    payload_1_plus_mb = {
        "text": "x" * int((1.2 * 1024 * 1024)),  # 1.2MB of text
        "temperature": 0.0,
    }
    with requests.Session() as session:
        response = session.post(
            f"{base_url}/generate",
            json=payload_1_plus_mb,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 413


def test_api_key(router_manager):
    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = router_manager.ensure_router(
        model=model,
        dp_size=1,
        policy="round_robin",
        api_key="correct_api_key",
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    )

    # Test case 1: request without api key should fail
    with requests.Session() as session:
        response = session.post(
            f"{base_url}/generate",
            json={"text": "Kanye west is, ", "temperature": 0},
        )
        assert response.status_code == 401

    # Test case 2: request with invalid api key should fail
    with requests.Session() as session:
        response = requests.post(
            f"{base_url}/generate",
            json={"text": "Kanye west is, ", "temperature": 0},
            headers={"Authorization": "Bearer 123"},
        )
        assert response.status_code == 401

    # Test case 3: request with correct api key should succeed
    with requests.Session() as session:
        response = session.post(
            f"{base_url}/generate",
            json={"text": "Kanye west is ", "temperature": 0},
            headers={"Authorization": "Bearer correct_api_key"},
        )
        assert response.status_code == 200


def test_mmlu_with_dp_aware(router_manager):
    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = router_manager.ensure_router(
        model=model,
        dp_size=2,
        policy="cache_aware",
        dp_aware=True,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    )

    args = SimpleNamespace(
        base_url=base_url,
        model=model,
        eval_name="mmlu",
        num_examples=64,
        num_threads=32,
        temperature=0.1,
    )

    metrics = run_eval(args)
    score = metrics["score"]
    THRESHOLD = 0.635
    assert score >= THRESHOLD, (
        f"dp aware MMLU test failed with score {score:.3f} (threshold: {THRESHOLD})"
    )


def test_add_and_remove_worker_with_dp_aware(router_manager):
    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = router_manager.ensure_router(
        model=model,
        dp_size=1,
        policy="round_robin",
        dp_aware=True,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    )

    # 1. Start a worker
    worker_url = router_manager.start_worker(model=model)

    # 2. Add it to the router
    with requests.Session() as session:
        response = session.post(f"{base_url}/add_worker?url={worker_url}")
        assert response.status_code == 200

    # 3. Run mmlu
    args = SimpleNamespace(
        base_url=base_url,
        model=model,
        eval_name="mmlu",
        num_examples=64,
        num_threads=32,
        temperature=0.1,
    )
    metrics = run_eval(args)
    score = metrics["score"]
    THRESHOLD = 0.635
    assert score >= THRESHOLD, (
        f"MMLU test failed with score {score:.3f} (threshold: {THRESHOLD})"
    )

    # 4. Remove it from the router
    with requests.Session() as session:
        response = session.post(f"{base_url}/remove_worker?url={worker_url}")
        assert response.status_code == 200

    # 5. Run mmlu again
    metrics = run_eval(args)
    score = metrics["score"]
    assert score >= THRESHOLD, (
        f"MMLU test failed with score {score:.3f} (threshold: {THRESHOLD})"
    )

    # 6. Start another worker with api_key set
    prev_proc = router_manager._workers.get(worker_url)
    if prev_proc is not None:
        terminate_and_wait(prev_proc)

    worker_url2 = router_manager.start_worker(
        model=model,
        api_key="correct_api_key",
    )

    # 7. Adding this worker should fail without the router knowing the API key
    with requests.Session() as session:
        response = session.post(f"{base_url}/add_worker?url={worker_url2}")
        assert response.status_code != 200


def test_lazy_fault_tolerance_with_dp_aware(router_manager):
    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = router_manager.ensure_router(
        model=model,
        dp_size=1,
        policy="round_robin",
        dp_aware=True,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    )

    # 1. Start a worker
    worker_url = router_manager.start_worker(model=model)

    # 2. Add to router
    with requests.Session() as session:
        response = session.post(f"{base_url}/add_worker?url={worker_url}")
        assert response.status_code == 200

    # Kill after 10 seconds
    proc = router_manager._workers[worker_url]

    def kill_worker():
        time.sleep(10)
        kill_process_tree(proc.pid)

    threading.Thread(target=kill_worker, daemon=True).start()

    # 3. Run mmlu
    args = SimpleNamespace(
        base_url=base_url,
        model=model,
        eval_name="mmlu",
        num_examples=256,
        num_threads=32,
        temperature=0.1,
    )
    metrics = run_eval(args)
    score = metrics["score"]
    THRESHOLD = 0.635
    assert score >= THRESHOLD, (
        f"MMLU test failed with score {score:.3f} (threshold: {THRESHOLD})"
    )


def test_payload_size_with_dp_aware(router_manager):
    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = router_manager.ensure_router(
        model=model,
        dp_size=1,
        policy="round_robin",
        max_payload_size=1 * 1024 * 1024,  # 1MB limit
        dp_aware=True,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    )

    # Test case 1: Payload just under 1MB should succeed
    payload_0_5_mb = {
        "text": "x" * int(0.5 * 1024 * 1024),  # 0.5MB of text
        "temperature": 0.0,
    }
    with requests.Session() as session:
        response = session.post(
            f"{base_url}/generate",
            json=payload_0_5_mb,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200

    # Test case 2: Payload over 1MB should fail
    payload_1_plus_mb = {
        "text": "x" * int((1.2 * 1024 * 1024)),  # 1.2MB of text
        "temperature": 0.0,
    }
    with requests.Session() as session:
        response = session.post(
            f"{base_url}/generate",
            json=payload_1_plus_mb,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 413


def test_api_key_with_dp_aware(router_manager):
    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = router_manager.ensure_router(
        model=model,
        dp_size=1,
        policy="round_robin",
        api_key="correct_api_key",
        dp_aware=True,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    )

    # Test case 1: request without api key should fail
    with requests.Session() as session:
        response = session.post(
            f"{base_url}/generate",
            json={"text": "Kanye west is, ", "temperature": 0},
        )
        assert response.status_code == 401

    # Test case 2: request with invalid api key should fail
    with requests.Session() as session:
        response = requests.post(
            f"{base_url}/generate",
            json={"text": "Kanye west is, ", "temperature": 0},
            headers={"Authorization": "Bearer 123"},
        )
        assert response.status_code == 401

    # Test case 3: request with correct api key should succeed
    with requests.Session() as session:
        response = session.post(
            f"{base_url}/generate",
            json={"text": "Kanye west is ", "temperature": 0},
            headers={"Authorization": "Bearer correct_api_key"},
        )
        assert response.status_code == 200

