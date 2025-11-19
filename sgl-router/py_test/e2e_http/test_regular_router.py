import threading
import time
from types import SimpleNamespace

import pytest
import requests

from sglang.test.run_eval import run_eval


def _wait_for_workers(
    base_url: str, expected_count: int, timeout: float = 60.0, headers: dict = None
) -> None:
    """Poll /workers endpoint until expected number of workers are registered."""
    start = time.perf_counter()
    with requests.Session() as session:
        while time.perf_counter() - start < timeout:
            try:
                r = session.get(f"{base_url}/workers", headers=headers, timeout=5)
                if r.status_code == 200:
                    workers = r.json().get("workers", [])
                    if len(workers) >= expected_count:
                        return
            except requests.RequestException:
                pass
            time.sleep(0.5)
    raise TimeoutError(
        f"Expected {expected_count} workers at {base_url}, timed out after {timeout}s"
    )


@pytest.mark.e2e
def test_mmlu(e2e_router_only_rr, e2e_two_workers_dp2, e2e_model):
    # Attach two dp=2 workers (total 4 GPUs) to a fresh router-only instance
    base = e2e_router_only_rr.url
    for w in e2e_two_workers_dp2:
        r = requests.post(f"{base}/workers", json={"url": w.url}, timeout=180)
        assert (
            r.status_code == 202
        ), f"Expected 202 ACCEPTED, got {r.status_code}: {r.text}"

    # Wait for workers to be registered
    _wait_for_workers(base, expected_count=2, timeout=60.0)

    args = SimpleNamespace(
        base_url=base,
        model=e2e_model,
        eval_name="mmlu",
        num_examples=64,
        num_threads=32,
        temperature=0.1,
    )
    metrics = run_eval(args)
    assert metrics["score"] >= 0.65


@pytest.mark.e2e
def test_genai_bench(
    e2e_router_only_rr, e2e_two_workers_dp2, e2e_model, genai_bench_runner
):
    """Attach a worker to the regular router and run a short genai-bench."""
    base = e2e_router_only_rr.url
    for w in e2e_two_workers_dp2:
        r = requests.post(f"{base}/workers", json={"url": w.url}, timeout=180)
        assert (
            r.status_code == 202
        ), f"Expected 202 ACCEPTED, got {r.status_code}: {r.text}"

    # Wait for workers to be registered
    _wait_for_workers(base, expected_count=2, timeout=60.0)

    genai_bench_runner(
        router_url=base,
        model_path=e2e_model,
        experiment_folder="benchmark_round_robin_regular",
        thresholds={
            "ttft_mean_max": 6,
            "e2e_latency_mean_max": 14,
            "input_throughput_mean_min": 800,  # temp relax from 1000 to 800 for now
            "output_throughput_mean_min": 12,
            # Enforce GPU utilization p50 >= 99% during the run.
            "gpu_util_p50_min": 99,
        },
        kill_procs=e2e_two_workers_dp2,
    )


@pytest.mark.e2e
def test_add_and_remove_worker_live(e2e_router_only_rr, e2e_primary_worker, e2e_model):
    base = e2e_router_only_rr.url
    worker_url = e2e_primary_worker.url

    r = requests.post(f"{base}/workers", json={"url": worker_url}, timeout=180)
    assert r.status_code == 202, f"Expected 202 ACCEPTED, got {r.status_code}: {r.text}"

    # Wait for worker to be registered
    _wait_for_workers(base, expected_count=1, timeout=60.0)

    with requests.Session() as s:
        for i in range(8):
            r = s.post(
                f"{base}/v1/completions",
                json={
                    "model": e2e_model,
                    "prompt": f"x{i}",
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=120,
            )
            r.raise_for_status()

    # Remove the worker
    from urllib.parse import quote

    encoded_url = quote(worker_url, safe="")
    r = requests.delete(f"{base}/workers/{encoded_url}", timeout=60)
    assert r.status_code == 202, f"Expected 202 ACCEPTED, got {r.status_code}: {r.text}"


@pytest.mark.e2e
def test_lazy_fault_tolerance_live(e2e_router_only_rr, e2e_primary_worker, e2e_model):
    base = e2e_router_only_rr.url
    worker = e2e_primary_worker

    r = requests.post(f"{base}/workers", json={"url": worker.url}, timeout=180)
    assert r.status_code == 202, f"Expected 202 ACCEPTED, got {r.status_code}: {r.text}"

    # Wait for worker to be registered
    _wait_for_workers(base, expected_count=1, timeout=60.0)

    def killer():
        time.sleep(10)
        try:
            worker.proc.terminate()
        except Exception:
            pass

    t = threading.Thread(target=killer, daemon=True)
    t.start()

    args = SimpleNamespace(
        base_url=base,
        model=e2e_model,
        eval_name="mmlu",
        num_examples=32,
        num_threads=16,
        temperature=0.0,
    )
    metrics = run_eval(args)
    assert 0.0 <= metrics["score"] <= 1.0


@pytest.mark.e2e
def test_dp_aware_worker_expansion_and_api_key(
    e2e_model,
    e2e_router_only_rr_dp_aware_api,
    e2e_worker_dp2_api,
):
    """
    Launch a router-only instance in dp_aware mode and a single worker with dp_size=2
    and API key protection. Verify expansion, auth enforcement, and basic eval.
    """
    import os

    router_url = e2e_router_only_rr_dp_aware_api.url
    worker_url = e2e_worker_dp2_api.url
    api_key = e2e_router_only_rr_dp_aware_api.api_key

    # Attach worker; router should expand to dp_size logical workers
    r = requests.post(
        f"{router_url}/workers",
        json={"url": worker_url, "api_key": api_key},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=180,
    )
    assert r.status_code == 202, f"Expected 202 ACCEPTED, got {r.status_code}: {r.text}"

    # Wait for workers to be registered and expanded
    _wait_for_workers(
        router_url,
        expected_count=2,
        timeout=60.0,
        headers={"Authorization": f"Bearer {api_key}"},
    )

    # Verify the expanded workers have correct URLs
    r = requests.get(
        f"{router_url}/workers",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    r.raise_for_status()
    workers = r.json().get("workers", [])
    urls = [w["url"] for w in workers]
    assert len(urls) == 2
    assert set(urls) == {f"{worker_url}@0", f"{worker_url}@1"}

    # Verify API key enforcement
    # 1) Without Authorization -> Should get 401 Unauthorized
    r = requests.post(
        f"{router_url}/v1/completions",
        json={"model": e2e_model, "prompt": "hi", "max_tokens": 1},
        timeout=60,
    )
    assert r.status_code == 401

    # 2) With correct Authorization -> 200
    r = requests.post(
        f"{router_url}/v1/completions",
        json={"model": e2e_model, "prompt": "hi", "max_tokens": 1},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    assert r.status_code == 200

    # Finally, run MMLU eval through the router with auth
    os.environ["OPENAI_API_KEY"] = api_key
    args = SimpleNamespace(
        base_url=router_url,
        model=e2e_model,
        eval_name="mmlu",
        num_examples=64,
        num_threads=32,
        temperature=0.1,
    )
    metrics = run_eval(args)
    assert metrics["score"] >= 0.65
