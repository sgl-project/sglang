import threading
import time
from types import SimpleNamespace

import pytest
import requests

from sglang.test.run_eval import run_eval


@pytest.mark.e2e
def test_mmlu(e2e_router_only_rr, e2e_primary_worker, e2e_model):
    # Attach the primary worker to a fresh router-only instance (single model)
    base = e2e_router_only_rr.url
    r = requests.post(
        f"{base}/add_worker", params={"url": e2e_primary_worker.url}, timeout=180
    )
    r.raise_for_status()

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
def test_add_and_remove_worker_live(e2e_router_only_rr, e2e_primary_worker, e2e_model):
    base = e2e_router_only_rr.url
    worker_url = e2e_primary_worker.url

    r = requests.post(f"{base}/add_worker", params={"url": worker_url}, timeout=180)
    r.raise_for_status()

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
    r = requests.post(f"{base}/remove_worker", params={"url": worker_url}, timeout=60)
    r.raise_for_status()


@pytest.mark.e2e
def test_lazy_fault_tolerance_live(e2e_router_only_rr, e2e_primary_worker, e2e_model):
    base = e2e_router_only_rr.url
    worker = e2e_primary_worker

    r = requests.post(f"{base}/add_worker", params={"url": worker.url}, timeout=180)
    r.raise_for_status()

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
        f"{router_url}/add_worker", params={"url": worker_url}, timeout=180
    )
    r.raise_for_status()

    r = requests.get(f"{router_url}/list_workers", timeout=30)
    r.raise_for_status()
    urls = r.json().get("urls", [])
    assert len(urls) == 2
    assert set(urls) == {f"{worker_url}@0", f"{worker_url}@1"}

    # Verify API key enforcement path-through
    # 1) Without Authorization -> 401 from backend
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
