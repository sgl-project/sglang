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
        f"{base}/add_worker", params={"url": e2e_primary_worker.url}, timeout=60
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

    r = requests.post(f"{base}/add_worker", params={"url": worker_url}, timeout=60)
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

    r = requests.post(f"{base}/add_worker", params={"url": worker.url}, timeout=60)
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
def test_dp_aware_worker_expansion_and_api_key(e2e_model):
    """
    Launch a router-only instance in dp_aware mode and a single worker with dp_size=2
    and API key protection. Verify that:
      - Adding the worker expands into 3 logical workers (suffix @0,@1,@2)
      - Requests without Authorization fail with 401
      - Requests with the correct API key succeed
    """
    import socket
    import subprocess

    def _free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _wait_health(url: str, timeout: float = 120.0):
        start = time.perf_counter()
        with requests.Session() as s:
            while time.perf_counter() - start < timeout:
                try:
                    r = s.get(f"{url}/health", timeout=5)
                    if r.status_code == 200:
                        return
                except requests.RequestException:
                    pass
                time.sleep(1)
        raise TimeoutError("Router did not become healthy in time")

    def _terminate(p: subprocess.Popen):
        if p is None:
            return
        p.terminate()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()

    # Spin up router-only with dp-aware enabled and API key
    router_port = _free_port()
    router_url = f"http://127.0.0.1:{router_port}"
    prom_port = _free_port()
    api_key = "secret"

    router_cmd = [
        "python3",
        "-m",
        "sglang_router.launch_router",
        "--host",
        "127.0.0.1",
        "--port",
        str(router_port),
        "--policy",
        "round_robin",
        "--dp-aware",
        "--api-key",
        api_key,
        "--prometheus-port",
        str(prom_port),
        "--prometheus-host",
        "127.0.0.1",
    ]
    router_proc = subprocess.Popen(router_cmd)
    try:
        _wait_health(router_url, timeout=180.0)

        # Spin up a DP=3 worker that requires the same API key
        worker_port = _free_port()
        worker_url = f"http://127.0.0.1:{worker_port}"
        worker_cmd = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            e2e_model,
            "--host",
            "127.0.0.1",
            "--port",
            str(worker_port),
            "--base-gpu-id",
            "0",
            "--dp-size",
            "2",
            "--api-key",
            api_key,
        ]
        worker_proc = subprocess.Popen(worker_cmd)
        try:
            # Attach worker to router; router should expand to dp_size logical workers
            r = requests.post(
                f"{router_url}/add_worker", params={"url": worker_url}, timeout=120
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
            import os

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
        finally:
            _terminate(worker_proc)
    finally:
        _terminate(router_proc)
