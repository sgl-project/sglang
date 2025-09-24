import logging
import os
import socket
import subprocess
import time
from types import SimpleNamespace
from typing import Optional

import pytest
import requests

from sglang.test.run_eval import run_eval

logger = logging.getLogger(__name__)


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_health(url: str, timeout: float = 180.0) -> None:
    start = time.perf_counter()
    with requests.Session() as session:
        while time.perf_counter() - start < timeout:
            try:
                r = session.get(f"{url}/health", timeout=5)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
    raise TimeoutError(f"Service at {url} failed to become healthy in time")


def _detect_ib_device() -> Optional[str]:
    """Return first active IB device name (e.g., mlx5_0) or None if unavailable."""
    # Fast check that ibv_devinfo exists
    try:
        subprocess.run(
            ["ibv_devinfo", "-l"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    for i in range(12):
        dev = f"mlx5_{i}"
        try:
            res = subprocess.run(
                ["ibv_devinfo", dev],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0 and ("state:" in res.stdout):
                for line in res.stdout.splitlines():
                    if "state:" in line and "PORT_ACTIVE" in line:
                        return dev
        except Exception:
            pass
    return None


def _popen_launch_prefill_worker(
    model: str,
    bootstrap_port: int,
    ib_device: Optional[str] = None,
    base_gpu_id: int = 0,
) -> SimpleNamespace:
    port = _find_available_port()
    url = f"http://127.0.0.1:{port}"
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--disaggregation-mode",
        "prefill",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--disaggregation-bootstrap-port",
        str(bootstrap_port),
        "--base-gpu-id",
        str(base_gpu_id),
    ]
    if ib_device:
        cmd += ["--disaggregation-ib-device", ib_device]
    proc = subprocess.Popen(cmd)
    _wait_health(url, timeout=300.0)
    return SimpleNamespace(proc=proc, url=url, bootstrap_port=bootstrap_port)


def _popen_launch_decode_worker(
    model: str, ib_device: Optional[str] = None, base_gpu_id: int = 0
) -> SimpleNamespace:
    port = _find_available_port()
    url = f"http://127.0.0.1:{port}"
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--disaggregation-mode",
        "decode",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--base-gpu-id",
        str(base_gpu_id),
    ]
    if ib_device:
        cmd += ["--disaggregation-ib-device", ib_device]
    proc = subprocess.Popen(cmd)
    _wait_health(url, timeout=300.0)
    return SimpleNamespace(proc=proc, url=url)


def _terminate(proc: subprocess.Popen, timeout: float = 120) -> None:
    if proc is None:
        return
    proc.terminate()
    start = time.perf_counter()
    while proc.poll() is None:
        if time.perf_counter() - start > timeout:
            proc.kill()
            break
        time.sleep(1)


@pytest.fixture(scope="module")
def pd_cluster(e2e_model: str):
    """Start 2 prefill + 2 decode workers and one PD router, once per module."""
    # Environment capability checks: require sgl_kernel and GPU backend
    try:
        import sgl_kernel  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.fail(f"PD e2e requires sgl_kernel but it is not available: {e}")

    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.fail(
            f"PD e2e requires torch but it is not available or misconfigured: {e}"
        )

    if not torch.cuda.is_available():  # pragma: no cover - environment dependent
        pytest.fail("PD e2e requires CUDA backend, but CUDA is not available")

    workers: list[SimpleNamespace] = []
    router_proc = None
    try:
        ib_device = _detect_ib_device()

        # Launch 4 workers across 4 GPUs: prefill on 0,1 and decode on 2,3
        pf1 = _popen_launch_prefill_worker(
            e2e_model,
            bootstrap_port=_find_available_port(),
            ib_device=ib_device,
            base_gpu_id=0,
        )
        pf2 = _popen_launch_prefill_worker(
            e2e_model,
            bootstrap_port=_find_available_port(),
            ib_device=ib_device,
            base_gpu_id=1,
        )
        dc1 = _popen_launch_decode_worker(e2e_model, ib_device=ib_device, base_gpu_id=2)
        dc2 = _popen_launch_decode_worker(e2e_model, ib_device=ib_device, base_gpu_id=3)
        prefills = [pf1, pf2]
        decodes = [dc1, dc2]
        workers.extend(prefills + decodes)

        # PD router with two prefill and two decode endpoints
        rport = _find_available_port()
        router_url = f"http://127.0.0.1:{rport}"
        pport = _find_available_port()

        prefill = [(pf.url, pf.bootstrap_port) for pf in prefills]
        decode = [dc.url for dc in decodes]

        cmd = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--host",
            "127.0.0.1",
            "--port",
            str(rport),
            "--policy",
            "round_robin",
            "--pd-disaggregation",
        ]
        for url, bport in prefill:
            cmd += ["--prefill", url, str(bport)]
        for url in decode:
            cmd += ["--decode", url]
        cmd += [
            "--prometheus-port",
            str(pport),
            "--prometheus-host",
            "127.0.0.1",
        ]

        router_proc = subprocess.Popen(cmd)
        _wait_health(router_url, timeout=180.0)

        yield SimpleNamespace(
            router_url=router_url, workers=workers, router_proc=router_proc
        )
    finally:
        if router_proc is not None:
            _terminate(router_proc)
        for w in workers:
            _terminate(w.proc)


@pytest.mark.e2e
def test_pd_mmlu(e2e_model: str, pd_cluster):
    """
    Launch 4 workers, start a PD router (2 prefill + 2 decode), then run MMLU.
    """
    args = SimpleNamespace(
        base_url=pd_cluster.router_url,
        model=e2e_model,
        eval_name="mmlu",
        num_examples=64,
        num_threads=32,
        temperature=0.1,
    )
    metrics = run_eval(args)
    assert metrics["score"] >= 0.65


@pytest.mark.e2e
def test_pd_genai_bench(e2e_model: str, pd_cluster, genai_bench_runner):
    """
    Launch 4 workers, start a PD router (2 prefill + 2 decode), then run a
    short genai-bench benchmark and validate aggregate metrics.
    """
    # Run genai-bench against the shared router
    policy_label = "benchmark_round_robin_pd"
    genai_bench_runner(
        router_url=pd_cluster.router_url,
        model_path=e2e_model,
        experiment_folder=policy_label,
        thresholds={
            "ttft_mean_max": 12,
            "e2e_latency_mean_max": 15,
            "input_throughput_mean_min": 400,
            "output_throughput_mean_min": 20,
            "gpu_util_p50_min": 99,
        },
        kill_procs=pd_cluster.workers,
    )


@pytest.mark.e2e
def test_pd_logprobs_merge(e2e_model: str, pd_cluster):
    """
    Verify PD router returns full input/output logprobs and input_top_logprobs merged
    across prefill and decode for a generate request.
    """
    url = f"{pd_cluster.router_url}/generate"

    payload = {
        "input_ids": [[128000, 3923, 892, 374, 433, 30, 318, 281, 220, 50256]],
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 4},
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": 2,
    }

    r = requests.post(url, json=payload, timeout=60)
    assert r.status_code == 200, f"unexpected status {r.status_code}: {r.text}"
    data = r.json()
    assert "meta_info" in data
    meta = data["meta_info"]

    assert "input_token_logprobs" in meta
    assert "output_token_logprobs" in meta

    input_len = len(payload["input_ids"][0])
    assert isinstance(meta["input_token_logprobs"], list)
    assert (
        len(meta["input_token_logprobs"])
        >= max(1, input_len - payload["logprob_start_len"]) - 1
    )

    assert isinstance(meta["output_token_logprobs"], list)
    assert (
        len(meta["output_token_logprobs"])
        == payload["sampling_params"]["max_new_tokens"]
    )

    if payload["top_logprobs_num"] > 0:
        assert "input_top_logprobs" in meta
        assert isinstance(meta["input_top_logprobs"], list)
        assert len(meta["input_top_logprobs"]) >= len(meta["input_token_logprobs"])

    payload_stream = dict(payload)
    payload_stream["stream"] = True
    with requests.post(url, json=payload_stream, stream=True, timeout=60) as rs:
        assert rs.status_code == 200
        line_count = 0
        first_event = None
        for line in rs.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                content = line[len("data: ") :]
                if content == "[DONE]":
                    break
                first_event = content
                line_count += 1
                break
        assert line_count >= 1 and first_event is not None
        try:
            ev = rs.json()
        except Exception:
            import json as _json

            ev = _json.loads(first_event)
        assert "meta_info" in ev
        assert "input_token_logprobs" in ev["meta_info"]
        assert "completion_tokens" in ev["meta_info"]
