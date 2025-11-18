import logging
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
            "--log-level",
            "warn",
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
            "ttft_mean_max": 13,
            "e2e_latency_mean_max": 16,
            "input_throughput_mean_min": 350,
            "output_throughput_mean_min": 18,
            "gpu_util_p50_min": 99,
        },
        kill_procs=pd_cluster.workers,
    )
