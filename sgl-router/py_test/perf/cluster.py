from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class WorkerProcess:
    role: str  # "prefill" or "decode"
    host: str
    port: int
    bootstrap_port: Optional[int]
    gpu_id: int
    process: subprocess.Popen

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class PDCluster:
    model_path: str
    ib_device: Optional[str]
    prefills: list[WorkerProcess]
    decodes: list[WorkerProcess]

    def all_workers(self) -> Iterable[WorkerProcess]:
        yield from self.prefills
        yield from self.decodes

    def prefill_endpoints(self) -> list[tuple[str, int]]:
        return [(worker.url, worker.bootstrap_port or 0) for worker in self.prefills]

    def decode_endpoints(self) -> list[str]:
        return [worker.url for worker in self.decodes]

    def stop(self) -> None:
        for worker in self.all_workers():
            _terminate(worker.process)


def ensure_environment_ready(model_path: str, require_ib: bool = True) -> None:
    if require_ib and detect_ib_device() is None:
        raise RuntimeError(
            "InfiniBand device not detected via ibv_devinfo; cannot run PD perf benchmark"
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for PD perf tests") from exc

    if not torch.cuda.is_available():  # pragma: no cover - environment dependent
        raise RuntimeError("CUDA backend unavailable; PD perf tests require GPUs")

    gpu_count = torch.cuda.device_count()
    if gpu_count < 8:
        raise RuntimeError(
            f"PD perf tests require at least 8 GPUs; detected {gpu_count}"
        )

    if not Path(model_path).exists():
        raise RuntimeError(f"Model path not found: {model_path}")


def detect_ib_device() -> Optional[str]:
    """Return the first active InfiniBand device (e.g., mlx5_0) if present."""
    if shutil.which("ibv_devinfo") is None:
        return None

    for idx in range(12):
        dev = f"mlx5_{idx}"
        try:
            res = subprocess.run(
                ["ibv_devinfo", dev],
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
        if res.returncode != 0:
            continue
        if "PORT_ACTIVE" in res.stdout:
            logger.info("Detected active IB device: %s", dev)
            return dev
    return None


def launch_disaggregated_cluster(
    *,
    model_path: str,
    ib_device: Optional[str],
    health_timeout: float = 300.0,
) -> PDCluster:
    prefills: list[WorkerProcess] = []
    decodes: list[WorkerProcess] = []
    launched: list[WorkerProcess] = []
    try:
        for idx, gpu_id in enumerate(range(4)):
            host = f"127.0.0.{idx + 1}"
            port = 30001 + idx
            bootstrap_port = 9001 + idx
            proc = _launch_worker_process(
                role="prefill",
                gpu_id=gpu_id,
                host=host,
                port=port,
                model_path=model_path,
                ib_device=ib_device,
                bootstrap_port=bootstrap_port,
            )
            _wait_for_health(host, port, timeout=health_timeout)
            worker = WorkerProcess(
                role="prefill",
                host=host,
                port=port,
                bootstrap_port=bootstrap_port,
                gpu_id=gpu_id,
                process=proc,
            )
            prefills.append(worker)
            launched.append(worker)

        for offset, gpu_id in enumerate(range(4, 8)):
            host = f"127.0.0.{offset + 5}"
            port = 30001 + gpu_id
            proc = _launch_worker_process(
                role="decode",
                gpu_id=gpu_id,
                host=host,
                port=port,
                model_path=model_path,
                ib_device=ib_device,
            )
            _wait_for_health(host, port, timeout=health_timeout)
            worker = WorkerProcess(
                role="decode",
                host=host,
                port=port,
                bootstrap_port=None,
                gpu_id=gpu_id,
                process=proc,
            )
            decodes.append(worker)
            launched.append(worker)

        return PDCluster(
            model_path=model_path,
            ib_device=ib_device,
            prefills=prefills,
            decodes=decodes,
        )
    except Exception:
        for worker in launched:
            _terminate(worker.process)
        raise


def _launch_worker_process(
    *,
    role: str,
    gpu_id: int,
    host: str,
    port: int,
    model_path: str,
    ib_device: Optional[str],
    bootstrap_port: Optional[int] = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--disaggregation-mode",
        role,
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]

    if role == "prefill" and bootstrap_port is not None:
        cmd += ["--disaggregation-bootstrap-port", str(bootstrap_port)]
    if role == "decode":
        cmd += ["--base-gpu-id", "0"]
    if ib_device:
        cmd += ["--disaggregation-ib-device", ib_device]

    logger.info("Launching %s worker on GPU %d at %s:%d", role, gpu_id, host, port)
    proc = subprocess.Popen(cmd, env=env)
    return proc


def _wait_for_health(host: str, port: int, timeout: float) -> None:
    url = f"http://{host}:{port}/health"
    start = time.time()
    session = requests.Session()
    try:
        while time.time() - start < timeout:
            try:
                resp = session.get(url, timeout=5)
                if resp.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(5)
    finally:
        session.close()
    raise TimeoutError(f"Worker at {url} failed to report healthy within {timeout}s")


def _terminate(proc: subprocess.Popen, timeout: float = 60.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    start = time.time()
    while proc.poll() is None and (time.time() - start) < timeout:
        time.sleep(1)
    if proc.poll() is None:
        proc.kill()
