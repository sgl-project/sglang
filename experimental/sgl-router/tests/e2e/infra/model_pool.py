"""Minimal SGLang worker spawner for sgl-router e2e tests.

Adapted from SMG's e2e_test/infra/model_pool.py — the 1200-line original
manages a pool of long-lived workers across many tests; here we only
need a thin wrapper around ``sglang.launch_server`` that:

  - allocates GPU(s) for the worker (via ``CUDA_VISIBLE_DEVICES``),
  - spawns ``python3 -m sglang.launch_server`` with the right args,
  - waits for ``/health`` to come up,
  - optionally injects ``--kv-events-config`` so the worker exposes
    the ``kv_events`` block on ``/server_info``.

A test owns a ``ModelInstance`` for its duration; teardown shuts the
worker down. No cross-test pooling — the acceptance tests are slow
enough already (model load dominates) that pooling complexity wasn't
worth porting.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field

import httpx

from .model_specs import get_model_spec

logger = logging.getLogger(__name__)


def _get_open_port() -> int:
    """Allocate an ephemeral TCP port in the range [20000, 55535].

    SGLang derives its internal gRPC port as ``http_port + 10000``; if the
    kernel hands us an ephemeral port above 55535, that derivation overflows
    65535 and ``ServerArgs.__post_init__`` rejects it. Retrying a bounded
    number of times keeps us safely below the ceiling without hand-rolling
    a port registry.
    """
    for _ in range(50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if 20000 <= port <= 55535:
            return port
    raise RuntimeError(
        "could not allocate an ephemeral port in [20000, 55535] after 50 tries; "
        "SGLang derives its internal gRPC port as http_port + 10000 and "
        "rejects values above 65535"
    )


@dataclass
class ModelInstance:
    """A running ``sglang.launch_server`` process.

    Use as a context manager:

        with spawn_worker("qwen3-0.6b", gpu_ids=[0]) as inst:
            httpx.post(f"{inst.url}/generate", ...)
    """

    url: str
    port: int
    process: subprocess.Popen
    model_id: str
    gpu_ids: list[int] = field(default_factory=list)
    kv_events_endpoint: str | None = None

    def __enter__(self) -> "ModelInstance":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.send_signal(signal.SIGTERM)
                try:
                    self.process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            except ProcessLookupError:
                pass


def spawn_worker(
    model_id: str,
    *,
    gpu_ids: list[int],
    port: int | None = None,
    enable_kv_events: bool = False,
    kv_events_port: int | None = None,
    disagg_mode: str | None = None,
    bootstrap_port: int | None = None,
    extra_args: list[str] | None = None,
    timeout: float = 600.0,
) -> ModelInstance:
    """Spawn a single ``sglang.launch_server`` and wait for ``/health``.

    Args:
        model_id: Key into :data:`model_specs.MODEL_SPECS`.
        gpu_ids: Concrete GPU indices to bind via ``CUDA_VISIBLE_DEVICES``.
        port: HTTP port; auto-assigned if None.
        enable_kv_events: If True, inject ``--kv-events-config`` with a
            ZMQ publisher so the router's introspection picks up the
            kv_events block from ``/server_info`` (Patch 1).
        kv_events_port: ZMQ publisher port. Auto-assigned if None and
            ``enable_kv_events`` is True.
        disagg_mode: "prefill" or "decode" for PD-disagg launches; passed
            through as ``--disaggregation-mode``.
        bootstrap_port: PD-disagg bootstrap port (prefill side only).
        extra_args: Additional CLI args appended verbatim.
        timeout: Health-check timeout. Cold-start on a fresh GPU can be
            slow; default is 10 minutes.
    """
    spec = get_model_spec(model_id)
    port = port or _get_open_port()
    base_url = f"http://127.0.0.1:{port}"

    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        spec["model"],
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--tp",
        str(spec.get("tp", 1)),
    ]
    cmd.extend(spec.get("worker_args", []) or [])

    kv_events_endpoint: str | None = None
    if enable_kv_events:
        kv_port = kv_events_port or _get_open_port()
        kv_events_endpoint = f"tcp://*:{kv_port}"
        kv_cfg = {
            "publisher": "zmq",
            "endpoint": kv_events_endpoint,
            "topic": "kv",
        }
        cmd.extend(["--kv-events-config", json.dumps(kv_cfg)])

    if disagg_mode is not None:
        cmd.extend(["--disaggregation-mode", disagg_mode])
        if bootstrap_port is not None:
            cmd.extend(["--disaggregation-bootstrap-port", str(bootstrap_port)])

    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    logger.info(
        "spawning sglang worker: model=%s port=%d gpus=%s disagg=%s",
        model_id,
        port,
        gpu_ids,
        disagg_mode,
    )

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    inst = ModelInstance(
        url=base_url,
        port=port,
        process=proc,
        model_id=model_id,
        gpu_ids=list(gpu_ids),
        kv_events_endpoint=kv_events_endpoint,
    )

    # Wait for /health. Cold-start on H200 with weights uncached can take
    # ~5 minutes; CI configurations should pre-warm.
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            out = b""
            try:
                if proc.stdout is not None:
                    out = proc.stdout.read() or b""
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(
                f"sglang worker exited during startup with code {proc.returncode}; "
                f"cmd: {' '.join(cmd)}\noutput:\n{out.decode(errors='replace')}",
            )
        try:
            resp = httpx.get(f"{base_url}/health", timeout=2.0)
            if resp.status_code == 200:
                logger.info("sglang worker ready at %s", base_url)
                return inst
        except (httpx.RequestError, httpx.TimeoutException):
            pass
        time.sleep(2.0)

    inst.shutdown()
    raise TimeoutError(
        f"sglang worker did not become healthy at {base_url} within {timeout}s",
    )
