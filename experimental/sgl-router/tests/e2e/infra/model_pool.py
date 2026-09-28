"""Minimal SGLang worker spawner for sgl-router e2e tests.

Adapted from SMG's e2e_test/infra/model_pool.py — the 1200-line original
manages a pool of long-lived workers across many tests; here we only
need a thin wrapper around ``sglang.launch_server`` that:

  - binds the worker to GPU(s) inside this process's own
    ``CUDA_VISIBLE_DEVICES`` allotment,
  - spawns ``python3 -m sglang.launch_server`` with the right args,
  - waits for ``/health`` to come up,
  - optionally injects ``--kv-events-config`` so the worker exposes
    the ``kv_events`` block on ``/server_info``.

A test owns a ``ModelInstance`` for its duration; teardown shuts the
worker down. No cross-test pooling — the acceptance tests are slow
enough already (model load dominates) that pooling complexity wasn't
worth porting.

It also owns the logical-to-physical device mapping
(:func:`visible_devices`, :func:`resolve_device_ids`), which ``conftest.py``
uses to size the GPU allocator and to pin the session-scoped server.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .model_specs import get_model_spec

logger = logging.getLogger(__name__)


def _wait_for_process_group_exit(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def _is_selectable_device(token: str) -> bool:
    """Whether CUDA would accept *token* as a device to expose."""
    if token.lstrip("-").isdigit():
        return not token.startswith("-")
    return token.startswith(("GPU-", "MIG-"))


def visible_devices() -> list[str] | None:
    """The devices this process owns, as ``CUDA_VISIBLE_DEVICES`` spells them.

    ``None`` means the variable is unset, so CUDA exposes every GPU and this
    harness falls back to treating the whole box as available. Entries are kept
    verbatim (indices, ``GPU-<uuid>``, ``MIG-<uuid>``) because
    :func:`resolve_device_ids` needs only their position.

    Parsing follows CUDA's own rule: enumeration stops at the first entry that
    is invalid or already seen, so ``0,-1,1`` exposes one device, ``0,0``
    exposes one, and ``-1`` (the idiom for hiding every GPU) exposes none.
    Counting raw tokens instead would publish a logical index that resolves
    onto a card another worker already holds, or onto no card at all.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    devices: list[str] = []
    for token in (token.strip() for token in raw.split(",")):
        if not _is_selectable_device(token) or token in devices:
            break
        devices.append(token)
    if len(devices) != len([token for token in raw.split(",") if token.strip()]):
        logger.warning(
            "CUDA_VISIBLE_DEVICES=%r exposes only %d device(s): CUDA stops "
            "enumerating at the first invalid or repeated entry",
            raw,
            len(devices),
        )
    return devices


def resolve_device_ids(gpu_ids: list[int]) -> list[str]:
    """Map logical GPU indices onto the devices this process actually owns.

    CI gives each runner a slice of a shared 8-GPU box through
    ``CUDA_VISIBLE_DEVICES`` (e.g. ``2,3``), and a child's
    ``CUDA_VISIBLE_DEVICES`` is absolute rather than relative to the parent's:
    writing a bare ``1`` there puts the worker on physical GPU 1 — a card
    another job owns, whose memory this job cannot see, cannot clean up before
    the run, and will lose a race against. Index into the inherited list
    instead. An index past the end of the allotment raises here rather than
    silently escaping it.
    """
    visible = visible_devices()
    if visible is None:
        return [str(gpu) for gpu in gpu_ids]
    resolved: list[str] = []
    for gpu in gpu_ids:
        if not 0 <= gpu < len(visible):
            raise ValueError(
                f"logical GPU {gpu} is outside this process's allotment of "
                f"{len(visible)}: CUDA_VISIBLE_DEVICES={','.join(visible)}"
            )
        resolved.append(visible[gpu])
    return resolved


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
    # Logical, as passed to spawn_worker; the child's CUDA_VISIBLE_DEVICES
    # holds the resolved devices. Callers hand these back to the allocator.
    gpu_ids: list[int] = field(default_factory=list)
    kv_events_endpoint: str | None = None
    log_path: Path | None = None
    _shutdown_started: bool = field(default=False, init=False, repr=False)

    def log_tail(self, lines: int = 200) -> str:
        """Last `lines` of the worker's log, for failure diagnostics."""
        if self.log_path is None:
            return "(no log file)"
        try:
            return "\n".join(
                self.log_path.read_text(errors="replace").splitlines()[-lines:]
            )
        except OSError:
            return f"({self.log_path} unreadable)"

    def __enter__(self) -> "ModelInstance":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        if self.process is None or self._shutdown_started:
            return
        self._shutdown_started = True
        pgid = self.process.pid

        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return

        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass

        if _wait_for_process_group_exit(pgid, timeout=30):
            return

        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return
        self.process.wait()
        if not _wait_for_process_group_exit(pgid, timeout=5):
            raise RuntimeError(f"worker process group {pgid} did not exit")


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
        gpu_ids: Logical GPU indices, resolved against this process's own
            ``CUDA_VISIBLE_DEVICES`` allotment by :func:`resolve_device_ids`.
        port: HTTP port; auto-assigned if None.
        enable_kv_events: If True, inject ``--kv-events-config`` with a
            ZMQ publisher so the router's introspection picks up the
            kv_events block from ``/server_info``.
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
    devices = resolve_device_ids(gpu_ids)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    logger.info(
        "spawning sglang worker: model=%s port=%d gpus=%s devices=%s disagg=%s",
        model_id,
        port,
        gpu_ids,
        env["CUDA_VISIBLE_DEVICES"],
        disagg_mode,
    )

    # Stream the worker's output to a file rather than an unread
    # subprocess.PIPE. Nothing in this process drains that pipe, so once its
    # ~64 KB OS buffer fills the engine blocks on write and stops serving —
    # requests then hang until the client timeout with no log to explain it.
    # Startup alone (weight load, memory pool, CUDA-graph capture) can
    # approach that, and a long test's per-request logging goes past it.
    # `conftest.py`'s session-scoped fixture already learned this; this is the
    # same fix for the per-test workers.
    log_path = Path(tempfile.gettempdir()) / f"sglang-worker-{port}.log"
    log_handle = open(log_path, "w", buffering=1)  # line-buffered
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        # The child keeps its own descriptor, so the parent's copy is done
        # with. Holding it would leak one fd per worker for the session, and
        # leave the file open with nothing writing through it. Failures read
        # the log back from `log_path`, not from this handle.
        log_handle.close()

    inst = ModelInstance(
        url=base_url,
        port=port,
        process=proc,
        model_id=model_id,
        gpu_ids=list(gpu_ids),
        kv_events_endpoint=kv_events_endpoint,
        log_path=log_path,
    )

    # Wait for /health. Cold-start on H200 with weights uncached can take
    # ~5 minutes; CI configurations should pre-warm.
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"sglang worker exited during startup with code {proc.returncode}; "
                f"cmd: {' '.join(cmd)}\noutput:\n{inst.log_tail()}",
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
        f"sglang worker did not become healthy at {base_url} within {timeout}s; "
        f"last log lines:\n{inst.log_tail()}",
    )
