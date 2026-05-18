"""Minimal sgl-router Gateway class — adapted from SMG's e2e_test/infra/gateway.py.

Differences from SMG:
  - SMG drives a Python launcher (`python3 -m sglang_router.launch_router`)
    with worker URLs on the CLI.
  - sgl-router uses a Rust binary (`experimental/sgl-router/target/release/sgl-router`)
    with a TOML config file. Worker discovery is config-file-based; this
    Gateway writes a TOML to a tempfile and execs the binary with
    `--config <tempfile>`.

Supported lifecycles:
  - Regular mode: one model, N worker URLs, single policy.
  - PD mode: one model, prefill_workers + decode_workers (lists of URLs),
    discovery emits separate `WorkerMode::Prefill` / `WorkerMode::Decode`
    entries. The router resolves PD pool isolation at request time.

Use as a context manager:

    with Gateway() as gw:
        gw.start_regular(model_path="...", worker_urls=[...])
        resp = httpx.post(f"{gw.base_url}/v1/chat/completions", json=...)

or pytest fixture style (see e2e_test/conftest.py).
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Repo-relative path to the release binary. Set ``SGL_ROUTER_BINARY`` to
# override (e.g. a debug build, or a non-default ``CARGO_TARGET_DIR``).
DEFAULT_BINARY = (
    Path(__file__).resolve().parent.parent.parent / "target" / "release" / "sgl-router"
)


def _get_open_port() -> int:
    """Reserve an ephemeral TCP port. Returns the port and closes the socket
    so the caller can immediately bind it. Small race window — acceptable
    for test infra; production code would pass `port=0` to the binder
    directly.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class WorkerInfo:
    """Worker visible to the gateway via ``/v1/models``-style introspection.

    Mirrors SMG's WorkerInfo shape so test code reads the same. sgl-router
    does not currently surface a `/v1/workers` admin API — this is a
    placeholder for the M5 surface; M4 tests scrape `/metrics` for
    per-worker observability instead.
    """

    id: str
    url: str
    model: str | None = None
    status: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


class Gateway:
    """Lifecycle-managed sgl-router instance for e2e tests.

    Not thread-safe; assume one Gateway per test (or per fixture scope).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = None,
        binary: Path | None = None,
    ):
        self.host = host
        self.port = port or _get_open_port()
        self.base_url = f"http://{self.host}:{self.port}"
        # Resolve binary from env override, explicit arg, or repo default.
        env_binary = os.environ.get("SGL_ROUTER_BINARY")
        if binary is not None:
            self.binary = Path(binary)
        elif env_binary:
            self.binary = Path(env_binary)
        else:
            self.binary = DEFAULT_BINARY

        self.process: subprocess.Popen | None = None
        self._config_path: Path | None = None
        self._started: bool = False
        # Track child workers we spawned so __exit__ can tear them down.
        self._owned_workers: list[subprocess.Popen] = []

    # ----- context manager -------------------------------------------------

    def __enter__(self) -> "Gateway":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()

    # ----- start ----------------------------------------------------------

    def start_regular(
        self,
        *,
        model_id: str,
        tokenizer_path: str,
        worker_urls: list[str],
        policy: str = "round_robin",
        extra_models: list[dict] | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Start the router in regular (non-PD) mode.

        Args:
            model_id: The model identifier the router will dispatch under.
            tokenizer_path: Path or HF ID for the tokenizer the router uses
                            for cache-aware tokenization.
            worker_urls: URLs of already-running ``sglang.launch_server``
                         instances. The router uses static-file discovery.
            policy: Policy kind — ``round_robin``, ``random``, ``power_of_two``,
                    or ``cache_aware_zmq``.
            timeout: How long to wait for ``/readyz`` before giving up.
        """
        self._launch(
            self._build_regular_config(
                model_id=model_id,
                tokenizer_path=tokenizer_path,
                worker_urls=worker_urls,
                policy=policy,
                extra_models=extra_models or [],
            ),
            timeout=timeout,
        )

    def start_pd(
        self,
        *,
        model_id: str,
        tokenizer_path: str,
        prefill_urls: list[str],
        decode_urls: list[str],
        policy: str = "round_robin",
        timeout: float = 60.0,
    ) -> None:
        """Start the router in PD-disaggregated mode.

        Discovery emits ``WorkerMode::Prefill`` for ``prefill_urls`` and
        ``WorkerMode::Decode`` for ``decode_urls``; the router's
        `PdPoolResolver` filters per-pool at request time.
        """
        self._launch(
            self._build_pd_config(
                model_id=model_id,
                tokenizer_path=tokenizer_path,
                prefill_urls=prefill_urls,
                decode_urls=decode_urls,
                policy=policy,
            ),
            timeout=timeout,
        )

    # ----- shutdown --------------------------------------------------------

    def shutdown(self) -> None:
        """SIGTERM the router; SIGKILL after 30s. Idempotent."""
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.send_signal(signal.SIGTERM)
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            except ProcessLookupError:
                pass
        self.process = None
        if self._config_path and self._config_path.exists():
            self._config_path.unlink(missing_ok=True)
        self._config_path = None
        self._started = False
        # Tear down any owned upstream workers.
        for w in self._owned_workers:
            if w.poll() is None:
                try:
                    w.send_signal(signal.SIGTERM)
                    try:
                        w.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        w.kill()
                        w.wait()
                except ProcessLookupError:
                    pass
        self._owned_workers.clear()

    # ----- HTTP introspection helpers -------------------------------------

    def healthy(self, timeout: float = 5.0) -> bool:
        try:
            resp = httpx.get(f"{self.base_url}/healthz", timeout=timeout)
            return resp.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    def ready(self, timeout: float = 5.0) -> bool:
        try:
            resp = httpx.get(f"{self.base_url}/readyz", timeout=timeout)
            return resp.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    def metrics_text(self, timeout: float = 5.0) -> str | None:
        try:
            resp = httpx.get(f"{self.base_url}/metrics", timeout=timeout)
            if resp.status_code == 200:
                return resp.text
            return None
        except (httpx.RequestError, httpx.TimeoutException):
            return None

    # ----- internals ------------------------------------------------------

    def _build_regular_config(
        self,
        *,
        model_id: str,
        tokenizer_path: str,
        worker_urls: list[str],
        policy: str,
        extra_models: list[dict],
    ) -> str:
        # Static-file discovery: a separate worker-list TOML, polled by
        # the router. We write that file as well and reference it from the
        # main config.
        workers_toml = "\n".join(
            f'[[workers]]\nid = "w{i}"\nurl = "{u}"\nmode = "Plain"\nmodel_ids = ["{model_id}"]'
            for i, u in enumerate(worker_urls)
        )
        return self._compose_main_config(
            model_id=model_id,
            tokenizer_path=tokenizer_path,
            policy=policy,
            workers_toml=workers_toml,
            extra_models=extra_models,
        )

    def _build_pd_config(
        self,
        *,
        model_id: str,
        tokenizer_path: str,
        prefill_urls: list[str],
        decode_urls: list[str],
        policy: str,
    ) -> str:
        entries = []
        for i, u in enumerate(prefill_urls):
            entries.append(
                f'[[workers]]\nid = "p{i}"\nurl = "{u}"\nmode = "Prefill"\nmodel_ids = ["{model_id}"]'
            )
        for i, u in enumerate(decode_urls):
            entries.append(
                f'[[workers]]\nid = "d{i}"\nurl = "{u}"\nmode = "Decode"\nmodel_ids = ["{model_id}"]'
            )
        workers_toml = "\n".join(entries)
        return self._compose_main_config(
            model_id=model_id,
            tokenizer_path=tokenizer_path,
            policy=policy,
            workers_toml=workers_toml,
            extra_models=[],
        )

    def _compose_main_config(
        self,
        *,
        model_id: str,
        tokenizer_path: str,
        policy: str,
        workers_toml: str,
        extra_models: list[dict],
    ) -> str:
        # Static-file discovery requires a separate worker-list path that
        # the router polls. We point at a sibling file written next to the
        # main config.
        workers_path = self._workers_path()
        # Write the workers list to that path.
        workers_path.write_text(workers_toml + "\n", encoding="utf-8")

        extra_model_toml = ""
        for em in extra_models:
            extra_model_toml += (
                f'\n[[models]]\nid = "{em["id"]}"\n'
                f'tokenizer_path = "{em["tokenizer_path"]}"\n'
                f'policy = "{em.get("policy", policy)}"\n'
            )

        return f"""\
[server]
host = "{self.host}"
port = {self.port}

[[models]]
id = "{model_id}"
tokenizer_path = "{tokenizer_path}"
policy = "{policy}"
{extra_model_toml}

[discovery]
backend = "static_file"

[discovery.static_file]
path = "{workers_path}"
poll_interval_ms = 200
"""

    def _workers_path(self) -> Path:
        # Persist alongside the main config so cleanup is local.
        if self._config_path is None:
            # _launch hasn't allocated the main config yet — use the temp
            # directory it will write into.
            return Path(tempfile.gettempdir()) / f"sgl-router-workers-{self.port}.toml"
        return self._config_path.with_suffix(".workers.toml")

    def _launch(self, config_text: str, *, timeout: float) -> None:
        if not self.binary.exists():
            raise RuntimeError(
                f"sgl-router binary not found at {self.binary}. "
                "Build it first: `cd experimental/sgl-router && cargo build --release` "
                "or set SGL_ROUTER_BINARY to the binary path."
            )
        # Write the main config.
        fd, path = tempfile.mkstemp(suffix=".toml", prefix="sgl-router-")
        os.close(fd)
        self._config_path = Path(path)
        self._config_path.write_text(config_text, encoding="utf-8")
        logger.info("sgl-router config: %s", self._config_path)
        logger.debug("sgl-router config text:\n%s", config_text)

        self.process = subprocess.Popen(
            [str(self.binary), "--config", str(self._config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        try:
            self._wait_ready(timeout=timeout)
        except Exception:
            self.shutdown()
            raise
        self._started = True

    def _wait_ready(self, *, timeout: float) -> None:
        deadline = time.time() + timeout
        last_exc: Exception | None = None
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                # Process exited early — surface stdout/stderr.
                out = b""
                try:
                    if self.process.stdout is not None:
                        out = self.process.stdout.read() or b""
                except Exception:  # noqa: BLE001
                    pass
                raise RuntimeError(
                    f"sgl-router exited during startup with code "
                    f"{self.process.returncode}. output:\n{out.decode(errors='replace')}",
                )
            try:
                resp = httpx.get(f"{self.base_url}/readyz", timeout=2.0)
                if resp.status_code == 200:
                    return
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                last_exc = exc
            time.sleep(0.5)
        raise TimeoutError(
            f"sgl-router did not become ready at {self.base_url} within {timeout}s "
            f"(last error: {last_exc})"
        )
