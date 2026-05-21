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
# This file is at `experimental/sgl-router/tests/e2e/infra/gateway.py`,
# so four `.parent` hops to reach the sgl-router workspace root
# (infra → e2e → tests → sgl-router). Cargo lands the binary at
# `experimental/sgl-router/target/release/sgl-router`. A previous
# version used three hops and pointed at `tests/target/`, which
# would have broken any test that actually launches the router via
# this helper.
DEFAULT_BINARY = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "target"
    / "release"
    / "sgl-router"
)


def _get_open_port() -> int:
    """Reserve an ephemeral TCP port in [20000, 55535].

    The router itself doesn't have the ``port + 10000`` gRPC-derivation
    constraint that SGLang's launch_server does, but we cap the range
    anyway so the e2e helpers behave consistently across components.
    """
    for _ in range(50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if 20000 <= port <= 55535:
            return port
    raise RuntimeError(
        "could not allocate an ephemeral port in [20000, 55535] after 50 tries"
    )


def _resolve_tokenizer_path(tokenizer_path: str) -> str:
    """Resolve a HuggingFace repo ID to a local ``tokenizer.json`` path.

    sgl-router's tokenizer loader treats the input as a filesystem path and
    inspects its extension; a bare HF id like ``Qwen/Qwen3-0.6B`` looks
    like a file with extension ``.6B`` and is rejected. When the HF Hub
    cache already has the tokenizer, point the loader at the on-disk
    ``tokenizer.json`` directly. Pass paths/URLs through unchanged.
    """
    p = Path(tokenizer_path)
    if p.exists():
        return str(p)
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore[import]

        cached = try_to_load_from_cache(tokenizer_path, "tokenizer.json")
        if cached and Path(cached).is_file():
            return str(cached)
    except Exception:  # noqa: BLE001
        pass
    return tokenizer_path


@dataclass
class WorkerInfo:
    """Worker visible to the gateway via ``/v1/models``-style introspection.

    Mirrors SMG's WorkerInfo shape so test code reads the same. sgl-router
    does not currently surface a `/v1/workers` admin API — this is a
    placeholder for a future admin surface; current tests scrape
    `/metrics` for per-worker observability instead.
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
        proxy_request_timeout_secs: int | None = None,
        stale_request_timeout_secs: int | None = None,
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

        # Test-side overrides for the router's tunables. Both default to
        # `None`, in which case the router uses its production defaults
        # (60 s proxy timeout, 300 s stale-request timeout). Tests set
        # these short so per-request failures and stale-request expiry
        # surface within the test's wall-time budget.
        self.proxy_request_timeout_secs = proxy_request_timeout_secs
        self.stale_request_timeout_secs = stale_request_timeout_secs

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
                         instances. The router uses ``static_urls`` discovery;
                         each worker's mode (plain) and any disaggregation
                         metadata are learned from ``/server_info``.
            policy: Policy kind — ``round_robin``, ``random``, ``power_of_two``,
                    or ``cache_aware_zmq``.
            timeout: How long to wait for ``/readyz`` before giving up.
        """
        self._launch(
            self._build_config(
                model_id=model_id,
                tokenizer_path=tokenizer_path,
                urls=list(worker_urls),
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

        All prefill + decode URLs go into one ``static_urls`` list. The
        router seeds each worker as ``WorkerMode::Plain`` and the
        manager's ``/server_info`` introspect step overrides mode +
        ``bootstrap_port`` from the worker's self-disclosure. Workers
        must have been launched with ``--disaggregation-mode`` and
        ``--disaggregation-bootstrap-port`` for the PD role to be
        picked up (see ``model_pool.spawn_worker``); modern SGLang is
        assumed.
        """
        self._launch(
            self._build_config(
                model_id=model_id,
                tokenizer_path=tokenizer_path,
                urls=list(prefill_urls) + list(decode_urls),
                policy=policy,
                extra_models=[],
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

    def _build_config(
        self,
        *,
        model_id: str,
        tokenizer_path: str,
        urls: list[str],
        policy: str,
        extra_models: list[dict],
    ) -> str:
        resolved_tokenizer = _resolve_tokenizer_path(tokenizer_path)

        extra_model_toml = ""
        for em in extra_models:
            extra_model_toml += (
                f'\n[[models]]\nid = "{em["id"]}"\n'
                f'tokenizer_path = "{_resolve_tokenizer_path(em["tokenizer_path"])}"\n'
                f'policy = "{em.get("policy", policy)}"\n'
            )

        # Optional tunables — only emit the [proxy] and [active_load]
        # sections if a test has overridden them, so production defaults
        # apply otherwise.
        proxy_section = ""
        if self.proxy_request_timeout_secs is not None:
            proxy_section = (
                f"\n[proxy]\nrequest_timeout_secs = {self.proxy_request_timeout_secs}\n"
            )
        active_load_section = ""
        if self.stale_request_timeout_secs is not None:
            active_load_section = (
                f"\n[active_load]\nstale_request_timeout_secs = "
                f"{self.stale_request_timeout_secs}\n"
            )

        urls_toml = ", ".join(f'"{u}"' for u in urls)

        return f"""\
[server]
host = "{self.host}"
port = {self.port}

[[models]]
id = "{model_id}"
tokenizer_path = "{resolved_tokenizer}"
policy = "{policy}"
{extra_model_toml}

[discovery]
backend = "static_urls"

[discovery.static_urls]
urls = [{urls_toml}]
{proxy_section}{active_load_section}"""

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
