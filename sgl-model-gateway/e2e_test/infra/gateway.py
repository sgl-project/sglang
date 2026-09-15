"""Gateway class for managing sgl-model-gateway router instances."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from .constants import DEFAULT_HOST, DEFAULT_ROUTER_TIMEOUT, ENV_SHOW_ROUTER_LOGS
from .gpu_allocator import get_open_port
from .process_utils import kill_process_tree, wait_for_health, wait_for_workers_ready

if TYPE_CHECKING:
    from .model_pool import ModelInstance

logger = logging.getLogger(__name__)


@dataclass
class WorkerInfo:
    """Information about a worker connected to the gateway."""

    id: str
    url: str
    model: str | None = None
    status: str = "unknown"
    pending_requests: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class Gateway:
    """Manages a sgl-model-gateway router instance.

    Provides lifecycle management and API access for:
    - Starting/stopping the router
    - Worker management (list, add, remove)
    - Health and metrics endpoints

    Four startup modes:
    1. Regular mode: Start with worker URLs
    2. PD mode: Start with prefill/decode workers
    3. IGW mode: Start empty, add workers via API
    4. Cloud mode: Start with cloud backend (OpenAI, xAI)

    Example (regular mode):
        gateway = Gateway()
        gateway.start(
            worker_urls=["http://127.0.0.1:30000"],
            model_path="/path/to/model",
        )

    Example (PD disaggregation mode):
        gateway = Gateway()
        gateway.start(
            prefill_workers=prefill_instances,
            decode_workers=decode_instances,
        )

    Example (IGW mode):
        gateway = Gateway()
        gateway.start(igw_mode=True)
        gateway.add_worker("http://127.0.0.1:30000")
        gateway.add_worker("http://127.0.0.1:30001")

        # Use gateway
        workers = gateway.list_workers()
        health = gateway.health()

        # Cleanup
        gateway.shutdown()

    Example (cloud mode):
        gateway = Gateway()
        gateway.start(cloud_backend="openai")  # or "xai"
        # Requires OPENAI_API_KEY or XAI_API_KEY env var
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int | None = None,
        prometheus_port: int | None = None,
    ):
        """Initialize gateway configuration.

        Args:
            host: Host to bind the router to.
            port: Port for the router. If None, auto-assigns.
            prometheus_port: Port for prometheus metrics. If None, auto-assigns.
        """
        self.host = host
        self.port = port or get_open_port()
        self.prometheus_port = prometheus_port or get_open_port()
        self.base_url = f"http://{self.host}:{self.port}"
        self.metrics_url = f"http://{self.host}:{self.prometheus_port}"

        self.process: subprocess.Popen | None = None
        self.model_path: str | None = None
        self.policy: str = "round_robin"
        self.pd_mode: bool = False
        self.igw_mode: bool = False
        self.cloud_mode: bool = False
        self.cloud_backend: str | None = None
        self._started: bool = False
        self._env: dict[str, str] | None = None  # Custom env for subprocess

    @property
    def is_running(self) -> bool:
        """Check if the gateway process is running."""
        return self.process is not None and self.process.poll() is None

    def start(
        self,
        *,
        # Regular mode arguments
        worker_urls: list[str] | None = None,
        model_path: str | None = None,
        # PD mode arguments
        prefill_workers: list["ModelInstance"] | None = None,
        decode_workers: list["ModelInstance"] | None = None,
        # IGW mode arguments
        igw_mode: bool = False,
        # Cloud mode arguments
        cloud_backend: str | None = None,
        history_backend: str = "memory",
        # Common arguments
        policy: str = "round_robin",
        timeout: float = DEFAULT_ROUTER_TIMEOUT,
        show_output: bool | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        """Start the gateway.

        Can be started in four modes:
        1. Regular mode: Provide worker_urls and model_path
        2. PD mode: Provide prefill_workers and decode_workers
        3. IGW mode: Set igw_mode=True, add workers later via add_worker()
        4. Cloud mode: Provide cloud_backend ("openai" or "xai")

        Args:
            worker_urls: List of worker URLs for regular mode.
            model_path: Model path for regular mode.
            prefill_workers: List of prefill ModelInstance objects for PD mode.
            decode_workers: List of decode ModelInstance objects for PD mode.
            igw_mode: Start in IGW mode (no workers, add via API).
            cloud_backend: Cloud backend type ("openai" or "xai").
            history_backend: History backend for cloud mode ("memory" or "oracle").
            policy: Routing policy (round_robin, random, etc.)
            timeout: Startup timeout in seconds.
            show_output: Show subprocess output (env var override).
            extra_args: Additional router arguments.

        Raises:
            RuntimeError: If gateway is already started.
            ValueError: If arguments are invalid for the mode.
        """
        if self._started:
            raise RuntimeError("Gateway already started")

        # Determine mode based on arguments
        is_pd_mode = prefill_workers is not None or decode_workers is not None
        is_regular_mode = worker_urls is not None
        is_igw_mode = igw_mode
        is_cloud_mode = cloud_backend is not None

        # Validate mode exclusivity
        modes_specified = sum([is_pd_mode, is_regular_mode, is_igw_mode, is_cloud_mode])
        if modes_specified > 1:
            raise ValueError(
                "Cannot specify multiple modes. Choose one of: "
                "worker_urls (regular), prefill/decode_workers (PD), "
                "igw_mode, or cloud_backend"
            )

        if modes_specified == 0:
            raise ValueError(
                "Must specify one mode: worker_urls (regular), "
                "prefill/decode_workers (PD), igw_mode=True, or cloud_backend"
            )

        if show_output is None:
            show_output = os.environ.get(ENV_SHOW_ROUTER_LOGS, "0") == "1"

        self.policy = policy

        if is_igw_mode:
            # IGW mode: start empty, add workers via API
            self.pd_mode = False
            self.igw_mode = True
            self._launch(
                mode_args=["--enable-igw"],
                timeout=timeout,
                show_output=show_output,
                extra_args=extra_args,
                log_msg="IGW gateway (no workers)",
            )
        elif is_pd_mode:
            # PD mode: prefill/decode disaggregation
            self.pd_mode = True
            self.igw_mode = False
            prefills = prefill_workers or []
            decodes = decode_workers or []

            mode_args = ["--pd-disaggregation"]
            for pf in prefills:
                mode_args += ["--prefill", pf.base_url, str(pf.bootstrap_port)]
            for dc in decodes:
                mode_args += ["--decode", dc.base_url]

            self._launch(
                mode_args=mode_args,
                timeout=timeout,
                show_output=show_output,
                extra_args=extra_args,
                log_msg=f"PD gateway ({len(prefills)} prefill, {len(decodes)} decode)",
            )
        elif is_cloud_mode:
            # Cloud mode: OpenAI/xAI backend
            self.pd_mode = False
            self.igw_mode = False
            self.cloud_mode = True
            self.cloud_backend = cloud_backend

            # Get worker URL and API key based on backend
            if cloud_backend == "openai":
                worker_url = "https://api.openai.com"
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable required")
                self._env = os.environ.copy()
                self._env["OPENAI_API_KEY"] = api_key
            elif cloud_backend == "xai":
                worker_url = "https://api.x.ai"
                api_key = os.environ.get("XAI_API_KEY")
                if not api_key:
                    raise ValueError("XAI_API_KEY environment variable required")
                self._env = os.environ.copy()
                self._env["XAI_API_KEY"] = api_key
            else:
                raise ValueError(f"Unsupported cloud backend: {cloud_backend}")

            mode_args = [
                "--backend",
                "openai",  # Both OpenAI and xAI use openai backend type
                "--worker-urls",
                worker_url,
                "--history-backend",
                history_backend,
            ]

            self._launch(
                mode_args=mode_args,
                timeout=timeout,
                show_output=show_output,
                extra_args=extra_args,
                log_msg=f"{cloud_backend} cloud gateway",
            )
        else:
            # Regular mode: worker URLs
            if model_path is None:
                raise ValueError("model_path is required for regular mode")
            self.model_path = model_path
            self.pd_mode = False
            self.igw_mode = False

            self._launch(
                mode_args=["--model-path", model_path, "--worker-urls", *worker_urls],
                timeout=timeout,
                show_output=show_output,
                extra_args=extra_args,
                num_workers=len(worker_urls),
                log_msg=f"gateway with {len(worker_urls)} worker(s)",
            )

    def _launch(
        self,
        mode_args: list[str],
        timeout: float,
        show_output: bool,
        extra_args: list[str] | None,
        num_workers: int | None = None,
        log_msg: str = "",
    ) -> None:
        """Launch the gateway process.

        Args:
            mode_args: Mode-specific CLI arguments.
            timeout: Startup timeout in seconds.
            show_output: Show subprocess output.
            extra_args: Additional router arguments.
            num_workers: If set, wait for this many workers to be ready.
                         If None, just wait for health check.
            log_msg: Log message describing the startup.
        """
        cmd = self._build_base_cmd()
        cmd.extend(mode_args)

        if extra_args:
            cmd.extend(extra_args)

        logger.info("Starting %s on port %d", log_msg or "gateway", self.port)
        logger.debug("Gateway command: %s", " ".join(cmd))

        self.process = subprocess.Popen(
            cmd,
            env=self._env,  # Use custom env if set (e.g., for cloud mode API keys)
            stdout=None if show_output else subprocess.PIPE,
            stderr=None if show_output else subprocess.PIPE,
            start_new_session=True,
        )

        try:
            if num_workers is not None:
                wait_for_workers_ready(self.base_url, num_workers, timeout=timeout)
            else:
                wait_for_health(self.base_url, timeout=timeout)
        except TimeoutError:
            self.shutdown()
            raise

        self._started = True
        logger.info("Gateway ready at %s", self.base_url)

    def shutdown(self) -> None:
        """Shutdown the gateway process."""
        if self.process is not None:
            logger.info("Shutting down gateway (PID %d)", self.process.pid)
            kill_process_tree(self.process.pid)
            self.process = None
        self._started = False

    def _build_base_cmd(self) -> list[str]:
        """Build the base command for launching the router."""
        return [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--prometheus-port",
            str(self.prometheus_port),
            "--prometheus-host",
            self.host,
            "--policy",
            self.policy,
            "--log-level",
            "warn",
        ]

    # -------------------------------------------------------------------------
    # Health & Metrics APIs
    # -------------------------------------------------------------------------

    def health(self, timeout: float = 5.0) -> bool:
        """Check gateway health.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=timeout)
            return resp.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    def get_metrics(self, timeout: float = 5.0) -> str | None:
        """Get Prometheus metrics.

        Returns:
            Metrics text or None if unavailable.
        """
        try:
            resp = httpx.get(f"{self.metrics_url}/metrics", timeout=timeout)
            if resp.status_code == 200:
                return resp.text
            return None
        except (httpx.RequestError, httpx.TimeoutException):
            return None

    # -------------------------------------------------------------------------
    # Worker Management APIs
    # -------------------------------------------------------------------------

    def _worker_from_api_response(self, w: dict) -> WorkerInfo:
        """Convert API response dict to WorkerInfo.

        Args:
            w: Worker dict from API response.

        Returns:
            WorkerInfo object.
        """
        status = "healthy" if w.get("is_healthy", False) else "unhealthy"
        return WorkerInfo(
            id=w.get("id", ""),
            url=w.get("url", ""),
            model=w.get("model_id"),
            status=status,
            pending_requests=w.get("load", 0),
            metadata={
                "worker_type": w.get("worker_type"),
                "connection_mode": w.get("connection_mode"),
                "priority": w.get("priority"),
                "cost": w.get("cost"),
            },
        )

    def list_workers(self, timeout: float = 5.0) -> list[WorkerInfo]:
        """List all workers connected to the gateway.

        Returns:
            List of WorkerInfo objects.
        """
        try:
            resp = httpx.get(f"{self.base_url}/workers", timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                return [
                    self._worker_from_api_response(w) for w in data.get("workers", [])
                ]
            return []
        except (httpx.RequestError, httpx.TimeoutException):
            return []

    def get_worker(self, worker_id: str, timeout: float = 5.0) -> WorkerInfo | None:
        """Get information about a specific worker.

        Args:
            worker_id: The worker ID.

        Returns:
            WorkerInfo or None if not found.
        """
        try:
            resp = httpx.get(f"{self.base_url}/workers/{worker_id}", timeout=timeout)
            if resp.status_code == 200:
                return self._worker_from_api_response(resp.json())
            return None
        except (httpx.RequestError, httpx.TimeoutException):
            return None

    def add_worker(
        self,
        worker_url: str,
        timeout: float = 10.0,
        wait_ready: bool = True,
        ready_timeout: float = 60.0,
    ) -> tuple[bool, str | None]:
        """Add a worker to the gateway.

        Args:
            worker_url: URL of the worker to add.
            timeout: HTTP request timeout.
            wait_ready: If True, wait for worker to become ready.
            ready_timeout: Timeout for waiting for worker to be ready.

        Returns:
            Tuple of (success, worker_id or error message).
        """
        try:
            resp = httpx.post(
                f"{self.base_url}/workers",
                json={"url": worker_url},
                timeout=timeout,
            )
            # API returns 200 OK or 202 Accepted for async processing
            if resp.status_code in (200, 202):
                data = resp.json()
                worker_id = data.get("worker_id")

                if wait_ready and worker_id:
                    # Wait for worker to appear in list
                    import time

                    start = time.time()
                    while time.time() - start < ready_timeout:
                        workers = self.list_workers()
                        for w in workers:
                            if w.id == worker_id:
                                return True, worker_id
                        time.sleep(1.0)
                    return (
                        False,
                        f"Worker {worker_id} not ready within {ready_timeout}s",
                    )

                return True, worker_id
            return False, resp.text
        except (httpx.RequestError, httpx.TimeoutException) as e:
            return False, str(e)

    def remove_worker(self, worker_url: str, timeout: float = 10.0) -> tuple[bool, str]:
        """Remove a worker from the gateway by URL.

        Args:
            worker_url: URL of the worker to remove.

        Returns:
            Tuple of (success, message).
        """
        # Find worker_id by URL
        workers = self.list_workers(timeout=timeout)
        worker_id = None
        for w in workers:
            if w.url == worker_url:
                worker_id = w.id
                break

        if not worker_id:
            return False, f"Worker with URL {worker_url} not found"

        try:
            resp = httpx.delete(
                f"{self.base_url}/workers/{worker_id}",
                timeout=timeout,
            )
            if resp.status_code == 200:
                return True, "Worker removed"
            return False, resp.text
        except (httpx.RequestError, httpx.TimeoutException) as e:
            return False, str(e)

    # -------------------------------------------------------------------------
    # Model APIs
    # -------------------------------------------------------------------------

    def list_models(self, timeout: float = 5.0) -> list[dict]:
        """List available models (OpenAI-compatible).

        Returns:
            List of model info dicts.
        """
        try:
            resp = httpx.get(f"{self.base_url}/v1/models", timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("data", [])
            return []
        except (httpx.RequestError, httpx.TimeoutException):
            return []

    # -------------------------------------------------------------------------
    # Context manager support
    # -------------------------------------------------------------------------

    def __enter__(self) -> "Gateway":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()


def launch_cloud_gateway(
    runtime: str,  # "openai" or "xai"
    *,
    history_backend: str = "memory",
    extra_args: list[str] | None = None,
    timeout: float = 60,
    show_output: bool | None = None,
) -> Gateway:
    """Launch gateway with cloud API runtime.

    Args:
        runtime: Cloud runtime ("openai" or "xai")
        history_backend: History storage backend ("memory" or "oracle")
        extra_args: Additional router arguments
        timeout: Startup timeout in seconds
        show_output: Show subprocess output

    Returns:
        Gateway instance with running router
    """
    from .model_specs import THIRD_PARTY_MODELS

    if runtime not in THIRD_PARTY_MODELS:
        raise ValueError(
            f"Unknown cloud runtime: {runtime}. "
            f"Available: {list(THIRD_PARTY_MODELS.keys())}"
        )

    gateway = Gateway()
    gateway.start(
        cloud_backend=runtime,
        history_backend=history_backend,
        timeout=timeout,
        show_output=show_output,
        extra_args=extra_args,
    )
    return gateway
