"""Model pool for managing pre-loaded models across GPUs."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    import openai

from .constants import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_STARTUP_TIMEOUT,
    ENV_SHOW_WORKER_LOGS,
    HEALTH_CHECK_INTERVAL,
    INITIAL_GRACE_PERIOD,
    LAUNCH_STAGGER_DELAY,
    LOCAL_MODES,
    ConnectionMode,
    WorkerType,
)
from .gpu_allocator import GPUAllocator, GPUSlot, get_open_port
from .model_specs import MODEL_SPECS, get_model_spec
from .process_utils import detect_ib_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerIdentity:
    """Unique identity for a single worker instance.

    Each worker is uniquely identified by (model_id, mode, worker_type, index).
    For example:
    - llama-8b:http (regular worker, index 0)
    - llama-8b:http:prefill_0 (first prefill worker)
    - llama-8b:http:prefill_1 (second prefill worker)
    - llama-8b:http:decode_0 (first decode worker)

    Frozen/hashable so it can be used in sets and as dict keys for deduplication.
    """

    model_id: str
    mode: ConnectionMode = ConnectionMode.HTTP
    worker_type: WorkerType = WorkerType.REGULAR
    index: int = 0

    @property
    def is_prefill(self) -> bool:
        """Check if this is a prefill worker."""
        return self.worker_type == WorkerType.PREFILL

    @property
    def is_decode(self) -> bool:
        """Check if this is a decode worker."""
        return self.worker_type == WorkerType.DECODE

    @property
    def is_regular(self) -> bool:
        """Check if this is a regular worker."""
        return self.worker_type == WorkerType.REGULAR

    @property
    def key(self) -> str:
        """Unique key for this worker instance."""
        if self.worker_type == WorkerType.REGULAR:
            if self.index == 0:
                return f"{self.model_id}:{self.mode.value}"
            return f"{self.model_id}:{self.mode.value}:{self.index}"
        return (
            f"{self.model_id}:{self.mode.value}:{self.worker_type.value}_{self.index}"
        )

    def __str__(self) -> str:
        """String representation for logging."""
        return self.key


@dataclass
class ModelInstance:
    """A running model instance.

    Contains both identity (model_id, mode, worker_type) and runtime state
    (process, port, gpu_slot, etc.).
    """

    model_id: str
    mode: ConnectionMode
    model_path: str
    base_url: str
    port: int
    process: subprocess.Popen
    gpu_slot: GPUSlot | None
    key: str  # Unique instance key (e.g., "llama-8b:http:prefill_0")
    worker_type: WorkerType = WorkerType.REGULAR
    bootstrap_port: int | None = None  # For prefill workers in PD mode
    last_used: float = 0.0  # Timestamp for MRU eviction
    _healthy: bool = False  # Track if initial health check passed

    # Reference counting for safe parallel test execution
    _ref_count: int = 0
    _ref_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def identity(self) -> WorkerIdentity:
        """Get the identity (model_id, mode, worker_type) of this instance."""
        return WorkerIdentity(
            model_id=self.model_id,
            mode=self.mode,
            worker_type=self.worker_type,
        )

    @property
    def is_in_use(self) -> bool:
        """Check if this instance has active references (tests using it)."""
        with self._ref_lock:
            return self._ref_count > 0

    def acquire(self) -> None:
        """Acquire a reference to this instance.

        Call this before using the instance in a test to prevent eviction.
        Must be paired with a release() call when done.
        Also updates last_used timestamp atomically with ref count.
        """
        with self._ref_lock:
            self._ref_count += 1
            self.last_used = time.time()
            logger.debug(
                "Acquired reference to %s (ref_count=%d)", self.key, self._ref_count
            )

    def release(self) -> None:
        """Release a reference to this instance.

        Call this when done using the instance in a test.
        """
        with self._ref_lock:
            if self._ref_count > 0:
                self._ref_count -= 1
                logger.debug(
                    "Released reference to %s (ref_count=%d)",
                    self.key,
                    self._ref_count,
                )
            else:
                logger.warning(
                    "Attempted to release reference to %s with ref_count=0", self.key
                )

    @property
    def worker_url(self) -> str:
        """URL to use when connecting router to this worker."""
        if self.mode == ConnectionMode.GRPC:
            return f"grpc://{DEFAULT_HOST}:{self.port}"
        return self.base_url

    def is_alive(self) -> bool:
        """Check if the process is still running."""
        return self.process.poll() is None

    def health_check(self, timeout: float = 5.0) -> bool:
        """Check if the model server is healthy.

        Uses HTTP /health endpoint for HTTP workers, gRPC health check for gRPC workers.
        """
        if self.mode == ConnectionMode.GRPC:
            return self._grpc_health_check(timeout)
        return self._http_health_check(timeout)

    def _http_health_check(self, timeout: float = 5.0) -> bool:
        """Check health via HTTP /health endpoint."""
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=timeout)
            return resp.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    def deep_health_check(self, timeout: float = 30.0) -> bool:
        """Deep health check that verifies the model can actually generate.

        Uses /health_generate for HTTP workers (runs actual inference).
        For gRPC workers, falls back to standard health check.
        """
        if self.mode == ConnectionMode.GRPC:
            # For gRPC, use standard health check (no /health_generate equivalent)
            return self._grpc_health_check(timeout)

        try:
            resp = httpx.get(f"{self.base_url}/health_generate", timeout=timeout)
            return resp.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    def _grpc_health_check(self, timeout: float = 5.0) -> bool:
        """Check health via gRPC health check protocol."""
        try:
            import grpc
            from grpc_health.v1 import health_pb2, health_pb2_grpc
        except ImportError as e:
            logger.debug("gRPC libraries not available: %s", e)
            return False

        try:
            channel = grpc.insecure_channel(f"{DEFAULT_HOST}:{self.port}")
            try:
                stub = health_pb2_grpc.HealthStub(channel)
                request = health_pb2.HealthCheckRequest(service="")
                response = stub.Check(request, timeout=timeout)
                is_serving = response.status == health_pb2.HealthCheckResponse.SERVING
                if is_serving:
                    logger.debug(
                        "gRPC health check passed for port %d (status: SERVING)",
                        self.port,
                    )
                return is_serving
            finally:
                channel.close()
        except grpc.RpcError as e:
            # gRPC-specific errors (connection refused, deadline exceeded, etc.)
            logger.debug(
                "gRPC health check failed for port %d: %s",
                self.port,
                e.code() if hasattr(e, "code") else str(e),
            )
            return False
        except Exception as e:
            # Other errors
            logger.debug(
                "gRPC health check error for port %d: %s",
                self.port,
                str(e),
            )
            return False

    def terminate(self, timeout: float = 10.0) -> None:
        """Terminate the model server process and all child processes.

        Since workers are started with start_new_session=True, they run in their
        own process group. We must kill the entire process group to ensure child
        processes (e.g., TP workers) are also terminated and GPU memory is freed.
        """
        if self.process.poll() is not None:
            return  # Already terminated

        pid = self.process.pid
        logger.info("Terminating %s (PID %d)", self.key, pid)

        # Try graceful shutdown of the entire process group first
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError) as e:
            logger.debug("Could not send SIGTERM to process group: %s", e)
            # Fall back to terminating just the main process
            self.process.terminate()

        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("%s did not terminate, killing process group", self.key)
            # Force kill the entire process group
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError) as e:
                logger.debug("Could not send SIGKILL to process group: %s", e)
                self.process.kill()

            try:
                self.process.wait(timeout=5)  # Brief timeout after kill
            except subprocess.TimeoutExpired:
                logger.error("%s did not die after SIGKILL, abandoning", self.key)


class ModelPool:
    """Manages long-running SGLang worker processes across GPUs.

    Workers are expensive to start (~30-60s due to model loading), so this pool
    keeps them running and allows reuse across multiple tests. Routers can then
    be launched cheaply (~1-2s) pointing to these workers.

    Startup behavior:
    - Workers are pre-launched at startup until GPUs are full
    - When a test needs a model that isn't running, MRU model is evicted
      (models just used are likely done, models not yet used are waiting)
    - The needed model is then launched on-demand

    Instance keys:
    - Regular workers: "model_id:mode" (e.g., "llama-8b:http")
    - PD workers: "model_id:mode:worker_type" (e.g., "llama-8b:http:prefill")

    Limitations:
    - Currently one worker instance per (model_id, mode) combination
    - @pytest.mark.workers(count=n) duplicates URLs to router, not distinct workers
    - For true multi-worker LB testing, extend to support multiple instances

    Usage:
        pool = ModelPool()
        pool.startup(requirements=[("llama-8b", ConnectionMode.HTTP)])
        instance = pool.get("llama-8b", "http")  # Pre-launched or on-demand
    """

    def __init__(self, allocator: GPUAllocator | None = None):
        """Initialize the model pool.

        Args:
            allocator: GPU allocator to use. If None, creates a new one.
        """
        self.allocator = allocator or GPUAllocator()
        self.instances: dict[str, ModelInstance] = {}  # key = "model_id:mode"
        self._startup_timeout = DEFAULT_STARTUP_TIMEOUT
        self._lock = threading.RLock()  # Protects instances dict

    def startup(
        self,
        requirements: list[WorkerIdentity] | None = None,
        startup_timeout: int = DEFAULT_STARTUP_TIMEOUT,
    ) -> None:
        """Start worker processes for the required workers in order.

        Workers are launched sequentially (one Popen at a time) but boot up
        concurrently since model loading happens in parallel across processes.
        This method blocks until all workers pass health checks.

        All worker types (regular, prefill, decode) are handled uniformly.
        Each WorkerIdentity uniquely identifies a worker by (model_id, mode,
        worker_type, index).

        Thread-safe: Protected by internal lock.

        Args:
            requirements: List of WorkerIdentity specifying what to start.
                         If None, starts default model in HTTP mode.
            startup_timeout: Timeout in seconds for all models to become healthy.
        """
        with self._lock:
            self._startup_unlocked(requirements, startup_timeout)

    def _startup_unlocked(
        self,
        requirements: list[WorkerIdentity] | None = None,
        startup_timeout: int = DEFAULT_STARTUP_TIMEOUT,
    ) -> None:
        """Internal startup logic. Caller must hold _lock."""
        self._startup_timeout = startup_timeout

        if requirements is None:
            requirements = [WorkerIdentity(DEFAULT_MODEL, ConnectionMode.HTTP)]

        # Validate requirements
        valid_requirements: list[WorkerIdentity] = []
        for identity in requirements:
            if identity.model_id not in MODEL_SPECS:
                logger.warning("Unknown model %s, skipping", identity.model_id)
                continue
            if identity.mode not in LOCAL_MODES:
                logger.warning(
                    "Invalid mode %s for %s, skipping", identity.mode, identity.model_id
                )
                continue
            valid_requirements.append(identity)

        if not valid_requirements:
            logger.warning("No valid requirements to start")
            return

        logger.info(
            "Starting model pool with %d workers: %s",
            len(valid_requirements),
            [str(r) for r in valid_requirements],
        )

        # Detect IB device once for PD workers
        has_pd = any(r.is_prefill or r.is_decode for r in valid_requirements)
        ib_device = detect_ib_device() if has_pd else None
        if ib_device:
            logger.info("Detected InfiniBand device: %s", ib_device)

        deferred: list[str] = []
        launched_count = 0

        # Process requirements in order - all workers treated uniformly
        for identity in valid_requirements:
            spec = get_model_spec(identity.model_id)
            tp = spec.get("tp", 1)

            # Check if we have enough GPUs
            available_gpus = self.allocator.available_gpus()
            if len(available_gpus) < tp:
                logger.info(
                    "Not enough GPUs for %s (need %d, have %d), deferring",
                    identity,
                    tp,
                    len(available_gpus),
                )
                deferred.append(str(identity))
                continue

            # Allocate GPU slot
            allocation_specs = {
                identity.key: {
                    "model": spec["model"],
                    "memory_gb": spec.get("memory_gb", 16),
                    "tp": tp,
                }
            }
            slots = self.allocator.allocate_slots(allocation_specs, preserve_order=True)
            if not slots:
                deferred.append(str(identity))
                continue

            # Each prefill worker needs its own bootstrap port for PD communication
            bootstrap_port = get_open_port() if identity.is_prefill else None

            # Stagger launches to avoid resource contention during model loading
            if launched_count > 0 and LAUNCH_STAGGER_DELAY > 0:
                logger.info(
                    "Staggering launch by %ds to reduce resource contention",
                    LAUNCH_STAGGER_DELAY,
                )
                time.sleep(LAUNCH_STAGGER_DELAY)

            # Launch the worker
            self._launch_model(
                model_id=identity.model_id,
                mode=identity.mode,
                gpu_slot=slots[0],
                worker_type=identity.worker_type,
                bootstrap_port=bootstrap_port,
                ib_device=(
                    ib_device if (identity.is_prefill or identity.is_decode) else None
                ),
                instance_key=identity.key,
            )
            launched_count += 1

        # Log deferred workers
        if deferred:
            logger.info(
                "%d workers deferred for on-demand launch: %s",
                len(deferred),
                deferred,
            )

        # Wait for all launched models to be healthy
        self._wait_all_healthy()

    def _launch_model(
        self,
        model_id: str,
        mode: ConnectionMode,
        gpu_slot: GPUSlot | None = None,
        worker_type: WorkerType = WorkerType.REGULAR,
        bootstrap_port: int | None = None,
        ib_device: str | None = None,
        instance_key: str | None = None,
    ) -> ModelInstance:
        """Launch a model instance.

        Args:
            model_id: Model identifier from MODEL_SPECS.
            mode: Connection mode (HTTP or GRPC).
            gpu_slot: GPU slot assignment, or None for auto.
            worker_type: Worker type (REGULAR, PREFILL, or DECODE).
            bootstrap_port: Bootstrap port for prefill workers in PD mode.
            ib_device: InfiniBand device for PD disaggregation.
            instance_key: Custom instance key, or None to auto-generate.

        Returns:
            The launched ModelInstance.
        """
        spec = get_model_spec(model_id)
        model_path = spec["model"]
        tp_size = spec.get("tp", 1)
        features = spec.get("features", [])

        # Get port - use slot's port if available, otherwise find open port
        port = gpu_slot.port if gpu_slot else get_open_port()

        # Build environment
        env = os.environ.copy()
        if gpu_slot:
            env["CUDA_VISIBLE_DEVICES"] = gpu_slot.cuda_visible_devices()

        # Build command
        cmd = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--host",
            DEFAULT_HOST,
            "--port",
            str(port),
            "--tp-size",
            str(tp_size),
            "--log-level",
            "warning",
        ]

        if mode == ConnectionMode.GRPC:
            cmd.append("--grpc-mode")

        # Embedding model flag
        if "embedding" in features:
            cmd.append("--is-embedding")

        # PD disaggregation arguments
        if worker_type == WorkerType.PREFILL:
            cmd.extend(["--disaggregation-mode", "prefill"])
            if bootstrap_port:
                cmd.extend(["--disaggregation-bootstrap-port", str(bootstrap_port)])
            if ib_device:
                cmd.extend(["--disaggregation-ib-device", ib_device])
        elif worker_type == WorkerType.DECODE:
            cmd.extend(["--disaggregation-mode", "decode"])
            # Base GPU ID 0 since CUDA_VISIBLE_DEVICES remaps the GPU
            cmd.extend(["--base-gpu-id", "0"])
            if ib_device:
                cmd.extend(["--disaggregation-ib-device", ib_device])

        # Additional worker args from model spec (e.g., --context-length)
        worker_args = spec.get("worker_args", [])
        if worker_args:
            cmd.extend(worker_args)

        # Build key based on worker type (or use custom key)
        if instance_key:
            key = instance_key
        elif worker_type == WorkerType.REGULAR:
            key = f"{model_id}:{mode.value}"
        else:
            key = f"{model_id}:{mode.value}:{worker_type.value}"

        gpu_info = gpu_slot.gpu_ids if gpu_slot else "auto"
        logger.info("Launching %s on GPUs %s port %d", key, gpu_info, port)

        show_output = os.environ.get(ENV_SHOW_WORKER_LOGS, "0") == "1"

        # Start the process
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=None if show_output else subprocess.PIPE,
            stderr=None if show_output else subprocess.PIPE,
            start_new_session=True,
        )

        base_url = f"http://{DEFAULT_HOST}:{port}"
        instance = ModelInstance(
            model_id=model_id,
            mode=mode,
            model_path=model_path,
            base_url=base_url,
            port=port,
            process=proc,
            gpu_slot=gpu_slot,
            key=key,
            worker_type=worker_type,
            bootstrap_port=bootstrap_port,
            last_used=time.time(),
        )
        self.instances[key] = instance
        return instance

    def _wait_all_healthy(self) -> None:
        """Wait for all model instances to become healthy.

        Only checks workers that haven't been marked healthy yet,
        avoiding redundant health checks on already-verified workers.
        """
        start_time = time.time()
        # Only wait for workers that haven't been verified healthy yet
        pending = {key for key, inst in self.instances.items() if not inst._healthy}
        check_count = 0

        if not pending:
            logger.info("All workers already healthy, skipping health check")
            return

        logger.info(
            "Waiting for %d workers to become healthy (timeout: %ds)...",
            len(pending),
            self._startup_timeout,
        )

        # Initial grace period to allow models to load before health checks
        if INITIAL_GRACE_PERIOD > 0:
            logger.info(
                "Waiting %ds for initial model loading before health checks...",
                INITIAL_GRACE_PERIOD,
            )
            time.sleep(INITIAL_GRACE_PERIOD)

        while pending and (time.time() - start_time) < self._startup_timeout:
            check_count += 1
            elapsed = time.time() - start_time

            for key in list(pending):
                instance = self.instances[key]

                # Check if process died
                if not instance.is_alive():
                    logger.error(
                        "[%.1fs] %s (PID %d) died during startup",
                        elapsed,
                        key,
                        instance.process.pid,
                    )
                    # Read stderr for debugging (non-blocking to avoid hangs)
                    if instance.process.stderr:
                        try:
                            import os
                            import select

                            # Use select for non-blocking read with short timeout
                            # to avoid hanging if child processes keep stderr open
                            ready, _, _ = select.select(
                                [instance.process.stderr], [], [], 0.5
                            )
                            if ready:
                                # Use os.read with limited size instead of .read()
                                # which reads until EOF and can block if pipe stays open
                                fd = instance.process.stderr.fileno()
                                stderr = os.read(fd, 65536)  # Read up to 64KB
                                if stderr:
                                    logger.error(
                                        "Stderr: %s",
                                        stderr.decode(errors="replace")[-2000:],
                                    )
                        except Exception as e:
                            logger.warning("Could not read stderr: %s", e)
                    # Evict dead instance and release GPUs
                    self._evict_instance(key)
                    pending.discard(key)
                    continue

                # Check health
                if instance.health_check():
                    logger.info(
                        "[%.1fs] %s is healthy at %s (router url: %s) (check #%d)",
                        elapsed,
                        key,
                        instance.base_url,
                        instance.worker_url,
                        check_count,
                    )
                    instance._healthy = True
                    pending.discard(key)

            if pending:
                # Log progress every 30 seconds
                if check_count % 15 == 0:  # ~30s at 2s interval
                    logger.info(
                        "[%.1fs] Still waiting for %d workers: %s",
                        elapsed,
                        len(pending),
                        list(pending),
                    )
                time.sleep(HEALTH_CHECK_INTERVAL)

        if pending:
            elapsed = time.time() - start_time
            logger.error(
                "[%.1fs] Models failed to start within %ds: %s",
                elapsed,
                self._startup_timeout,
                pending,
            )
            # Log stderr from failed workers for debugging
            for key in pending:
                instance = self.instances.get(key)
                if instance and instance.process.stderr:
                    try:
                        import os
                        import select

                        # Use select for non-blocking read with short timeout
                        # to avoid hanging if worker is unresponsive
                        ready, _, _ = select.select(
                            [instance.process.stderr], [], [], 0.1
                        )
                        if ready:
                            # Use os.read with limited size instead of .read()
                            # which reads until EOF and can block if pipe stays open
                            fd = instance.process.stderr.fileno()
                            stderr = os.read(fd, 65536)  # Read up to 64KB
                            if stderr:
                                logger.error(
                                    "[%s] Last stderr output:\n%s",
                                    key,
                                    stderr.decode(errors="replace")[-3000:],
                                )
                    except Exception as e:
                        logger.error("[%s] Could not read stderr: %s", key, e)
            # Terminate failed instances and release their GPUs
            for key in pending:
                self._evict_instance(key)
        else:
            elapsed = time.time() - start_time
            logger.info(
                "[%.1fs] All %d workers healthy after %d health checks",
                elapsed,
                len(self.instances),
                check_count,
            )

    def get(
        self,
        model_id: str,
        mode: ConnectionMode | str,
        worker_type: WorkerType | str = WorkerType.REGULAR,
        wait_for_gpus: bool = True,
        gpu_wait_timeout: int = 300,
    ) -> ModelInstance:
        """Get a model instance by model_id, mode, and worker_type.

        If the model is not running, it will be launched on-demand with MRU
        eviction if GPU resources are constrained.

        Thread-safe: Protected by internal lock. The returned instance has its
        reference count incremented (via acquire()) to prevent eviction.
        Caller MUST call release() on the instance when done.

        Args:
            model_id: The model ID (e.g., "llama-8b")
            mode: The mode (ConnectionMode.HTTP or ConnectionMode.GRPC, or string)
            worker_type: The worker type (REGULAR, PREFILL, DECODE). Defaults to REGULAR.
            wait_for_gpus: If True, wait for GPUs to become available when all
                are in use by other tests. Defaults to True.
            gpu_wait_timeout: Max seconds to wait for GPUs (default 5 min).

        Returns:
            ModelInstance for the requested model/mode/worker_type (already acquired).

        Raises:
            RuntimeError: If worker process died, failed health check, or
                timeout waiting for GPUs.
        """
        deadline = time.time() + gpu_wait_timeout
        poll_interval = 2.0  # seconds

        while True:
            with self._lock:
                instance = self._get_unlocked(model_id, mode, worker_type)
                if instance is not None:
                    # Acquire while holding lock to prevent race with eviction
                    instance.acquire()
                    return instance

                # _get_unlocked returns None when GPUs unavailable after eviction
                if not wait_for_gpus:
                    raise RuntimeError(
                        f"Cannot get {model_id}: GPUs unavailable and waiting disabled"
                    )

                if time.time() >= deadline:
                    raise RuntimeError(
                        f"Timeout waiting for GPUs for {model_id} after {gpu_wait_timeout}s"
                    )

            # Release lock while waiting so other tests can release workers
            logger.info(
                "All GPUs in use by other tests, waiting %.1fs for %s...",
                poll_interval,
                model_id,
            )
            time.sleep(poll_interval)

    def _get_unlocked(
        self,
        model_id: str,
        mode: ConnectionMode | str,
        worker_type: WorkerType | str = WorkerType.REGULAR,
    ) -> ModelInstance | None:
        """Internal get logic. Caller must hold _lock.

        Returns:
            ModelInstance if successful, None if GPUs unavailable (signals retry).

        Raises:
            RuntimeError: If worker died or failed health check.
        """
        # Accept both enum and string for convenience
        if isinstance(mode, str):
            mode = ConnectionMode(mode)
        if isinstance(worker_type, str):
            worker_type = WorkerType(worker_type)

        if worker_type == WorkerType.REGULAR:
            key = f"{model_id}:{mode.value}"
        else:
            key = f"{model_id}:{mode.value}:{worker_type.value}"

        # Check if instance exists - if not, launch on-demand with eviction
        if key not in self.instances:
            logger.info(
                "Model %s not running, launching on-demand with MRU eviction if needed",
                key,
            )
            if not self._ensure_gpu_available(model_id):
                # GPUs not available after eviction - signal retry
                return None

            # Allocate GPU slot for this model
            spec = get_model_spec(model_id)
            allocation_specs = {
                key: {
                    "model": spec["model"],
                    "memory_gb": spec.get("memory_gb", 16),
                    "tp": spec.get("tp", 1),
                }
            }
            slots = self.allocator.allocate_slots(allocation_specs)
            if not slots:
                raise RuntimeError(
                    f"Failed to allocate GPU slot for {model_id} after eviction"
                )
            gpu_slot = slots[0]

            self._launch_model(model_id, mode, gpu_slot=gpu_slot)
            self._wait_for_instance(key)

        instance = self.instances[key]

        # Note: last_used is updated in acquire() which should be called by fixtures
        # to prevent eviction during test execution

        # Verify worker is still alive and healthy
        if not instance.is_alive():
            raise RuntimeError(f"Worker {key} process died (was healthy at startup)")

        if not instance.deep_health_check(timeout=30.0):
            raise RuntimeError(
                f"Worker {key} failed deep health check (health_generate) - "
                "model may be stuck or crashed"
            )

        logger.info("Worker %s passed deep health check", key)
        return instance

    def _evict_for_gpus(
        self,
        required_gpus: int,
        exclude_model_id: str | None = None,
        exclude_mode: ConnectionMode | None = None,
        exclude_worker_types: set[WorkerType] | None = None,
    ) -> None:
        """Evict models until we have enough GPUs available.

        Uses MRU (most recently used) eviction strategy - evicts models that
        were just used first, keeping models that haven't been used yet
        (which are likely waiting for upcoming tests).

        Args:
            required_gpus: Number of GPUs needed.
            exclude_model_id: Model ID to exclude from eviction.
            exclude_mode: Connection mode to exclude from eviction (optional).
            exclude_worker_types: Worker types to exclude from eviction.
                If None, falls back to excluding by model_id only (backward compatible).
        """
        available = self.allocator.available_gpus()
        if len(available) >= required_gpus:
            return  # Already have enough

        # Sort by last_used descending (MRU eviction) - evict most recently used first
        # Store (dict_key, instance) tuples to preserve the actual key for eviction
        # Note: Make a copy of items to avoid RuntimeError if dict is modified during iteration
        evictable: list[tuple[str, ModelInstance]] = []
        for dict_key, inst in list(self.instances.items()):
            # Skip instances with active references (tests using them)
            if inst.is_in_use:
                logger.debug(
                    "Skipping eviction of %s - has active references", dict_key
                )
                continue
            if exclude_worker_types is not None:
                # Precise matching with worker types
                # Must match model_id AND worker_type, mode is optional
                if (
                    exclude_model_id is not None
                    and inst.model_id == exclude_model_id
                    and inst.worker_type in exclude_worker_types
                ):
                    # If mode is specified, also require mode match
                    if exclude_mode is None or inst.mode == exclude_mode:
                        continue
            else:
                # Backward compatible: exclude by model_id only
                if exclude_model_id is not None and inst.model_id == exclude_model_id:
                    continue
            evictable.append((dict_key, inst))

        evictable.sort(key=lambda x: x[1].last_used, reverse=True)

        freed_gpus = len(available)
        for dict_key, inst in evictable:
            if freed_gpus >= required_gpus:
                break

            logger.info("Evicting model %s (MRU) to free GPUs", dict_key)
            self._evict_instance(dict_key)
            if inst.gpu_slot:
                freed_gpus += len(inst.gpu_slot.gpu_ids)

    def _ensure_gpu_available(self, model_id: str) -> bool:
        """Ensure GPU is available for a model, evicting if needed.

        Args:
            model_id: Model ID that needs GPU resources.

        Returns:
            True if GPUs are available, False if not (all in use by other tests).
        """
        spec = get_model_spec(model_id)
        required_gpus = spec.get("tp", 1)

        # Exclude REGULAR workers of same model from eviction (keep them)
        # but allow evicting PD workers (PREFILL/DECODE) to free GPUs
        self._evict_for_gpus(
            required_gpus,
            exclude_model_id=model_id,
            exclude_worker_types={WorkerType.REGULAR},
        )

        available = self.allocator.available_gpus()
        if len(available) < required_gpus:
            logger.info(
                "Cannot launch %s: need %d GPUs, only %d available after eviction "
                "(all workers in use by other tests)",
                model_id,
                required_gpus,
                len(available),
            )
            return False
        return True

    def _evict_instance(self, key: str) -> None:
        """Evict a model instance and free its resources.

        Args:
            key: Instance key to evict.
        """
        if key not in self.instances:
            return

        instance = self.instances[key]
        instance.terminate()

        # Release GPU slot back to allocator
        if instance.gpu_slot:
            self.allocator.release_slot(instance.gpu_slot)

        del self.instances[key]
        logger.info("Evicted instance %s", key)

    def _wait_for_instance(self, key: str, timeout: float | None = None) -> None:
        """Wait for a specific instance to become healthy.

        Args:
            key: Instance key to wait for.
            timeout: Timeout in seconds. Defaults to _startup_timeout.
        """
        if timeout is None:
            timeout = self._startup_timeout

        start_time = time.time()
        instance = self.instances.get(key)
        if not instance:
            raise KeyError(f"Instance {key} not found")

        while (time.time() - start_time) < timeout:
            if not instance.is_alive():
                raise RuntimeError(f"Worker {key} died during startup")

            if instance.health_check():
                logger.info("Instance %s is healthy", key)
                instance._healthy = True
                return

            time.sleep(HEALTH_CHECK_INTERVAL)

        raise TimeoutError(f"Instance {key} did not become healthy within {timeout}s")

    def get_workers_by_type(
        self, model_id: str, worker_type: WorkerType
    ) -> list[ModelInstance]:
        """Get all workers of a specific type for a model.

        Thread-safe: Protected by internal lock. All returned instances have their
        reference count incremented (via acquire()) to prevent eviction.
        Caller MUST call release() on each instance when done.

        Args:
            model_id: The model ID.
            worker_type: The worker type to filter by.

        Returns:
            List of matching ModelInstance objects (already acquired).
        """
        with self._lock:
            workers = [
                inst
                for inst in self.instances.values()
                if inst.model_id == model_id and inst.worker_type == worker_type
            ]
            # Acquire all while holding lock to prevent race with eviction
            for worker in workers:
                worker.acquire()
            return workers

    def launch_workers(
        self,
        workers: list[WorkerIdentity],
        startup_timeout: int = DEFAULT_STARTUP_TIMEOUT,
        allow_eviction: bool = True,
        wait_for_gpus: bool = True,
        gpu_wait_timeout: int = 300,
    ) -> list[ModelInstance]:
        """Launch workers of any type.

        This is the unified method for launching workers. It handles all worker
        types (regular, prefill, decode) uniformly.

        Thread-safe: Protected by internal lock.

        Args:
            workers: List of WorkerIdentity objects specifying workers to launch.
            startup_timeout: Timeout for workers to become healthy.
            allow_eviction: If True, evict MRU models to free GPUs.
            wait_for_gpus: If True, wait for GPUs to become available when all
                are in use by other tests (with eviction enabled).
            gpu_wait_timeout: Max seconds to wait for GPUs (default 5 min).

        Returns:
            List of launched ModelInstance objects.
        """
        deadline = time.time() + gpu_wait_timeout
        poll_interval = 2.0  # seconds

        while True:
            with self._lock:
                result = self._launch_workers_unlocked(
                    workers, startup_timeout, allow_eviction
                )
                if result is not None:
                    return result

                # _launch_workers_unlocked returns None when GPUs unavailable
                # after eviction attempt (all workers in use by other tests)
                if not wait_for_gpus or not allow_eviction:
                    return []

                if time.time() >= deadline:
                    logger.warning(
                        "Timeout waiting for GPUs after %ds, giving up",
                        gpu_wait_timeout,
                    )
                    return []

            # Release lock while waiting so other tests can release workers
            logger.info(
                "All GPUs in use by other tests, waiting %.1fs for availability...",
                poll_interval,
            )
            time.sleep(poll_interval)

    def _launch_workers_unlocked(
        self,
        workers: list[WorkerIdentity],
        startup_timeout: int = DEFAULT_STARTUP_TIMEOUT,
        allow_eviction: bool = True,
    ) -> list[ModelInstance] | None:
        """Internal launch logic. Caller must hold _lock.

        Returns:
            List of launched instances, empty list if no valid workers,
            or None if GPUs unavailable (signals caller to wait and retry).
        """
        if not workers:
            return []

        self._startup_timeout = startup_timeout

        # Validate all workers
        valid_workers: list[WorkerIdentity] = []
        for w in workers:
            if w.model_id not in MODEL_SPECS:
                logger.warning("Unknown model %s, skipping", w.model_id)
                continue
            if w.mode not in LOCAL_MODES:
                logger.warning("Invalid mode %s, skipping", w.mode)
                continue
            valid_workers.append(w)

        if not valid_workers:
            return []

        # Calculate total GPUs needed
        total_gpus = 0
        for w in valid_workers:
            spec = get_model_spec(w.model_id)
            total_gpus += spec.get("tp", 1)

        # Check if we have enough GPUs
        available = self.allocator.available_gpus()
        if len(available) < total_gpus:
            if allow_eviction:
                logger.info(
                    "Need %d GPUs for %d workers, only %d available. Evicting...",
                    total_gpus,
                    len(valid_workers),
                    len(available),
                )
                self._evict_for_gpus(total_gpus)

                # Check again after eviction
                available = self.allocator.available_gpus()
                if len(available) < total_gpus:
                    # Still not enough - all workers are in use by other tests
                    # Return None to signal caller to wait and retry
                    logger.info(
                        "Still need %d GPUs, only %d available after eviction. "
                        "All workers in use by other tests.",
                        total_gpus,
                        len(available),
                    )
                    return None
            else:
                logger.warning(
                    "Need %d GPUs, only %d available. Skipping launch.",
                    total_gpus,
                    len(available),
                )
                return []

        # Build allocation specs
        allocation_specs = {}
        for w in valid_workers:
            spec = get_model_spec(w.model_id)
            allocation_specs[w.key] = {
                "model": spec["model"],
                "memory_gb": spec.get("memory_gb", 16),
                "tp": spec.get("tp", 1),
            }

        # Allocate GPU slots
        slots = self.allocator.allocate_slots(allocation_specs, preserve_order=True)
        slot_map = {s.assigned_model: s for s in slots}

        if not slots:
            raise RuntimeError(
                f"Failed to allocate GPU slots for {len(valid_workers)} workers"
            )

        # Detect IB device for PD workers
        has_pd = any(w.is_prefill or w.is_decode for w in valid_workers)
        ib_device = detect_ib_device() if has_pd else None

        instances: list[ModelInstance] = []
        for w in valid_workers:
            # Each prefill worker needs its own bootstrap port for PD communication
            bootstrap_port = get_open_port() if w.is_prefill else None

            instance = self._launch_model(
                model_id=w.model_id,
                mode=w.mode,
                gpu_slot=slot_map.get(w.key),
                worker_type=w.worker_type,
                bootstrap_port=bootstrap_port,
                ib_device=ib_device if (w.is_prefill or w.is_decode) else None,
                instance_key=w.key,
            )
            instances.append(instance)

        self._wait_all_healthy()
        return instances

    def get_client(
        self, model_id: str, mode: ConnectionMode | str = ConnectionMode.HTTP
    ) -> "openai.OpenAI":
        """Get OpenAI client for a specific model.

        Args:
            model_id: The model ID to get a client for.
            mode: The mode (ConnectionMode.HTTP or ConnectionMode.GRPC). Defaults to HTTP.

        Returns:
            OpenAI client configured for this model.
        """
        import openai

        instance = self.get(model_id, mode)
        return openai.OpenAI(
            base_url=f"{instance.base_url}/v1",
            api_key="not-used",
        )

    def get_base_url(
        self, model_id: str, mode: ConnectionMode | str = ConnectionMode.HTTP
    ) -> str:
        """Get the base URL for a specific model."""
        return self.get(model_id, mode).base_url

    def shutdown(self) -> None:
        """Tear down all models.

        Thread-safe: Protected by internal lock.
        """
        with self._lock:
            logger.info("Shutting down model pool (%d instances)", len(self.instances))
            for instance in self.instances.values():
                instance.terminate()
            self.instances.clear()

    def __enter__(self) -> "ModelPool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
