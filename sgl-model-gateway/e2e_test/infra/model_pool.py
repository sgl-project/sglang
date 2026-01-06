"""Model pool for managing pre-loaded models across GPUs."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
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
    LOCAL_MODES,
    ConnectionMode,
    WorkerType,
)
from .gpu_allocator import GPUAllocator, GPUSlot, get_open_port
from .model_specs import MODEL_SPECS, get_model_spec
from .process_utils import detect_ib_device

logger = logging.getLogger(__name__)


@dataclass
class ModelInstance:
    """A running model instance."""

    model_id: str
    mode: ConnectionMode
    model_path: str
    base_url: str
    port: int
    process: subprocess.Popen
    gpu_slot: GPUSlot | None
    worker_type: WorkerType = WorkerType.REGULAR
    bootstrap_port: int | None = None  # For prefill workers in PD mode

    @property
    def key(self) -> str:
        """Unique key for this instance.

        Regular: 'model_id:mode' (e.g., 'llama-8b:http')
        PD workers: 'model_id:mode:worker_type' (e.g., 'llama-8b:http:prefill')
        """
        if self.worker_type == WorkerType.REGULAR:
            return f"{self.model_id}:{self.mode.value}"
        return f"{self.model_id}:{self.mode.value}:{self.worker_type.value}"

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
        """Terminate the model server process."""
        if self.process.poll() is not None:
            return  # Already terminated

        logger.info("Terminating %s (PID %d)", self.key, self.process.pid)

        # Try graceful shutdown first
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("%s did not terminate, killing", self.key)
            self.process.kill()
            self.process.wait()


class ModelPool:
    """Manages long-running SGLang worker processes across GPUs.

    Workers are expensive to start (~30-60s due to model loading), so this pool
    keeps them running and allows reuse across multiple tests. Routers can then
    be launched cheaply (~1-2s) pointing to these workers.

    Startup behavior:
    - Workers are launched sequentially (one subprocess.Popen at a time)
    - But they boot up concurrently (overlapping model loading)
    - _wait_all_healthy() blocks until all workers respond to health checks

    Instance keys:
    - Regular workers: "model_id:mode" (e.g., "llama-8b:http")
    - PD workers: "model_id:mode:worker_type" (e.g., "llama-8b:http:prefill")

    Limitations:
    - Currently one worker instance per (model_id, mode) combination
    - @pytest.mark.workers(n) duplicates URLs to router, not distinct workers
    - For true multi-worker LB testing, extend to support multiple instances

    Usage:
        pool = ModelPool()
        pool.startup(requirements=[("llama-8b", ConnectionMode.HTTP)])

        instance = pool.get("llama-8b", "http")
        # instance.base_url -> "http://127.0.0.1:30000"
        # instance.worker_url -> URL for router to connect to
    """

    def __init__(self, allocator: GPUAllocator | None = None):
        """Initialize the model pool.

        Args:
            allocator: GPU allocator to use. If None, creates a new one.
        """
        self.allocator = allocator or GPUAllocator()
        self.instances: dict[str, ModelInstance] = {}  # key = "model_id:mode"
        self._startup_timeout = DEFAULT_STARTUP_TIMEOUT

    def startup(
        self,
        requirements: list[tuple[str, ConnectionMode]] | None = None,
        startup_timeout: int = DEFAULT_STARTUP_TIMEOUT,
    ) -> None:
        """Start worker processes for the required models.

        Workers are launched sequentially (one Popen at a time) but boot up
        concurrently since model loading happens in parallel across processes.
        This method blocks until all workers pass health checks.

        Args:
            requirements: List of (model_id, mode) tuples specifying what to start.
                         mode is ConnectionMode.HTTP or ConnectionMode.GRPC.
                         If None, starts default model in HTTP mode.
            startup_timeout: Timeout in seconds for all models to become healthy.
        """
        self._startup_timeout = startup_timeout

        if requirements is None:
            requirements = [(DEFAULT_MODEL, ConnectionMode.HTTP)]

        # Deduplicate and validate
        requirements = list(set(requirements))
        valid_requirements = []
        for model_id, mode in requirements:
            if model_id not in MODEL_SPECS:
                logger.warning("Unknown model %s, skipping", model_id)
                continue
            if mode not in LOCAL_MODES:
                logger.warning("Invalid mode %s for %s, skipping", mode, model_id)
                continue
            valid_requirements.append((model_id, mode))

        if not valid_requirements:
            logger.warning("No valid requirements to start")
            return

        logger.info("Starting model pool with: %s", valid_requirements)

        # Build allocation specs - each (model, mode) combo needs its own slot
        # Use "model_id:mode" as the allocation key
        allocation_specs = {}
        for model_id, mode in valid_requirements:
            spec = MODEL_SPECS[model_id]
            key = f"{model_id}:{mode.value}"
            allocation_specs[key] = {
                "model": spec["model"],
                "memory_gb": spec.get("memory_gb", 16),
                "tp": spec.get("tp", 1),
            }

        # Allocate GPU slots
        slots = self.allocator.allocate_slots(allocation_specs)

        if not slots:
            logger.warning("No GPU slots allocated, launching without GPU assignment")
            # Fallback: launch without specific GPU assignment
            for model_id, mode in valid_requirements:
                self._launch_model(model_id, mode, gpu_slot=None)
        else:
            # Launch on allocated slots
            for slot in slots:
                if slot.assigned_model:
                    # Parse "model_id:mode" back
                    model_id, mode_str = slot.assigned_model.rsplit(":", 1)
                    mode = ConnectionMode(mode_str)
                    self._launch_model(model_id, mode, gpu_slot=slot)

        # Wait for all to be healthy
        self._wait_all_healthy()

    def _launch_model(
        self,
        model_id: str,
        mode: ConnectionMode,
        gpu_slot: GPUSlot | None = None,
        worker_type: WorkerType = WorkerType.REGULAR,
        bootstrap_port: int | None = None,
        ib_device: str | None = None,
    ) -> ModelInstance:
        """Launch a model instance.

        Args:
            model_id: Model identifier from MODEL_SPECS.
            mode: Connection mode (HTTP or GRPC).
            gpu_slot: GPU slot assignment, or None for auto.
            worker_type: Worker type (REGULAR, PREFILL, or DECODE).
            bootstrap_port: Bootstrap port for prefill workers in PD mode.
            ib_device: InfiniBand device for PD disaggregation.

        Returns:
            The launched ModelInstance.
        """
        spec = get_model_spec(model_id)
        model_path = spec["model"]
        tp_size = spec.get("tp", 1)

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

        # PD disaggregation arguments
        if worker_type == WorkerType.PREFILL:
            cmd.extend(["--disaggregation-mode", "prefill"])
            if bootstrap_port:
                cmd.extend(["--disaggregation-bootstrap-port", str(bootstrap_port)])
            if ib_device:
                cmd.extend(["--disaggregation-ib-device", ib_device])
        elif worker_type == WorkerType.DECODE:
            cmd.extend(["--disaggregation-mode", "decode"])
            if ib_device:
                cmd.extend(["--disaggregation-ib-device", ib_device])

        # Build key based on worker type
        if worker_type == WorkerType.REGULAR:
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
            worker_type=worker_type,
            bootstrap_port=bootstrap_port,
        )
        self.instances[key] = instance
        return instance

    def _wait_all_healthy(self) -> None:
        """Wait for all model instances to become healthy."""
        start_time = time.time()
        pending = set(self.instances.keys())
        check_count = 0

        logger.info(
            "Waiting for %d workers to become healthy (timeout: %ds)...",
            len(pending),
            self._startup_timeout,
        )

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
                    # Read stderr for debugging
                    if instance.process.stderr:
                        stderr = instance.process.stderr.read()
                        if stderr:
                            logger.error("Stderr: %s", stderr.decode()[-2000:])
                    pending.discard(key)
                    continue

                # Check health
                if instance.health_check():
                    logger.info(
                        "[%.1fs] %s is healthy at %s (check #%d)",
                        elapsed,
                        key,
                        instance.base_url,
                        check_count,
                    )
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
            # Terminate failed instances
            for key in pending:
                self.instances[key].terminate()
                del self.instances[key]
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
    ) -> ModelInstance:
        """Get a model instance by model_id, mode, and worker_type.

        Args:
            model_id: The model ID (e.g., "llama-8b")
            mode: The mode (ConnectionMode.HTTP or ConnectionMode.GRPC, or string)
            worker_type: The worker type (REGULAR, PREFILL, DECODE). Defaults to REGULAR.

        Returns:
            ModelInstance for the requested model/mode/worker_type.

        Raises:
            KeyError: If model/mode/worker_type combination is not running.
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

        if key not in self.instances:
            raise KeyError(
                f"{key} not running. Available: {list(self.instances.keys())}"
            )

        instance = self.instances[key]

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

    def get_workers_by_type(
        self, model_id: str, worker_type: WorkerType
    ) -> list[ModelInstance]:
        """Get all workers of a specific type for a model.

        Args:
            model_id: The model ID.
            worker_type: The worker type to filter by.

        Returns:
            List of matching ModelInstance objects.
        """
        return [
            inst
            for inst in self.instances.values()
            if inst.model_id == model_id and inst.worker_type == worker_type
        ]

    def launch_pd_workers(
        self,
        model_id: str,
        num_prefill: int = 1,
        num_decode: int = 1,
        mode: ConnectionMode = ConnectionMode.HTTP,
        startup_timeout: int = DEFAULT_STARTUP_TIMEOUT,
    ) -> tuple[list[ModelInstance], list[ModelInstance]]:
        """Launch prefill and decode workers for PD disaggregation.

        Args:
            model_id: Model identifier from MODEL_SPECS.
            num_prefill: Number of prefill workers to launch. Defaults to 1.
            num_decode: Number of decode workers to launch. Defaults to 1.
            mode: Connection mode (HTTP or GRPC).
            startup_timeout: Timeout for workers to become healthy.

        Returns:
            Tuple of (prefill_instances, decode_instances).
        """
        self._startup_timeout = startup_timeout

        if model_id not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {model_id}")

        spec = get_model_spec(model_id)
        ib_device = detect_ib_device()
        if ib_device:
            logger.info("Detected InfiniBand device: %s", ib_device)

        # Build allocation specs for all PD workers
        # Each worker needs its own GPU slot
        allocation_specs = {}
        for i in range(num_prefill):
            key = f"{model_id}:{mode.value}:prefill_{i}"
            allocation_specs[key] = {
                "model": spec["model"],
                "memory_gb": spec.get("memory_gb", 16),
                "tp": spec.get("tp", 1),
            }
        for i in range(num_decode):
            key = f"{model_id}:{mode.value}:decode_{i}"
            allocation_specs[key] = {
                "model": spec["model"],
                "memory_gb": spec.get("memory_gb", 16),
                "tp": spec.get("tp", 1),
            }

        # Allocate GPU slots
        slots = self.allocator.allocate_slots(allocation_specs)
        slot_map = {slot.assigned_model: slot for slot in slots}

        if not slots:
            logger.warning(
                "No GPU slots allocated for PD workers, launching without GPU assignment"
            )

        prefill_instances: list[ModelInstance] = []
        decode_instances: list[ModelInstance] = []

        # Launch prefill workers
        for i in range(num_prefill):
            key = f"{model_id}:{mode.value}:prefill_{i}"
            gpu_slot = slot_map.get(key)
            bootstrap_port = get_open_port()
            instance = self._launch_model(
                model_id=model_id,
                mode=mode,
                gpu_slot=gpu_slot,
                worker_type=WorkerType.PREFILL,
                bootstrap_port=bootstrap_port,
                ib_device=ib_device,
            )
            prefill_instances.append(instance)

        # Launch decode workers
        for i in range(num_decode):
            key = f"{model_id}:{mode.value}:decode_{i}"
            gpu_slot = slot_map.get(key)
            instance = self._launch_model(
                model_id=model_id,
                mode=mode,
                gpu_slot=gpu_slot,
                worker_type=WorkerType.DECODE,
                ib_device=ib_device,
            )
            decode_instances.append(instance)

        # Wait for all to be healthy
        self._wait_all_healthy()

        return prefill_instances, decode_instances

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
        """Tear down all models."""
        logger.info("Shutting down model pool (%d instances)", len(self.instances))
        for instance in self.instances.values():
            instance.terminate()
        self.instances.clear()

    def __enter__(self) -> "ModelPool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
