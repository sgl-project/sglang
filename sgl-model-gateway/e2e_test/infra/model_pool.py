"""Model pool for managing pre-loaded models across GPUs."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    import openai

from .gpu_allocator import GPUAllocator, GPUSlot
from .model_specs import MODEL_SPECS, get_model_spec

logger = logging.getLogger(__name__)

# Default timeout for model startup (seconds)
DEFAULT_STARTUP_TIMEOUT = 300
# Health check interval (seconds)
HEALTH_CHECK_INTERVAL = 5
# Host for model servers
DEFAULT_HOST = "127.0.0.1"


@dataclass
class ModelInstance:
    """A running model instance."""

    model_id: str
    model_path: str
    base_url: str
    process: subprocess.Popen
    gpu_slot: GPUSlot
    grpc_mode: bool = False

    def is_alive(self) -> bool:
        """Check if the process is still running."""
        return self.process.poll() is None

    def health_check(self, timeout: float = 5.0) -> bool:
        """Check if the model server is healthy via HTTP."""
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=timeout)
            return resp.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    def terminate(self, timeout: float = 10.0) -> None:
        """Terminate the model server process."""
        if self.process.poll() is not None:
            return  # Already terminated

        logger.info("Terminating model %s (PID %d)", self.model_id, self.process.pid)

        # Try graceful shutdown first
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Model %s did not terminate, killing", self.model_id)
            self.process.kill()
            self.process.wait()


class ModelPool:
    """Manages a pool of pre-loaded models across GPUs."""

    def __init__(self, allocator: GPUAllocator | None = None):
        """Initialize the model pool.

        Args:
            allocator: GPU allocator to use. If None, creates a new one.
        """
        self.allocator = allocator or GPUAllocator()
        self.instances: dict[str, ModelInstance] = {}
        self._startup_timeout = DEFAULT_STARTUP_TIMEOUT

    def startup(
        self,
        model_ids: list[str] | None = None,
        grpc_mode: bool = False,
        startup_timeout: int = DEFAULT_STARTUP_TIMEOUT,
    ) -> None:
        """Spin up models in parallel on assigned GPU slots.

        Args:
            model_ids: List of model IDs to start. If None, starts all in MODEL_SPECS.
            grpc_mode: If True, launch workers in gRPC mode.
            startup_timeout: Timeout in seconds for each model to become healthy.
        """
        self._startup_timeout = startup_timeout

        # Determine which models to start
        if model_ids is None:
            model_ids = list(MODEL_SPECS.keys())

        # Filter to models we have specs for
        specs_to_start = {
            mid: MODEL_SPECS[mid] for mid in model_ids if mid in MODEL_SPECS
        }

        if not specs_to_start:
            logger.warning("No valid model specs to start")
            return

        # Allocate GPU slots
        slots = self.allocator.allocate_slots(specs_to_start)

        if not slots:
            logger.warning("No GPU slots allocated")
            return

        logger.info(self.allocator.summary())

        # Launch all models in parallel
        for slot in slots:
            if slot.assigned_model:
                self._launch_model(slot, grpc_mode=grpc_mode)

        # Wait for all to be healthy
        self._wait_all_healthy()

    def _launch_model(self, slot: GPUSlot, grpc_mode: bool = False) -> None:
        """Launch a model on the given GPU slot."""
        model_id = slot.assigned_model
        if not model_id:
            return

        spec = get_model_spec(model_id)
        model_path = spec["model"]
        tp_size = spec.get("tp", 1)
        port = slot.port

        # Build environment with CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = slot.cuda_visible_devices()

        # Build command
        cmd = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--port",
            str(port),
            "--tp-size",
            str(tp_size),
            "--log-level",
            "warning",
        ]

        if grpc_mode:
            cmd.append("--grpc-mode")

        logger.info(
            "Launching %s on GPUs %s port %d: %s",
            model_id,
            slot.gpu_ids,
            port,
            " ".join(cmd),
        )

        # Start the process
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Use process group for clean shutdown
            start_new_session=True,
        )

        base_url = f"http://{DEFAULT_HOST}:{port}"
        instance = ModelInstance(
            model_id=model_id,
            model_path=model_path,
            base_url=base_url,
            process=proc,
            gpu_slot=slot,
            grpc_mode=grpc_mode,
        )
        self.instances[model_id] = instance

    def _wait_all_healthy(self) -> None:
        """Wait for all model instances to become healthy."""
        start_time = time.time()
        pending = set(self.instances.keys())

        while pending and (time.time() - start_time) < self._startup_timeout:
            for model_id in list(pending):
                instance = self.instances[model_id]

                # Check if process died
                if not instance.is_alive():
                    logger.error(
                        "Model %s (PID %d) died during startup",
                        model_id,
                        instance.process.pid,
                    )
                    # Read stderr for debugging
                    if instance.process.stderr:
                        stderr = instance.process.stderr.read()
                        if stderr:
                            logger.error("Stderr: %s", stderr.decode()[-2000:])
                    pending.discard(model_id)
                    continue

                # Check health
                if instance.health_check():
                    logger.info(
                        "Model %s is healthy at %s", model_id, instance.base_url
                    )
                    pending.discard(model_id)

            if pending:
                time.sleep(HEALTH_CHECK_INTERVAL)

        if pending:
            logger.error(
                "Models failed to start within %ds: %s",
                self._startup_timeout,
                pending,
            )
            # Terminate failed instances
            for model_id in pending:
                self.instances[model_id].terminate()
                del self.instances[model_id]

    def get_client(self, model_id: str) -> "openai.OpenAI":
        """Get OpenAI client for a specific model.

        Args:
            model_id: The model ID to get a client for.

        Returns:
            OpenAI client configured for this model.

        Raises:
            KeyError: If model is not running.
        """
        import openai

        if model_id not in self.instances:
            raise KeyError(
                f"Model {model_id} not running. Available: {list(self.instances.keys())}"
            )

        instance = self.instances[model_id]
        return openai.OpenAI(
            base_url=f"{instance.base_url}/v1",
            api_key="not-used",
        )

    def get_base_url(self, model_id: str) -> str:
        """Get the base URL for a specific model."""
        if model_id not in self.instances:
            raise KeyError(
                f"Model {model_id} not running. Available: {list(self.instances.keys())}"
            )
        return self.instances[model_id].base_url

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
