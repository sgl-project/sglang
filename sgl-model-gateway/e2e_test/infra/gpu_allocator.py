"""GPU detection and slot allocation for parallel test execution."""

from __future__ import annotations

import logging
import os
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import nvidia-ml-py for GPU detection
try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logger.debug("nvidia-ml-py not available, GPU detection will be limited")


@contextmanager
def nvml_context():
    """Context manager for NVML initialization/shutdown.

    Usage:
        with nvml_context():
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            ...
    """
    if not NVML_AVAILABLE:
        yield
        return

    try:
        pynvml.nvmlInit()
        yield
    finally:
        pynvml.nvmlShutdown()


@dataclass
class GPUInfo:
    """Information about a single GPU."""

    id: int
    name: str
    memory_mb: int

    @property
    def memory_gb(self) -> float:
        return self.memory_mb / 1024


@dataclass
class GPUSlot:
    """A slot representing one or more GPUs allocated for a model."""

    gpu_ids: list[int]
    total_memory_mb: int
    assigned_model: str | None = None
    port: int | None = None

    @property
    def total_memory_gb(self) -> float:
        return self.total_memory_mb / 1024

    def cuda_visible_devices(self) -> str:
        """Return CUDA_VISIBLE_DEVICES string for this slot."""
        return ",".join(str(g) for g in self.gpu_ids)


def get_open_port() -> int:
    """Get an available port by binding to port 0 and reading the assigned port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def get_physical_device_indices(devices: list[int]) -> list[int]:
    """Map logical device indices to physical indices based on CUDA_VISIBLE_DEVICES."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return devices

    visible_indices = [int(x) for x in visible_devices.split(",")]
    index_mapping = {i: physical for i, physical in enumerate(visible_indices)}
    return [index_mapping[i] for i in devices if i in index_mapping]


def get_gpu_memory_usage(device_id: int) -> tuple[float, float]:
    """Get GPU memory usage in GB (used, total).

    Args:
        device_id: Physical GPU device ID

    Returns:
        Tuple of (used_gb, total_gb)
    """
    if not NVML_AVAILABLE:
        return (0.0, 0.0)

    with nvml_context():
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return (mem_info.used / (1024**3), mem_info.total / (1024**3))


def wait_for_gpu_memory_to_clear(
    *,
    devices: list[int],
    threshold_bytes: int | None = None,
    threshold_ratio: float | None = None,
    timeout_s: float = 120,
) -> None:
    """Wait for GPU memory to be freed below a threshold.

    Args:
        devices: List of logical GPU device IDs to check
        threshold_bytes: Memory threshold in bytes (used <= threshold)
        threshold_ratio: Memory threshold as ratio (used/total <= ratio)
        timeout_s: Timeout in seconds

    Raises:
        ValueError: If memory doesn't clear within timeout
    """
    if not NVML_AVAILABLE:
        logger.warning("nvidia-ml-py not available, skipping memory wait")
        return

    if threshold_bytes is None and threshold_ratio is None:
        raise ValueError("Must specify threshold_bytes or threshold_ratio")

    physical_devices = get_physical_device_indices(devices)
    start_time = time.time()

    with nvml_context():
        while True:
            output: dict[int, str] = {}
            output_raw: dict[int, tuple[float, float]] = {}

            for device in physical_devices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gb_used = mem_info.used / (1024**3)
                gb_total = mem_info.total / (1024**3)
                output_raw[device] = (gb_used, gb_total)
                output[device] = f"{gb_used:.02f}/{gb_total:.02f}"

            logger.debug(
                "GPU memory used/total (GiB): %s",
                " ".join(f"{k}={v}" for k, v in output.items()),
            )

            if threshold_bytes is not None:

                def is_free(used: float, total: float) -> bool:
                    return used <= threshold_bytes / (1024**3)

                threshold_desc = f"{threshold_bytes / (1024**3):.1f} GiB"
            else:

                def is_free(used: float, total: float) -> bool:
                    return used / total <= threshold_ratio  # type: ignore[operator]

                threshold_desc = f"{threshold_ratio:.2%}"  # type: ignore[str-format]

            dur_s = time.time() - start_time
            if all(is_free(used, total) for used, total in output_raw.values()):
                logger.info(
                    "GPU memory cleared on devices %s (threshold=%s) in %.1fs",
                    devices,
                    threshold_desc,
                    dur_s,
                )
                return

            if dur_s >= timeout_s:
                raise ValueError(
                    f"GPU memory on devices {devices} not freed after {dur_s:.1f}s "
                    f"(threshold={threshold_desc})"
                )

            time.sleep(5)


class GPUAllocator:
    """Detects GPUs and assigns them to model slots using bin-packing."""

    def __init__(self, gpus: list[GPUInfo] | None = None):
        """Initialize the allocator.

        Args:
            gpus: Optional list of GPUs. If None, auto-detects via nvidia-ml-py.
        """
        self.gpus = gpus if gpus is not None else self._detect_gpus()
        self.slots: list[GPUSlot] = []
        self._used_gpus: set[int] = set()  # Track GPUs used across all allocations

    def _detect_gpus(self) -> list[GPUInfo]:
        """Auto-detect available GPUs via nvidia-ml-py (NVML)."""
        if not NVML_AVAILABLE:
            logger.warning("nvidia-ml-py not available - no GPUs detected")
            return []

        # Check for CUDA_VISIBLE_DEVICES restriction
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        allowed_ids: set[int] | None = None
        if visible_devices:
            allowed_ids = set(int(x) for x in visible_devices.split(",") if x.strip())

        try:
            with nvml_context():
                device_count = pynvml.nvmlDeviceGetCount()

                gpus = []
                for idx in range(device_count):
                    # Skip GPUs not in CUDA_VISIBLE_DEVICES if set
                    if allowed_ids is not None and idx not in allowed_ids:
                        continue

                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    name = pynvml.nvmlDeviceGetName(handle)
                    # Handle bytes vs string return type (varies by pynvml version)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    # Convert bytes to MB
                    memory_mb = mem_info.total // (1024 * 1024)

                    gpus.append(GPUInfo(idx, name, memory_mb))

            logger.info("Detected %d GPUs: %s", len(gpus), [g.name for g in gpus])
            return gpus

        except pynvml.NVMLError as e:
            logger.warning("NVML error during GPU detection: %s", e)
            return []
        except Exception as e:
            logger.warning("Failed to detect GPUs: %s", e)
            return []

    def allocate_slots(self, model_specs: dict[str, dict]) -> list[GPUSlot]:
        """Allocate GPU slots based on model memory requirements.

        Uses a first-fit decreasing bin-packing algorithm:
        1. Sort models by memory requirement (largest first)
        2. For each model, find the first GPU(s) that can fit it
        3. For multi-GPU models, find consecutive GPUs

        Note: This method tracks used GPUs across multiple calls, so subsequent
        allocations will use different GPUs than previous ones.

        Args:
            model_specs: Dict of model_id -> spec dict with 'memory_gb' and 'tp' keys

        Returns:
            List of GPUSlots with assigned models (only the newly allocated slots)
        """
        if not self.gpus:
            logger.warning("No GPUs available for allocation")
            return []

        # Sort models by memory requirement (largest first for better packing)
        sorted_models = sorted(
            model_specs.items(),
            key=lambda x: x[1].get("memory_gb", 0),
            reverse=True,
        )

        # Track new slots allocated in this call
        new_slots: list[GPUSlot] = []

        for model_id, spec in sorted_models:
            memory_gb = spec.get("memory_gb", 16)
            tp_size = spec.get("tp", 1)

            # Find available GPUs (not used by any previous allocation)
            available = [g for g in self.gpus if g.id not in self._used_gpus]

            if tp_size == 1:
                # Single GPU - find one with enough memory
                for gpu in available:
                    if gpu.memory_gb >= memory_gb:
                        slot = GPUSlot(
                            gpu_ids=[gpu.id],
                            total_memory_mb=gpu.memory_mb,
                            assigned_model=model_id,
                            port=get_open_port(),
                        )
                        new_slots.append(slot)
                        self._used_gpus.add(gpu.id)
                        logger.info(
                            "Allocated GPU %d (%s, %.1fGB) for %s",
                            gpu.id,
                            gpu.name,
                            gpu.memory_gb,
                            model_id,
                        )
                        break
                else:
                    logger.warning(
                        "No GPU with %.1fGB available for %s (used: %s)",
                        memory_gb,
                        model_id,
                        self._used_gpus,
                    )
            else:
                # Multi-GPU - find consecutive GPUs with enough total memory
                # Sort available by ID for consecutive allocation
                available_sorted = sorted(available, key=lambda g: g.id)

                for i in range(len(available_sorted) - tp_size + 1):
                    candidate_gpus = available_sorted[i : i + tp_size]
                    total_mem = sum(g.memory_mb for g in candidate_gpus)

                    if total_mem >= memory_gb * 1024:
                        gpu_ids = [g.id for g in candidate_gpus]
                        slot = GPUSlot(
                            gpu_ids=gpu_ids,
                            total_memory_mb=total_mem,
                            assigned_model=model_id,
                            port=get_open_port(),
                        )
                        new_slots.append(slot)
                        self._used_gpus.update(gpu_ids)
                        logger.info(
                            "Allocated GPUs %s (%.1fGB total) for %s (tp=%d)",
                            gpu_ids,
                            total_mem / 1024,
                            model_id,
                            tp_size,
                        )
                        break
                else:
                    logger.warning(
                        "No %d consecutive GPUs with %.1fGB available for %s (used: %s)",
                        tp_size,
                        memory_gb,
                        model_id,
                        self._used_gpus,
                    )

        # Add new slots to existing slots list
        self.slots.extend(new_slots)
        return new_slots

    def get_slot_for_model(self, model_id: str) -> GPUSlot | None:
        """Get the slot assigned to a specific model."""
        for slot in self.slots:
            if slot.assigned_model == model_id:
                return slot
        return None

    def release_gpus(self, gpu_ids: list[int]) -> None:
        """Release GPUs back to the available pool.

        Args:
            gpu_ids: List of GPU IDs to release.
        """
        for gpu_id in gpu_ids:
            self._used_gpus.discard(gpu_id)
        # Remove slots that used these GPUs
        self.slots = [s for s in self.slots if not any(g in gpu_ids for g in s.gpu_ids)]
        logger.info("Released GPUs %s, now used: %s", gpu_ids, self._used_gpus)

    def summary(self) -> str:
        """Return a summary of GPU allocations."""
        lines = ["GPU Allocation Summary:"]
        lines.append(f"  Total GPUs: {len(self.gpus)}")
        lines.append(f"  Used GPUs: {sorted(self._used_gpus)}")
        lines.append(f"  Allocated Slots: {len(self.slots)}")
        for slot in self.slots:
            lines.append(
                f"    - {slot.assigned_model}: GPUs {slot.gpu_ids} "
                f"({slot.total_memory_gb:.1f}GB) port={slot.port}"
            )
        return "\n".join(lines)
