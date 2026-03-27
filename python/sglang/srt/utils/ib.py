"""InfiniBand device discovery and validation utilities."""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _validate_ib_devices(devices_csv: str) -> str:
    """Validate a comma-separated list of IB device names against sysfs.

    Checks for empty input, duplicates, sysfs availability, and unknown devices.

    Returns:
        The normalized comma-separated device string.
    """
    devices = [d.strip() for d in devices_csv.split(",") if d.strip()]
    if not devices:
        raise ValueError(f"No valid IB devices in: {devices_csv!r}")

    # Deduplicate while preserving order
    unique_devices = list(dict.fromkeys(devices))
    if len(unique_devices) != len(devices):
        logger.warning(
            "Duplicate IB devices specified: %s. Deduplicating to: %s",
            devices_csv,
            ",".join(unique_devices),
        )
        devices = unique_devices

    ib_sysfs_path = "/sys/class/infiniband"
    if not os.path.isdir(ib_sysfs_path):
        raise RuntimeError(
            f"InfiniBand sysfs path not found: {ib_sysfs_path}. "
            "Please ensure InfiniBand drivers are installed."
        )

    available = set(os.listdir(ib_sysfs_path))
    if not available:
        raise RuntimeError(f"No IB devices found in {ib_sysfs_path}")

    invalid = [d for d in devices if d not in available]
    if invalid:
        raise ValueError(
            f"Invalid IB devices specified: {invalid}. "
            f"Available devices: {sorted(available)}"
        )

    return ",".join(devices)


def get_ib_devices_for_gpu(ib_device_str: Optional[str], gpu_id: int) -> Optional[str]:
    """
    Parse IB device string, validate against sysfs, and return the devices
    for a specific GPU ID.

    Supports the following formats:
    1. Comma-separated (same devices for all GPUs): "mlx5_0,mlx5_1"
    2. JSON GPU mapping: '{"0":"mlx5_0,mlx5_1","1":"mlx5_2,mlx5_3"}'
    3. JSON file path: path to a .json file containing the GPU mapping

    Args:
        ib_device_str: The original IB device string or path to JSON file
        gpu_id: The GPU ID to get devices for

    Returns:
        Validated IB devices string for the GPU, or None if input is None.
    """
    if ib_device_str is None or not ib_device_str.strip():
        return None

    ib_device_str = ib_device_str.strip()

    # Check if it's a JSON file first and load its content
    is_json_file = ib_device_str.endswith(".json")
    original_path = ib_device_str
    if is_json_file:
        try:
            if os.path.isfile(ib_device_str):
                with open(ib_device_str, "r") as f:
                    ib_device_str = f.read()
            else:
                raise RuntimeError(f"File {ib_device_str} does not exist.")
        except (IOError, OSError) as e:
            raise RuntimeError(f"Failed to read JSON file {ib_device_str}: {e}") from e

    # Try to parse as JSON (GPU mapping format)
    try:
        parsed_json = json.loads(ib_device_str)
        if isinstance(parsed_json, dict):
            gpu_mapping = {}
            for gpu_key, ib_devices in parsed_json.items():
                if (
                    isinstance(gpu_key, str)
                    and gpu_key.isdigit()
                    and isinstance(ib_devices, str)
                ):
                    gpu_mapping[int(gpu_key)] = ib_devices.strip()
                elif isinstance(gpu_key, int) and isinstance(ib_devices, str):
                    gpu_mapping[gpu_key] = ib_devices.strip()
                else:
                    raise ValueError(
                        "Invalid format: keys must be integers (or string "
                        "representations of integers) and values must be strings"
                    )

            if not gpu_mapping:
                raise ValueError("No valid GPU mappings found in JSON")

            if gpu_id not in gpu_mapping:
                raise ValueError(
                    f"No IB devices configured for GPU {gpu_id}. "
                    f"Available GPUs: {list(gpu_mapping.keys())}"
                )

            return _validate_ib_devices(gpu_mapping[gpu_id])

    except json.JSONDecodeError:
        if is_json_file:
            raise RuntimeError(
                f"Failed to parse JSON content from file {original_path}"
            )

    # Not JSON format — comma-separated, same devices for all GPUs
    return _validate_ib_devices(ib_device_str)
