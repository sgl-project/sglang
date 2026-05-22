import json
import logging
import os
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _split_devices(devices_str: str) -> List[str]:
    return [d.strip() for d in devices_str.split(",") if d.strip()]


def _parse_json_mapping(mapping_str: str) -> Dict[int, List[str]]:
    parsed_json = json.loads(mapping_str)
    if not isinstance(parsed_json, dict):
        raise ValueError("IB device JSON mapping must be a dictionary")

    gpu_mapping: Dict[int, List[str]] = {}
    for gpu_key, ib_devices in parsed_json.items():
        if isinstance(gpu_key, str) and gpu_key.isdigit():
            gpu_id = int(gpu_key)
        elif isinstance(gpu_key, int):
            gpu_id = gpu_key
        else:
            raise ValueError(
                "Invalid IB device JSON mapping: keys must be integers "
                "or string representations of integers"
            )

        if not isinstance(ib_devices, str):
            raise ValueError("Invalid IB device JSON mapping: values must be strings")

        devices = _split_devices(ib_devices)
        if len(devices) == 0:
            raise ValueError(f"No valid IB devices specified for GPU {gpu_id}")
        gpu_mapping[gpu_id] = devices

    if not gpu_mapping:
        raise ValueError("No valid GPU mappings found in IB device JSON")
    return gpu_mapping


def validate_ib_devices(device_str: Optional[str]) -> Optional[str]:
    """
    Validate IB devices before passing to mooncake.

    Args:
        device_str: Comma-separated IB device names (e.g., "mlx5_0,mlx5_1"),
            a JSON GPU-to-IB mapping, or a path to a JSON mapping file.

    Returns:
        Normalized comma-separated string of validated device names, or None if input is None.
    """
    if device_str is None:
        logger.warning(
            "No IB devices specified for Mooncake backend, falling back to auto discovery."
        )
        return None

    device_str = device_str.strip()

    json_mapping: Optional[Dict[int, List[str]]] = None
    json_mapping_path = device_str if device_str.endswith(".json") else None
    if json_mapping_path is not None:
        if not os.path.isfile(json_mapping_path):
            raise RuntimeError(f"File {json_mapping_path} does not exist.")
        try:
            with open(json_mapping_path, "r") as f:
                json_mapping = _parse_json_mapping(f.read())
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse JSON content from file {json_mapping_path}"
            ) from e
    else:
        try:
            json_mapping = _parse_json_mapping(device_str)
        except (json.JSONDecodeError, ValueError):
            json_mapping = None

    # Strip whitespace from device names
    devices = (
        [
            device
            for mapping_devices in json_mapping.values()
            for device in mapping_devices
        ]
        if json_mapping is not None
        else _split_devices(device_str)
    )
    if len(devices) == 0:
        raise ValueError("No valid IB devices specified")

    if json_mapping is None:
        # Deduplicate while preserving order for the legacy comma-separated format.
        unique_devices = list(dict.fromkeys(devices))
        if len(unique_devices) != len(devices):
            logger.warning(
                "Duplicate IB devices specified: %s. Deduplicating to: %s",
                device_str,
                ",".join(unique_devices),
            )
            devices = unique_devices

    # Get available IB devices from sysfs
    ib_sysfs_path = "/sys/class/infiniband"
    if not os.path.isdir(ib_sysfs_path):
        raise RuntimeError(
            f"InfiniBand sysfs path not found: {ib_sysfs_path}. "
            "Please ensure InfiniBand drivers are installed."
        )

    available_devices = set(os.listdir(ib_sysfs_path))
    if len(available_devices) == 0:
        raise RuntimeError(f"No IB devices found in {ib_sysfs_path}")

    # Check for invalid devices
    invalid_devices = [d for d in devices if d not in available_devices]
    if len(invalid_devices) != 0:
        raise ValueError(
            f"Invalid IB devices specified: {invalid_devices}. "
            f"Available devices: {sorted(available_devices)}"
        )

    return device_str if json_mapping is not None else ",".join(devices)


def validate_mooncake_ib_device(server_args: "ServerArgs") -> None:
    """Validate disaggregation IB devices when a Mooncake transfer backend is used."""
    if (
        server_args.disaggregation_transfer_backend == "mooncake"
        and server_args.disaggregation_mode in ("prefill", "decode")
    ) or server_args.encoder_transfer_backend == "mooncake":
        server_args.disaggregation_ib_device = validate_ib_devices(
            server_args.disaggregation_ib_device
        )


def validate_elastic_ep_mooncake_ib_device(server_args: "ServerArgs") -> None:
    """Validate Elastic EP IB devices when Mooncake is used."""
    if server_args.elastic_ep_backend == "mooncake":
        server_args.mooncake_ib_device = validate_ib_devices(
            server_args.mooncake_ib_device
        )
