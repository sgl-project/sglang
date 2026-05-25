from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

from sglang.srt.mem_cache.shared_hicache.plan import normalize_endpoint


SHARED_HICACHE_TRANSFER_BACKEND_CHOICES = ["auto", "mooncake"]


@dataclass(frozen=True)
class SharedHiCacheConfig:
    worker_id: int
    control_endpoint: Optional[str]
    timeout_secs: float
    transfer_backend: str


SharedHiCacheConfigInput = Union[str, Dict[str, Any], SharedHiCacheConfig]
SharedHiCacheConfigView = Union[SharedHiCacheConfig, Mapping[str, Any]]


def shared_hicache_config(server_args) -> Optional[SharedHiCacheConfigView]:
    config = getattr(server_args, "shared_hicache_config", None)
    if isinstance(config, SharedHiCacheConfig):
        return config
    if isinstance(config, Mapping):
        return config
    return None


def shared_hicache_config_value(server_args, key: str, default=None):
    config = shared_hicache_config(server_args)
    if config is None:
        return default
    if isinstance(config, SharedHiCacheConfig):
        return getattr(config, key, default)
    return config.get(key, default)


def shared_hicache_transfer_backend_name(server_args, default: str = "auto") -> str:
    return str(
        shared_hicache_config_value(server_args, "transfer_backend", default)
    ).lower()


def shared_hicache_timeout_secs(server_args, default: float = 1.0) -> float:
    return float(shared_hicache_config_value(server_args, "timeout_secs", default))


def _load_json_object_config(
    raw: Optional[SharedHiCacheConfigInput], arg_name: str
) -> Optional[Union[SharedHiCacheConfig, Dict[str, Any]]]:
    if raw is None:
        return None
    if isinstance(raw, SharedHiCacheConfig):
        return raw
    if isinstance(raw, dict):
        data = dict(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        if text.startswith("{"):
            data = json.loads(text)
        else:
            path = os.path.expanduser(os.path.expandvars(text))
            with open(path) as f:
                data = json.load(f)
    else:
        raise ValueError(f"{arg_name} must be a JSON object or path")

    if not isinstance(data, dict):
        raise ValueError(f"{arg_name} must be a JSON object")
    return data


def _normalize_endpoint(value: object, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{field_name} must be non-empty")
        return normalize_endpoint(value)
    raise ValueError(f"{field_name} must be a non-empty string")


def normalize_shared_hicache_server_config(
    *,
    enable_shared_hicache: bool,
    raw_config: Optional[SharedHiCacheConfigInput],
    worker_id: Optional[int],
    enable_hierarchical_cache: bool,
) -> tuple[bool, Optional[int], Optional[SharedHiCacheConfig]]:
    config_data = _load_json_object_config(raw_config, "--shared-hicache-config")
    if config_data is not None:
        enable_shared_hicache = True

    if not enable_shared_hicache:
        return False, worker_id, None

    if not enable_hierarchical_cache:
        raise ValueError("--enable-shared-hicache requires --enable-hierarchical-cache")
    if isinstance(config_data, SharedHiCacheConfig):
        return True, config_data.worker_id, config_data

    config: Dict[str, Any] = dict(config_data or {})
    if any(
        key in config
        for key in (
            "endpoint",
            "peer_endpoints",
            "static_peer_endpoints",
            "http_control",
        )
    ):
        raise ValueError(
            "shared_hicache_config static endpoint maps are not supported; put the "
            "local bind endpoint under control.endpoint and let each SharedHiCache "
            "plan provide source_endpoint"
        )
    if "fetch_workers" in config:
        raise ValueError(
            "shared_hicache_config.fetch_workers is not supported; use "
            "SGLANG_SHARED_HICACHE_FETCH_WORKERS"
        )
    if "transfer_parallelism" in config:
        raise ValueError(
            "shared_hicache_config.transfer_parallelism is not supported; use "
            "SGLANG_SHARED_HICACHE_TRANSFER_PARALLELISM"
        )

    control_config = config.get("control") or {}
    if not isinstance(control_config, dict):
        raise ValueError("shared_hicache_config.control must be a JSON object")
    if any(
        key in control_config
        for key in ("workers", "peer_endpoints", "static_peer_endpoints")
    ):
        raise ValueError(
            "shared_hicache_config.control static worker maps are not supported; "
            "use source_endpoint from each SharedHiCache plan"
        )

    transfer_config = config.get("transfer") or {}
    if not isinstance(transfer_config, dict):
        raise ValueError("shared_hicache_config.transfer must be a JSON object")
    if "parallelism" in transfer_config:
        raise ValueError(
            "shared_hicache_config.transfer.parallelism is not supported; use "
            "SGLANG_SHARED_HICACHE_TRANSFER_PARALLELISM"
        )

    if "worker_id" in config:
        worker_id = config["worker_id"]
    if worker_id is None:
        raise ValueError("--enable-shared-hicache requires --shared-hicache-worker-id")
    if not isinstance(worker_id, int) or isinstance(worker_id, bool) or worker_id < 0:
        raise ValueError("shared_hicache_worker_id must be a non-negative integer")

    if "control_backend" in config or "backend" in control_config:
        raise ValueError(
            "shared_hicache_config.control.backend is not supported; "
            "SharedHiCache plans carry the source endpoint directly"
        )

    transfer_backend = str(
        config.get("transfer_backend", transfer_config.get("backend", "auto"))
    ).lower()
    if transfer_backend not in SHARED_HICACHE_TRANSFER_BACKEND_CHOICES:
        raise ValueError(
            "shared_hicache_config.transfer_backend must be one of "
            f"{SHARED_HICACHE_TRANSFER_BACKEND_CHOICES}, got {transfer_backend!r}"
        )

    timeout_secs = config.get(
        "timeout_secs",
        transfer_config.get("timeout_secs", control_config.get("timeout_secs", 1.0)),
    )
    if not isinstance(timeout_secs, (int, float)) or isinstance(timeout_secs, bool):
        raise ValueError("shared_hicache_config.timeout_secs must be a positive number")
    timeout_secs = float(timeout_secs)
    if timeout_secs <= 0:
        raise ValueError("shared_hicache_config.timeout_secs must be > 0")

    return True, worker_id, SharedHiCacheConfig(
        worker_id=worker_id,
        control_endpoint=_normalize_endpoint(
            control_config.get("endpoint", config.get("control_endpoint")),
            "shared_hicache_config.control.endpoint",
        ),
        timeout_secs=timeout_secs,
        transfer_backend=transfer_backend,
    )
