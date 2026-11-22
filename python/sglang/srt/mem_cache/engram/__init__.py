"""HiCache storage for immutable Engram tables."""

import json
import math

from sglang.srt.mem_cache.hicache_storage import HiCacheEngramStorage


def _parse_config(config: str | None) -> dict:
    """Validate user options; worker identity and caller counts are runtime-owned."""
    try:
        options = {} if config is None else json.loads(config)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("--hi-engram-config must be a JSON object") from exc
    if not isinstance(options, dict):
        raise ValueError("--hi-engram-config must be a JSON object")
    backend = options.get("backend", "local")
    if backend == "remote":
        raise NotImplementedError("Remote backend name is reserved; select deepep_rdma")
    if backend not in ("local", "deepep_rdma"):
        raise ValueError(f"Unknown Engram storage backend: {backend!r}")
    allowed = (
        {"backend", "local_dir"}
        if backend == "local"
        else {
            "backend",
            "assignment",
            "group_size",
            "group_id",
            "shard_id",
            "rdma_device",
            "timeout",
            "max_lookup_rows",
        }
    )
    unknown = options.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown --hi-engram-config options: {sorted(unknown)}")
    if backend == "local":
        options.setdefault("local_dir", "/dev/shm/sglang-engram")
        if not isinstance(options["local_dir"], str) or not options["local_dir"]:
            raise ValueError("--hi-engram-config local_dir must be a nonempty string")
        return options
    if options.get("assignment") not in ("auto", "manual"):
        raise ValueError("deepep_rdma requires assignment='auto' or 'manual'")
    size = options.get("group_size")
    if type(size) is not int or not 1 <= size <= 64:
        raise ValueError("group_size must be an integer in [1,64]")
    if options["assignment"] == "auto":
        if "group_id" in options or "shard_id" in options:
            raise ValueError("Auto assignment does not accept group_id/shard_id")
    elif (
        not isinstance(options.get("group_id"), str)
        or not options["group_id"]
        or type(options.get("shard_id")) is not int
        or not 0 <= options["shard_id"] < size
    ):
        raise ValueError("Manual assignment requires group_id and a valid shard_id")
    if "max_lookup_rows" in options and (
        type(options["max_lookup_rows"]) is not int or options["max_lookup_rows"] < 1
    ):
        raise ValueError("max_lookup_rows must be a positive integer")
    if "timeout" in options and (
        isinstance(options["timeout"], bool)
        or not isinstance(options["timeout"], (int, float))
        or not math.isfinite(options["timeout"])
        or options["timeout"] <= 0
    ):
        raise ValueError("timeout must be finite and positive")
    if "rdma_device" in options:
        devices = options["rdma_device"]
        devices = devices if isinstance(devices, list) else [devices]
        if not devices or any(
            not isinstance(device, str) or not device.strip() for device in devices
        ):
            raise ValueError("rdma_device must be a nonempty string or device list")
    return options


def create_storage(
    config: str | None = None, *, head_bounds=None, namespace=None, row_bytes=None
) -> HiCacheEngramStorage:
    """Create Engram storage without allocating a KV cache or a host KV pool."""
    from sglang.srt.mem_cache.storage.mooncake_engram import MooncakeEngramStorage

    options = _parse_config(config)
    backend = options.pop("backend", "local")
    if backend == "local":
        return MooncakeEngramStorage(**options)
    return MooncakeEngramStorage(
        remote=options,
        head_bounds=head_bounds,
        namespace=namespace,
        row_bytes=row_bytes,
    )
