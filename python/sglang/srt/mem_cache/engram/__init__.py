"""HiCache storage for immutable Engram tables."""

import json

from sglang.srt.mem_cache.hicache_storage import HiCacheEngramStorage


def _parse_config(config: str | None, mode: str = "local") -> dict:
    """Parse and validate the HiCache Engram storage options."""
    try:
        options = {} if config is None else json.loads(config)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("--hi-engram-config must be a JSON object") from exc
    if not isinstance(options, dict):
        raise ValueError("--hi-engram-config must be a JSON object")
    allowed = {"local_dir"}
    if mode == "remote":
        allowed |= {
            "shard_id",
            "num_shards",
            "master_addr",
            "session_id",
            "caller_id",
            "max_lookup_rows",
            "rdma_device",
            "read_policy",
            "opt_max_rows",
            "native_qps",
            "opt_qps",
            "timeout",
            "num_callers",
        }
    unknown = options.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown --hi-engram-config options: {sorted(unknown)}")
    options.setdefault("local_dir", "/dev/shm/sglang-engram")
    if not isinstance(options["local_dir"], str) or not options["local_dir"]:
        raise ValueError("--hi-engram-config local_dir must be a nonempty string")
    if mode == "remote":
        required = {
            "shard_id",
            "num_shards",
            "master_addr",
            "session_id",
            "max_lookup_rows",
        }
        if required - options.keys():
            raise ValueError(
                f"Missing remote Engram options: {sorted(required - options.keys())}"
            )
        for name in ("shard_id", "num_shards", "max_lookup_rows"):
            if type(options[name]) is not int:
                raise ValueError(f"Remote Engram {name} must be an integer")
        if (
            not 0 <= options["shard_id"] < options["num_shards"]
            or options["max_lookup_rows"] < 1
        ):
            raise ValueError("Invalid node shard ID/count or lookup capacity")
        for name in ("master_addr", "session_id"):
            if not isinstance(options[name], str) or not options[name]:
                raise ValueError(f"Remote Engram {name} must be a nonempty string")
    return options


def create_storage(
    mode: str, config: str | None = None, *, head_bounds=None, namespace=None
) -> HiCacheEngramStorage:
    """Create row storage without requiring a KV cache or host KV pool."""
    if mode in ("local", "remote"):
        from sglang.srt.mem_cache.storage.mooncake_engram import (
            MooncakeEngramStorage,
        )

        options = _parse_config(config, mode)
        if mode == "remote":
            local_dir = options.pop("local_dir")
            return MooncakeEngramStorage(
                local_dir=local_dir,
                remote=options,
                head_bounds=head_bounds,
                namespace=namespace,
            )
        return MooncakeEngramStorage(**options)
    raise ValueError(f"Unknown Engram storage mode: {mode!r}")
