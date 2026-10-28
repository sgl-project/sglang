"""HiCache storage for immutable Engram tables."""

import json

from sglang.srt.mem_cache.hicache_storage import HiCacheEngramStorage


def _parse_config(config: str | None) -> dict:
    """Parse and validate the HiCache Engram storage options."""
    try:
        options = {} if config is None else json.loads(config)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("--hi-engram-config must be a JSON object") from exc
    if not isinstance(options, dict):
        raise ValueError("--hi-engram-config must be a JSON object")
    unknown = options.keys() - {"local_dir"}
    if unknown:
        raise ValueError(f"Unknown --hi-engram-config options: {sorted(unknown)}")
    options.setdefault("local_dir", "/dev/shm/sglang-engram")
    if not isinstance(options["local_dir"], str) or not options["local_dir"]:
        raise ValueError("--hi-engram-config local_dir must be a nonempty string")
    return options


def create_storage(mode: str, config: str | None = None) -> HiCacheEngramStorage:
    """Create row storage without requiring a KV cache or host KV pool."""
    if mode == "remote":
        raise NotImplementedError("Remote HiCache Engram storage is not supported yet")
    if mode == "local":
        from sglang.srt.mem_cache.storage.mooncake_engram import (
            MooncakeEngramStorage,
        )

        return MooncakeEngramStorage(**_parse_config(config))
    raise ValueError(f"Unknown Engram storage mode: {mode!r}")
