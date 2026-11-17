"""Storage options shared by native DSV4 Decode startup and runtime attach."""

from __future__ import annotations

import os
from typing import Optional

from sglang.srt.mem_cache.dsv4_storage_namespace import (
    validate_dsv4_mooncake_transport,
    validate_dsv4_storage_namespace,
)


def validate_dsv4_decode_storage_config(
    *,
    backend: Optional[str],
    host_memory_mode: str,
    write_policy: str,
    prefetch_policy: str,
    extra_config: Optional[dict] = None,
    file_storage_path: Optional[str] = None,
    file_storage_path_override: Optional[str] = None,
) -> dict:
    """Validate before changing policies or constructing a storage backend.

    File objects have no self-describing checkpoint/layout header. The operator
    must dedicate the explicit directory to one compatible configuration.
    """
    if host_memory_mode != "cache":
        raise ValueError("DeepSeek-V4 Decode Host L2/File L3 requires cache mode.")
    if write_policy != "write_through":
        raise ValueError(
            "DeepSeek-V4 decode Host L2/File L3 requires "
            "--hicache-write-policy write_through."
        )
    if backend is None:
        return {}
    if backend not in ("file", "mooncake"):
        raise ValueError(
            "DeepSeek-V4 Decode storage supports File or namespaced Mooncake L3."
        )
    if prefetch_policy not in ("wait_complete", "timeout"):
        raise ValueError(
            "DeepSeek-V4 Decode File L3 requires wait_complete or timeout prefetch policy."
        )
    if extra_config is not None and not isinstance(extra_config, dict):
        raise ValueError("Decode L3 extra configuration must be an object.")
    extra = dict(extra_config or {})
    if backend == "mooncake":
        validate_dsv4_storage_namespace(extra)
        validate_dsv4_mooncake_transport(extra)
        return extra
    # Match UnifiedRadixCache startup: an explicit backend option precedes the
    # CLI fallback. The backend's environment override must name the same domain.
    path = (
        extra["file_storage_path"]
        if "file_storage_path" in extra
        else file_storage_path
    )
    if not isinstance(path, str) or not path.strip():
        raise ValueError(
            "DeepSeek-V4 Decode File L3 requires an explicit file_storage_path "
            "dedicated to its checkpoint, layout, pool schema, page geometry, "
            "topology and compute recipe."
        )
    if file_storage_path_override and os.path.abspath(
        file_storage_path_override
    ) != os.path.abspath(path):
        raise ValueError(
            "Decode File L3 environment storage directory differs from the explicit "
            "file_storage_path; use one compatible directory."
        )
    extra["file_storage_path"] = path
    return extra
