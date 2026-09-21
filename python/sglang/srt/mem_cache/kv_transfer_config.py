# SPDX-License-Identifier: Apache-2.0
"""Serializable configuration for an out-of-tree SGLang KV connector.

Keep this module independent of cache implementations and accelerator imports:
argument validation runs in the launcher, before scheduler workers are spawned.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KVTransferConfig:
    kv_connector: str
    kv_connector_module_path: str
    kv_role: str = "kv_both"
    kv_connector_extra_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> KVTransferConfig:
        if not isinstance(value, dict):
            raise ValueError("--kv-transfer-config must be a JSON object")
        unknown = value.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown --kv-transfer-config fields: {sorted(unknown)}")
        for name in ("kv_connector", "kv_connector_module_path"):
            item = value.get(name)
            if not isinstance(item, str) or not item or item != item.strip():
                raise ValueError(f"--kv-transfer-config requires a non-empty {name}")
            parts = item.split(".") if name.endswith("module_path") else [item]
            if not all(part.isidentifier() for part in parts):
                raise ValueError(f"Invalid --kv-transfer-config {name}: {item!r}")
        role = value.get("kv_role", "kv_both")
        if role not in ("kv_producer", "kv_consumer", "kv_both"):
            raise ValueError(
                "kv_role must be 'kv_producer', 'kv_consumer', or 'kv_both'"
            )
        extra = value.get("kv_connector_extra_config", {})
        if not isinstance(extra, dict):
            raise ValueError("kv_connector_extra_config must be a JSON object")
        return cls(
            kv_connector=value["kv_connector"],
            kv_connector_module_path=value["kv_connector_module_path"],
            kv_role=role,
            kv_connector_extra_config=copy.deepcopy(extra),
        )


def validate_kv_transfer_config(args: Any) -> None:
    """Reject ambiguous cache ownership before importing any provider code."""
    if args.kv_transfer_config is None:
        return
    KVTransferConfig.from_dict(args.kv_transfer_config)
    for name in (
        "radix_cache_backend",
        "enable_lmcache",
        "enable_flexkv",
        "enable_hierarchical_cache",
        "hicache_storage_backend",
        "enable_unified_cache_external_linker",
        "disable_radix_cache",
        "enable_hisparse",
    ):
        if getattr(args, name):
            raise ValueError(
                f"--kv-transfer-config cannot be combined with --{name.replace('_', '-')}"
            )
    if args.disaggregation_mode != "null":
        raise ValueError("--kv-transfer-config currently requires colocated serving")
    if args.enable_streaming_session:
        raise ValueError(
            "--kv-transfer-config does not yet support --enable-streaming-session"
        )
