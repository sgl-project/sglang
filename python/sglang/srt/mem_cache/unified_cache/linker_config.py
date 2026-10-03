"""Serializable configuration for an explicitly selected external linker.

Keep validation free of backend and device imports: argument parsing also runs
in the frontend, while the plugin is imported only in the cache-owning worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UnifiedCacheLinkerConfig:
    linker: str
    linker_module_path: str
    linker_extra_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> UnifiedCacheLinkerConfig:
        flag = "--unified-cache-external-linker-config"
        if not isinstance(value, dict):
            raise ValueError(f"{flag} must be a JSON object.")
        unknown = value.keys() - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"{flag}: unknown fields: {sorted(unknown)}")
        for name in ("linker", "linker_module_path"):
            entry = value.get(name)
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(f"{flag}: {name} must be a non-empty string.")
        extra = value.get("linker_extra_config", {})
        if not isinstance(extra, dict):
            raise ValueError(f"{flag}: linker_extra_config must be a JSON object.")
        return cls(value["linker"], value["linker_module_path"], dict(extra))
