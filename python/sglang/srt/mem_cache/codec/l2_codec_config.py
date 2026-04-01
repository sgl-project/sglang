from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HiCacheL2CodecConfig:
    name: str
    ratio: float = 4.0
    level: int = 1
    params_path: Optional[str] = None


def _load_extra_config(storage_backend_extra_config: Optional[str]) -> dict:
    if not storage_backend_extra_config:
        return {}
    if storage_backend_extra_config.startswith("@"):
        path = storage_backend_extra_config[1:]
        ext = os.path.splitext(path)[1].lower()
        with open(path, "rb" if ext == ".toml" else "r") as f:
            if ext == ".json":
                return json.load(f)
            if ext == ".toml":
                import tomllib

                return tomllib.load(f)
            if ext in (".yaml", ".yml"):
                import yaml

                return yaml.safe_load(f)
            raise ValueError(f"Unsupported config file {path} (config format: {ext})")
    return json.loads(storage_backend_extra_config)


def parse_hicache_l2_codec_config(
    storage_backend_extra_config: Optional[str],
) -> Optional[HiCacheL2CodecConfig]:
    extra = _load_extra_config(storage_backend_extra_config)
    name = extra.get("l2_codec")
    if not name:
        return None
    ratio = float(extra.get("l2_codec_ratio", 4.0))
    level = int(extra.get("l2_codec_level", 1))
    params_path = extra.get("l2_codec_params_path") or extra.get("kvtc_params_path")
    return HiCacheL2CodecConfig(
        name=str(name), ratio=ratio, level=level, params_path=params_path
    )


