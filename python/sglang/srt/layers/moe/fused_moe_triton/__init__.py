from contextlib import contextmanager
from typing import Any, Dict, Optional

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    fused_experts,
    fused_moe,
    get_config_file_name,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)

_config: Optional[Dict[str, Any]] = None


@contextmanager
def override_config(config):
    global _config
    old_config = _config
    _config = config
    yield
    _config = old_config


def get_config() -> Optional[Dict[str, Any]]:
    return _config


__all__ = [
    "FusedMoE",
    "FusedMoEMethodBase",
    "FusedMoeWeightScaleSupported",
    "override_config",
    "get_config",
    "fused_moe",
    "fused_experts",
    "get_config_file_name",
]
