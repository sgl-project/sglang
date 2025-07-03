from contextlib import contextmanager
from typing import Any, Dict, Optional

# Import only what we need without creating circular dependencies
# We'll use lazy imports in the __all__ section

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


# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "FusedMoE":
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        return FusedMoE
    elif name == "FusedMoEMethodBase":
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoEMethodBase
        return FusedMoEMethodBase
    elif name == "FusedMoeWeightScaleSupported":
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoeWeightScaleSupported
        return FusedMoeWeightScaleSupported
    elif name == "fused_experts":
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
        return fused_experts
    elif name == "get_config_file_name":
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import get_config_file_name
        return get_config_file_name
    elif name == "fused_moe":
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
        return fused_moe
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

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
