from contextlib import contextmanager
from typing import Any, Dict, Optional

<<<<<<< HEAD
import sglang.srt.layers.moe.fused_moe_triton.fused_moe  # noqa
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    fused_experts,
    get_config_file_name,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import (
    FusedMoE,
    FusedMoEMethodBase,
=======
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    fused_experts,
    get_config_file_name,
    moe_align_block_size,
    try_get_optimal_moe_config,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import (
    FusedMoE,
>>>>>>> origin/main
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
<<<<<<< HEAD
    "FusedMoEMethodBase",
    "FusedMoeWeightScaleSupported",
    "override_config",
    "get_config",
    "fused_moe",
    "fused_experts",
    "get_config_file_name",
=======
    "FusedMoeWeightScaleSupported",
    "override_config",
    "get_config",
    "fused_experts",
    "get_config_file_name",
    "moe_align_block_size",
    "try_get_optimal_moe_config",
>>>>>>> origin/main
]
