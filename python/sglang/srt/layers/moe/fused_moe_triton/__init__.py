from contextlib import contextmanager
from typing import Any, Dict, Optional

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
    get_config_file_name,
    try_get_optimal_moe_config,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)
from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import (
    moe_align_block_size,
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
    "FusedMoeWeightScaleSupported",
    "override_config",
    "get_config",
    "fused_experts",
    "get_config_file_name",
    "moe_align_block_size",
    "try_get_optimal_moe_config",
]
