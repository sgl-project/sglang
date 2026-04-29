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


@contextmanager
def override_split_config(up_cfg: Dict[str, Any], down_cfg: Dict[str, Any]):
    """Force ``try_get_optimal_moe_config`` to return separate up and down
    tiles for the duration of the ``with`` block.

    The plain ``override_config`` only injects ONE tile — both up and
    down end up using it. This monkey-patches ``try_get_optimal_moe_config``
    so the runtime sees per-direction tiles, which is what the heter-MoE
    autotuned ``bf16_sparse_configs_sep.json`` provides.

    Required: ``up_cfg["BLOCK_SIZE_M"] == down_cfg["BLOCK_SIZE_M"]``
    (asserted in fused_moe_triton_config.py:300-303). The autotune already
    enforces this.
    """
    from sglang.srt.layers.moe.fused_moe_triton import (
        fused_moe_triton_config as _cfg_mod,
    )

    orig = _cfg_mod.try_get_optimal_moe_config
    block_m = max(up_cfg["BLOCK_SIZE_M"], down_cfg["BLOCK_SIZE_M"])

    def patched(*args, return_down_config=False, **kwargs):
        if return_down_config:
            return up_cfg, (down_cfg, block_m)
        return up_cfg

    _cfg_mod.try_get_optimal_moe_config = patched
    try:
        yield
    finally:
        _cfg_mod.try_get_optimal_moe_config = orig


__all__ = [
    "FusedMoE",
    "FusedMoeWeightScaleSupported",
    "override_config",
    "override_split_config",
    "get_config",
    "fused_experts",
    "get_config_file_name",
    "moe_align_block_size",
    "try_get_optimal_moe_config",
]
