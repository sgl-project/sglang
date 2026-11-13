from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    get_config_file_name,
    try_get_optimal_moe_config,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
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


_config_lookup: Optional[Tuple[int, int]] = None


@contextmanager
def override_config_lookup(num_tokens: int, top_k: int):
    """Look up kernel configs as for a call of ``num_tokens`` tokens with ``top_k`` routed
    experts each: a caller that runs one such call as several smaller ones keeps that
    call's configs, and with them its numerics."""
    global _config_lookup
    old_lookup = _config_lookup
    _config_lookup = (num_tokens, top_k)
    try:
        yield
    finally:
        _config_lookup = old_lookup


def get_config_lookup() -> Optional[Tuple[int, int]]:
    return _config_lookup


__all__ = [
    "override_config",
    "get_config",
    "override_config_lookup",
    "get_config_lookup",
    "fused_experts",
    "get_config_file_name",
    "moe_align_block_size",
    "try_get_optimal_moe_config",
]
