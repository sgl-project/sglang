"""
Configuration loader for the auto-tuned MoE LoRA shrink (LoRA A) kernel.

Follows the same pattern as fused_moe_triton_config.py and lora_tuning_config.py:
- An offline tuning script writes one JSON file per (E, N, K, device), keyed by the
  number of input tokens M (mirroring the fused_moe config layout).
- At runtime the shrink launcher (`_invoke_moe_lora_shrink_splitk`) looks up the
  config for the closest tuned M and uses it instead of the heuristic default.

The kernel tuned here is `_moe_lora_shrink_splitk_kernel` (LoRA A / shrink stage),
whose weight is shaped [E, N, K] where:
    E = num virtual experts (weight dim 0, e.g. 64, 128, 256)
    N = max LoRA rank   (the kernel's N / output dim, e.g. 16, 32, 64)
    K = hidden size     (the kernel's K / reduction dim, e.g. 512, 768, 2048, 7168)

Config file naming (kept close to the fused_moe `E=..,N=..` style):
    moe_lora_shrink,E={experts},N={rank},K={hidden},device_name={device}.json

Config file format (keyed by token count M). BLOCK_SIZE_N (= rank N) and
BLOCK_SIZE_K (= 256) are not stored: they are derived by the runtime launcher.
{
    "16":  {"BLOCK_SIZE_M": 16, "GROUP_SIZE_M": 1, "num_warps": 2,
            "num_stages": 3, "SPLIT_K": 4},
    "512": {"BLOCK_SIZE_M": 64, "GROUP_SIZE_M": 1, "num_warps": 4,
            "num_stages": 4, "SPLIT_K": 2}
}

Usage:
    python3 benchmark/kernels/lora_moe_shrink/tune_lora_moe_shrink.py \
        --num-experts 64 128 256 --ranks 16 32 64 --hidden-sizes 512 768 2048 7168

    # Configs are written to
    #   python/sglang/srt/lora/triton_ops/moe_shrink_configs/triton_<ver>/
    # and picked up automatically at runtime.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from typing import Any, Dict, Optional

import triton

from sglang.srt.utils import get_device_name

logger = logging.getLogger(__name__)


def get_moe_lora_shrink_config_file_name(E: int, N: int, K: int) -> str:
    """Filename for the MoE LoRA shrink config of a given (experts E, rank N, hidden K)."""
    device_name = get_device_name().replace(" ", "_")
    return f"moe_lora_shrink,E={E},N={N},K={K},device_name={device_name}.json"


@functools.lru_cache
def get_moe_lora_shrink_configs(
    E: int, N: int, K: int
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Load pre-tuned shrink configs from JSON, keyed by token count M.

    Returns a dict mapping M -> config dict, or None if no file is found.
    Reuses the SGLANG_LORA_CONFIG_DIR override shared with lora_tuning_config.py.
    """
    json_file_name = get_moe_lora_shrink_config_file_name(E, N, K)

    config_dir = os.environ.get(
        "SGLANG_LORA_CONFIG_DIR", os.path.dirname(os.path.realpath(__file__))
    )
    configs_root = os.path.join(config_dir, "moe_shrink_configs")

    triton_version = triton.__version__
    version_dir = f"triton_{triton_version.replace('.', '_')}"

    # Try exact triton version first.
    config_file_path = os.path.join(configs_root, version_dir, json_file_name)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info(f"Using MoE LoRA shrink config from {config_file_path}.")
            return {int(key): val for key, val in json.load(f).items()}

    # Fall back to other triton versions (newest first).
    if os.path.isdir(configs_root):
        version_dirs = sorted(
            (d for d in os.listdir(configs_root) if d.startswith("triton_")),
            reverse=True,
        )
        for vdir in version_dirs:
            if vdir == version_dir:
                continue
            try_path = os.path.join(configs_root, vdir, json_file_name)
            if os.path.exists(try_path):
                with open(try_path) as f:
                    logger.warning(
                        f"MoE LoRA shrink config not found for Triton {triton_version}. "
                        f"Falling back to {try_path}. Performance might be sub-optimal!"
                    )
                    return {int(key): val for key, val in json.load(f).items()}

    return None


# Track which (E, N, K, M) combos have been logged to avoid spamming every forward.
_logged_configs: set = set()


def get_moe_lora_shrink_config(
    E: int, N: int, K: int, M: int
) -> Optional[Dict[str, Any]]:
    """Return the tuned shrink config for (experts E, rank N, hidden K) at token count M.

    Picks the config tuned for the closest M. Returns None when no tuned config
    file is available, in which case the caller should fall back to its heuristic
    default. The returned dict is a fresh copy and safe to mutate.
    """
    configs = get_moe_lora_shrink_configs(E, N, K)
    if configs is None:
        return None

    closest = min(configs.keys(), key=lambda x: abs(x - M))
    config = dict(configs[closest])

    log_key = (E, N, K, M)
    if log_key not in _logged_configs:
        _logged_configs.add(log_key)
        logger.info(
            f"MoE LoRA shrink (E={E}, N={N}, K={K}, M={M}): using tuned config "
            f"for M={closest}: {config}"
        )
    return config
