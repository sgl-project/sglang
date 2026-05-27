"""
Configuration loader for auto-tuned LoRA CSGMV kernel block sizes.

Follows the same pattern as fused_moe_triton_config.py:
- Offline tuning script writes JSON files keyed by chunk_size (BLOCK_M)
- At server startup, the config loader reads the best block sizes for each kernel
- Kernels use these instead of hardcoded defaults

Config file naming: lora_{kernel},K={K},R={R},S={S},device={device}.json
Where kernel is "shrink" or "expand", K is input_dim, R is max_rank, S is num_slices.

Config file format (keyed by chunk_size):
{
    "16": {"BLOCK_N": 16, "BLOCK_K": 256, "num_warps": 4, "num_stages": 3},
    "32": {"BLOCK_N": 32, "BLOCK_K": 128, "num_warps": 4, "num_stages": 4},
    "128": {"BLOCK_N": 64, "BLOCK_K": 256, "num_warps": 8, "num_stages": 3}
}

Usage:
    python3 benchmark/kernels/lora_csgmv/tune_lora_csgmv.py \
        --model Qwen/Qwen3-Embedding-0.6B --max-lora-rank 64

    # Configs saved to python/sglang/srt/lora/triton_ops/configs/

    # Server automatically picks them up:
    python3 -m sglang.launch_server --model ... --enable-lora --lora-backend csgmv
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


def get_lora_config_file_name(
    kernel: str,
    K: int,
    R: int,
    S: int,
) -> str:
    """Generate config filename for a LoRA kernel configuration.

    Args:
        kernel: "shrink" or "expand"
        K: The large dimension (input_dim for shrink, output_dim for expand)
        R: The max LoRA rank
        S: num_slices (qkv=3, gate_up=2, others=1)
    """
    device_name = get_device_name().replace(" ", "_")
    return f"lora_{kernel},K={K},R={R},S={S},device={device_name}.json"


@functools.lru_cache
def get_lora_configs(
    kernel: str,
    K: int,
    R: int,
    S: int,
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Load pre-tuned LoRA kernel configs from JSON files.

    Returns a dict mapping chunk_size (BLOCK_M) to block size configs,
    or None if no config file is found.
    """
    json_file_name = get_lora_config_file_name(kernel, K, R, S)

    config_dir = os.environ.get(
        "SGLANG_LORA_CONFIG_DIR", os.path.dirname(os.path.realpath(__file__))
    )
    configs_root = os.path.join(config_dir, "csgmv_configs")

    triton_version = triton.__version__
    version_dir = f"triton_{triton_version.replace('.', '_')}"

    # Try exact triton version first
    config_file_path = os.path.join(configs_root, version_dir, json_file_name)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info(f"Using LoRA {kernel} config from {config_file_path}.")
            return {int(key): val for key, val in json.load(f).items()}

    # Scan existing version directories as fallback (newest first)
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
                        f"LoRA {kernel} config not found for Triton {triton_version}. "
                        f"Falling back to {try_path}."
                    )
                    return {int(key): val for key, val in json.load(f).items()}

    return None


# Default block sizes (current hardcoded values)
DEFAULT_SHRINK_CONFIG = {"BLOCK_N": 16, "BLOCK_K": 256}
DEFAULT_EXPAND_CONFIG = {"BLOCK_N": 64, "BLOCK_K": 16}

# Track which configs have been logged to avoid spamming on every forward pass
_logged_configs: set = set()


def get_lora_shrink_config(
    K: int,
    R: int,
    num_slices: int,
    chunk_size: int,
) -> Dict[str, int]:
    """Get block sizes for the CSGMV shrink (lora_a) kernel.

    Args:
        K: input_dim
        R: max_rank
        num_slices: number of slices (qkv=3, gate_up=2, others=1)
        chunk_size: BLOCK_M value (= batch_info.max_len)
    """
    log_key = ("shrink", K, R, num_slices, chunk_size)
    configs = get_lora_configs("shrink", K, R, num_slices)
    if configs is not None:
        config = configs.get(chunk_size)
        if config is None:
            closest = min(configs.keys(), key=lambda x: abs(x - chunk_size))
            config = configs[closest]
            if log_key not in _logged_configs:
                _logged_configs.add(log_key)
                logger.info(
                    f"LoRA shrink (K={K}, R={R}): no config for chunk_size={chunk_size}, "
                    f"using closest={closest}: {config}"
                )
        else:
            if log_key not in _logged_configs:
                _logged_configs.add(log_key)
                logger.info(
                    f"LoRA shrink (K={K}, R={R}, chunk_size={chunk_size}): tuned config {config}"
                )
        return config
    if log_key not in _logged_configs:
        _logged_configs.add(log_key)
        logger.info(
            f"LoRA shrink (K={K}, R={R}): no tuned config, using defaults {DEFAULT_SHRINK_CONFIG}"
        )
    return dict(DEFAULT_SHRINK_CONFIG)


def get_lora_expand_config(
    K: int,
    R: int,
    num_slices: int,
    chunk_size: int,
) -> Dict[str, int]:
    """Get block sizes for the CSGMV expand (lora_b) kernel.

    Args:
        K: output_dim
        R: max_rank
        num_slices: number of slices (qkv=3, gate_up=2, others=1)
        chunk_size: BLOCK_M value (= batch_info.max_len)
    """
    log_key = ("expand", K, R, num_slices, chunk_size)
    configs = get_lora_configs("expand", K, R, num_slices)
    if configs is not None:
        config = configs.get(chunk_size)
        if config is None:
            closest = min(configs.keys(), key=lambda x: abs(x - chunk_size))
            config = configs[closest]
            if log_key not in _logged_configs:
                _logged_configs.add(log_key)
                logger.info(
                    f"LoRA expand (K={K}, R={R}): no config for chunk_size={chunk_size}, "
                    f"using closest={closest}: {config}"
                )
        else:
            if log_key not in _logged_configs:
                _logged_configs.add(log_key)
                logger.info(
                    f"LoRA expand (K={K}, R={R}, chunk_size={chunk_size}): tuned config {config}"
                )
        return config
    if log_key not in _logged_configs:
        _logged_configs.add(log_key)
        logger.info(
            f"LoRA expand (K={K}, R={R}): no tuned config, using defaults {DEFAULT_EXPAND_CONFIG}"
        )
    return dict(DEFAULT_EXPAND_CONFIG)
