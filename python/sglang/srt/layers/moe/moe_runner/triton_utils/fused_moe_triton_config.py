from __future__ import annotations

import functools
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton

from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import get_device_name, is_hip

logger = logging.getLogger(__name__)
_is_hip = is_hip()
_LOW_SMEM_FP8_DEFAULT_CUTOFF_BYTES = 128 * 1024


@functools.lru_cache(maxsize=None)
def _get_cuda_shared_memory_per_block_optin() -> Optional[int]:
    if _is_hip or not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
    except (AssertionError, RuntimeError):
        return None
    return getattr(props, "shared_memory_per_block_optin", None)


def _use_low_smem_fp8_default() -> bool:
    smem_limit = _get_cuda_shared_memory_per_block_optin()
    return smem_limit is not None and smem_limit < _LOW_SMEM_FP8_DEFAULT_CUTOFF_BYTES


def get_config_file_name(
    E: int,
    N: int,
    dtype: Optional[str],
    block_shape: Optional[int] = None,
    per_channel_quant: bool = False,
    down_moe: bool = False,
) -> str:
    device_name = get_device_name().replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = (
        "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"
    )
    per_channel_quant_selector = ",per_channel_quant=True" if per_channel_quant else ""
    down_moe_selector = "_down" if down_moe else ""
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}{per_channel_quant_selector}{down_moe_selector}.json"


@functools.lru_cache
def get_moe_configs(
    E: int,
    N: int,
    dtype: Optional[str],
    block_n: Optional[int] = 0,
    block_k: Optional[int] = 0,
    per_channel_quant: bool = False,
    down_moe: bool = False,
) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """
    if get_exec().deterministic.enable_deterministic_inference:
        logger.warning(
            "Deterministic inference is enabled, using default MoE kernel config."
        )
        return None

    # First look up if an optimized configuration is available in the configs
    # directory
    json_file_name = get_config_file_name(
        E,
        N,
        dtype,
        [block_n, block_k],
        per_channel_quant,
        down_moe=down_moe,
    )

    # We found that using the fused_moe_kernel config from Triton 3.1.0 with Triton 3.2.0 results in negative performance gains,
    # so we also include the Triton version as a key for finding the fused_moe_kernel config to achieve the best performance.
    config_dir = os.environ.get(
        "SGLANG_MOE_CONFIG_DIR", os.path.dirname(os.path.realpath(__file__))
    )

    triton_version = triton.__version__
    version_dir = f"triton_{triton_version.replace('.', '_')}"
    config_file_path = os.path.join(
        config_dir,
        "configs",
        version_dir,
        json_file_name,
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            # Please note that although we find the config files, performance might still be suboptimal.
            # This is because the tuning environment might differ from your current environment.
            # For example, updating the Triton version might cause all old configs to become suboptimal.
            # To achieve the best performance, consider re-tuning the Triton fused MOE kernel in your environment.
            # For the tuning method, refer to: https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
            logger.info(f"Using MoE kernel config from {config_file_path}.")
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # Discover available triton config dirs on disk and search newest-first.
    configs_root = os.path.join(config_dir, "configs")
    available_versions = sorted(
        (
            d.removeprefix("triton_").replace("_", ".")
            for d in os.listdir(configs_root)
            if d.startswith("triton_")
        ),
        key=lambda v: tuple(int(x) for x in v.split(".")),
        reverse=True,
    )

    for try_triton_version in available_versions:
        if try_triton_version == triton_version:
            continue
        try_config_file_path = os.path.join(
            configs_root,
            f"triton_{try_triton_version.replace('.', '_')}",
            json_file_name,
        )
        if os.path.exists(try_config_file_path):
            with open(try_config_file_path) as f:
                logger.warning(
                    f"Config file not found at {config_file_path}. Fallback to triton version {try_triton_version} and use MoE kernel config from {try_config_file_path}. Performance might be sub-optimal!",
                )
                # If a configuration has been found, return it
                return {int(key): val for key, val in json.load(f).items()}

    if down_moe:
        # A separate down-projection config enables the TMA path, but it is
        # optional. Reuse a tuned up-projection config when it is absent so
        # the second GEMM does not silently fall back to the heuristic.
        up_configs = get_moe_configs(
            E,
            N,
            dtype,
            block_n,
            block_k,
            per_channel_quant=per_channel_quant,
            down_moe=False,
        )
        if up_configs is not None:
            logger.warning(
                "Down MoE config file not found at %s; reusing the tuned "
                "up-projection config without TMA. Performance might be sub-optimal.",
                config_file_path,
            )
            return up_configs
        logger.warning(
            (
                "Using default MoE kernel config. Performance might be sub-optimal! "
                "Config file not found at %s, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton"
            ),
            config_file_path,
        )
    else:
        logger.warning(
            (
                "Using default MoE kernel config. Performance might be sub-optimal! "
                "Config file not found at %s, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton"
            ),
            config_file_path,
        )
    return None


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
    is_marlin: bool,
    block_shape: Optional[List[int]] = None,
) -> Dict[str, int]:
    if get_exec().deterministic.enable_deterministic_inference:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
        return config
    if dtype == "fp8_w8a8":
        if block_shape is None:
            if _use_low_smem_fp8_default():
                config = {
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 256,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 4,
                }
                if M > E:
                    config = {
                        "BLOCK_SIZE_M": 64,
                        "BLOCK_SIZE_N": 128,
                        "BLOCK_SIZE_K": 256,
                        "GROUP_SIZE_M": 64,
                        "num_warps": 4,
                        "num_stages": 2,
                    }
            else:
                config = {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 32,
                    "num_warps": 8,
                    "num_stages": 2 if _is_hip else 4,
                }
                if M <= E:
                    config = {
                        "BLOCK_SIZE_M": 64,
                        "BLOCK_SIZE_N": 128,
                        "BLOCK_SIZE_K": 128,
                        "GROUP_SIZE_M": 1,
                        "num_warps": 4,
                        "num_stages": 2 if _is_hip else 4,
                    }
        else:
            # Block-wise quant: BLOCK_SIZE_K must be divisible by block_shape[1]
            config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": block_shape[0],
                "BLOCK_SIZE_K": block_shape[1],
                "GROUP_SIZE_M": 32,
                "num_warps": 4,
                "num_stages": 2 if _is_hip else 3,
            }
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
        # A heuristic: fused marlin works faster with this config for small M
        if M <= E or (is_marlin and M <= 32):
            config = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
            }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    is_marlin: bool = False,
    block_shape: Optional[List[int]] = None,
    per_channel_quant: bool = False,
    return_down_config: bool = False,
):
    from sglang.srt.layers.moe.moe_runner.triton_utils import get_config

    down_config = None
    max_block_m = None
    override_config = get_config()
    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        E, _, N = w2_shape
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        configs = get_moe_configs(
            E,
            N,
            dtype,
            block_n,
            block_k,
            per_channel_quant=per_channel_quant,
            down_moe=False,
        )

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = get_default_config(
                M, E, N, w1_shape[2], top_k, dtype, is_marlin, block_shape
            )
        if return_down_config:
            down_configs = get_moe_configs(
                E,
                N,
                dtype,
                block_n,
                block_k,
                per_channel_quant=per_channel_quant,
                down_moe=True,
            )
            if down_configs:
                down_config = down_configs[
                    min(down_configs.keys(), key=lambda x: abs(x - M))
                ]
                down_config = dict(**down_config)
                max_block_m = max(
                    [cfg["BLOCK_SIZE_M"] for cfg in down_configs.values()]
                )
    if return_down_config:
        if (
            down_config is not None
            and config["BLOCK_SIZE_M"] != down_config["BLOCK_SIZE_M"]
        ):
            # Both kernels share one moe_align_block_size sort, so the down
            # config must use the up config's BLOCK_SIZE_M.
            logger.warning_once(
                "down_moe config BLOCK_SIZE_M=%d does not match up config "
                "BLOCK_SIZE_M=%d at M=%d; overriding down BLOCK_SIZE_M to match.",
                down_config["BLOCK_SIZE_M"],
                config["BLOCK_SIZE_M"],
                M,
            )
            down_config["BLOCK_SIZE_M"] = config["BLOCK_SIZE_M"]
        return config, (down_config, max_block_m)
    return config


def get_config_dtype_str(
    dtype: torch.dtype,
    use_int8_w8a16: Optional[bool] = False,
    use_int4_w4a16: Optional[bool] = False,
    use_fp8_w8a8: Optional[bool] = False,
    use_int8_w8a8: Optional[bool] = False,
):
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a8:
        return "int8_w8a8"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None
