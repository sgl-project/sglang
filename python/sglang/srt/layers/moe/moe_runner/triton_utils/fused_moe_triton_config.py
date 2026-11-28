from __future__ import annotations

import functools
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton

from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import get_device_name, is_hip

logger = logging.getLogger(__name__)
_is_hip = is_hip()
_LOW_SMEM_FP8_DEFAULT_CUTOFF_BYTES = 128 * 1024
_MIN_DOT_BLOCK_K = 32  # Smallest K tl.dot accepts for INT8 operands.


def _get_standard_dispatcher_ep_size() -> int:
    # StandardDispatcher keeps global (M, topk) ids with non-local experts masked to -1,
    # so only M * topk / ep_size slots reach this rank's E_local experts.
    if get_moe_a2a_backend().is_none():
        return get_parallel().moe_ep_size
    return 1


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
    config_dir = envs.SGLANG_MOE_CONFIG_DIR.get()
    if config_dir is None:
        config_dir = os.path.dirname(os.path.realpath(__file__))

    triton_version = triton.__version__
    version_dir = f"triton_{triton_version.replace('.', '_')}"
    configs_root = os.path.join(config_dir, "configs")
    config_file_path = os.path.join(
        configs_root,
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
    available_versions = (
        sorted(
            (
                d.removeprefix("triton_").replace("_", ".")
                for d in os.listdir(configs_root)
                if d.startswith("triton_")
            ),
            key=lambda v: tuple(int(x) for x in v.split(".")),
            reverse=True,
        )
        if os.path.isdir(configs_root)
        else []
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


def _get_meff_block_size_m(M: int, E: int, topk: int) -> int:
    # Expected routed rows per expert; E is E_local, so scale back to global under EP.
    m_per_expert = (M * topk) / (max(E, 1) * _get_standard_dispatcher_ep_size())
    if m_per_expert <= 4.0:
        return 16
    if m_per_expert <= 16.0:
        return 32
    return 64


def _get_fp8_per_tensor_config(block_size_m: int, N: int) -> Dict[str, int]:
    small_m = block_size_m <= 32
    if _use_low_smem_fp8_default():
        return {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": 64 if (small_m or N < 128) else 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1 if small_m else 64,
            "num_warps": 4,
            "num_stages": 4 if small_m else 2,
        }
    if N < 128:
        block_size_n = 64
    elif small_m or N < 256:
        block_size_n = 128
    else:
        block_size_n = 256
    return {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1 if small_m else 32,
        "num_warps": 4 if small_m else 8,
        "num_stages": 2 if _is_hip else 4,
    }


def _get_fp8_block_wise_config(
    block_size_m: int, block_shape: List[int]
) -> Dict[str, int]:
    # The kernel applies one scale per K tile, so block_shape[1] must be divisible by BLOCK_SIZE_K.
    return {
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_shape[0],
        "BLOCK_SIZE_K": block_shape[1],
        "GROUP_SIZE_M": 1 if block_size_m == 16 else 16,
        "num_warps": 4,
        "num_stages": 2 if _is_hip else 3,
    }


def _get_unquantized_config(block_size_m: int, N: int) -> Dict[str, int]:
    if block_size_m <= 32:
        return {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128 if N >= 128 else 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 16,
        "num_warps": 4,
        "num_stages": 4,
    }


def _get_legacy_unquantized_config(M: int, E: int, is_marlin: bool) -> Dict[str, int]:
    if M <= E or (is_marlin and M <= 32):
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }


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
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
    if dtype == "float32":
        # FP32 tiles are twice as wide in shared memory; keep the pre-M_eff config,
        # which fits the 99 KiB SM86/SM89 per-block limit.
        return _get_legacy_unquantized_config(M=M, E=E, is_marlin=is_marlin)
    block_size_m = _get_meff_block_size_m(M=M, E=E, topk=topk)
    if dtype == "fp8_w8a8":
        if block_shape is None:
            return _get_fp8_per_tensor_config(block_size_m=block_size_m, N=N)
        return _get_fp8_block_wise_config(
            block_size_m=block_size_m, block_shape=block_shape
        )
    # A heuristic: fused marlin works faster with this config for small M
    if is_marlin and M <= 32:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    config = _get_unquantized_config(block_size_m=block_size_m, N=N)
    if dtype == "int8_w8a8" and block_shape is not None:
        # Block-wise INT8 shares the one-scale-per-K-tile kernel path with block-wise FP8.
        if block_shape[1] < _MIN_DOT_BLOCK_K:
            raise ValueError(
                f"Block-wise INT8 MoE needs block_shape[1] >= {_MIN_DOT_BLOCK_K} "
                f"(one scale per K tile, tl.dot needs K >= {_MIN_DOT_BLOCK_K}); "
                f"got {block_shape[1]}."
            )
        config["BLOCK_SIZE_K"] = math.gcd(config["BLOCK_SIZE_K"], block_shape[1])
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
    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        get_config,
        get_config_lookup,
    )

    lookup = get_config_lookup()
    if lookup is not None:
        M, top_k = lookup
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
