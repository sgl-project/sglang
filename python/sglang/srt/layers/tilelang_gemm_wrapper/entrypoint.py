"""Runtime entrypoints for TileLang FP8 blockwise GEMM."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Iterable, Tuple

import torch

from sglang.srt.layers.tilelang_gemm_wrapper.configurer import assert_available

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_M_MAX = 1024 * 16
_DO_COMPILE = True
_SELECTED_CONFIGS: Dict[Tuple[int, int, int, str], dict] = {}
_PARTIAL_BUFFER_CACHE: Dict[Tuple[str, int, int, int, str, str], torch.Tensor] = {}

_DEFAULT_CONFIG = {
    "kernel_type": "base",
    "block_M": 128,
    "block_N": 128,
    "block_K": 128,
    "num_stages": 2,
    "threads": 128,
    "split_k": 1,
    "out_dtype": "bfloat16",
    "accum_dtype": "float32",
    "c_scale_local": False,
    "a_scale_shm": False,
    "b_scale_shm": False,
}

_SPLIT_K_KERNEL_TYPES = {"splitK", "splitK_swapAB"}
_SWAP_AB_KERNEL_TYPES = {"swapAB", "splitK_swapAB"}
_SUPPORTED_KERNEL_TYPES = {"base", "swapAB", "splitK", "splitK_swapAB"}


def update_tilelang_config(gpu_id: int, server_args: "ServerArgs") -> None:
    """Update TileLang runtime config.

    The first-rank policy is wired here so warmup/autotune can run before CUDA
    graph capture.
    """

    global _M_MAX, _DO_COMPILE

    assert_available()

    m_max = 1024 * 16
    if server_args.chunked_prefill_size < 1:
        m_max = 1024 * 64
    elif server_args.chunked_prefill_size > 8192:
        m_max = server_args.chunked_prefill_size * 2
    _M_MAX = min(1024 * 128, m_max)
    _DO_COMPILE = server_args.base_gpu_id == gpu_id

    logger.info(
        "TileLang FP8 GEMM config updated: m_max=%s, do_compile=%s",
        _M_MAX,
        _DO_COMPILE,
    )


def _select_config(M: int, N: int, K: int) -> dict:
    """Return the selected kernel config for a concrete shape.

    The current default is intentionally conservative until SM89/SM90 validation
    selects tuned configs. The full kernel family is available for autotuning.
    """

    config = dict(_DEFAULT_CONFIG)
    config.update({"M": M, "N": N, "K": K})
    return config


def _validate_config(config: dict) -> None:
    kernel_type = config["kernel_type"]
    if kernel_type not in _SUPPORTED_KERNEL_TYPES:
        raise RuntimeError(
            "TileLang FP8 GEMM got unsupported kernel_type="
            f"{kernel_type}; expected one of {sorted(_SUPPORTED_KERNEL_TYPES)}."
        )

    split_k = config["split_k"]
    if kernel_type in _SPLIT_K_KERNEL_TYPES:
        K = config["K"]
        if split_k <= 1:
            raise RuntimeError(
                f"TileLang {kernel_type} requires split_k > 1, got {split_k}."
            )
        if K % split_k != 0 or (K // split_k) % 128 != 0:
            raise RuntimeError(
                f"TileLang {kernel_type} requires K/split_k to be divisible by 128; "
                f"got K={K}, split_k={split_k}."
            )
    elif split_k != 1:
        raise RuntimeError(
            f"TileLang {kernel_type} does not use split_k, got split_k={split_k}."
        )


@lru_cache(maxsize=256)
def _get_kernel(
    kernel_type: str,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    num_stages: int,
    threads: int,
    split_k: int,
    out_dtype: str,
    accum_dtype: str,
    c_scale_local: bool,
    a_scale_shm: bool,
    b_scale_shm: bool,
):
    assert_available()

    from sglang.srt.layers.tilelang_gemm_wrapper.kernels import (
        fp8_blockwise_gemm_base_kernel,
        fp8_blockwise_gemm_split_k_kernel,
        fp8_blockwise_gemm_split_k_swap_ab_kernel,
        fp8_blockwise_gemm_swap_ab_kernel,
    )

    common = {
        "N": N,
        "K": K,
        "block_M": block_M,
        "block_N": block_N,
        "block_K": block_K,
        "num_stages": num_stages,
        "threads": threads,
        "out_dtype": out_dtype,
        "accum_dtype": accum_dtype,
        "c_scale_local": c_scale_local,
    }

    if kernel_type == "base":
        return fp8_blockwise_gemm_base_kernel(**common, a_scale_shm=a_scale_shm)
    if kernel_type == "swapAB":
        return fp8_blockwise_gemm_swap_ab_kernel(**common, b_scale_shm=b_scale_shm)
    if kernel_type == "splitK":
        return fp8_blockwise_gemm_split_k_kernel(
            **common, split_k=split_k, a_scale_shm=a_scale_shm
        )
    if kernel_type == "splitK_swapAB":
        return fp8_blockwise_gemm_split_k_swap_ab_kernel(
            **common, split_k=split_k, b_scale_shm=b_scale_shm
        )

    raise RuntimeError(f"Unknown TileLang FP8 GEMM kernel type: {kernel_type}.")


def _compile_from_config(config: dict):
    return _get_kernel(
        config["kernel_type"],
        config["N"],
        config["K"],
        config["block_M"],
        config["block_N"],
        config["block_K"],
        config["num_stages"],
        config["threads"],
        config["split_k"],
        config["out_dtype"],
        config["accum_dtype"],
        config["c_scale_local"],
        config["a_scale_shm"],
        config["b_scale_shm"],
    )


def _record_selected_config(config: dict) -> None:
    key = (config["M"], config["N"], config["K"], config["kernel_type"])
    _SELECTED_CONFIGS[key] = dict(config)


def _get_partial_buffer(
    kernel_type: str,
    split_k: int,
    M: int,
    N: int,
    device: torch.device,
    dtype: str,
) -> torch.Tensor:
    device_key = str(device)
    key = (kernel_type, split_k, M, N, device_key, dtype)
    if key not in _PARTIAL_BUFFER_CACHE:
        torch_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
        _PARTIAL_BUFFER_CACHE[key] = torch.zeros(
            (split_k, M, N), device=device, dtype=torch_dtype
        )
    return _PARTIAL_BUFFER_CACHE[key]


def gemm_nt_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
) -> None:
    """Compute out = A @ B^T for FP8 inputs with FP32 block scales."""

    A_fp8, A_scale = lhs
    B_fp8, B_scale = rhs
    M, K = A_fp8.shape
    N, weight_k = B_fp8.shape
    if K != weight_k:
        raise RuntimeError(
            f"TileLang FP8 GEMM got mismatched K dimensions: A K={K}, B K={weight_k}."
        )
    config = _select_config(M, N, K)
    _validate_config(config)
    kernel = _compile_from_config(config)
    _record_selected_config(config)

    kernel_type = config["kernel_type"]
    if kernel_type in _SPLIT_K_KERNEL_TYPES:
        partial = _get_partial_buffer(
            kernel_type,
            config["split_k"],
            M,
            N,
            out.device,
            config["accum_dtype"],
        )
        if kernel_type in _SWAP_AB_KERNEL_TYPES:
            kernel(B_fp8, B_scale, A_fp8, A_scale, partial, out)
        else:
            kernel(A_fp8, A_scale, B_fp8, B_scale, partial, out)
    elif kernel_type in _SWAP_AB_KERNEL_TYPES:
        kernel(B_fp8, B_scale, A_fp8, A_scale, out)
    else:
        kernel(A_fp8, A_scale, B_fp8, B_scale, out)


def warmup_or_autotune_shapes(shapes: Iterable[Tuple[int, int, int]]) -> None:
    """Compile kernels for concrete (M, N, K) shapes on the compile rank.

    The default config is conservative, but this entrypoint compiles through the
    same config dispatch path used at runtime.
    """

    if not _DO_COMPILE:
        return

    assert_available()
    for M, N, K in shapes:
        config = _select_config(M, N, K)
        _validate_config(config)
        _compile_from_config(config)
        _record_selected_config(config)


def export_selected_configs(path: str) -> None:
    """Export selected configs for reproducible benchmark runs."""

    payload = {
        "backend": "tilelang_fp8_gemm",
        "configs": [v for _, v in sorted(_SELECTED_CONFIGS.items())],
    }
    with open(path, "w") as fout:
        json.dump(payload, fout, indent=2)
        fout.write("\n")


def clear_cache() -> None:
    _get_kernel.cache_clear()
    _SELECTED_CONFIGS.clear()
    _PARTIAL_BUFFER_CACHE.clear()
