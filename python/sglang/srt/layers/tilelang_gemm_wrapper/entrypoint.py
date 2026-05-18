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
_SELECTED_CONFIGS: Dict[Tuple[int, int], dict] = {}


def update_tilelang_config(gpu_id: int, server_args: "ServerArgs") -> None:
    """Update TileLang runtime config.

    Phase 1 uses a fixed baseline kernel, but this hook is intentionally wired
    now so later autotuning can reuse the same first-rank/precompile policy.
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


@lru_cache(maxsize=128)
def _get_kernel(N: int, K: int):
    assert_available()

    from sglang.srt.layers.tilelang_gemm_wrapper.kernels import (
        fp8_blockwise_gemm_kernel,
    )

    kernel = fp8_blockwise_gemm_kernel(N=N, K=K)
    _SELECTED_CONFIGS[(N, K)] = {
        "kernel_type": "base",
        "N": N,
        "K": K,
        "block_M": 128,
        "block_N": 128,
        "block_K": 128,
        "num_stages": 2,
        "threads": 128,
    }
    return kernel


def gemm_nt_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
) -> None:
    """Compute out = A @ B^T for FP8 inputs with FP32 block scales."""

    A_fp8, A_scale = lhs
    B_fp8, B_scale = rhs
    N, K = B_fp8.shape
    kernel = _get_kernel(N, K)
    kernel(A_fp8, A_scale, B_fp8, B_scale, out)


def warmup_or_autotune_shapes(shapes: Iterable[Tuple[int, int, int]]) -> None:
    """Compile kernels for concrete (M, N, K) shapes on the compile rank.

    Phase 1 has a symbolic M baseline kernel, so only N and K participate in the
    kernel cache key. Autotuned kernels can expand this entrypoint later.
    """

    if not _DO_COMPILE:
        return

    assert_available()
    for _, N, K in shapes:
        _get_kernel(N, K)


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
