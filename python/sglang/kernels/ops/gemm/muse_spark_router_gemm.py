"""Skinny fp32 router GEMM for MuseSpark v1.2 decode; torch.mm above MAX_TOKENS."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

MAX_TOKENS = 8
# Swept 16/32/64 on B300 at T=8, K=8192, N=256: 11.3 / 6.8 / 10.0 us.
_SPLIT_K = 32
_N_TILE = 64
_THREADS = 256

_workspaces: Dict[tuple, torch.Tensor] = {}


@cache_once
def _jit_module(max_tokens: int, hidden_size: int) -> Module:
    args = make_cpp_args(
        max_tokens, hidden_size, _SPLIT_K, _N_TILE, _THREADS, is_arch_support_pdl()
    )
    return load_jit(
        "muse_spark_router_gemm",
        *args,
        cuda_files=["muse_spark/router_gemm.cuh"],
        cuda_wrappers=[("run", f"MuseSparkRouterGemm<{args}>::run")],
    )


def _max_tokens_bucket(num_tokens: int) -> int:
    for cap in (MAX_TOKENS,):
        if num_tokens <= cap:
            return cap
    raise ValueError(num_tokens)


def _workspace(max_tokens: int, num_experts: int, device: torch.device) -> torch.Tensor:
    key = (max_tokens, num_experts, device)
    ws = _workspaces.get(key)
    if ws is None:
        n_tiles = num_experts // _N_TILE
        ws = torch.zeros(
            _SPLIT_K * max_tokens * num_experts + n_tiles,
            dtype=torch.float32,
            device=device,
        )
        _workspaces[key] = ws
    return ws


def router_gemm_supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.float32
        and w.dtype == torch.float32
        and x.dim() == 2
        and 1 <= x.shape[0] <= MAX_TOKENS
        and w.shape[0] % (_SPLIT_K * _THREADS // (_N_TILE // 4)) == 0
        and w.shape[1] % _N_TILE == 0
        and w.is_contiguous()
    )


def router_gemm(
    x: torch.Tensor, w: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``x @ w`` for fp32 ``x: [T, K]`` (T <= MAX_TOKENS) and ``w: [K, N]``."""
    assert router_gemm_supported(x, w)
    x = x.contiguous()
    num_tokens, _ = x.shape
    num_experts = w.shape[1]
    if out is None:
        out = torch.empty(num_tokens, num_experts, dtype=torch.float32, device=x.device)
    max_tokens = _max_tokens_bucket(num_tokens)
    _jit_module(max_tokens, w.shape[0]).run(
        x, w, out, _workspace(max_tokens, num_experts, x.device)
    )
    return out
