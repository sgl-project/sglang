"""Single-launch MoE router front end for 512-expert / top-10 / 4096-hidden models.

Replaces router GEMM + softmax top-k + NVFP4 activation quant + routing-table
build (moe_sort) at decode token counts (M <= 16). Outputs are written into a
preallocated :class:`RouterFrontBuffers` so the launch is CUDA-graph safe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

HIDDEN = 4096
NUM_EXPERTS = 512
TOP_K = 10
MAX_TOKENS = 16


def _max_m_bucket(m: int) -> int:
    return 1 if m <= 1 else 8 if m <= 8 else 16


@cache_once
def _jit_module(max_m: int, trace: bool = False) -> Module:
    args = make_cpp_args(max_m, is_arch_support_pdl(), trace)
    return load_jit(
        "qwen_router_front",
        *args,
        cuda_files=["moe/qwen_router_front.cuh"],
        cuda_wrappers=[
            ("run", f"RouterFrontKernel<{args}>::run"),
            ("run_topk", f"RouterFrontKernel<{args}>::run_topk"),
        ],
        # No fast-math: the top-k and quantization arithmetic is pinned to
        # explicit instructions and must not be contracted or flushed.
        extra_cuda_cflags=["-O3"],
    )


class RouterFrontBuffers:
    """Outputs of one call site (one MoE layer); reuse across CUDA graph replays."""

    def __init__(
        self,
        num_tokens: int,
        tile_size: int,
        device: torch.device | str = "cuda",
        arrival: torch.Tensor | None = None,
    ) -> None:
        assert 0 < num_tokens <= MAX_TOKENS
        m = num_tokens
        self.num_tokens = m
        self._max_tile_size = tile_size
        self.set_tile_size(tile_size)
        max_tiles = min(m * TOP_K, NUM_EXPERTS)
        kw = dict(device=device)
        self.logits = torch.empty((m, NUM_EXPERTS), dtype=torch.bfloat16, **kw)
        self.topk_weights = torch.empty((m, TOP_K), dtype=torch.float32, **kw)
        self.topk_ids = torch.empty((m, TOP_K), dtype=torch.int32, **kw)
        self.x_fp4 = torch.empty((m, HIDDEN // 2), dtype=torch.uint8, **kw)
        self.x_sf = torch.empty((m, HIDDEN // 16), dtype=torch.uint8, **kw)
        self.tile_idx_to_expert_idx = torch.empty((max_tiles,), dtype=torch.int32, **kw)
        self.tile_idx_to_mn_limit = torch.empty((max_tiles,), dtype=torch.int32, **kw)
        self.expanded_idx_to_permuted_idx = torch.empty(
            (m, TOP_K), dtype=torch.int32, **kw
        )
        self.permuted_idx_to_expanded_idx = torch.empty(
            (max_tiles * tile_size,), dtype=torch.int32, **kw
        )
        self.total_num_padded_tokens = torch.empty((1,), dtype=torch.int32, **kw)
        self.num_non_exiting_tiles = torch.empty((1,), dtype=torch.int32, **kw)
        # Monotonic counter; only its value modulo the CTA count matters. Pass a
        # persistent one per call site when launches are captured in CUDA graphs.
        self.arrival = (
            arrival if arrival is not None else torch.zeros((1,), dtype=torch.int32, **kw)
        )

    def set_tile_size(self, tile_size: int) -> None:
        """Routing-table tile size; must not exceed the size the buffers were built for."""
        assert tile_size & (tile_size - 1) == 0 and 16 <= tile_size <= self._max_tile_size
        self.tile_size = tile_size
        self.tile_log2 = tile_size.bit_length() - 1


def router_front(
    x: torch.Tensor,
    weight: torch.Tensor,
    global_scale: torch.Tensor,
    out: RouterFrontBuffers,
    trace: torch.Tensor | None = None,
) -> RouterFrontBuffers:
    """x [M, 4096] bf16, weight [512, 4096] bf16, global_scale [1] fp32 (quant domain).

    ``trace`` (int64 [129 * 8]) selects an instrumented build that records %globaltimer stamps.
    """
    module = _jit_module(_max_m_bucket(x.shape[0]), trace is not None)
    module.run(
        x,
        weight,
        global_scale,
        out.logits,
        out.topk_weights,
        out.topk_ids,
        out.x_fp4,
        out.x_sf,
        out.tile_idx_to_expert_idx,
        out.tile_idx_to_mn_limit,
        out.expanded_idx_to_permuted_idx,
        out.permuted_idx_to_expanded_idx,
        out.total_num_padded_tokens,
        out.num_non_exiting_tiles,
        out.arrival,
        trace if trace is not None else out.arrival,
        out.tile_log2,
    )
    return out


def router_topk(
    x: torch.Tensor,
    weight: torch.Tensor,
    arrival: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Router logits + softmax top-10 + renormalize only: (weights fp32 [M, 10], ids int32 [M, 10]).

    ``arrival`` is the call site's persistent int32 [1] counter (see RouterFrontBuffers).
    """
    m = x.shape[0]
    logits = torch.empty((m, NUM_EXPERTS), dtype=torch.bfloat16, device=x.device)
    topk_weights = torch.empty((m, TOP_K), dtype=torch.float32, device=x.device)
    topk_ids = torch.empty((m, TOP_K), dtype=torch.int32, device=x.device)
    _jit_module(_max_m_bucket(m)).run_topk(
        x, weight, logits, topk_weights, topk_ids, arrival
    )
    return topk_weights, topk_ids
