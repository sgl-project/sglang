"""Kimi-K3.1 decode routed MoE (TP-MoE rank: 896 experts, intermediate shard 384, top-16), MXFP4 weights in the
engine's flashinfer_mxfp4 layout x MXFP8 activations, SiTU, on SM100: a routing kernel and a persistent FC1 + FC2
kernel chained with PDL. The output is flashinfer's deferred-finalize contract (permuted BF16 FC2 rows +
expanded_idx); the engine applies the top-k weights in its finalize / all-reduce pass.

Kernel source: ``kernels/jit/csrc/moe/k31_decode/moe_mxfp4.cuh``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

NUM_EXPERTS, TOP_K, HIDDEN, INTER = 896, 16, 3584, 384
MAX_TOKENS = 256
N_CAP, STAGES = 32, 5  # tokens per virtual expert (MMA N), smem ring depth
# SiTU constants compiled into the kernel: beta (gemm1_alpha) and linear_beta (gemm1_clamp_limit)
SITU_BETA, SITU_LINEAR_BETA = 4.0, 25.0
_MAX_V = 1024  # virtual experts (expert, chunk of <= 32 tokens)
_COUNTER_WORDS = _MAX_V + 2


@cache_once
def _jit_module(n_cap: int, stages: int) -> Module:
    args = make_cpp_args(n_cap, stages)
    return load_jit(
        "k31_moe_mxfp4",
        *args,
        cuda_files=["moe/k31_decode/moe_mxfp4.cuh"],
        cuda_wrappers=[
            ("route", f"k31_moe_route<{n_cap}>"),
            ("run", f"k31_moe_mxfp4<{args}>"),
        ],
    )


def k31_moe_mxfp4(
    x: torch.Tensor,  # [T, 3584] float8_e4m3fn
    x_sf: torch.Tensor,  # [T, 112] E8M0 bytes (linear)
    topk_ids: torch.Tensor,  # [T, 16] int32 ids, or the packed (id << 16) | bf16(weight) form
    w13: torch.Tensor,  # [896, 768, 1792] uint8, flashinfer_mxfp4 layout
    w13_sf: torch.Tensor,  # [896, 768, 112] E8M0 bytes, 128x4 atoms
    w2: torch.Tensor,  # [896, 3584, 192] uint8
    w2_sf: torch.Tensor,  # [896, 3584, 12]
    *,
    packed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Routing kernel + MoE kernel. Returns (gemm2_out [T * 16, 3584] bf16, expanded_idx [T * 16] int32, weights):
    out[t] = sum_k topk_weights[t, k] * gemm2_out[expanded_idx[t * 16 + k]]. With `packed`, `weights` is the [T, 16]
    bf16 half of the packed form, else None. The scratch holds no state across calls (the routing kernel resets the
    counters), so it is allocated per call like the outputs."""
    module = _jit_module(N_CAP, STAGES)
    num_tokens, device = x.shape[0], x.device
    rows = num_tokens * TOP_K
    gemm2_out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device)
    expanded_idx = torch.empty(rows, dtype=torch.int32, device=device)
    weights = (
        torch.empty(num_tokens, TOP_K, dtype=torch.bfloat16, device=device)
        if packed
        else None
    )
    token_bits = torch.empty(_MAX_V * 8, dtype=torch.int32, device=device)
    vinfo = torch.empty(_MAX_V + 1, dtype=torch.int32, device=device)
    counters = torch.empty(_COUNTER_WORDS, dtype=torch.int32, device=device)
    act = torch.empty(rows + N_CAP, INTER, dtype=torch.uint8, device=device)
    act_sf = torch.empty(_MAX_V * 3 * 512, dtype=torch.uint8, device=device)
    module.route(topk_ids, expanded_idx, token_bits, vinfo, counters, weights)
    module.run(
        x.view(torch.uint8),
        x_sf.view(torch.uint8),
        w13.view(torch.uint8),
        w13_sf.view(torch.uint8),
        w2.view(torch.uint8),
        w2_sf.view(torch.uint8),
        token_bits,
        vinfo,
        act,
        act_sf,
        gemm2_out,
        counters,
    )
    return gemm2_out, expanded_idx, weights
