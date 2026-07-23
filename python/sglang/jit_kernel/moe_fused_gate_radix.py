"""Native-CUDA radix-select router for the K3 decode regime.

Replaces the triton router's 16 dependent argmax rounds with byte-histogram
radix narrowing (single CTA per token). Selection semantics match
_router_triton_kernel exactly (ids bit-identical incl. adversarial ties/NaN;
weights <= 2.4e-7 rel); 1.79x over the tuned triton router at [1, 896] top-16
(graphed A/B on GB300), CUDA-graph capturable.

Not dispatched in serving (moe_route_radix_v2 superseded it at every batch
size); kept as the bit-exact selection reference for the v2 test/bench.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.layers.moe import single_token_handoff

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_NUM_EXPERTS = 896
_TOPK = 16


@cache_once
def _jit_moe_fused_gate_radix_module() -> Module:
    return load_jit(
        "moe_fused_gate_radix",
        cuda_files=["moe/moe_fused_gate_radix.cuh"],
        cuda_wrappers=[
            ("run", "MoeFusedGateRadixKernel::run"),
            ("run_align", "MoeFusedGateRadixKernel::run_align"),
        ],
        # No fast-math: expert-id selection must stay bit-identical to the
        # triton router under ties/NaN.
        extra_cuda_cflags=["-O3"],
    )


def covered(scores: torch.Tensor, bias: torch.Tensor, topk: int) -> bool:
    """The kernel is specialized for K3 decode routing: [M, 896] bf16
    row-contiguous scores, fp32 bias, top-16."""
    return (
        scores.dim() == 2
        and scores.size(1) == _NUM_EXPERTS
        and int(topk) == _TOPK
        and scores.dtype == torch.bfloat16
        and bias.dtype == torch.float32
        and scores.stride(1) == 1
        and bias.is_contiguous()
    )


def moe_fused_gate_radix(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_scale: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (weights [M, topk] fp32, ids [M, topk] int32). Caller must
    have checked covered()."""
    M = scores.shape[0]
    device = scores.device
    out_w = torch.empty((M, topk), dtype=torch.float32, device=device)
    out_i = torch.empty((M, topk), dtype=torch.int32, device=device)
    if M == 1:
        # Emit the single-token moe_align_block_size outputs in the same
        # kernel and stash them for fused_marlin_moe (attempt-and-verify;
        # unconsumed entries just fall back to the align kernel).
        block_size = single_token_handoff.compute_single_token_block_size(
            topk, _NUM_EXPERTS
        )
        sorted_ids = torch.empty((topk * block_size,), dtype=torch.int32, device=device)
        expert_ids = torch.empty((topk,), dtype=torch.int32, device=device)
        num_post = torch.empty((1,), dtype=torch.int32, device=device)
        _jit_moe_fused_gate_radix_module().run_align(
            scores,
            bias,
            out_w,
            out_i,
            topk,
            float(routed_scaling_factor),
            bool(renormalize),
            bool(apply_scale),
            sorted_ids,
            expert_ids,
            num_post,
            block_size,
        )
        single_token_handoff.stash_alignment(
            out_i, sorted_ids, expert_ids, num_post, block_size
        )
        return out_w, out_i
    _jit_moe_fused_gate_radix_module().run(
        scores,
        bias,
        out_w,
        out_i,
        topk,
        float(routed_scaling_factor),
        bool(renormalize),
        bool(apply_scale),
    )
    return out_w, out_i
