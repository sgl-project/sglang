"""Fused MuseSpark MoE routing (CUDA JIT).

Drop-in replacement for the Triton ``fused_muse_spark_routing`` topk8 path in
``sglang.kernels.ops.muse_spark.fused_muse_spark_routing``: sigmoid/softmax normalization,
bias-scored iterative top-k, and renormalization by the selected unbiased score
sum, fused into one kernel with one warp per token.

Bit-exact with the Triton kernel (same emitted expert ids, same fp32 weights)
for the sigmoid path; see ``routing.cuh``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_NORMALIZATION_MODES = {"sigmoid": 0, "softmax": 1}
# Measured on B300 over warps/CTA in {1, 2, 4, 8, 16}: a single warp per CTA
# wins up to ~2k tokens (finer-grained scheduling), 4 wins above it.
_LARGE_BATCH_TOKENS = 2048


def _warps_per_block(num_tokens: int) -> int:
    return 1 if num_tokens <= _LARGE_BATCH_TOKENS else 4


@cache_once
def _jit_module(
    topk: int, experts_per_lane: int, norm_mode: int, warps_per_block: int
) -> Module:
    args = make_cpp_args(
        topk, experts_per_lane, norm_mode, warps_per_block, is_arch_support_pdl()
    )
    return load_jit(
        "muse_spark_routing",
        *args,
        cuda_files=["muse_spark/routing.cuh"],
        cuda_wrappers=[
            ("fused_muse_spark_routing", f"fused_muse_spark_routing<{args}>")
        ],
    )


def _experts_per_lane(num_experts: int) -> int:
    per_lane = max(4, -(-num_experts // 32))
    return -(-per_lane // 4) * 4


def fused_muse_spark_routing(
    gating_output: torch.Tensor,  # [num_tokens, num_experts] fp32
    bias: torch.Tensor,  # [num_experts] fp32
    topk: int,
    route_scale: float,
    router_output_multiplier: Optional[float] = None,
    router_score_normalization: str = "sigmoid",
    router_score_division_eps: Optional[float] = None,
    topk_weights: Optional[torch.Tensor] = None,  # [num_tokens, topk] fp32
    topk_ids: Optional[torch.Tensor] = None,  # [num_tokens, topk] int32
    local_expert_mapping: Optional[torch.Tensor] = None,  # [num_experts] int32
    num_valid_tokens: Optional[torch.Tensor] = None,  # [1] int32, device scalar
    packed_topk_ids: Optional[torch.Tensor] = None,  # [num_tokens, topk] int32
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert gating_output.dim() == 2 and gating_output.is_contiguous()
    assert gating_output.dtype == torch.float32
    num_tokens, num_experts = gating_output.shape
    assert bias.shape[0] == num_experts and bias.is_contiguous()

    norm_mode = _NORMALIZATION_MODES[router_score_normalization]
    bias_f32 = bias if bias.dtype == torch.float32 else bias.float()

    if topk_weights is None:
        topk_weights = torch.empty(
            (num_tokens, topk), dtype=torch.float32, device=gating_output.device
        )
    if topk_ids is None:
        topk_ids = torch.empty(
            (num_tokens, topk), dtype=torch.int32, device=gating_output.device
        )
    topk_weights_view = topk_weights[:num_tokens, :topk]
    topk_ids_view = topk_ids[:num_tokens, :topk]
    packed_view = (
        packed_topk_ids[:num_tokens, :topk] if packed_topk_ids is not None else None
    )

    module = _jit_module(
        topk,
        _experts_per_lane(num_experts),
        norm_mode,
        _warps_per_block(num_tokens),
    )
    module.fused_muse_spark_routing(
        gating_output,
        bias_f32,
        topk_weights_view,
        topk_ids_view,
        packed_view,
        local_expert_mapping,
        num_valid_tokens,
        float(
            router_output_multiplier if router_output_multiplier is not None else 1.0
        ),
        float(route_scale),
        float(
            router_score_division_eps if router_score_division_eps is not None else 0.0
        ),
    )
    return topk_weights_view, topk_ids_view, packed_view
