# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Portions adapted from sgl-project/sglang #46475 / #46484 and from SGLang's moe_route_quant_fused.py.
"""K3.1 decode MoE routing in one launch, trtllm-gen metadata included.

Extends the fused route+quant launch (moe_route_quant_fused: radix top-16 of
896 and the mxfp8 group quant of the routed activations) with the padded-row
mask and the routing metadata the trtllm-gen routed MoE GEMMs consume. The
routing CTA that finishes last builds the expert-sorted, tile-padded
permutation from every row's ids, so the routed MoE skips its own routing
kernel (trtllm-gen routingIndices{DynBlock,Cluster}Kernel) and the separate
mask_topk_ids kernel. Ids, weights and quantized activations are bit-identical
to route_quant_fused; the metadata follows trtllm-gen's routingPermutation
contract (see the kernel's header comment).

Up to 16 rows the metadata comes from per-row expert bitmaps, the layout of
moe_route_quant_fused's M <= 8 prepared routing (sgl-project/sglang #46475 /
#46484) extended to an expert range (EP), padded rows, any tile and experts
spanning several tiles; above 16 rows each routing CTA counts its routes into a
per-expert counter (global atomics) and the last CTA scans the counts.

Specialized like route_quant_fused: 896 experts, top-16, a 3584-wide
activation row, and up to _MAX_TOKENS rows (DFLASH verify decode batches up to
bs 85 at 6 draft tokens). Wired into serving through
sglang.srt.layers.moe.route_quant_handoff.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.moe import moe_route_quant_fused, moe_route_radix

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_HIDDEN = 3584
_NUM_EXPERTS = 896
_TOPK = 16
# Serving coverage (the kernel takes up to 4096 rows): DFLASH verify batches up
# to bs 85 at 6 tokens; the sharded MoE front / latent-up caps are 512 too.
_MAX_TOKENS = 512


class K31RoutingMetadata(NamedTuple):
    """The trtllm-gen routing record, in FlashInfer's RoutingMetadataBuffers
    FFI order (trtllm_fused_moe_kernel_launcher.cu)."""

    total_num_padded_tokens: torch.Tensor  # [1] int32
    expanded_idx_to_permuted_idx: torch.Tensor  # [M * 16] int32
    permuted_idx_to_token_idx: torch.Tensor  # [max padded rows + 1] int32
    expert_weights: torch.Tensor  # [M, 16] bf16 or fp32
    expert_count_histogram: torch.Tensor  # scratch, not written
    num_tokens_per_expert: torch.Tensor  # [896] int32
    cta_idx_xy_to_batch_idx: torch.Tensor  # [max ctas] int32
    cta_idx_xy_to_mn_limit: torch.Tensor  # [max ctas] int32
    num_non_exiting_ctas: torch.Tensor  # [1] int32


class K31FusedRouting(NamedTuple):
    weights: torch.Tensor  # [M, 16] fp32
    ids: torch.Tensor  # [M, 16] int32, -1 on padded rows
    x_q: torch.Tensor  # [M, 3584] fp8_e4m3
    x_s: torch.Tensor  # [M, 28] int32, row-major packed UE8M0
    metadata: K31RoutingMetadata


def max_num_ctas(num_tokens: int, tile_tokens: int) -> int:
    """trtllm-gen Routing::getMaxNumCtasInBatchDim over the global experts (the
    bound the routed MoE sizes its own routing buffers with)."""
    remaining = num_tokens * _TOPK
    filled = min(_NUM_EXPERTS, remaining)
    return filled + (remaining - filled) // tile_tokens


def max_num_padded_tokens(num_tokens: int, tile_tokens: int) -> int:
    """trtllm-gen Routing::getMaxPermutedPaddedCount."""
    return max_num_ctas(num_tokens, tile_tokens) * tile_tokens


@cache_once
def _jit_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "moe_k31_fused_routing",
        *args,
        cuda_files=["moe/k31_fused_routing.cuh"],
        cuda_wrappers=[("run", f"K31FusedRoutingKernel<{args}>::run")],
        # No fast-math, as route_quant_fused: the routing half must stay
        # bit-identical to route_radix.
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def available() -> bool:
    import logging

    try:
        _jit_module()
        return True
    except Exception as e:  # pragma: no cover - toolchain dependent
        logging.getLogger(__name__).warning(
            f"Failed to load the JIT K3.1 fused routing kernel: {e}"
        )
        return False


_counters: dict[int, torch.Tensor] = {}


def _counters_for(device: torch.device) -> torch.Tensor:
    """One per device: [0] the routing CTAs' arrival counter, [1:] the
    per-expert route counters (rows above 16). The last CTA of each launch
    resets what it used, so they stay zero between launches (launches on one
    device must not overlap, which the routed MoE's stream order guarantees)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    counters = _counters.get(index)
    if counters is None:
        counters = torch.zeros(1 + _NUM_EXPERTS, dtype=torch.int32, device=device)
        _counters[index] = counters
    return counters


def covered(
    scores: torch.Tensor, bias: torch.Tensor, topk: int, x: torch.Tensor
) -> bool:
    """route_quant_fused coverage, up to _MAX_TOKENS rows."""
    return (
        moe_route_radix.covered(scores, bias, topk)
        and x.dim() == 2
        and 0 < x.shape[0] <= _MAX_TOKENS
        and x.shape[1] == _HIDDEN
        and x.dtype in (torch.bfloat16, torch.float32)
        and x.stride(1) == 1
        and x.data_ptr() % 32 == 0
        and (x.stride(0) * x.element_size()) % 32 == 0
    )


def k31_fused_routing(
    scores: torch.Tensor,
    bias: torch.Tensor,
    x: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_scale: bool,
    *,
    num_token_non_padded: Optional[torch.Tensor],
    local_expert_offset: int,
    num_local_experts: int,
    tile_tokens: int,
    weights_dtype: torch.dtype,
) -> K31FusedRouting:
    """Caller must have checked covered(). Winners come out in expert-id order
    (route_quant_fused's order); ``weights_dtype`` (bf16 or fp32) is the dtype
    of the trtllm-gen expert weights the routed MoE finalizes with."""
    M = scores.shape[0]
    device = scores.device
    num_ctas = max_num_ctas(M, tile_tokens)
    counters = _counters_for(device)

    def i32(*shape):
        return torch.empty(shape, dtype=torch.int32, device=device)

    weights = torch.empty((M, topk), dtype=torch.float32, device=device)
    ids = i32(M, topk)
    x_q = torch.empty((M, _HIDDEN), dtype=torch.float8_e4m3fn, device=device)
    x_s = i32(M, _HIDDEN // 32 // 4)
    metadata = K31RoutingMetadata(
        total_num_padded_tokens=i32(1),
        expanded_idx_to_permuted_idx=i32(M * topk),
        # +1: the routed batched-GEMM kernels read one entry past the end
        # (same WAR as FlashInfer's own allocation).
        permuted_idx_to_token_idx=i32(num_ctas * tile_tokens + 1),
        expert_weights=torch.empty((M, topk), dtype=weights_dtype, device=device),
        expert_count_histogram=i32(2 * max(_NUM_EXPERTS, 256)),
        num_tokens_per_expert=i32(_NUM_EXPERTS),
        cta_idx_xy_to_batch_idx=i32(num_ctas),
        cta_idx_xy_to_mn_limit=i32(num_ctas),
        num_non_exiting_ctas=i32(1),
    )
    _jit_module().run(
        scores,
        bias,
        weights,
        ids,
        x,
        x_q,
        x_s,
        metadata.expert_weights,
        metadata.expanded_idx_to_permuted_idx,
        metadata.permuted_idx_to_token_idx,
        metadata.cta_idx_xy_to_batch_idx,
        metadata.cta_idx_xy_to_mn_limit,
        metadata.num_non_exiting_ctas,
        metadata.total_num_padded_tokens,
        metadata.num_tokens_per_expert,
        counters[:1],
        counters[1:],
        num_token_non_padded,
        topk,
        float(routed_scaling_factor),
        bool(renormalize),
        bool(apply_scale),
        int(local_expert_offset),
        int(num_local_experts),
        int(tile_tokens),
    )
    return K31FusedRouting(weights, ids, x_q, x_s, metadata)


# The fused route+quant launch's coverage: at or below it today's routed MoE
# takes bf16-packed routing weights, above it the router's fp32 weights.
PACKED_WEIGHTS_MAX_TOKENS = moe_route_quant_fused._MAX_TOKENS

# The serving dispatch keeps the unfused chain below this many rows. With the
# bitmap layout up to 16 rows the fused kernel also beats FlashInfer's
# single-block DynBlock routing, so every decode batch takes it.
SERVING_MIN_TOKENS = 1
