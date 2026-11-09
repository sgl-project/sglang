"""Qwen3.8 decode MoE front in one launch: top-10 of 512 routing, trtllm-gen
routing metadata for a prepared-routing (DA) body, and the NVFP4 activation quant.

Specialized for Qwen3.8 Flash-Next at TP1/EP1: 512 experts, RenormalizeNaive
top-10 over BF16 router logits, a 2560-wide BF16 activation row quantized to
NVFP4 (16-element E4M3 scales in linear layout, one per-tensor global scale),
1-512 rows (one CTA routes 1-32 rows, a CTA cluster 33-512). Ids, weights,
quantized activations and every metadata field but the row order inside an
expert are bit-identical to FlashInfer 0.7.2's routing kernels and CuTe-DSL
fp4_quantize; see the kernel header. Adapted from the Kimi-K3 fused routing
(sgl-project/sglang #48586).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

NUM_EXPERTS = 512
TOP_K = 10
HIDDEN_SIZE = 2560
MAX_TOKENS = 32  # single-CTA kernel; the cluster kernel takes the rest
MAX_ROWS = 512  # cluster kernel: 16 CTAs x 32 rows (8 x 64 where 16-CTA clusters cannot run)
_SF_VEC = 16


class FusedRouting(NamedTuple):
    ids: torch.Tensor  # [M, 10] int32
    metadata: Any  # flashinfer TrtllmMoERoutingMetadataSlot; expert_weights is [M, 10] bf16


def max_num_ctas(num_tokens: int, tile_tokens: int) -> int:
    # FlashInfer Routing::getMaxNumCtasInBatchDim(num_tokens, 10, 512, tile).
    remaining = num_tokens * TOP_K
    filled = min(NUM_EXPERTS, remaining)
    return filled + (remaining - filled) // tile_tokens


@cache_once
def _jit_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "q38_fused_routing",
        *args,
        cuda_files=["moe/q38_fused_routing.cuh"],
        cuda_wrappers=[
            ("run", f"Q38FusedRoutingKernel<{args}>::run"),
            ("cluster_ctas", f"Q38FusedRoutingKernel<{args}>::cluster_ctas"),
        ],
        # FlashInfer builds its routing kernels with fast math; the softmax must
        # lower identically. The quant half is inline PTX and unaffected.
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


@cache_once
def available() -> bool:
    import logging

    try:
        _jit_module()
        return True
    except Exception as e:  # pragma: no cover - toolchain dependent
        logging.getLogger(__name__).warning(
            f"Failed to load the JIT Q38 fused routing kernel: {e}"
        )
        return False


def cluster_ctas() -> int:
    """CTAs per routing cluster this process launches for 33-512 rows (16, or 8 on a device without 16-CTA clusters)."""
    return int(_jit_module().cluster_ctas())


def _rows_aligned(t: torch.Tensor) -> bool:
    return t.stride(1) == 1 and t.stride(0) % 8 == 0 and t.data_ptr() % 16 == 0


def covered(router_logits: torch.Tensor, x: torch.Tensor, global_scale: torch.Tensor) -> bool:
    return (
        x.dim() == 2
        and 1 <= x.shape[0] <= MAX_ROWS
        and x.shape[1] == HIDDEN_SIZE
        and x.dtype == torch.bfloat16
        and _rows_aligned(x)
        and router_logits.shape == (x.shape[0], NUM_EXPERTS)
        and router_logits.dtype == torch.bfloat16
        and _rows_aligned(router_logits)
        and global_scale.dtype == torch.float32
        and global_scale.numel() == 1
        and global_scale.is_contiguous()
        and global_scale.device == x.device
    )


def alloc_quant_outputs(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """([M, 1280] uint8 packed e2m1 pairs, low nibble first; [M, 160] uint8 e4m3
    block scales, linear layout), as quantize_hidden_states_fp4 returns them."""
    M = x.shape[0]
    x_fp4 = torch.empty((M, HIDDEN_SIZE // 2), dtype=torch.uint8, device=x.device)
    x_sf = torch.empty((M, HIDDEN_SIZE // _SF_VEC), dtype=torch.uint8, device=x.device)
    return x_fp4, x_sf


def fused_routing(
    router_logits: torch.Tensor,
    x: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    tile_tokens: int,
) -> FusedRouting:
    """Writes x_fp4, x_sf, ids [M, 10] int32 and weights [M, 10] bf16 in place.
    Caller must have checked covered(). The metadata is sized as FlashInfer's
    routed call sizes its own (getMaxPermutedPaddedCount + 1 guard row)."""
    from flashinfer.fused_moe.core import TrtllmMoERoutingMetadataSlot

    M = x.shape[0]
    device = x.device
    num_ctas = max_num_ctas(M, tile_tokens)

    def i32(*shape):
        return torch.empty(shape, dtype=torch.int32, device=device)

    metadata = TrtllmMoERoutingMetadataSlot(
        tile_n=tile_tokens,
        total_num_padded_tokens=i32(1),
        expanded_idx_to_permuted_idx=i32(M * TOP_K),
        permuted_idx_to_token_idx=i32(num_ctas * tile_tokens + 1),
        expert_weights=weights,
        expert_count_histogram=i32(2 * NUM_EXPERTS),
        num_tokens_per_expert=i32(NUM_EXPERTS),
        cta_idx_xy_to_batch_idx=i32(num_ctas),
        cta_idx_xy_to_mn_limit=i32(num_ctas),
        num_non_exiting_ctas=i32(1),
    )
    _jit_module().run(
        router_logits,
        x,
        global_scale.reshape(1),
        x_fp4,
        x_sf.view(torch.uint8),
        ids,
        weights,
        metadata.expanded_idx_to_permuted_idx,
        metadata.permuted_idx_to_token_idx,
        metadata.cta_idx_xy_to_batch_idx,
        metadata.cta_idx_xy_to_mn_limit,
        metadata.num_non_exiting_ctas,
        metadata.total_num_padded_tokens,
        metadata.num_tokens_per_expert,
        int(tile_tokens),
    )
    return FusedRouting(ids, metadata)
