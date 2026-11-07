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
    from flashinfer.fused_moe.core import TrtllmMoERoutingMetadataSlot
    from tvm_ffi.module import Module

# NOTE(nvpohanh): The Kimi-K3 / K3.1 MoE only: sigmoid + bias top-16 of 896, no groups.
_KIMI_K3_HIDDEN_SIZE = 3584
_KIMI_K3_NUM_EXPERTS = 896
_KIMI_K3_TOP_K = 16
_MAX_TOKENS = 256


class FusedRouting(NamedTuple):
    weights: torch.Tensor
    ids: torch.Tensor
    x_q: torch.Tensor
    x_s: torch.Tensor
    metadata: TrtllmMoERoutingMetadataSlot


def max_num_ctas(num_tokens: int, tile_tokens: int) -> int:
    remaining = num_tokens * _KIMI_K3_TOP_K
    filled = min(_KIMI_K3_NUM_EXPERTS, remaining)
    return filled + (remaining - filled) // tile_tokens


@cache_once
def _jit_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "kimi_k3_fused_routing",
        *args,
        cuda_files=["moe/kimi_k3_fused_routing.cuh"],
        cuda_wrappers=[("run", f"KimiK3FusedRoutingKernel<{args}>::run")],
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
            f"Failed to load the JIT Kimi-K3 fused routing kernel: {e}"
        )
        return False


_counters: dict[int, torch.Tensor] = {}


def _counter(device: torch.device) -> torch.Tensor:
    index = device.index if device.index is not None else torch.cuda.current_device()
    counter = _counters.get(index)
    if counter is None:
        counter = torch.zeros(1, dtype=torch.int32, device=device)
        _counters[index] = counter
    return counter


def covered(
    scores: torch.Tensor, bias: torch.Tensor, topk: int, x: torch.Tensor
) -> bool:
    return (
        moe_route_radix.covered(scores, bias, topk)
        and x.dim() == 2
        and 0 < x.shape[0] <= _MAX_TOKENS
        and x.shape[1] == _KIMI_K3_HIDDEN_SIZE
        and x.dtype in (torch.bfloat16, torch.float32)
        and x.stride(1) == 1
        and x.data_ptr() % 32 == 0
        and (x.stride(0) * x.element_size()) % 32 == 0
    )


def fused_routing(
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
) -> FusedRouting:
    from flashinfer.fused_moe.core import TrtllmMoERoutingMetadataSlot

    M = scores.shape[0]
    device = scores.device
    num_ctas = max_num_ctas(M, tile_tokens)

    def i32(*shape):
        return torch.empty(shape, dtype=torch.int32, device=device)

    weights = torch.empty((M, topk), dtype=torch.float32, device=device)
    ids = i32(M, topk)
    x_q = torch.empty(
        (M, _KIMI_K3_HIDDEN_SIZE), dtype=torch.float8_e4m3fn, device=device
    )
    x_s = i32(M, _KIMI_K3_HIDDEN_SIZE // 32 // 4)
    metadata = TrtllmMoERoutingMetadataSlot(
        tile_n=tile_tokens,
        total_num_padded_tokens=i32(1),
        expanded_idx_to_permuted_idx=i32(M * topk),
        permuted_idx_to_token_idx=i32(num_ctas * tile_tokens + 1),
        expert_weights=torch.empty((M, topk), dtype=weights_dtype, device=device),
        expert_count_histogram=i32(2 * max(_KIMI_K3_NUM_EXPERTS, 256)),
        num_tokens_per_expert=i32(_KIMI_K3_NUM_EXPERTS),
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
        _counter(device),
        num_token_non_padded,
        topk,
        float(routed_scaling_factor),
        bool(renormalize),
        bool(apply_scale),
        int(local_expert_offset),
        int(num_local_experts),
        int(tile_tokens),
    )
    return FusedRouting(weights, ids, x_q, x_s, metadata)


def expert_weights_dtype(num_tokens: int) -> torch.dtype:
    if num_tokens <= moe_route_quant_fused._MAX_TOKENS:
        return torch.bfloat16
    return torch.float32
