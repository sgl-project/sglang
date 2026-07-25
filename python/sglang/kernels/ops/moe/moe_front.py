"""K3 MoE front: merged gate + routed_expert_down_proj GEMM, and the fp32 router.

`KimiK3MoE._forward_unfused` -- the path every EP-a2a / WideEP deployment takes --
runs three ops over the same `hidden_states [T, 7168]`:

    router_logits = gate(hidden_states)                     # [896, 7168]  12.85 MB
    topk_output   = topk(hidden_states, router_logits)
    routed_input  = routed_expert_down_proj(hidden_states)   # [3584, 7168] 51.4 MB

Two things make that faster, and which one to use depends only on the token count.

**route_radix_fp32** -- radix-select top-k over fp32 logits.  `route_radix` is
bf16-only, so the fp32 logits the cuBLAS gate GEMV produces silently fell back to
the Triton router: 7.04 us instead of 2.4 us per layer at T=1, across 92 MoE
layers.  Always a win, so it is unconditional.

**fused_front** -- the two GEMMs share their input, so their weights are merged
and one cuBLAS GEMM emits `[T, 896 + 3584]` fp32; a single epilogue kernel then
runs the top-k on the gate slice and casts the latent slice to bf16.  Routing
stays bit-identical to the fp32 path and routed_input comes out dense.

Measured in-graph on a GB300, us per MoE layer, with the copy a dense-input runner
(deep_gemm) needs:

    T            1     16    256   1024   1280   2560   4096  16384
    baseline   22.3   21.9   31.2   61.4   65.9  125.2  185.1  735.8
    merged     13.1   12.0   17.6   47.2   54.8  106.4  181.8  702.8
    radix-only 17.3   17.5   24.2   51.3   55.0  100.7  154.5  621.8

The merge stops paying once the GEMM is compute-bound: it saves one read of
`hidden_states` but costs doubled fp32 output traffic, and past ~1024 tokens the
second outweighs the first.  1024 is where the merged path still wins clearly
(47.2 vs 51.3); 1280-2048 is a wash and 2560 up belongs to the router-only path,
so the threshold sits at the last power of two that is unambiguous.

A bf16-output merged GEMM was measured too (fastest at T=1, 11.8 us).  It is not
used: bf16 rounds the router logits and moves the selected expert set on 2-25% of
rows depending on T, for ~1.2 us -- and from T>=8 it loses to the fp32 variant
anyway, because its routed_input is a strided slice a dense-input runner must copy.
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

NUM_EXPERTS = 896
TOPK = 16

# Above this token count the merged GEMM stops paying; see the table above.
MERGED_FRONT_MAX_TOKENS = 1024


@cache_once
def _jit_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "moe_front",
        *args,
        cuda_files=["moe/moe_front.cuh"],
        cuda_wrappers=[
            ("route_radix_fp32", f"RouteRadixFp32Kernel<{args}>::run"),
            ("front_epilogue", f"FusedFrontEpilogueKernel<{args}>::run"),
        ],
        # No fast-math: scoring and expert-id selection must stay comparable to
        # route_radix / the Triton router under ties and NaN.
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def available() -> bool:
    import logging

    try:
        _jit_module()
        return True
    except Exception as e:  # pragma: no cover - toolchain dependent
        logging.getLogger(__name__).warning(f"Failed to load the JIT MoE front kernels: {e}")
        return False


# --------------------------------------------------------------------------
# router-only: fp32 logits -> top-k
# --------------------------------------------------------------------------


def route_radix_fp32_covered(
    logits: torch.Tensor, bias: Optional[torch.Tensor], topk: int
) -> bool:
    """Row-dense [M, 896] fp32 logits, fp32 bias, top-16."""
    return (
        logits.dim() == 2
        and logits.dtype == torch.float32
        and logits.shape[1] == NUM_EXPERTS
        and logits.stride(1) == 1
        and logits.stride(0) == NUM_EXPERTS
        and bias is not None
        and bias.dtype == torch.float32
        and bias.numel() == NUM_EXPERTS
        and int(topk) == TOPK
        and logits.shape[0] > 0
    )


def route_radix_fp32(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int = TOPK,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Radix-select top-k over fp32 logits.

    Winners come out in expert-id order, the same contract as
    ``moe_route_radix.route_radix(..., sorted=False)``.
    """
    M = logits.shape[0]
    weights = torch.empty((M, topk), dtype=torch.float32, device=logits.device)
    ids = torch.empty((M, topk), dtype=torch.int32, device=logits.device)
    _jit_module().route_radix_fp32(
        logits,
        correction_bias,
        weights,
        ids,
        topk,
        float(routed_scaling_factor if routed_scaling_factor is not None else 1.0),
        bool(renormalize),
        bool(apply_routed_scaling_factor_on_output),
    )
    return weights, ids


# --------------------------------------------------------------------------
# merged front: [gate | down] GEMM -> top-k + routed_input
# --------------------------------------------------------------------------


def fused_front_covered(
    hidden_states: torch.Tensor,
    merged_weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    topk: int,
    latent: int,
) -> bool:
    """[T<=MERGED_FRONT_MAX_TOKENS, 7168] bf16 x [896 + latent, 7168] bf16, fp32
    bias, top-16, latent a multiple of 4."""
    return (
        hidden_states.dim() == 2
        and merged_weight.dim() == 2
        and hidden_states.dtype == torch.bfloat16
        and merged_weight.dtype == torch.bfloat16
        and bias is not None
        and bias.dtype == torch.float32
        and bias.numel() == NUM_EXPERTS
        and int(topk) == TOPK
        and merged_weight.shape[0] == NUM_EXPERTS + latent
        and merged_weight.shape[1] == hidden_states.shape[1]
        and latent % 4 == 0
        and 0 < hidden_states.shape[0] <= MERGED_FRONT_MAX_TOKENS
        and hidden_states.stride(1) == 1
        and merged_weight.stride(1) == 1
    )


def fused_front(
    hidden_states: torch.Tensor,
    merged_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    latent: int,
    topk: int = TOPK,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merged front GEMM + fused top-k/cast epilogue.

    Returns ``(topk_weights [M, topk] fp32, topk_ids [M, topk] int32,
    routed_input [M, latent] bf16)``.
    """
    M = hidden_states.shape[0]
    device = hidden_states.device

    # fp32 out keeps routing exact; the extra output traffic versus bf16 is
    # M x (896 + latent) x 2 bytes, negligible against the 64 MB weight read at
    # the sizes this path serves.
    merged = torch.mm(hidden_states, merged_weight.t(), out_dtype=torch.float32)

    weights = torch.empty((M, topk), dtype=torch.float32, device=device)
    ids = torch.empty((M, topk), dtype=torch.int32, device=device)
    routed = torch.empty((M, latent), dtype=torch.bfloat16, device=device)

    _jit_module().front_epilogue(
        merged,
        correction_bias,
        weights,
        ids,
        routed,
        topk,
        float(routed_scaling_factor if routed_scaling_factor is not None else 1.0),
        bool(renormalize),
        bool(apply_routed_scaling_factor_on_output),
    )
    return weights, ids, routed
