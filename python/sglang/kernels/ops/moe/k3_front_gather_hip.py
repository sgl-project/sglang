"""TP8 all-gather of Kimi-K3's column-parallel MoE front on gfx950 (see
csrc/kimi_k3/comm/front_gather_hip.cuh): every rank's [T, 112] router logits
and [T, 448] latent columns into the full [T, 896] logits and [T, 3584]
latent, over communication/k3_ll_comm_hip's scratch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.communication.k3_ll_comm_hip import MAX_ROWS, WORLD_SIZE, LLComm

if TYPE_CHECKING:
    from tvm_ffi.module import Module

NUM_EXPERTS: int = 896
LATENT_SIZE: int = 3584
SLICE_SIZE: int = (NUM_EXPERTS + LATENT_SIZE) // WORLD_SIZE
TOPK: int = 16
SHARED_INTER: int = 768  # the shared expert's TP8 intermediate width


@cache_once
def _module() -> Module:
    from aiter.jit.core import AITER_CSRC_DIR

    return load_jit(
        "kimi_k3_front_gather",
        cuda_files=["kimi_k3/comm/front_gather_hip.cuh"],
        cuda_wrappers=[("run", "k3_front_gather::K3FrontGatherKernel::run")],
        # AITER's CustomAllreduce, whose meta buffers hold the gather scratch
        extra_include_paths=[f"{AITER_CSRC_DIR}/include"],
        extra_cuda_cflags=["-O3"],
    )


def situ_covered(gate_up: torch.Tensor, slices: torch.Tensor) -> bool:
    """Whether front_gather can activate the shared expert's gate_up."""
    return (
        gate_up.dtype == torch.bfloat16
        and gate_up.shape == (slices.shape[0], 2 * SHARED_INTER)
        and gate_up.stride() == slices.stride()
        and gate_up.data_ptr() % 16 == 0
    )


def covered(slices: torch.Tensor) -> bool:
    return (
        0 < slices.shape[0] <= MAX_ROWS
        and slices.dtype == torch.bfloat16
        and slices.shape[1] == SLICE_SIZE
        and slices.stride(1) == 1
        and slices.stride(0) % 8 == 0
        and slices.data_ptr() % 16 == 0
    )


class FrontGather(NamedTuple):
    logits: torch.Tensor  # [T, 896] bf16
    latent: torch.Tensor  # [T, 3584] bf16
    # MXFP8 stage-1 activations of the latent (quant): [T, 3584] fp8 e4m3 and
    # [T, 112] e8m0 token-major scales, for aiter's mxfp4_moe_sort_fwd
    a1: Optional[torch.Tensor] = None
    a1_scale: Optional[torch.Tensor] = None
    # router top-k (bias given): [T, 16] fp32 weights and int32 ids
    topk_weights: Optional[torch.Tensor] = None
    topk_ids: Optional[torch.Tensor] = None
    # the shared expert's SiTU activation (gate_up given): [T, 768] bf16
    situ: Optional[torch.Tensor] = None


def front_gather(
    slices: torch.Tensor,
    comm: LLComm,
    *,
    quant: bool = False,
    bias: Optional[torch.Tensor] = None,
    routed_scaling_factor: float = 1.0,
    renormalize: bool = True,
    gate_up: Optional[torch.Tensor] = None,
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
) -> FrontGather:
    """Gather every rank's slices [T, 112 + 448] (rank r: experts
    [112 r, 112 r + 112), latent columns [448 r, 448 r + 448)) into the full
    logits and latent; quant and bias (the router's correction bias, bf16 or
    fp32 [896]) add the MXFP8 latent and the top-16 routes (sigmoid scores,
    biased selection, as kimi_k3_route_sort selects them). gate_up ([T, 1536]
    bf16 with the slices' row stride: the shared expert's columns of the same
    GEMM output) adds its SiTU activation as situ_and_mul computes it."""
    rows = slices.shape[0]
    device = slices.device
    logits = slices.new_empty((rows, NUM_EXPERTS))
    latent = slices.new_empty((rows, LATENT_SIZE))
    unread = logits
    a1 = (
        torch.empty((rows, LATENT_SIZE), dtype=torch.uint8, device=device)
        if quant
        else unread
    )
    a1_scale = (
        torch.empty((rows, LATENT_SIZE // 32), dtype=torch.uint8, device=device)
        if quant
        else unread
    )
    route = bias is not None
    if route:
        topk_ids = torch.empty((rows, TOPK), dtype=torch.int32, device=device)
        topk_weights = torch.empty((rows, TOPK), dtype=torch.float32, device=device)
    else:
        bias = topk_ids = topk_weights = unread
    situ = gate_up is not None
    situ_out = slices.new_empty((rows, SHARED_INTER)) if situ else unread
    _module().run(
        comm.comm._ptr,
        slices,
        logits,
        latent,
        a1,
        a1_scale,
        comm.epochs,
        quant,
        bias,
        topk_ids,
        topk_weights,
        route,
        float(routed_scaling_factor),
        renormalize,
        gate_up if situ else unread,
        situ_out,
        situ,
        float(situ_beta),
        float(situ_linear_beta),
    )
    return FrontGather(
        logits,
        latent,
        a1.view(torch.float8_e4m3fn) if quant else None,
        a1_scale.view(torch.float8_e8m0fnu) if quant else None,
        topk_weights if route else None,
        topk_ids if route else None,
        situ_out if situ else None,
    )
