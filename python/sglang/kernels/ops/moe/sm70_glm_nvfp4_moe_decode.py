"""Batch-1 GLM-5.3 NVFP4 MoE decode for SM70.

Gate and down use Volta HMMA with fp32 accumulation, split over K across
warps; the summation order differs from Marlin in the last fp16 bit. SiLU is
bitwise the fp16 swiglu_limit_func Marlin calls, and the down epilogue multiplies the
route weight before the global scale. Works for EP (whole experts) and TP
(experts sliced along the intermediate dim).
"""

from __future__ import annotations

import logging

import msgspec
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

HIDDEN = 4096
TOPK = 8
# One Marlin-layout scratch for prefill. Decode reads the in-place HMMA pack,
# so this is not touched inside the CUDA graph. Holds every local expert.
_MARLIN_SCRATCH: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


class Sm70SharedExpertPack(msgspec.Struct, frozen=True):
    """One dense NVFP4 shared expert in the HMMA pack, folded into the routed
    decode as one extra row per token. Packs are [N/32, K/16, 32] int64,
    scales [K/16, N] fp8 bytes, global scales one fp32."""

    w13_packed: torch.Tensor
    w13_scales: torch.Tensor
    w13_global: torch.Tensor
    w2_packed: torch.Tensor
    w2_scales: torch.Tensor
    w2_global: torch.Tensor


def sm70_glm_nvfp4_moe_decode_available() -> bool:
    if not envs.SGLANG_SM70_GLM_NVFP4_MOE_DECODE.get():
        return False
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability() == (7, 0)
    except (AssertionError, RuntimeError):
        return False


def convert_marlin_experts_to_hmma(weight: torch.Tensor) -> torch.Tensor:
    """Replace one layer's Marlin expert pack with the lane-major HMMA pack.

    The two layouts are the same number of bytes. Converting in place avoids
    a second copy of every routed expert (~18 GB), which does not fit beside
    the 23 GB weight footprint.
    """
    from sglang.kernels.ops.gemm.sm70_glm_nvfp4_gemv import _load

    ext = _load()
    if ext is None:
        raise RuntimeError("GLM NVFP4 HMMA extension is unavailable")
    experts, groups, packed_n = weight.shape
    n = packed_n // 2
    packed = torch.empty(
        (experts, n // 32, groups, 32), dtype=torch.int64, device=weight.device
    )
    for expert in range(experts):
        packed[expert].copy_(ext.repack(weight[expert].contiguous()))
    return packed


def glm_marlin_experts_supported(w13: torch.Tensor, w2: torch.Tensor) -> bool:
    """Marlin NVFP4 packs: w13 [E, HIDDEN/16, 4 * I], w2 [E, I/16, 2 * HIDDEN]."""
    if w13.dim() != 3 or w2.dim() != 3 or w13.shape[0] != w2.shape[0]:
        return False
    intermediate = w2.shape[1] * 16
    return (
        w13.shape[1] == HIDDEN // 16
        and w13.shape[2] == 4 * intermediate
        and w2.shape[2] == 2 * HIDDEN
        # unpack_experts_into needs N % 256 for the gate_up pack.
        and (2 * intermediate) % 256 == 0
    )


def glm_hmma_experts_packed(w13: torch.Tensor, w2: torch.Tensor) -> bool:
    """HMMA packs [E, N/32, K/16, 32] int64: w13 N = 2 * I, K = HIDDEN; w2 N = HIDDEN, K = I."""
    return (
        w13.dtype == torch.int64
        and w2.dtype == torch.int64
        and w13.dim() == 4
        and w2.dim() == 4
        and w13.shape[0] == w2.shape[0]
        and w13.shape[2] == HIDDEN // 16
        and w13.shape[3] == 32
        and w2.shape[1] == HIDDEN // 32
        and w2.shape[3] == 32
        and w13.shape[1] * 32 == 2 * w2.shape[2] * 16
    )


def ensure_glm_marlin_scratch(
    device: torch.device, w13_shape: torch.Size, w2_shape: torch.Size
) -> None:
    slot = _MARLIN_SCRATCH.get(device.index)
    if (
        slot is not None
        and slot[0].shape[1:] == w13_shape[1:]
        and slot[1].shape[1:] == w2_shape[1:]
        and slot[0].shape[0] >= w13_shape[0]
    ):
        return
    _MARLIN_SCRATCH[device.index] = (
        torch.empty(tuple(w13_shape), dtype=torch.int32, device=device),
        torch.empty(tuple(w2_shape), dtype=torch.int32, device=device),
    )


def materialize_glm_marlin(
    w13_packed: torch.Tensor, w2_packed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack HMMA experts into the shared Marlin scratch for a prefill GEMM."""
    from sglang.kernels.ops.gemm.sm70_glm_nvfp4_gemv import _load

    ext = _load()
    if ext is None:
        raise RuntimeError("GLM NVFP4 HMMA extension is unavailable")
    slot = _MARLIN_SCRATCH.get(w13_packed.device.index)
    if slot is None:
        raise RuntimeError("GLM Marlin scratch was not allocated at weight load")
    w13_scratch, w2_scratch = slot
    experts = w13_packed.shape[0]
    ext.unpack_experts_into(w13_packed.contiguous(), w13_scratch[:experts])
    ext.unpack_experts_into(w2_packed.contiguous(), w2_scratch[:experts])
    return w13_scratch[:experts], w2_scratch[:experts]


def sm70_glm_nvfp4_moe_decode(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    w13_global: torch.Tensor,
    w2_global: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    swiglu_limit: float = 0.0,
    routed_scale: float = 1.0,
    shared: Sm70SharedExpertPack | None = None,
) -> torch.Tensor:
    """Routed experts, plus the shared expert when `shared` is given; the result
    is then bitwise the routed output followed by `+= shared_expert(x)`."""
    from sglang.kernels.ops.gemm.sm70_glm_nvfp4_gemv import _load
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import moe_sum_reduce_triton
    from sglang.kernels.ops.moe.sm70_moe_glue import sum_routes_add_shared, swiglu_clamp

    ext = _load()
    if ext is None:
        raise RuntimeError("GLM NVFP4 HMMA extension is unavailable")
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    batch = hidden_states.shape[0]
    routes = batch * TOPK
    topk_ids = topk_ids.reshape(routes).to(torch.int32).contiguous()
    topk_weights = topk_weights.reshape(routes).float().contiguous()
    w13_scales = w13_scales.contiguous()
    w2_scales = w2_scales.contiguous()
    w13_global = w13_global.reshape(-1).float().contiguous()
    w2_global = w2_global.reshape(-1).float().contiguous()
    if w13.dtype != torch.int64 or w2.dtype != torch.int64:
        raise RuntimeError("GLM HMMA decode expects the in-place expert pack")
    w13_packed = w13 if w13.is_contiguous() else w13.contiguous()
    w2_packed = w2 if w2.is_contiguous() else w2.contiguous()
    # Packs are [E, N/32, K/16, 32]; w2's K is the local intermediate size.
    intermediate = w2_packed.shape[2] * 16
    rows = routes if shared is None else routes + batch
    device = hidden_states.device
    gate = torch.empty((rows, 2 * intermediate), dtype=torch.float16, device=device)
    activated = torch.empty((rows, intermediate), dtype=torch.float16, device=device)
    down = torch.empty((rows, HIDDEN), dtype=torch.float16, device=device)
    output = torch.empty_like(hidden_states)
    # Verify batches: stream each expert once for every token that routes to it.
    group = batch > 1
    if shared is None:
        ext.moe_hmma_splitk(
            hidden_states, w13_packed, w13_scales, w13_global,
            topk_ids, topk_weights, gate, False, False, 0, group,
        )
        swiglu_clamp(activated, gate, float(swiglu_limit))
        ext.moe_hmma_splitk(
            activated, w2_packed, w2_scales, w2_global,
            topk_ids, topk_weights, down, True, True, 0, group,
        )
        moe_sum_reduce_triton(down.view(batch, TOPK, HIDDEN), output, float(routed_scale))
        return output
    ext.moe_hmma_splitk_shared(
        hidden_states, w13_packed, w13_scales, w13_global, topk_ids, topk_weights,
        gate, False, 0, shared.w13_packed, shared.w13_scales, shared.w13_global, batch, group,
    )
    swiglu_clamp(activated, gate, float(swiglu_limit))
    ext.moe_hmma_splitk_shared(
        activated, w2_packed, w2_scales, w2_global, topk_ids, topk_weights,
        down, True, 0, shared.w2_packed, shared.w2_scales, shared.w2_global, batch, group,
    )
    sum_routes_add_shared(output, down, TOPK, float(routed_scale))
    return output
