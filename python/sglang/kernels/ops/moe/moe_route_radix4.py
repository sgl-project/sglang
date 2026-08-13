"""Four-bit radix-select router for K3 routing on CDNA (ROCm).

The ROCm counterpart to moe_route_radix, dispatched from
biased_grouped_topk_gpu's aiter branch for covered inputs; anything else falls
back to aiter. Kernel notes live in jit/csrc/moe/route_radix4_hip.cuh.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.jit.utils.common import is_hip_runtime

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_NUM_EXPERTS = 896
_TOPK = 16
# One block per token, so the grid outgrows the machine somewhere past a
# thousand tokens and the kernel turns throughput-bound, where spreading a token
# over four waves is a cost rather than a win. Measured break-even is ~1.5k
# tokens; below 1k the kernel still leads by 1.2x or more, and prefill-sized
# batches are far above either number.
_MAX_TOKENS = 1024

logger = logging.getLogger(__name__)


def _is_supported_cdna() -> bool:
    if not is_hip_runtime() or not torch.cuda.is_available():
        return False
    gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
    return any(arch in gcn_arch for arch in ("gfx942", "gfx950"))


@cache_once
def _jit_route_radix4_module() -> Module:
    return load_jit(
        "moe_route_radix4",
        cuda_files=["moe/route_radix4_hip.cuh"],
        cuda_wrappers=[("run", "RouteRadix4Kernel::run")],
        # No fast-math: expert-id selection must stay comparable to aiter under
        # ties and NaN.
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def available() -> bool:
    """CDNA only, and only if the kernel actually builds on this toolchain."""
    if not _is_supported_cdna():
        return False
    try:
        _jit_route_radix4_module()
        return True
    except Exception as e:  # pragma: no cover - toolchain dependent
        logger.warning(f"Failed to load the JIT ROCm radix router: {e}")
        return False


def covered(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    num_expert_group: Optional[int],
    topk_group: Optional[int],
) -> bool:
    """Specialized for K3 routing: [M, 896] row-contiguous scores, top-16,
    ungrouped, with the bias in the score dtype (what the aiter path feeds it).

    Grouped routing is excluded rather than emulated: the kernel ranks all 896
    experts at once and has no notion of masking whole groups out first.
    """
    return (
        scores.dim() == 2
        and scores.size(0) <= _MAX_TOKENS
        and scores.size(1) == _NUM_EXPERTS
        and int(topk) == _TOPK
        and scores.dtype in (torch.bfloat16, torch.float32)
        and bias.dtype == scores.dtype
        and scores.stride(1) == 1
        and bias.is_contiguous()
        and (num_expert_group or 1) == 1
        and (topk_group or 1) == 1
    )


def route_radix4(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (weights [M, topk] fp32, ids [M, topk] int32). Caller must have
    checked covered().

    Experts are ranked by sigmoid(score) + bias but the emitted weight is the
    plain sigmoid, scaled by routed_scaling_factor and, when renormalize is set,
    divided by the sum over the selected experts. A NaN ranking value is floored
    so it can never displace a finite expert, and experts that tie on the full
    ranking value go to the lower expert id, same as the CUDA router. aiter
    breaks those ties in its own wave64 traversal order, so on a tied row the two
    can name different experts; both are a valid top-k and the weights agree.
    Winners come out unordered, which the MoE sorting stage downstream does not
    care about.
    """
    M = scores.shape[0]
    out_w = torch.empty((M, topk), dtype=torch.float32, device=scores.device)
    out_i = torch.empty((M, topk), dtype=torch.int32, device=scores.device)
    _jit_route_radix4_module().run(
        scores,
        bias,
        out_w,
        out_i,
        topk,
        float(routed_scaling_factor),
        bool(renormalize),
    )
    return out_w, out_i
