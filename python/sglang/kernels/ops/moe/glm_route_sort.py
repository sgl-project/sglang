"""GLM-5.2 routing plus AITER's general MoE sort in one launch (ROCm).

Covers the routing aiter serves GLM-5.2 with (256 experts, ungrouped sigmoid +
fp32 correction bias, top-8, one fused shared expert appended at id 256) and
writes, from the same launch, the sort AITER's fused_moe would run on its
result under the EP expert mask, plus the zeroed accumulation buffer. Kernel
notes live in jit/csrc/moe/glm_route_sort_hip.cuh.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once, empty_sentinel, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

NUM_EXPERTS = 256
ROUTED_TOPK = 8
TOPK = ROUTED_TOPK + 1
MAX_TOKENS = 64
# Up to this many tokens one block routes (one wave each) and sorts without a
# hand-off; past it blocks of MULTI_BLOCK_TOKENS tokens route in parallel and
# the last to finish sorts. Measured on MI355X with GLM-5.2's EP4 shapes.
SINGLE_BLOCK_MAX_TOKENS = 8
MULTI_BLOCK_TOKENS = 4

logger = logging.getLogger(__name__)

# Per (device, stream) arrival counter for multi-block launches; launches in
# flight on two streams must not share it.
_SORT_WS: dict[tuple[torch.device, int], torch.Tensor] = {}


def supported_hardware() -> bool:
    hip = current_arch().hip
    return hip is not None and hip.gcn_arch in ("gfx942", "gfx950")


@cache_once
def build() -> Module:
    return load_jit(
        "glm_route_sort",
        cuda_files=["moe/glm_route_sort_hip.cuh"],
        cuda_wrappers=[("run", "GlmRouteSortKernel::run")],
        # No fast-math: the sigmoid has to round as aiter's does.
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def available() -> bool:
    if not supported_hardware():
        return False
    try:
        build()
        return True
    except Exception as e:  # pragma: no cover - toolchain dependent
        logger.warning(f"Failed to load the JIT GLM route+sort kernel: {e}")
        return False


def covered(scores: torch.Tensor, bias: torch.Tensor) -> bool:
    return (
        scores.dim() == 2
        and 1 <= scores.size(0) <= MAX_TOKENS
        and scores.size(1) == NUM_EXPERTS
        and scores.dtype in (torch.float32, torch.bfloat16)
        and scores.stride(1) == 1
        and bias.dim() == 1
        and bias.dtype == torch.float32
        and bias.is_contiguous()
    )


def tokens_per_block(n_tokens: int) -> int:
    return n_tokens if n_tokens <= SINGLE_BLOCK_MAX_TOKENS else MULTI_BLOCK_TOKENS


def ready(device: torch.device, n_tokens: int) -> bool:
    """Whether a launch for n_tokens may run on the current stream now. A
    multi-block launch's counter must start at zero, and one first allocated
    during graph capture would only be zeroed when that graph replays."""
    if n_tokens <= tokens_per_block(n_tokens):
        return True
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    return key in _SORT_WS or not torch.cuda.is_current_stream_capturing()


def _sort_workspace(device: torch.device) -> torch.Tensor:
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    ws = _SORT_WS.get(key)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "glm_route_sort first ran multi-block on this stream during graph "
                "capture; run it once eagerly on the capture stream first"
            )
        # Zeroed once; every launch leaves it ready for the next one.
        ws = torch.zeros(1, dtype=torch.int32, device=device)
        _SORT_WS[key] = ws
    return ws


def sorted_capacity(n_tokens: int, block_m: int) -> int:
    """aiter moe_sorting's allocation for the 257 global experts."""
    padded = n_tokens * TOPK + (NUM_EXPERTS + 1) * block_m - TOPK
    return (padded + block_m - 1) // block_m * block_m


def glm_route_sort(
    scores: torch.Tensor,
    bias: torch.Tensor,
    *,
    routed_begin: int,
    num_routed: int,
    block_m: int,
    routed_scaling_factor: float,
    shared_weight: float,
    renormalize: bool = True,
    moe_buf: Optional[torch.Tensor] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Returns (weights [M, 9] fp32, ids [M, 9] int32, sort) where sort holds
    sorted_token_ids, sorted_weights, sorted_expert_ids and num_valid_ids in
    aiter moe_sorting's layout for the mask keeping global experts
    [routed_begin, routed_begin + num_routed) and the shared expert 256.
    A given moe_buf [M, D] bf16 is zeroed in the same launch."""
    M = scores.shape[0]
    device = scores.device
    capacity = sorted_capacity(M, block_m)
    out_w = torch.empty((M, TOPK), dtype=torch.float32, device=device)
    out_i = torch.empty((M, TOPK), dtype=torch.int32, device=device)
    sorted_token_ids = torch.empty(capacity, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(capacity, dtype=torch.float32, device=device)
    sorted_expert_ids = torch.empty(
        capacity // block_m, dtype=torch.int32, device=device
    )
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    tpb = tokens_per_block(M)
    sort_ws = (
        _sort_workspace(device) if M > tpb else empty_sentinel(device, torch.int32)
    )
    build().run(
        scores,
        bias,
        out_w,
        out_i,
        sort_ws,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf if moe_buf is not None else empty_sentinel(device, torch.bfloat16),
        (
            num_token_non_padded.to(torch.int32).reshape(1)
            if num_token_non_padded is not None
            else empty_sentinel(device, torch.int32)
        ),
        int(routed_begin),
        int(num_routed),
        int(block_m),
        tpb,
        float(routed_scaling_factor),
        float(shared_weight),
        bool(renormalize),
    )
    sort = {
        "sorted_token_ids": sorted_token_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
    }
    return out_w, out_i, sort
