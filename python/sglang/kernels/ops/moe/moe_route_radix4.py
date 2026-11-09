"""Four-bit radix-select router for K3 routing on CDNA (ROCm).

The ROCm counterpart to moe_route_radix, dispatched from
biased_grouped_topk_gpu's aiter branch for covered inputs; anything else falls
back to aiter. Kernel notes live in jit/csrc/moe/route_radix4_hip.cuh.

route_radix4_with_sort also writes AITER's BM=16 MoE sort from the same launch,
for a caller that hands it to AITER's fused_moe in place of moe_sorting. Up to 64
tokens the last CTA to finish routing sorts; past that the last ceil(M / 16)
CTAs share the sort, one 16-token chunk each.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once, empty_sentinel, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_NUM_EXPERTS = 896
_TOPK = 16
_SORT_BLOCK_M = 16
# One block per token, so the grid outgrows the machine somewhere past a
# thousand tokens and the kernel turns throughput-bound, where spreading a token
# over four waves is a cost rather than a win. Measured break-even is ~1.5k
# tokens; below 1k the kernel still leads by 1.2x or more, and prefill-sized
# batches are far above either number.
_MAX_TOKENS = 1024

logger = logging.getLogger(__name__)

# Per (device, stream): the sort's arrival counters, which launches in flight on
# two streams must not share. Four int32 (see RouteRadix4Params::sort_ws).
_SORT_WS: dict[tuple[torch.device, int], torch.Tensor] = {}


def supported_hardware() -> bool:
    """Whether this device is one the kernel targets, before asking whether it
    builds. The kernel is wave64 and GFX9 DPP throughout, hence gfx942/gfx950."""
    hip = current_arch().hip
    return hip is not None and hip.gcn_arch in ("gfx942", "gfx950")


@cache_once
def build() -> Module:
    """Compile and load the kernel, raising if the toolchain cannot."""
    return load_jit(
        "moe_route_radix4",
        cuda_files=["moe/route_radix4_hip.cuh"],
        cuda_wrappers=[
            ("run", "RouteRadix4Kernel::run"),
            ("run_with_sort", "RouteRadix4SortKernel::run"),
        ],
        # No fast-math: expert-id selection must stay comparable to aiter under
        # ties and NaN.
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def available() -> bool:
    """Whether dispatch may use the kernel: targeted hardware, and a kernel that
    builds on this toolchain.

    A build failure is swallowed on purpose, since serving would rather fall back
    to aiter than refuse to start. That makes this the wrong gate for a test,
    which wants a kernel that stopped compiling to be a failure and not a skip --
    the tests pair supported_hardware() with build() instead.
    """
    if not supported_hardware():
        return False
    try:
        build()
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
    divided by the sum over the selected experts. A NaN ranking value keys below
    every number, so it can never displace one.

    A row is what aiter would have produced, expert for expert and column for
    column: winners come out highest ranking value first, and experts that tie on
    the full ranking value are separated the way aiter's wave64 walk reaches them
    (kAiterTieLaneRank in the kernel). Matching on tied rows too is what keeps a
    token's routing from depending on the batch size, since batches past
    _MAX_TOKENS fall back to aiter. None of it depends on how the kernel's
    compaction raced, so a row also repeats bit for bit run to run.
    """
    M = scores.shape[0]
    out_w = torch.empty((M, topk), dtype=torch.float32, device=scores.device)
    out_i = torch.empty((M, topk), dtype=torch.int32, device=scores.device)
    build().run(
        scores,
        bias,
        out_w,
        out_i,
        topk,
        float(routed_scaling_factor),
        bool(renormalize),
    )
    return out_w, out_i


def fused_sort_ready(device: torch.device) -> bool:
    """Whether route_radix4_with_sort may launch on the current stream now.

    Its arrival counters must start at zero, and counters first allocated
    during CUDA graph capture are only zeroed when that one graph replays.
    """
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    return key in _SORT_WS or not torch.cuda.is_current_stream_capturing()


def _sort_workspace(device: torch.device) -> torch.Tensor:
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    ws = _SORT_WS.get(key)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "route_radix4_with_sort first ran on this stream during graph "
                "capture; run it once eagerly on the capture stream first"
            )
        # Zeroed once; every launch leaves it ready for the next one.
        ws = torch.zeros(4, dtype=torch.int32, device=device)
        _SORT_WS[key] = ws
    return ws


def _sorted_capacity(n_tokens: int) -> int:
    """AITER's moe_sorting capacity: every route, plus up to BM - 1 padding
    rows for each expert that can be active."""
    routes = n_tokens * _TOPK
    active = min(_NUM_EXPERTS, routes)
    blocks = (routes + active * (_SORT_BLOCK_M - 1) + _SORT_BLOCK_M - 1) // (
        _SORT_BLOCK_M
    )
    return blocks * _SORT_BLOCK_M


def route_radix4_with_sort(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    *,
    moe_buf: Optional[torch.Tensor] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """route_radix4 plus AITER's BM=16 sort of its routes, in one launch.

    Returns (weights, ids, aux), where aux is the dict AITER's
    register_presorted_moe takes. A given moe_buf ([M, D] packed bf16, D a
    multiple of 8) is zeroed in the same launch, and aux records it. Rows at or
    past num_token_non_padded come out as expert 0 with weight 0, which is what
    select_experts masks CUDA-graph padding rows to on HIP.
    """
    M = scores.shape[0]
    device = scores.device
    capacity = _sorted_capacity(M)
    out_w = torch.empty((M, topk), dtype=torch.float32, device=device)
    out_i = torch.empty((M, topk), dtype=torch.int32, device=device)
    sorted_token_ids = torch.empty(capacity, dtype=torch.int32, device=device)
    sorted_expert_ids = torch.empty(
        capacity // _SORT_BLOCK_M, dtype=torch.int32, device=device
    )
    sorted_weights = torch.empty(capacity, dtype=torch.float32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    reverse_sorted = torch.empty(M * _TOPK, dtype=torch.int32, device=device)
    m_indices = torch.empty(capacity, dtype=torch.int32, device=device)
    build().run_with_sort(
        scores,
        bias,
        out_w,
        out_i,
        _sort_workspace(device),
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        reverse_sorted,
        m_indices,
        moe_buf if moe_buf is not None else empty_sentinel(device, torch.bfloat16),
        (
            num_token_non_padded.to(torch.int32).reshape(1)
            if num_token_non_padded is not None
            else empty_sentinel(device, torch.int32)
        ),
        topk,
        float(routed_scaling_factor),
        bool(renormalize),
    )
    aux = {
        "topk_ids_ptr": out_i.data_ptr(),
        "topk_weights_ptr": out_w.data_ptr(),
        "M": M,
        "block_m": _SORT_BLOCK_M,
        "num_experts": _NUM_EXPERTS,
        "topk": _TOPK,
        "sorted_token_ids": sorted_token_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "m_indices": m_indices,
        "reverse_sorted": reverse_sorted,
        "moe_buf_zeroed": moe_buf is not None,
        "moe_buf_ptr": moe_buf.data_ptr() if moe_buf is not None else 0,
    }
    return out_w, out_i, aux
