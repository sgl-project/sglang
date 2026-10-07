"""Four-bit radix-select router for K3 routing on CDNA (ROCm).

The ROCm counterpart to moe_route_radix, dispatched from
biased_grouped_topk_gpu's aiter branch for covered inputs; anything else falls
back to aiter. Kernel notes live in jit/csrc/moe/route_radix4_hip.cuh.

Decode M<=64 can fuse the BM=16 expert sort into the same launch so AITER's
follow-up ``mxfp4_moe_sort`` is skipped.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.jit.utils.common import is_hip_runtime

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_NUM_EXPERTS = 896
_TOPK = 16
_SORT_BM = 16
_K3_MODEL_DIM = 3584
_FUSE_SORT_MAX_TOKENS = 64
# One block per token, so the grid outgrows the machine somewhere past a
# thousand tokens and the kernel turns throughput-bound, where spreading a token
# over four waves is a cost rather than a win. Measured break-even is ~1.5k
# tokens; below 1k the kernel still leads by 1.2x or more, and prefill-sized
# batches are far above either number.
_MAX_TOKENS = 1024

logger = logging.getLogger(__name__)

_ARRIVAL: dict[str, torch.Tensor] = {}
_EMPTY_MOE_BUF: dict[str, torch.Tensor] = {}


def supported_hardware() -> bool:
    """Whether this device is one the kernel targets, before asking whether it
    builds. The kernel is wave64 and GFX9 DPP throughout, hence gfx942/gfx950."""
    if not is_hip_runtime() or not torch.cuda.is_available():
        return False
    gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
    return any(arch in gcn_arch for arch in ("gfx942", "gfx950"))


@cache_once
def build() -> Module:
    """Compile and load the kernel, raising if the toolchain cannot."""
    moe_csrc = Path(__file__).resolve().parents[2] / "jit" / "csrc" / "moe"
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
        extra_include_paths=[str(moe_csrc)],
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


def _max_sorted(n_tokens: int) -> int:
    active = min(_NUM_EXPERTS, n_tokens * _TOPK)
    return (
        (n_tokens * _TOPK + active * (_SORT_BM - 1) + _SORT_BM - 1) // _SORT_BM
    ) * _SORT_BM


def _device_key(device: torch.device) -> str:
    return f"{device.type}:{device.index}"


def _arrival_buf(device: torch.device) -> torch.Tensor:
    key = _device_key(device)
    buf = _ARRIVAL.get(key)
    if buf is None or buf.device != device:
        buf = torch.zeros(1, dtype=torch.int32, device=device)
        _ARRIVAL[key] = buf
    return buf


def _empty_moe_buf(device: torch.device) -> torch.Tensor:
    key = _device_key(device)
    buf = _EMPTY_MOE_BUF.get(key)
    if buf is None or buf.device != device:
        buf = torch.empty((0, 0), dtype=torch.bfloat16, device=device)
        _EMPTY_MOE_BUF[key] = buf
    return buf


def _register_presorted(aux: dict) -> None:
    try:
        from aiter.fused_moe import register_k3_presorted_moe
    except Exception:
        return
    register_k3_presorted_moe(aux)


def should_fuse_sort(n_tokens: int) -> bool:
    """BM=16 a4w4 decode rows in the K3 FMOE table: M in 1..64 except the
    token=3 BM=32 hole. fused_moe still ignores the aux unless block_m is 16."""
    from sglang.srt.environ import envs

    if not envs.SGLANG_ROCM_K3_RADIX4_FUSE_SORT.get():
        return False
    return 1 <= n_tokens <= _FUSE_SORT_MAX_TOKENS


def route_radix4(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    *,
    fuse_sort: Optional[bool] = None,
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

    When ``fuse_sort`` is true (default for decode-sized M when the env gate is
    on), the same launch also emits AITER BM=16 sort metadata and registers it
    so ``fused_moe`` can skip ``mxfp4_moe_sort``.
    """
    M = scores.shape[0]
    out_w = torch.empty((M, topk), dtype=torch.float32, device=scores.device)
    out_i = torch.empty((M, topk), dtype=torch.int32, device=scores.device)
    if fuse_sort is None:
        fuse_sort = should_fuse_sort(M)
    if not fuse_sort:
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
    return _route_radix4_with_sort(
        scores, bias, topk, renormalize, routed_scaling_factor, out_w, out_i
    )


def route_radix4_with_sort(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    moe_buf: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Route and BM=16-sort in one launch. Returns weights, ids, and the sort
    aux dict fused_moe expects. Does not register the aux; tests use this."""
    M = scores.shape[0]
    out_w = torch.empty((M, topk), dtype=torch.float32, device=scores.device)
    out_i = torch.empty((M, topk), dtype=torch.int32, device=scores.device)
    aux = _launch_with_sort(
        scores, bias, topk, renormalize, routed_scaling_factor, out_w, out_i, moe_buf
    )
    return out_w, out_i, aux


def _route_radix4_with_sort(
    scores,
    bias,
    topk,
    renormalize,
    routed_scaling_factor,
    out_w,
    out_i,
):
    moe_buf = None
    try:
        from sglang.srt.layers.zero_copy_context import get_moe_output_spec

        moe_buf = get_moe_output_spec(
            torch.Size((scores.shape[0], _K3_MODEL_DIM)),
            torch.bfloat16,
            scores.device,
        )
    except Exception:
        moe_buf = None
    aux = _launch_with_sort(
        scores, bias, topk, renormalize, routed_scaling_factor, out_w, out_i, moe_buf
    )
    _register_presorted(aux)
    return out_w, out_i


def _launch_with_sort(
    scores,
    bias,
    topk,
    renormalize,
    routed_scaling_factor,
    out_w,
    out_i,
    moe_buf,
):
    M = scores.shape[0]
    device = scores.device
    max_sorted = _max_sorted(M)
    sorted_token_ids = torch.empty(max_sorted, dtype=torch.int32, device=device)
    sorted_expert_ids = torch.empty(
        max_sorted // _SORT_BM, dtype=torch.int32, device=device
    )
    sorted_weights = torch.empty(max_sorted, dtype=torch.float32, device=device)
    reverse_sorted = torch.empty(M * _TOPK, dtype=torch.int32, device=device)
    m_indices = torch.empty(max_sorted, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    if moe_buf is None:
        moe_launch = _empty_moe_buf(device)
        moe_buf_zeroed = False
        moe_buf_ptr = 0
    else:
        moe_launch = moe_buf
        moe_buf_zeroed = True
        moe_buf_ptr = int(moe_buf.data_ptr())
    build().run_with_sort(
        scores,
        bias,
        out_w,
        out_i,
        _arrival_buf(device),
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        reverse_sorted,
        m_indices,
        moe_launch,
        topk,
        float(routed_scaling_factor),
        bool(renormalize),
    )
    return {
        "topk_ids_ptr": int(out_i.data_ptr()),
        "M": M,
        "block_m": _SORT_BM,
        "num_experts": _NUM_EXPERTS,
        "topk": _TOPK,
        "sorted_token_ids": sorted_token_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "m_indices": m_indices,
        "reverse_sorted": reverse_sorted,
        "moe_buf_zeroed": moe_buf_zeroed,
        "moe_buf_ptr": moe_buf_ptr,
    }
