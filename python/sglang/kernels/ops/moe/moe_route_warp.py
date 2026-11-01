"""One-warp-per-token CUDA MoE router (sigmoid / sqrtsoftplus + bias, softmax).

Softmax with a correction bias ranks and weighs ``softmax(logits + bias)``, as
the Triton router in ``moe_fused_gate``.

Same contract as the router in ``moe_fused_gate`` for the inputs ``covered``
accepts: fp32 weights and int32 ids in descending ranking order, lowest expert
id wins ties, rows past ``num_token_non_padded`` get zero weights and id -1,
fused shared-expert slots follow the routed ones. NaN logits rank last. Grouped
routing scores a group by the sum of its two largest biased values, a duplicated
maximum counted twice (DeepSeek reference and flashinfer ``fused_topk_deepseek``).
"""

from __future__ import annotations

import enum
import functools
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


class Scoring(str, enum.Enum):
    """Router scoring; the value is the ``scoring_func`` string srt passes."""

    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    SQRTSOFTPLUS = "sqrtsoftplus"

    @property
    def cpp_name(self) -> str:
        return self.name


_SCORING_FUNCS = frozenset(s.value for s in Scoring)


@cache_once
def _jit_route_warp_module(
    num_experts: int,
    topk: int,
    num_groups: int,
    topk_groups: int,
    scoring_func: str,
    dtype: torch.dtype,
    bias_dtype: torch.dtype,
    has_bias: bool,
    muse_spark: bool,
    token_bias: bool,
) -> Module:
    args = make_cpp_args(
        num_experts,
        topk,
        num_groups,
        topk_groups,
        Scoring(scoring_func).cpp_name,
        dtype,
        bias_dtype,
        has_bias,
        is_arch_support_pdl(),
        muse_spark,
        token_bias,
    )
    return load_jit(
        "moe_route_warp",
        *args,
        cuda_files=["moe/route_warp.cuh"],
        cuda_wrappers=[("run", f"moe::route_warp::RouteWarpKernel<{args}>::run")],
    )


@functools.cache
def _device_ok(device_index: Optional[int]) -> bool:
    # redux.sync needs SM80+; the kernel is CUDA-only.
    return (
        torch.version.hip is None
        and torch.cuda.get_device_capability(device_index)[0] >= 8
    )


@functools.cache
def covered(
    scores_dtype: torch.dtype,
    num_experts: int,
    topk: int,
    scoring_func: str,
    num_fused_shared_experts: int = 0,
    num_expert_group: int = 1,
    topk_group: int = 1,
    device_index: Optional[int] = None,
) -> bool:
    """Whether :func:`moe_route_warp` serves this router config. Static facts
    only; the launcher rejects misaligned or non-row-major logits."""
    if not _device_ok(device_index) or scoring_func not in _SCORING_FUNCS:
        return False
    if scores_dtype not in (torch.float32, torch.bfloat16):
        return False
    # Moonlight (64 experts) bs1 decode on H200 lost 20-30 us per step: the earlier
    # expert GEMM shares SMs with the shared-expert stream. Arbitrary cutoff.
    if num_experts % 32 or not 128 <= num_experts <= 512:
        return False
    k_routed = topk - num_fused_shared_experts
    if k_routed < 1 or topk > 32:
        return False
    if num_expert_group > 1:
        experts_per_group = num_experts // num_expert_group
        if scoring_func == Scoring.SOFTMAX or 32 % num_expert_group:
            return False
        if experts_per_group != (32 // num_expert_group) * (num_experts // 32):
            return False
        if topk_group * experts_per_group < k_routed:
            return False
    return True


def moe_route_warp(
    scores: torch.Tensor,
    bias: Optional[torch.Tensor],
    topk: int,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: Optional[float] = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
    num_expert_group: int = 1,
    topk_group: int = 1,
    num_token_non_padded: Optional[torch.Tensor] = None,
    renormalize_epsilon: float = 0.0,
    packed_out: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    bias_alt: Optional[torch.Tensor] = None,
    bias_alt_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (weights [T, topk] fp32, ids [T, topk] int32); ``topk`` counts the
    fused shared-expert slots. ``bias`` is optional for softmax, which then
    ranks and weighs ``logits + bias``. ``num_token_non_padded`` is an int32
    device scalar. ``packed_out`` ([T, topk] int32) also receives ``id << 16 |
    bf16_bits(weight)``. Rows whose ``input_ids`` equal ``bias_alt_token_id``
    rank with ``bias_alt`` instead of ``bias``. Caller must have checked
    :func:`covered`."""
    num_tokens, num_experts = scores.shape
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0
    grouped = num_expert_group > 1
    module = _jit_route_warp_module(
        num_experts,
        topk - num_fused_shared_experts,
        num_expert_group if grouped else 1,
        topk_group if grouped else 1,
        scoring_func,
        scores.dtype,
        torch.float32 if bias is None else bias.dtype,
        has_bias=bias is not None,
        muse_spark=False,
        token_bias=input_ids is not None,
    )
    out_w = torch.empty((num_tokens, topk), dtype=torch.float32, device=scores.device)
    out_i = torch.empty((num_tokens, topk), dtype=torch.int32, device=scores.device)
    module.run(
        scores,
        bias,
        out_w,
        out_i,
        packed_out,
        num_token_non_padded,
        num_fused_shared_experts,
        float(routed_scaling_factor),
        float(renormalize_epsilon),
        bool(renormalize),
        bool(apply_routed_scaling_factor_on_output),
        1.0,
        None,
        input_ids,
        bias_alt,
        0 if bias_alt_token_id is None else int(bias_alt_token_id),
    )
    return out_w, out_i


@functools.cache
def muse_spark_route_warp_covered(
    num_experts: int, topk: int, device_index: Optional[int] = None
) -> bool:
    """Whether :func:`muse_spark_route_warp` serves this MuseSpark router config
    (fp32 logits and bias, sigmoid scoring: the caller's checks)."""
    return (
        _device_ok(device_index)
        and num_experts % 32 == 0
        and num_experts <= 512
        and 1 <= topk <= 32
    )


def muse_spark_route_warp(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    router_output_multiplier: Optional[float] = None,
    division_eps: Optional[float] = None,
    local_expert_mapping: Optional[torch.Tensor] = None,
    num_valid_tokens: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    packed_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MuseSpark routing: ``sigmoid(scores * multiplier)`` ranked with ``+ bias``,
    weights ``score * (route_scale / (sum + eps))``, ids through the optional EP
    map; rows ``>= num_valid_tokens`` get id -1 and packed ``-1 << 16`` and keep
    their weights. Caller must have checked :func:`muse_spark_route_warp_covered`."""
    num_tokens, num_experts = scores.shape
    module = _jit_route_warp_module(
        num_experts,
        topk,
        1,
        1,
        Scoring.SIGMOID.value,
        scores.dtype,
        bias.dtype,
        has_bias=True,
        muse_spark=True,
        token_bias=False,
    )
    if topk_weights is None:
        topk_weights = torch.empty(
            (num_tokens, topk), dtype=torch.float32, device=scores.device
        )
    if topk_ids is None:
        topk_ids = torch.empty(
            (num_tokens, topk), dtype=torch.int32, device=scores.device
        )
    module.run(
        scores,
        bias,
        topk_weights,
        topk_ids,
        packed_out,
        num_valid_tokens,
        0,
        float(route_scale),
        float(division_eps if division_eps is not None else 0.0),
        True,
        True,
        float(
            router_output_multiplier if router_output_multiplier is not None else 1.0
        ),
        local_expert_mapping,
        None,
        None,
        0,
    )
    return topk_weights, topk_ids
