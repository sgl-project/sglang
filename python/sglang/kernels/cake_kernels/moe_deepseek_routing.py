"""Cake DeepSeek-V3 fused NoAuxTc routing (``fused_topk_deepseek(backend="cake")``) via FlashInfer.

FlashInfer entry: ``flashinfer.fused_moe.fused_topk_deepseek(..., backend="cake")``
(``flashinfer.fused_moe.fused_routing_dsv3``; JIT spec
``flashinfer.jit.dsv3_optimizations.gen_dsv3_fused_routing_module(backend="cake")``
building ``csrc/fused_moe/cake_deepseek_fused_routing``). Contract at FlashInfer
``46340689a5ab`` (``_is_cake_dsv3_fused_routing_supported``): CC (10,0) / (10,3)
only; ``scores [T, E]`` and ``bias [E]`` fp16 / bf16 / fp32;
``num_experts % n_group == 0``; ``1 <= topk <= 8``, ``topk <= num_experts``;
``topk_group <= n_group`` and ``topk_group * n_group >= topk``; with
``n_group == 1``: ``num_experts <= 384``; otherwise ``n_group <= 8``,
``topk_group <= 4``, ``num_experts <= 256``, ``2 <= E/n_group <= 32`` and
``(E/n_group) * topk_group <= 128``. Outputs written in place:
``topk_values [T, topk]`` (scores dtype), ``topk_indices [T, topk]`` int32,
optional int16 ``routing_replay_out`` with ``shape[0] >= T``, ``shape[1] == topk``
(oversizable for CUDA-graph reuse). Semantics: ``sigmoid(scores) + bias``,
group score = sum of each group's top-2, keep ``topk_group`` groups, top-k
experts, weights ``sigmoid / sum(sigmoid) * routed_scaling_factor``.

CUDA graphs: single in-place launch, no workspace; capturable. ``launch_with_pdl``
is a runtime flag.

Not supported here: ``backend="default"`` (the existing FlashInfer kernel; use
SGLang's current path), SM89/90/107/120/121, ``topk > 8``, fp64 scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.cake_kernels._support import SM100, SM103, cuda_tensor_on
from sglang.kernels.cake_kernels.moe_common import modules_available

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.fused_moe.fused_routing_dsv3"
FI_JIT_MODULE = "flashinfer.jit.dsv3_optimizations"
ARCHS = (SM100, SM103)
MAX_TOPK = 8


def _contract(
    *,
    num_tokens: int,
    num_experts: int,
    n_group: int,
    topk_group: int,
    topk: int,
) -> bool:
    if num_tokens <= 0 or num_experts <= 0 or n_group <= 0:
        return False
    if num_experts % n_group != 0:
        return False
    if topk <= 0 or topk > MAX_TOPK or topk > num_experts:
        return False
    if topk_group <= 0 or topk_group > n_group or topk_group * n_group < topk:
        return False
    if n_group == 1:
        return num_experts <= 384
    experts_per_group = num_experts // n_group
    return (
        n_group <= 8
        and topk_group <= 4
        and num_experts <= 256
        and 2 <= experts_per_group <= 32
        and experts_per_group * topk_group <= 128
    )


def supports_fused_topk_deepseek(
    scores: torch.Tensor,
    bias: torch.Tensor,
    *,
    n_group: int,
    topk_group: int,
    topk: int,
) -> bool:
    """Admission check mirroring ``_is_cake_dsv3_fused_routing_supported``; never raises."""
    import torch

    dtypes = (torch.float16, torch.bfloat16, torch.float32)
    try:
        return (
            modules_available(FI_MODULE, FI_JIT_MODULE)
            and cuda_tensor_on(scores, ARCHS)
            and scores.ndim == 2
            and scores.dtype in dtypes
            and bias.dtype in dtypes
            and bias.ndim == 1
            and bias.shape[0] == scores.shape[1]
            and bias.device == scores.device
            and _contract(
                num_tokens=int(scores.shape[0]),
                num_experts=int(scores.shape[1]),
                n_group=int(n_group),
                topk_group=int(topk_group),
                topk=int(topk),
            )
        )
    except Exception:
        return False


def fused_topk_deepseek(
    scores: torch.Tensor,
    bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    routed_scaling_factor: float,
    topk_values: torch.Tensor,
    topk_indices: torch.Tensor,
    launch_with_pdl: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
) -> None:
    """Forward to ``fused_topk_deepseek(backend="cake")``; writes outputs in place."""
    from flashinfer.fused_moe import fused_topk_deepseek as fi_fused_topk_deepseek

    fi_fused_topk_deepseek(
        scores,
        bias,
        n_group,
        topk_group,
        topk,
        routed_scaling_factor,
        topk_values,
        topk_indices,
        launch_with_pdl,
        routing_replay_out,
        backend="cake",
    )
