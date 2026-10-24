"""Cake BGMV MoE-LoRA shrink + expand pipeline via FlashInfer.

FlashInfer entry: ``flashinfer.fused_moe.prepare_bgmv_moe(..., backend="cake")``
returning ``BGMVMoECakePlan`` (``flashinfer.fused_moe.bgmv_moe``; JIT module
``flashinfer.jit.cake_bgmv_moe``). Contract at FlashInfer ``46340689a5ab``:
exact SM90 / SM100 / SM103 (one cubin per target); ``x [T, H]`` and the LoRA
weights share one dtype (BF16 or FP16); exactly one LoRA slice;
``lora_a [num_loras, E, rank, H]``, ``lora_b [num_loras, E, H, rank]``
(feat_out == H); ``rank in {8, 16, 32, 64}``; ``H`` a positive multiple of 8;
int64 ``sorted_token_ids [num_pairs]``, ``expert_ids [num_pairs]`` (in
``[0, E)``), ``lora_indices [T]`` (-1 or ``[0, num_loras)``), f32
``topk_weights [num_pairs]``. Specialized bodies for ``H in {2688, 3072}`` at
rank 32 up to 2048 tokens; runtime-hidden generic bundles otherwise.
Arbitrary routing order; the output has one owner per token so replays are
bitwise reproducible (no output atomics). Returns the FP32 accumulator
``y_accum [T, H]`` from ``plan.run()``.

CUDA graphs: ``prepare_bgmv_moe`` binds pointer-stable tensors (optionally
caller-owned ``shrink_out`` / ``y_accum``); the first eager ``run()`` captures
an internal CUDA graph (``_BGMVMoEGraphPlan``) that later ``run()`` calls
replay on the original stream, reading current tensor contents. Prepare and
run once eagerly before any outer capture.

Not supported here: the atomics-based ``BGMVMoEPortablePlan`` fallback (this
adapter defaults ``fallback=False`` so unsupported inputs raise instead of
silently running the non-Cake portable kernels), multiple slices,
``feat_out != hidden_size``, ranks outside {8, 16, 32, 64}.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from sglang.kernels.cake_kernels._support import SM90, SM100, SM103, cuda_tensor_on
from sglang.kernels.cake_kernels.moe_common import modules_available

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.fused_moe.bgmv_moe"
FI_JIT_MODULE = "flashinfer.jit.cake_bgmv_moe"
ARCHS = (SM90, SM100, SM103)
RANKS = (8, 16, 32, 64)
HIDDEN_MULTIPLE = 8
SPECIALIZED_HIDDEN = (2688, 3072)
SPECIALIZED_MAX_TOKENS = 2048


def supports_bgmv_moe(
    x: torch.Tensor,
    lora_a_weights: List[torch.Tensor],
    lora_b_weights: List[torch.Tensor],
) -> bool:
    """Admission check mirroring ``bgmv_moe._cake_unsupported_reason``; never raises."""
    import torch

    try:
        if not (
            modules_available(FI_MODULE, FI_JIT_MODULE) and cuda_tensor_on(x, ARCHS)
        ):
            return False
        if x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.float16):
            return False
        if len(lora_a_weights) != 1 or len(lora_b_weights) != 1:
            return False
        lora_a, lora_b = lora_a_weights[0], lora_b_weights[0]
        if lora_a.ndim != 4 or lora_b.ndim != 4:
            return False
        if lora_a.dtype != x.dtype or lora_b.dtype != x.dtype:
            return False
        hidden = int(x.shape[1])
        rank = int(lora_a.shape[2])
        return (
            hidden > 0
            and hidden % HIDDEN_MULTIPLE == 0
            and rank in RANKS
            and int(lora_a.shape[3]) == hidden
            and int(lora_b.shape[2]) == hidden
            and int(lora_b.shape[3]) == rank
            and lora_a.device == x.device
            and lora_b.device == x.device
        )
    except Exception:
        return False


def prepare_bgmv_moe(
    x: torch.Tensor,
    lora_a_weights: List[torch.Tensor],
    lora_b_weights: List[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    lora_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    *,
    fallback: bool = False,
    shrink_out: Optional[torch.Tensor] = None,
    y_accum: Optional[torch.Tensor] = None,
):
    """Forward to ``prepare_bgmv_moe(backend="cake")``; returns the plan (``plan.run()`` -> f32 ``[T, H]``).

    ``fallback`` defaults to ``False`` here: callers gate on
    :func:`supports_bgmv_moe` and get a ``ValueError`` instead of the portable
    (non-Cake, non-bitwise) plan.
    """
    from flashinfer.fused_moe import prepare_bgmv_moe as fi_prepare_bgmv_moe

    return fi_prepare_bgmv_moe(
        x,
        lora_a_weights,
        lora_b_weights,
        sorted_token_ids,
        expert_ids,
        lora_indices,
        topk_weights,
        num_experts,
        backend="cake",
        fallback=fallback,
        shrink_out=shrink_out,
        y_accum=y_accum,
    )


def get_bgmv_moe_cake_plan_class():
    from flashinfer.fused_moe import BGMVMoECakePlan

    return BGMVMoECakePlan
