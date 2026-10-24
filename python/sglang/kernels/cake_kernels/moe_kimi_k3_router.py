"""Cake Kimi-K3 fused MoE router (sigmoid + bias top-16 + aligned route plan) via FlashInfer.

FlashInfer entries: ``flashinfer.fused_moe.prepare_kimi_k3_fused_router``,
``flashinfer.fused_moe.kimi_k3_fused_router`` and
``flashinfer.fused_moe.allocate_kimi_k3_route_plan`` (thin wrappers over
``flashinfer.experimental.kimi_k3_fused_router.cake_backend``; JIT registry
``flashinfer.experimental.kimi_k3_fused_router.cake_jit``). Contract at
FlashInfer ``46340689a5ab``: sm_100a / sm_103a; contiguous f32
``logits [T, 896]``, f32 ``bias [896]``; exactly 896 experts, top-16 on
``sigmoid(logits) + bias``, weights renormalized over the selected sigmoid
scores; ``block_m in {8, 16}``; routed shapes ONLY ``T in {1, 2, 4, ..., 8192}``
(powers of two), other token counts raise ``NotImplementedError``. One launch
writes a ``KimiK3RoutePlan`` in ``moe_align_block_size`` layout
(``topk_weights [T,16]`` f32, ``topk_ids [T,16]`` int32 ascending,
``sorted_token_ids``, ``expert_ids``, ``num_tokens_post_padded [1]``,
``expert_counts [896]``, ``expert_offsets [897]``, ``expert_scatter_offsets``;
sentinel ``T*16``). Sigmoid uses ``__expf``/``__fdividef`` (last-bit
differences vs torch; ties within 2**-22 may reorder at the top-16 boundary).

CUDA graphs: ``prepare_*`` (outside capture) resolves the dispatch arm and
grid (persistent grids bounded by SM count, 4-CTA clusters bounded by the
co-resident cluster capacity, a 16-CTA non-portable cluster for T=16); the
returned runner launches with no allocation / host sync and is capturable; it
reads ``logits`` / ``bias`` on device so new values are honoured on replay.

Not supported here: other expert counts / top-k, non-power-of-two token
counts, BF16 logits, plans whose buffers overlap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from sglang.kernels.cake_kernels._support import SM100, SM103, cuda_tensor_on
from sglang.kernels.cake_kernels.moe_common import modules_available

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.fused_moe.kimi_k3_fused_router"
FI_JIT_MODULE = "flashinfer.experimental.kimi_k3_fused_router.cake_jit"
ARCHS = (SM100, SM103)
NUM_EXPERTS = 896
TOP_K = 16
BLOCK_M_VALUES = (8, 16)
MAX_TOKENS = 8192


def _routed_num_tokens(num_tokens: int) -> bool:
    return (
        isinstance(num_tokens, int)
        and not isinstance(num_tokens, bool)
        and 1 <= num_tokens <= MAX_TOKENS
        and num_tokens & (num_tokens - 1) == 0
    )


def supports_kimi_k3_fused_router(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    block_m: int = 8,
) -> bool:
    """Admission check mirroring the FlashInfer contract; never raises."""
    import torch

    try:
        return (
            modules_available(FI_MODULE, FI_JIT_MODULE)
            and cuda_tensor_on(logits, ARCHS)
            and logits.dtype == torch.float32
            and logits.ndim == 2
            and logits.is_contiguous()
            and logits.shape[1] == NUM_EXPERTS
            and _routed_num_tokens(int(logits.shape[0]))
            and bias.is_cuda
            and bias.device == logits.device
            and bias.dtype == torch.float32
            and tuple(bias.shape) == (NUM_EXPERTS,)
            and bias.is_contiguous()
            and block_m in BLOCK_M_VALUES
        )
    except Exception:
        return False


def allocate_kimi_k3_route_plan(num_tokens: int, block_m: int, device: torch.device):
    """Worst-case-capacity ``KimiK3RoutePlan`` (no launch)."""
    from flashinfer.fused_moe import allocate_kimi_k3_route_plan as fi_allocate

    return fi_allocate(num_tokens, block_m, device)


def prepare_kimi_k3_fused_router(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    block_m: int = 8,
    plan: Optional[Any] = None,
):
    """Forward to FlashInfer; returns ``KimiK3FusedRouterRunner`` (call it to launch)."""
    from flashinfer.fused_moe import prepare_kimi_k3_fused_router as fi_prepare

    return fi_prepare(logits, bias, block_m=block_m, plan=plan, backend="cake")


def kimi_k3_fused_router(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    block_m: int = 8,
    plan: Optional[Any] = None,
):
    """Prepare + one launch; returns the ``KimiK3RoutePlan``."""
    from flashinfer.fused_moe import kimi_k3_fused_router as fi_router

    return fi_router(logits, bias, block_m=block_m, plan=plan, backend="cake")
