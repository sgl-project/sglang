"""Cake (DeepGEMM-port) fused BF16 routing GEMM + expert mapping + normalized top-k via FlashInfer.

FlashInfer entry: ``flashinfer.mega_gate.prepare_mega_gate(...)`` ->
``MegaGatePlan`` (``run()`` -> ``(expert_indices, weights)``; implementation
``flashinfer.experimental.deepgemm_mega_gate.mega_gate``, catalog
``mega_gate.v2`` = ``mega_gate_catalog.json``). Contract at FlashInfer
``46340689a5ab``: SM100a (148 SMs) / SM103a (152 SMs) exact; ``x`` BF16
``[M, K]``, ``weight`` BF16 ``[E, K]``, optional FP32 expert ``bias [E]``;
returns int64 ``[M, topk + shared]`` physical ids + FP32 normalized weights x
``routed_scaling_factor``. Exported routes: ``K=5120, E=384, top_k=6``,
``scoring_func="sqrtsoftplus"`` with ``M in {1, 3, 16, 128, 512, 1024, 2048,
4096, 8192}`` and a physical map (``to_physical_map`` + ``logical_count``),
plus ``M=16`` deterministic (``ep_rank=7``, logical ``unmapped_topk_idx`` out)
and ``M=16`` logical routing without a map. Other scoring functions, image
bias, masks, fixed / random routing have no exported route (the prepare raises
``RuntimeError("No exported ...")``).

CUDA graphs: optional caller-owned ``scratch`` (FP32 split-K partials),
``score_barriers`` uint64 ``[ceil(M/block_tokens), 16]`` zeroed once, and a
descriptor workspace; ``run()`` is one fused kernel without allocation and is
capturable; bias / mapping contents may change in place between replays.

Not supported here: other (M, K, E, top_k) / SM counts, non-sqrtsoftplus
scoring, image bias, masks, forced / random routing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from sglang.kernels.cake_kernels._support import SM100, SM103, cuda_tensor_on
from sglang.kernels.cake_kernels.moe_common import device_sm_count, modules_available

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.mega_gate"
FI_JIT_MODULE = "flashinfer.experimental.deepgemm_mega_gate.mega_gate"
ARCHS = (SM100, SM103)
ARCH_NAMES = {SM100: "sm_100a", SM103: "sm_103a"}
K = 5120
NUM_EXPERTS = 384
TOP_K = 6
SCORING_FUNC = "sqrtsoftplus"
ROUTED_M = (1, 3, 16, 128, 512, 1024, 2048, 4096, 8192)


def supports_mega_gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    num_topk: int = TOP_K,
    *,
    scoring_func: str = SCORING_FUNC,
    has_physical_map: bool = True,
    deterministic: bool = False,
) -> bool:
    """Admission check mirroring the exported ``mega_gate`` routes; never raises."""
    import torch

    from sglang.kernels.cake_kernels._support import device_capability

    try:
        if not (
            modules_available(FI_MODULE, FI_JIT_MODULE) and cuda_tensor_on(x, ARCHS)
        ):
            return False
        if x.ndim != 2 or weight.ndim != 2:
            return False
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            return False
        if (
            weight.device != x.device
            or not x.is_contiguous()
            or not weight.is_contiguous()
        ):
            return False
        m = int(x.shape[0])
        if int(x.shape[1]) != K or tuple(weight.shape) != (NUM_EXPERTS, K):
            return False
        if int(num_topk) != TOP_K or scoring_func != SCORING_FUNC:
            return False
        from flashinfer.experimental.deepgemm_mega_gate import mega_gate as runtime

        arch = ARCH_NAMES[device_capability(x.device.index)]
        if device_sm_count(x.device.index) not in set(runtime.supported_num_sms(arch)):
            return False
        if has_physical_map and not deterministic:
            return m in ROUTED_M
        # M=16 deterministic (ep_rank 7) and M=16 logical routing without a map.
        return m == 16
    except Exception:
        return False


def prepare_mega_gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    num_topk: int = TOP_K,
    *,
    scoring_func: str = SCORING_FUNC,
    bias: Optional[torch.Tensor] = None,
    image_bias: Optional[torch.Tensor] = None,
    image_token_mask: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    to_physical_map: Optional[torch.Tensor] = None,
    logical_count: Optional[torch.Tensor] = None,
    fix_routing_mask: Optional[torch.Tensor] = None,
    force_random: Optional[Any] = None,
    unmapped_topk_idx: Optional[torch.Tensor] = None,
    use_shared_as_routed: bool = False,
    num_shared_experts: int = 1,
    routed_scaling_factor: float = 1.5,
    ep_rank: int = 0,
    out: Optional[Any] = None,
    deterministic: bool = False,
    scratch: Optional[torch.Tensor] = None,
    score_barriers: Optional[torch.Tensor] = None,
    descriptor_workspace: Optional[torch.Tensor] = None,
):
    """Forward to ``flashinfer.mega_gate.prepare_mega_gate``; returns ``MegaGatePlan``."""
    from flashinfer.mega_gate import prepare_mega_gate as fi_prepare

    return fi_prepare(
        x,
        weight,
        num_topk,
        scoring_func=scoring_func,
        bias=bias,
        image_bias=image_bias,
        image_token_mask=image_token_mask,
        mask=mask,
        to_physical_map=to_physical_map,
        logical_count=logical_count,
        fix_routing_mask=fix_routing_mask,
        force_random=force_random,
        unmapped_topk_idx=unmapped_topk_idx,
        use_shared_as_routed=use_shared_as_routed,
        num_shared_experts=num_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        ep_rank=ep_rank,
        out=out,
        deterministic=deterministic,
        scratch=scratch,
        score_barriers=score_barriers,
        descriptor_workspace=descriptor_workspace,
    )
