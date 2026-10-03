"""Cake (DeepGEMM-port) fused BF16 routing GEMM + expert mapping + normalized top-k via FlashInfer.

FlashInfer entry: ``flashinfer.mega_gate.prepare_mega_gate(x, weight, num_topk=6, *,
scoring_func="sqrtsoftplus", bias=None, image_bias=None, image_token_mask=None,
mask=None, to_physical_map=None, logical_count=None, fix_routing_mask=None,
force_random=None, unmapped_topk_idx=None, use_shared_as_routed=False,
num_shared_experts=1, routed_scaling_factor=1.5, ep_rank=0, out=None,
deterministic=False, scratch=None, score_barriers=None,
descriptor_workspace=None)`` -> ``MegaGatePlan`` (``run()`` -> ``(expert_indices,
weights)``; implementation ``flashinfer.experimental.deepgemm_mega_gate.mega_gate``).

Contract at FlashInfer ``e4f94f94`` (ead850398 "cake_deepgemm_mega_gate: take
the token count at runtime with one program per schedule template"): one
compiled program per physical schedule template; the token count ``M``, the
SM-count-derived worker stride and the physical-map / logical-output flags are
kernel arguments, so **any** ``M`` (``0 < M <= 2**20``) on any SM100a / SM103a
part is served (``mega_gate.select_template`` snaps DeepGEMM's configuration
for ``M`` onto the exported template set). The problem constants are fixed:
``x`` BF16 ``[M, 5120]``, ``weight`` BF16 ``[384, 5120]``, ``num_topk=6``,
``scoring_func="sqrtsoftplus"``, and an FP32 expert ``bias [384]`` is required.
Returns int64 ``[M, topk + shared]`` ids + FP32 normalized weights x
``routed_scaling_factor``. ``to_physical_map`` (int32 ``[E + shared, width]``)
and ``logical_count`` (int32 ``[E + shared]``) come together or not at all;
``unmapped_topk_idx`` (int64 ``[M, topk]``, unit column stride) optionally
receives the logical top-k; ``deterministic=True`` selects the split-K-free
template. ``plan.route`` is a dict (``arch``, ``template``, ``program``,
``launch_ctas``, ``num_workers``, ``block_tokens``, ``num_split_k``,
``num_sms``, ``route_flags``).

CUDA graphs: optional caller-owned ``scratch`` FP32 ``[ceil(M/block_tokens),
num_split_k, block_tokens, 384]`` and ``score_barriers`` uint64
``[ceil(M/block_tokens), 16]`` zeroed once; ``run()`` is one fused kernel
without allocation and is capturable; bias / mapping contents may change in
place between replays. The programs take their TMA descriptors by value:
``descriptor_workspace`` must be ``None`` (FlashInfer raises otherwise).

Not supported (``NotImplementedError`` / ``ValueError`` upstream, ``False``
here): other ``(K, E, top_k)``, other scoring functions, no bias, image bias,
token masks, fixed / random routing, a descriptor workspace, SM90 / SM12x.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from sglang.kernels.cake_kernels._support import SM100, SM103, cuda_tensor_on
from sglang.kernels.cake_kernels.moe_common import modules_available

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.mega_gate"
FI_JIT_MODULE = "flashinfer.experimental.deepgemm_mega_gate.mega_gate"
ARCHS = (SM100, SM103)
# The problem constants the exported programs were generated for
# (``mega_gate.EXPORTED``).
K = 5120
NUM_EXPERTS = 384
TOP_K = 6
SCORING_FUNC = "sqrtsoftplus"
INT32_MAX = 0x7FFFFFFF


def supports_mega_gate(
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
) -> bool:
    """Admission check mirroring ``MegaGatePlan.__init__``; never raises.

    Takes the same arguments as :func:`prepare_mega_gate`. The schedule
    template is resolved through FlashInfer's ``mega_gate.device_facts`` and
    ``mega_gate.select_template`` (lazy import), exactly as the plan does.
    """
    import torch

    del routed_scaling_factor  # any float is accepted upstream
    try:
        if not (
            modules_available(FI_MODULE, FI_JIT_MODULE) and cuda_tensor_on(x, ARCHS)
        ):
            return False
        if (
            x.ndim != 2
            or weight.ndim != 2
            or x.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
        ):
            return False
        m, k = (int(v) for v in x.shape)
        e, wk = (int(v) for v in weight.shape)
        if wk != k or not x.is_contiguous() or not weight.is_contiguous():
            return False
        if (k, e, int(num_topk), scoring_func) != (K, NUM_EXPERTS, TOP_K, SCORING_FUNC):
            return False
        if bias is None or any(
            t is not None
            for t in (
                image_bias,
                image_token_mask,
                mask,
                fix_routing_mask,
                force_random,
            )
        ):
            return False
        if descriptor_workspace is not None:
            return False
        if (to_physical_map is None) != (logical_count is None):
            return False
        if not 0 <= int(ep_rank) <= INT32_MAX:
            return False
        shared = int(num_shared_experts) if use_shared_as_routed else 0
        if shared and (
            shared not in (1, 2) or num_topk % shared or e % (num_topk // shared)
        ):
            return False
        if num_topk + shared > 32:
            return False
        from flashinfer.experimental.deepgemm_mega_gate import mega_gate as runtime

        index = x.device.index
        if index is None:
            index = torch.cuda.current_device()
        _arch, sms = runtime.device_facts(index)
        template_index, _launch_ctas, _workers = runtime.select_template(
            m,
            sms,
            has_physical_map=to_physical_map is not None,
            unmapped_output=unmapped_topk_idx is not None,
            deterministic=bool(deterministic),
        )
        template = runtime.TEMPLATES[template_index]
        block_tokens, num_split_k = template["block_tokens"], template["num_split_k"]
        blocks = (m + block_tokens - 1) // block_tokens
        aligned_e = (e + 127) // 128 * 128
        slots = num_topk + shared
        if out is not None and (
            len(out) != 2
            or any(tuple(t.shape) != (m, slots) for t in out)
            or out[0].dtype != torch.int64
            or out[1].dtype != torch.float32
        ):
            return False
        if scratch is not None and (
            tuple(scratch.shape) != (blocks, num_split_k, block_tokens, aligned_e)
            or scratch.dtype != torch.float32
        ):
            return False
        if score_barriers is not None and (
            tuple(score_barriers.shape) != (blocks, 16)
            or score_barriers.dtype != torch.uint64
        ):
            return False
        if bias.dtype != torch.float32 or tuple(bias.shape) != (e,):
            return False
        if logical_count is not None and (
            logical_count.dtype != torch.int32
            or tuple(logical_count.shape) != (e + shared,)
        ):
            return False
        if to_physical_map is not None and (
            to_physical_map.dtype != torch.int32
            or to_physical_map.ndim != 2
            or int(to_physical_map.shape[0]) != e + shared
        ):
            return False
        if unmapped_topk_idx is not None and (
            unmapped_topk_idx.dtype != torch.int64
            or tuple(unmapped_topk_idx.shape) != (m, num_topk)
            or unmapped_topk_idx.stride(1) != 1
            or unmapped_topk_idx.device != x.device
        ):
            return False
        tensors = [
            weight,
            bias,
            to_physical_map,
            logical_count,
            scratch,
            score_barriers,
        ] + (list(out) if out is not None else [])
        return not any(
            t.device != x.device or not t.is_contiguous()
            for t in tensors
            if t is not None
        )
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
