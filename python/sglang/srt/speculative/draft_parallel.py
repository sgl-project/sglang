from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.configs.draft_parallel import (
    DraftParallelLayout,
    DraftParallelSpec,
    DraftTPExecution,
)
from sglang.srt.runtime_context import draft_attn_parallel_group, get_parallel

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def legacy_draft_parallel_spec(
    spec_algorithm: SpeculativeAlgorithm, *, dspark_is_moe: bool = False
) -> DraftParallelSpec:
    """Compatibility policy for model implementations without a declaration."""
    single_group = (
        spec_algorithm.is_eagle3()
        or spec_algorithm.is_dflash()
        or (spec_algorithm.is_dspark() and not dspark_is_moe)
    )
    return DraftParallelSpec(
        tp_execution=(
            DraftTPExecution.SINGLE_GROUP
            if single_group
            else DraftTPExecution.TARGET_LAYOUT
        )
    )


def model_draft_parallel_spec(
    model_config: ModelConfig | None,
    *,
    spec_algorithm: SpeculativeAlgorithm,
    dspark_is_moe: bool = False,
) -> DraftParallelSpec:
    # Resolve the same registered class as the loader, including replacements.
    # This runs after target initialization, never during launcher resolution.
    if model_config is not None:
        from sglang.srt.model_loader.utils import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)
        declaration = getattr(model_cls, "draft_parallel_spec", None)
        if declaration is not None:
            return declaration
    return legacy_draft_parallel_spec(spec_algorithm, dspark_is_moe=dspark_is_moe)


def resolve_draft_parallel_layout(
    model_spec: DraftParallelSpec, *, collapse_cp: bool = False
) -> DraftParallelLayout:
    """Resolve once in target scope; CP-only execution is a worker policy."""
    parallel = get_parallel()
    independent = model_spec.tp_execution is DraftTPExecution.SINGLE_GROUP and (
        parallel.attn_dp_enabled or (collapse_cp and parallel.attn_cp_size > 1)
    )
    if not independent:
        return DraftParallelLayout(tp_group=None, attn_tp_size=parallel.attn_tp_size)
    group = getattr(parallel, f"{draft_attn_parallel_group()}_group")
    return DraftParallelLayout(
        tp_group=group,
        attn_tp_size=group.world_size,
        disable_cp=parallel.enable_prefill_cp,
    )


def legacy_draft_attn_tp_size(spec_algorithm: SpeculativeAlgorithm) -> int:
    """Geometry for direct runner callers without a resolved draft worker."""
    parallel = get_parallel()
    if spec_algorithm.is_dflash():
        return getattr(parallel, f"{draft_attn_parallel_group()}_size")
    return parallel.attn_tp_size


def resolve_prefill_context_parallel_layout(width: int) -> DraftParallelLayout:
    """Build encoder groups from actual DP/CP/attention-TP memberships."""
    import torch

    from sglang.srt.distributed.parallel_state import init_model_parallel_group

    parallel = get_parallel()
    records = parallel.tp_group.all_gather_object(
        (
            parallel.attn_dp_rank,
            parallel.attn_cp_rank,
            parallel.attn_tp_rank,
            parallel.tp_group.ranks[parallel.tp_group.rank_in_group],
        )
    )
    lanes = {}
    for dp, cp, atp, rank in sorted(records):
        lanes.setdefault(dp, []).append(rank)
    if width < 1 or any(len(ranks) % width for ranks in lanes.values()):
        raise ValueError("Context encoder TP must divide each target DP lane")
    groups = [
        ranks[i : i + width]
        for ranks in lanes.values()
        for i in range(0, len(ranks), width)
    ]
    group = init_model_parallel_group(
        groups,
        local_rank=parallel.tp_group.local_rank,
        backend=torch.distributed.get_backend(parallel.tp_group.device_group),
        use_pynccl=False,
        use_custom_allreduce=False,
        use_quick_allreduce=False,
        use_mscclpp=False,
        use_torch_symm_mem_allreduce=False,
        group_name="dflash_prefill_context",
    )
    return DraftParallelLayout(tp_group=group, attn_tp_size=width, disable_cp=True)
