from dataclasses import dataclass, field
from typing import Optional

import torch

from sglang.srt.debug_utils.comparator.aligner.unsharder.types import (
    ConcatParams,
    CpThdConcatParams,
    PickParams,
    ReduceSumParams,
    UnsharderParams,
    UnsharderPlan,
)
from sglang.srt.debug_utils.comparator.dims import (
    ParallelAxis,
    resolve_dim_by_name,
)
from sglang.srt.debug_utils.comparator.output_types import ReplicatedCheckResult
from sglang.srt.debug_utils.comparator.tensor_comparator.comparator import compute_diff

_REPLICATED_ATOL: float = 1e-6


@dataclass(frozen=True)
class UnsharderResult:
    tensors: list[torch.Tensor]
    replicated_checks: list[ReplicatedCheckResult] = field(default_factory=list)


def execute_unsharder_plan(
    plan: UnsharderPlan,
    tensors: list[torch.Tensor],
) -> UnsharderResult:
    result_tensors: list[torch.Tensor] = []
    all_checks: list[ReplicatedCheckResult] = []

    for group_idx, group in enumerate(plan.groups):
        group_tensors = [tensors[i] for i in group]
        tensor, checks = _apply_unshard(
            plan.params,
            group_tensors,
            axis=plan.axis,
            group_index=group_idx,
        )
        result_tensors.append(tensor)
        all_checks.extend(checks)

    return UnsharderResult(tensors=result_tensors, replicated_checks=all_checks)


def _apply_unshard(
    params: UnsharderParams,
    ordered_tensors: list[torch.Tensor],
    *,
    axis: ParallelAxis,
    group_index: int,
) -> tuple[torch.Tensor, list[ReplicatedCheckResult]]:
    if isinstance(params, PickParams):
        checks: list[ReplicatedCheckResult] = _verify_replicated_group(
            ordered_tensors,
            axis=axis,
            group_index=group_index,
        )
        return ordered_tensors[0], checks

    if isinstance(params, ConcatParams):
        dim: int = resolve_dim_by_name(ordered_tensors[0], params.dim_name)
        return torch.cat(ordered_tensors, dim=dim), []

    if isinstance(params, CpThdConcatParams):
        thd_dim: int = resolve_dim_by_name(ordered_tensors[0], params.dim_name)
        return (
            _thd_concat(
                ordered_tensors,
                dim=thd_dim,
                seq_lens_per_rank=params.seq_lens_per_rank,
            ),
            [],
        )

    if isinstance(params, ReduceSumParams):
        stripped: list[torch.Tensor] = [t.rename(None) for t in ordered_tensors]
        result: torch.Tensor = torch.stack(stripped).sum(dim=0)
        names: tuple[Optional[str], ...] = ordered_tensors[0].names
        if names[0] is not None:
            result = result.refine_names(*names)
        return result, []

    raise ValueError(f"Unsupported unshard operation: {type(params).__name__}")


def _verify_replicated_group(
    ordered_tensors: list[torch.Tensor],
    *,
    axis: ParallelAxis,
    group_index: int,
) -> list[ReplicatedCheckResult]:
    baseline: torch.Tensor = ordered_tensors[0].rename(None).float()
    checks: list[ReplicatedCheckResult] = []

    for i in range(1, len(ordered_tensors)):
        other: torch.Tensor = ordered_tensors[i].rename(None).float()
        diff_info = compute_diff(
            x_baseline=baseline,
            x_target=other,
            diff_threshold=_REPLICATED_ATOL,
        )
        passed: bool = diff_info.max_abs_diff <= _REPLICATED_ATOL
        checks.append(
            ReplicatedCheckResult(
                axis=axis.value,
                group_index=group_index,
                compared_index=i,
                baseline_index=0,
                passed=passed,
                atol=_REPLICATED_ATOL,
                diff=diff_info,
            )
        )

    return checks


def _thd_concat(
    ordered_tensors: list[torch.Tensor],
    *,
    dim: int,
    seq_lens_per_rank: list[int],
) -> torch.Tensor:
    """Per-seq concat across ranks for THD format.

    Each rank holds segments of each seq packed contiguously:
      rank_data = [seq0_tokens | seq1_tokens | ... | pad_tokens]

    This function splits each rank by seq_lens, then interleaves across ranks
    per-seq: [seqA_r0 + seqA_r1 + ... | seqB_r0 + seqB_r1 + ... | tail_pad].
    """
    names: tuple[Optional[str], ...] = ordered_tensors[0].names
    stripped: list[torch.Tensor] = [t.rename(None) for t in ordered_tensors]

    # Split each rank into [seq0, seq1, ..., tail_remainder]
    split_sizes: list[int] = list(seq_lens_per_rank)
    remainder: int = stripped[0].shape[dim] - sum(split_sizes)
    if remainder < 0:
        raise ValueError(
            f"sum(seq_lens_per_rank)={sum(split_sizes)} exceeds tensor dim size "
            f"{stripped[0].shape[dim]} along dim={dim}"
        )
    if remainder > 0:
        split_sizes.append(remainder)
    per_rank_splits: list[tuple[torch.Tensor, ...]] = [
        t.split(split_sizes, dim=dim) for t in stripped
    ]

    # Per-seq concat across ranks, then concatenate all seqs
    result: torch.Tensor = torch.cat(
        [torch.cat(rank_parts, dim=dim) for rank_parts in zip(*per_rank_splits)],
        dim=dim,
    )

    if names[0] is not None:
        result = result.refine_names(*names)
    return result
