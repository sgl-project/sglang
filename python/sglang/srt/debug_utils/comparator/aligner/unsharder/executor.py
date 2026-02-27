from typing import Optional

import torch

from sglang.srt.debug_utils.comparator.aligner.unsharder.types import (
    ConcatParams,
    CpThdConcatParams,
    PickParams,
    UnsharderParams,
    UnsharderPlan,
)
from sglang.srt.debug_utils.comparator.dims import (
    ParallelAxis,
    resolve_dim_by_name,
)
from sglang.srt.debug_utils.comparator.output_types import ReplicatedMismatchWarning
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink


def execute_unsharder_plan(
    plan: UnsharderPlan,
    tensors: list[torch.Tensor],
) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []

    for group_idx, group in enumerate(plan.groups):
        group_tensors = [tensors[i] for i in group]
        tensor = _apply_unshard(
            plan.params,
            group_tensors,
            axis=plan.axis,
            group_index=group_idx,
        )
        result.append(tensor)

    return result


def _apply_unshard(
    params: UnsharderParams,
    ordered_tensors: list[torch.Tensor],
    *,
    axis: ParallelAxis,
    group_index: int,
) -> torch.Tensor:
    if isinstance(params, PickParams):
        _verify_replicated_group(
            ordered_tensors,
            axis=axis,
            group_index=group_index,
        )
        return ordered_tensors[0]

    if isinstance(params, ConcatParams):
        dim: int = resolve_dim_by_name(ordered_tensors[0], params.dim_name)
        return torch.cat(ordered_tensors, dim=dim)

    if isinstance(params, CpThdConcatParams):
        thd_dim: int = resolve_dim_by_name(ordered_tensors[0], params.dim_name)
        return _thd_concat(
            ordered_tensors,
            dim=thd_dim,
            seq_lens_per_rank=params.seq_lens_per_rank,
        )

    raise ValueError(f"Unsupported unshard operation: {type(params).__name__}")


def _verify_replicated_group(
    ordered_tensors: list[torch.Tensor],
    *,
    axis: ParallelAxis,
    group_index: int,
) -> None:
    baseline = ordered_tensors[0].rename(None)

    for i in range(1, len(ordered_tensors)):
        other = ordered_tensors[i].rename(None)
        if not torch.allclose(baseline, other, atol=1e-6):
            warning_sink.add(
                ReplicatedMismatchWarning(
                    axis=axis.value,
                    group_index=group_index,
                    differing_index=i,
                    baseline_index=0,
                    max_abs_diff=(baseline - other).abs().max().item(),
                )
            )


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
