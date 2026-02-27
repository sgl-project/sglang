from typing import Optional

import torch

from sglang.srt.debug_utils.comparator.aligner.reorderer.types import (
    ReordererPlan,
    ZigzagToNaturalParams,
    ZigzagToNaturalThdParams,
)
from sglang.srt.debug_utils.comparator.dims import (
    resolve_dim_by_name,
    strip_dim_names,
)


def execute_reorderer_plan(
    plan: ReordererPlan,
    tensors: list[torch.Tensor],
) -> list[torch.Tensor]:
    if isinstance(plan.params, ZigzagToNaturalThdParams):
        thd_dim: int = resolve_dim_by_name(tensors[0], plan.params.dim_name)
        return [
            _reorder_zigzag_to_natural_thd(
                tensor,
                dim=thd_dim,
                cp_size=plan.params.cp_size,
                seq_lens=plan.params.seq_lens,
            )
            for tensor in tensors
        ]

    if isinstance(plan.params, ZigzagToNaturalParams):
        dim: int = resolve_dim_by_name(tensors[0], plan.params.dim_name)
        return [
            _reorder_zigzag_to_natural(tensor, dim=dim, cp_size=plan.params.cp_size)
            for tensor in tensors
        ]

    raise ValueError(f"Unsupported reorderer params type: {type(plan.params).__name__}")


def _reorder_zigzag_to_natural_thd(
    tensor: torch.Tensor, *, dim: int, cp_size: int, seq_lens: list[int]
) -> torch.Tensor:
    """Undo CP zigzag interleaving for THD (packed-seq) format.

    Each seq in seq_lens is independently reordered from zigzag to natural order
    along the given dim.
    """
    stripped: torch.Tensor = strip_dim_names(tensor)
    names: tuple[Optional[str], ...] = tensor.names

    split_sizes: list[int] = list(seq_lens)
    remainder: int = stripped.shape[dim] - sum(split_sizes)
    if remainder < 0:
        raise ValueError(
            f"sum(seq_lens)={sum(split_sizes)} exceeds tensor dim size "
            f"{stripped.shape[dim]} along dim={dim}"
        )
    if remainder > 0:
        split_sizes.append(remainder)

    segments: list[torch.Tensor] = list(stripped.split(split_sizes, dim=dim))

    reordered_segments: list[torch.Tensor] = [
        _reorder_zigzag_to_natural(seg, dim=dim, cp_size=cp_size)
        for seg in segments[: len(seq_lens)]
    ]

    # Tail padding — pass through unchanged
    if remainder > 0:
        reordered_segments.append(segments[-1])

    result: torch.Tensor = torch.cat(reordered_segments, dim=dim)

    if names[0] is not None:
        result = result.refine_names(*names)
    return result


def _reorder_zigzag_to_natural(
    tensor: torch.Tensor, *, dim: int, cp_size: int
) -> torch.Tensor:
    """Undo CP zigzag interleaving, restoring natural chunk order.

    Generalized from Megatron-LM _undo_attention_load_balancing
    (megatron/core/ssm/mamba_context_parallel.py:360-373).
    """
    stripped: torch.Tensor = strip_dim_names(tensor)
    names: tuple[Optional[str], ...] = tensor.names

    num_chunks: int = cp_size * 2
    chunks: tuple[torch.Tensor, ...] = stripped.chunk(num_chunks, dim=dim)
    order: list[int] = [2 * i for i in range(cp_size)] + [
        num_chunks - 2 * i - 1 for i in range(cp_size)
    ]
    result: torch.Tensor = torch.cat([chunks[i] for i in order], dim=dim)

    if names[0] is not None:
        result = result.refine_names(*names)
    return result
