import torch

from sglang.srt.debug_utils.comparator.aligner.reorderer.types import ReordererPlan


def execute_reorderer_plan(
    plan: ReordererPlan,
    tensors: list[torch.Tensor],
) -> list[torch.Tensor]:
    return [
        _reorder_zigzag_to_natural(
            tensor, dim=plan.params.dim, cp_size=plan.params.cp_size
        )
        for tensor in tensors
    ]


def _reorder_zigzag_to_natural(
    tensor: torch.Tensor, *, dim: int, cp_size: int
) -> torch.Tensor:
    """Undo CP zigzag interleaving, restoring natural chunk order.

    Generalized from Megatron-LM _undo_attention_load_balancing
    (megatron/core/ssm/mamba_context_parallel.py:360-373).
    """
    num_chunks: int = cp_size * 2
    chunks: tuple[torch.Tensor, ...] = tensor.chunk(num_chunks, dim=dim)
    order: list[int] = [2 * i for i in range(cp_size)] + [
        num_chunks - 2 * i - 1 for i in range(cp_size)
    ]
    return torch.cat([chunks[i] for i in order], dim=dim)
