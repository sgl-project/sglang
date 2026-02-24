from typing import Literal

import torch

from sglang.srt.debug_utils.comparator.aligner.unshard.types import AxisInfo
from sglang.srt.debug_utils.comparator.dims import DimSpec, Ordering, ParallelAxis
from sglang.srt.debug_utils.comparator.utils import _FrozenBase


class ZigzagToNaturalParams(_FrozenBase):
    op: Literal["zigzag_to_natural"] = "zigzag_to_natural"
    dim: int
    cp_size: int


ReorderParams = ZigzagToNaturalParams


class ReorderPlan(_FrozenBase):
    params: ReorderParams


_ALLOWED_ZIGZAG_DIM_NAMES: set[str] = {"s"}


def compute_reorder_plans(
    dim_specs: list[DimSpec],
    parallel_infos: list[dict[ParallelAxis, AxisInfo]],
) -> list[ReorderPlan]:
    plans: list[ReorderPlan] = []

    for dim_index, spec in enumerate(dim_specs):
        if (
            spec.ordering is not None
            and spec.ordering != Ordering.NATURAL
            and spec.parallel is not None
        ):
            if spec.name not in _ALLOWED_ZIGZAG_DIM_NAMES:
                raise ValueError(
                    f"Zigzag ordering is only supported on sequence dims "
                    f"(bshd/sbhd format, dim name must be one of "
                    f"{sorted(_ALLOWED_ZIGZAG_DIM_NAMES)}), "
                    f"but got dim name {spec.name!r} in {spec}"
                )

            assert spec.ordering == Ordering.ZIGZAG
            axis_size: int = parallel_infos[0][spec.parallel].axis_size
            plans.append(
                ReorderPlan(
                    params=ZigzagToNaturalParams(dim=dim_index, cp_size=axis_size),
                )
            )

    return plans


def execute_reorder_plan(
    plan: ReorderPlan,
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
