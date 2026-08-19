"""Model-facing adapter for distributed token layouts."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sglang.srt.layers.cp.base import is_cp_enabled
from sglang.srt.layers.cp.utils import cp_materialize_global_token_order, is_cp_active
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.runtime_context import get_parallel


@dataclass(frozen=True)
class ModelExecutionLayout:
    """Hide layout-specific communication behind a model-facing interface."""

    has_rank_local_tokens: bool

    @property
    def allows_parallel_model_branches(self) -> bool:
        return not self.has_rank_local_tokens

    @property
    def requires_full_kv(self) -> bool:
        return self.has_rank_local_tokens

    @property
    def requires_moe_reduce_scatter(self) -> bool:
        return self.has_rank_local_tokens

    def materialize_kv(
        self, kv: torch.Tensor, forward_batch, stream=None
    ) -> torch.Tensor:
        if not self.has_rank_local_tokens:
            return kv
        return cp_materialize_global_token_order(kv, forward_batch, stream)

    def prepare_moe_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.has_rank_local_tokens:
            return hidden_states

        moe_a2a_backend = get_moe_a2a_backend()
        if moe_a2a_backend.is_none():
            from sglang.srt.layers.communicator_dsa_cp import (
                dsa_cp_gather_hidden_states,
            )

            return dsa_cp_gather_hidden_states(hidden_states)

        assert moe_a2a_backend.is_deepep() or moe_a2a_backend.is_megamoe(), (
            "The distributed model-input layout requires DeepEP or megaMoE "
            "when an MoE A2A backend is configured. "
            f"Got {moe_a2a_backend.value}."
        )
        return hidden_states

    def restore_moe_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.has_rank_local_tokens or not get_moe_a2a_backend().is_none():
            return hidden_states

        from sglang.srt.layers.communicator_dsa_cp import (
            dsa_cp_reduce_scatter_hidden_states,
        )

        return dsa_cp_reduce_scatter_hidden_states(hidden_states)


def get_model_execution_layout(forward_batch) -> ModelExecutionLayout:
    return ModelExecutionLayout(has_rank_local_tokens=is_cp_active(forward_batch))


def resolve_attention_partition(
    rank: Optional[int], size: Optional[int]
) -> Tuple[int, int]:
    if rank is not None and size is not None:
        return rank, size
    if is_cp_enabled():
        return 0, 1
    return get_parallel().attn_tp_rank, get_parallel().attn_tp_size
