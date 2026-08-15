"""Attention-owned helpers for model execution over distributed token layouts."""

from typing import Optional, Tuple

import torch

from sglang.srt.layers.cp.base import is_cp_enabled
from sglang.srt.layers.cp.utils import cp_materialize_global_token_order, is_cp_active
from sglang.srt.runtime_context import get_parallel


def uses_sharded_prefill_layout(forward_batch) -> bool:
    return is_cp_active(forward_batch)


def resolve_model_attention_partition(
    rank: Optional[int], size: Optional[int]
) -> Tuple[int, int]:
    if rank is not None and size is not None:
        return rank, size
    if is_cp_enabled():
        return 0, 1
    return get_parallel().attn_tp_rank, get_parallel().attn_tp_size


def materialize_global_kv(kv: torch.Tensor, forward_batch, stream=None) -> torch.Tensor:
    if not uses_sharded_prefill_layout(forward_batch):
        return kv
    return cp_materialize_global_token_order(kv, forward_batch, stream)


def gather_sharded_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    from sglang.srt.layers.communicator_dsa_cp import dsa_cp_gather_hidden_states

    return dsa_cp_gather_hidden_states(hidden_states)


def reduce_scatter_sharded_hidden_states(
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    from sglang.srt.layers.communicator_dsa_cp import (
        dsa_cp_reduce_scatter_hidden_states,
    )

    return dsa_cp_reduce_scatter_hidden_states(hidden_states)
