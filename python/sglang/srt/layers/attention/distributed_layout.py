"""Attention-owned helpers for model execution over distributed token layouts."""

from typing import Optional, Tuple

import torch

from sglang.srt.layers.cp.base import is_cp_enabled
from sglang.srt.layers.cp.utils import (
    cp_materialize_global_token_order,
    enable_cp_v2,
    is_cp_v2_active,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_hip


def uses_sharded_prefill_layout(forward_batch) -> bool:
    if is_cp_v2_active(forward_batch):
        return True
    if is_hip():
        from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp

        return dsa_use_prefill_cp(forward_batch)
    return False


def resolve_model_attention_partition(
    rank: Optional[int], size: Optional[int]
) -> Tuple[int, int]:
    if rank is not None and size is not None:
        return rank, size
    if enable_cp_v2() and is_cp_enabled():
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
