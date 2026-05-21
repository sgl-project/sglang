# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Compatibility shim for the pre-strategy CP API.

The real implementation lives in ``cp_strategy.py`` and ``cp_collectives.py``;
this module forwards to them so existing import sites keep working.  New code
should import from ``cp_strategy`` directly."""

from typing import Callable, List, Optional

import torch

from sglang.srt.layers.utils.cp_collectives import (
    cp_all_gather_reorganized_into_tensor,
    cp_all_gather_reorganized_into_tensor_kv_cache,
)
from sglang.srt.layers.utils.cp_strategy import (
    ContextParallelMetadata,
    get_cp_strategy,
    is_cp_enabled,
    is_zigzag,
)

__all__ = [
    "ContextParallelMetadata",
    "can_cp_split",
    "cp_all_gather_reorganized_into_tensor",
    "cp_all_gather_reorganized_into_tensor_kv_cache",
    "cp_all_gather_rerange_kv_cache",
    "cp_all_gather_rerange_output",
    "cp_allgather_and_save_kv_cache",
    "cp_attn_forward_extend",
    "cp_split_and_rebuild_data",
    "cp_split_and_rebuild_position",
    "is_prefill_context_parallel_enabled",
    "is_prefill_cp_in_seq_split",
    "prepare_context_parallel_metadata",
]


# ---------------------------------------------------------------------------
# Predicates kept as compat helpers; new code should use cp_strategy directly.
# ---------------------------------------------------------------------------


def is_prefill_context_parallel_enabled() -> bool:
    return is_cp_enabled()


def is_prefill_cp_in_seq_split() -> bool:
    return is_zigzag()


def can_cp_split(seq_len: int, cp_size: int, forward_batch) -> bool:
    s = get_cp_strategy()
    if s is None:
        return False
    return s.can_apply(seq_len, forward_batch)


# ---------------------------------------------------------------------------
# Forwarders to the active strategy.
# ---------------------------------------------------------------------------


def prepare_context_parallel_metadata(
    kv_len: int, cp_rank: int, cp_size: int, seqs_len: Optional[List[int]]
) -> ContextParallelMetadata:
    """Kept for callers that still pass ``cp_rank``/``cp_size`` explicitly. The
    strategy reads its own ``cp_size``/``cp_rank`` so those positional args
    are silently ignored when the active strategy disagrees — callers that
    need to pin a specific shape should call ``get_cp_strategy().build_metadata``
    directly."""
    s = get_cp_strategy()
    if s is None:
        return ContextParallelMetadata()
    return s.build_metadata(kv_len, seqs_len)


def cp_split_and_rebuild_data(forward_batch, x: torch.Tensor) -> torch.Tensor:
    return get_cp_strategy().shard_tokens(x, forward_batch)


def cp_split_and_rebuild_position(
    forward_batch, positions: torch.Tensor
) -> torch.Tensor:
    return get_cp_strategy().shard_positions(positions, forward_batch)


def cp_all_gather_rerange_output(
    input_tensor: torch.Tensor,
    cp_size: int,
    forward_batch,
    stream,
) -> torch.Tensor:
    return get_cp_strategy().gather_tokens(input_tensor, forward_batch, stream)


def cp_all_gather_rerange_kv_cache(
    input_tensor: torch.Tensor,
    cp_size: int,
    forward_batch,
    stream,
) -> torch.Tensor:
    return get_cp_strategy().gather_kv_cache(input_tensor, forward_batch, stream)


def cp_allgather_and_save_kv_cache(forward_batch, layer, k, v, cp_size: int) -> None:
    get_cp_strategy().materialize_full_kv(forward_batch, layer, k, v)


def cp_attn_forward_extend(
    forward_batch,
    q: torch.Tensor,
    device: torch.device,
    attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    return get_cp_strategy().run_attention(q, forward_batch, device, attn_fn)
