# Copyright 2023-2026 SGLang Team
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

"""Padding helpers shared by context-parallel strategies."""

from typing import Any

import torch

from sglang.srt.runtime_context import get_parallel


def get_cp_padding_align_size() -> int:
    """Return the token-count alignment required by the active CP strategy."""
    from sglang.srt.layers.attention.dsa.utils import is_dsa_prefill_cp_in_seq_split
    from sglang.srt.layers.utils.cp_utils import is_prefill_cp_in_seq_split

    attn_cp_size = get_parallel().attn_cp_size
    if is_prefill_cp_in_seq_split() or is_dsa_prefill_cp_in_seq_split():
        return attn_cp_size * 2
    return attn_cp_size


def pad_logical_token_to_physical(metadata: Any) -> None:
    """Align each CP rank's physical token count for CP collectives."""
    logical_tokens = list(metadata.per_rank_actual_token)
    align_size = get_cp_padding_align_size()
    physical_rank_len = (
        (max(logical_tokens) + align_size - 1) // align_size * align_size
    )
    metadata.per_rank_logical_token = logical_tokens
    metadata.per_rank_actual_token = [physical_rank_len] * len(logical_tokens)
    metadata.max_rank_len = [physical_rank_len] * len(logical_tokens)


def pad_local_rows(x: torch.Tensor, metadata: Any, dim: int) -> torch.Tensor:
    """Pad a local CP tensor from its logical length to its physical length."""
    if (
        metadata.per_rank_logical_token is None
        or metadata.per_rank_logical_token == metadata.per_rank_actual_token
    ):
        return x

    target_len = metadata.per_rank_actual_token[0]
    pad_size = target_len - x.shape[dim]
    assert pad_size >= 0
    if pad_size == 0:
        return x

    pad_shape = list(x.shape)
    pad_shape[dim] = pad_size
    return torch.cat([x, x.new_zeros(pad_shape)], dim=dim)
