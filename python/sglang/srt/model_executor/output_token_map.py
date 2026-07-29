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

from __future__ import annotations

from typing import Any, Sequence

import torch


def validate_output_token_ids(
    token_ids: Sequence[int] | torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    output_token_ids = torch.as_tensor(token_ids, dtype=torch.int64, device="cpu")
    if output_token_ids.ndim != 1 or output_token_ids.numel() == 0:
        raise ValueError("The output token map must be a non-empty 1D tensor.")
    if output_token_ids.min().item() < 0:
        raise ValueError("The output token map contains a negative token ID.")
    if output_token_ids.max().item() >= vocab_size:
        raise ValueError(
            "The output token map contains a token ID outside the model vocabulary."
        )
    if torch.unique(output_token_ids).numel() != output_token_ids.numel():
        raise ValueError("The output token map contains duplicate token IDs.")
    return output_token_ids.contiguous()


def project_vocab_tensor(
    tensor: torch.Tensor,
    output_token_ids: torch.Tensor | None,
) -> torch.Tensor:
    if output_token_ids is None or tensor.shape[-1] == output_token_ids.numel():
        return tensor
    return tensor.index_select(-1, output_token_ids.to(tensor.device))


def apply_projected_vocab_mask(
    logits: torch.Tensor,
    vocab_mask: torch.Tensor,
    output_token_ids: torch.Tensor,
) -> None:
    token_ids = output_token_ids.to(vocab_mask.device)
    word_indices = torch.div(token_ids, 32, rounding_mode="floor")
    bit_indices = torch.remainder(token_ids, 32).to(torch.int32)
    words = vocab_mask.index_select(-1, word_indices).to(torch.int32)
    allowed = ((words >> bit_indices) & 1).bool().to(logits.device)
    logits.masked_fill_(~allowed, float("-inf"))


def map_output_token_indices(value: Any, output_token_ids: torch.Tensor) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        indices = value.to(torch.long)
        mapped = output_token_ids.to(value.device)[indices.clamp_min(0)]
        return torch.where(indices >= 0, mapped, indices)
    if isinstance(value, list):
        return [map_output_token_indices(item, output_token_ids) for item in value]
    if isinstance(value, tuple):
        return tuple(map_output_token_indices(item, output_token_ids) for item in value)
    index = int(value)
    return output_token_ids[index].item() if index >= 0 else index
