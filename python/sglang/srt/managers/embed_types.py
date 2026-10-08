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
"""
Structs for embedding injection.

These are placed in a separate module to avoid circular imports between
io_struct.py and schedule_batch.py.
"""

from typing import List

import msgspec
import torch


class PositionalEmbeds(msgspec.Struct, array_like=True):
    """Embeddings to place at specific token positions.

    Accepts either a list of [1, hidden_dim] tensors or a pre-stacked [N, hidden_dim] tensor.
    In both cases, __post_init__ stacks into a single [N, hidden_dim] tensor to reduce
    ZMQ serialization overhead.

    Attributes:
        embeds: Stacked tensor of shape [N, hidden_dim] after __post_init__.
        positions: List of positions where embeddings should be injected.
    """

    embeds: torch.Tensor
    positions: List[int]

    def __post_init__(self):
        # Normalize list of tensors into a single [N, hidden_dim] tensor.
        # Dispatch by element rank to avoid a per-element unsqueeze.
        if isinstance(self.embeds, list):
            if not self.embeds:
                raise ValueError(
                    "positional_embed_overrides embeds must be a non-empty list of "
                    "tensors or a pre-stacked [N, hidden_dim] tensor."
                )
            if not all(isinstance(embed, torch.Tensor) for embed in self.embeds):
                raise ValueError(
                    "positional_embed_overrides embeds must contain only tensors."
                )
            try:
                if self.embeds[0].dim() == 1:
                    # [hidden_dim] elements → stack adds the leading dim.
                    self.embeds = torch.stack(self.embeds, dim=0)
                else:
                    # [1, hidden_dim] (already has the leading dim) → plain concat.
                    self.embeds = torch.cat(self.embeds, dim=0)
            except (RuntimeError, TypeError) as exc:
                raise ValueError(
                    "positional_embed_overrides embeds must have compatible tensor shapes."
                ) from exc
        self.validate()

    def validate(self) -> None:
        """Validate the normalized object before it crosses the scheduler boundary."""
        if not isinstance(self.embeds, torch.Tensor) or self.embeds.dim() != 2:
            raise ValueError(
                "positional_embed_overrides embeds must be a 2-D tensor [N, hidden_dim]."
            )
        if not isinstance(self.positions, list) or any(
            type(position) is not int for position in self.positions
        ):
            raise ValueError(
                "positional_embed_overrides positions must be a list of integers."
            )
        if self.embeds.shape[0] != len(self.positions):
            raise ValueError(
                f"embeds length ({self.embeds.shape[0]}) != "
                f"positions length ({len(self.positions)})"
            )

    def validate_hidden_dim(self, expected_hidden_dim: int) -> None:
        """Reject incompatible embeddings before the scheduler's scatter operation."""
        self.validate()
        actual = self.embeds.shape[-1]
        if actual != expected_hidden_dim:
            raise ValueError(
                f"positional_embed_overrides hidden_dim ({actual}) does not match "
                f"model hidden_size ({expected_hidden_dim}). Each embed tensor must "
                "have shape [hidden_size] or [1, hidden_size]."
            )
