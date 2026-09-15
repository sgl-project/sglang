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
                self.embeds = torch.cat(self.embeds, dim=0)  # raises — empty is invalid
            elif self.embeds[0].dim() == 1:
                # [hidden_dim] elements → stack adds the leading dim.
                self.embeds = torch.stack(self.embeds, dim=0)
            else:
                # [1, hidden_dim] (already has the leading dim) → plain concat.
                self.embeds = torch.cat(self.embeds, dim=0)
        if self.embeds.shape[0] != len(self.positions):
            raise ValueError(
                f"embeds length ({self.embeds.shape[0]}) != "
                f"positions length ({len(self.positions)})"
            )

    @classmethod
    def from_json(cls, value: dict) -> "PositionalEmbeds":
        """Materialize HTTP rows as an owned CPU tensor, matching Rust's EXT 2 wire."""
        rows, positions = value.get("embeds"), value.get("positions")
        if (
            not isinstance(rows, list)
            or not rows
            or not all(isinstance(row, list) and row for row in rows)
            or any(len(row) != len(rows[0]) for row in rows)
            or any(type(item) not in (int, float) for row in rows for item in row)
            or not isinstance(positions, list)
            or any(type(position) is not int for position in positions)
        ):
            raise ValueError(
                "positional_embed_overrides requires a nonempty numeric embeds "
                "matrix and an integer positions list"
            )
        tensor = torch.tensor(rows, dtype=torch.float32, device="cpu")
        if not torch.isfinite(tensor).all():
            raise ValueError("positional_embed_overrides embeds must be finite")
        return cls(tensor, positions)

    def validate(self, input_len: int, hidden_size: int) -> None:
        """Validate positions after tokenization, media expansion, and truncation."""
        if (
            self.embeds.ndim != 2
            or self.embeds.shape != (len(self.positions), hidden_size)
            or not torch.isfinite(self.embeds).all()
        ):
            raise ValueError(
                "positional_embed_overrides must have one embedding with "
                f"{hidden_size} finite values per position"
            )
        if any(
            type(position) is not int or not 0 <= position < input_len
            for position in self.positions
        ):
            raise ValueError(
                f"positional_embed_overrides positions must be in [0, {input_len})"
            )
