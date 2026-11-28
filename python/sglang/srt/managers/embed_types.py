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

from __future__ import annotations

from typing import Callable, List

import msgspec
import torch

from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy


class PositionalEmbeds(msgspec.Struct, array_like=True):
    """Embeddings to place at specific token positions.

    Accepts either a list of [1, hidden_dim] tensors or a pre-stacked [N, hidden_dim] tensor.
    In both cases, __post_init__ stacks into a single [N, hidden_dim] tensor to reduce
    ZMQ serialization overhead.

    Attributes:
        embeds: Stacked tensor of shape [N, hidden_dim] after __post_init__.
        positions: List of positions where embeddings should be injected.
    """

    # torch.Tensor or CudaIpcTensorTransportProxy; msgspec cannot type that union.
    embeds: object
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
            elif self.embeds[0].dim() == 1:
                # [hidden_dim] elements → stack adds the leading dim.
                self.embeds = torch.stack(self.embeds, dim=0)
            else:
                # [1, hidden_dim] (already has the leading dim) → plain concat.
                self.embeds = torch.cat(self.embeds, dim=0)
        if isinstance(self.embeds, torch.Tensor):
            # validate_positional_embed_overrides reads only the last dim,
            # and a [N, 1, hidden_dim] input would abort its whole scheduler batch.
            if self.embeds.dim() != 2:
                raise ValueError(
                    f"positional_embed_overrides embeds must be 2-D [N, hidden_dim], "
                    f"got shape {tuple(self.embeds.shape)}."
                )
            if self.embeds.shape[0] != len(self.positions):
                raise ValueError(
                    f"embeds length ({self.embeds.shape[0]}) != "
                    f"positions length ({len(self.positions)})"
                )

    def map_object_fields(self, transform: Callable[[list], list]) -> PositionalEmbeds:
        """Copy with transform applied to [embeds]; returning that list unchanged returns self."""
        values = [self.embeds]
        new_values = transform(values)
        if new_values is values:
            return self
        (embeds,) = new_values
        return msgspec.structs.replace(self, embeds=embeds)

    def reconstruct(self, target_device: int) -> None:
        """Copy embeds out of the CUDA IPC pool onto target_device and release the slot."""
        if isinstance(self.embeds, CudaIpcTensorTransportProxy):
            self.embeds = self.embeds.reconstruct_on_target_device(target_device)


# Raising here returns HTTP 400, where a bad value would abort its scheduler batch.
def validate_positional_embed_overrides(value: object, *, hidden_size: int) -> None:
    if not isinstance(value, PositionalEmbeds):
        raise ValueError(
            "positional_embed_overrides must be a PositionalEmbeds, got "
            f"{type(value).__name__}."
        )
    actual = value.embeds.shape[-1]
    if actual != hidden_size:
        raise ValueError(
            f"positional_embed_overrides hidden_dim ({actual}) does not match "
            f"model hidden_size ({hidden_size}). Each embed tensor must have shape "
            f"[hidden_size] or [1, hidden_size]."
        )
