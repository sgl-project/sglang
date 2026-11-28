# Copyright 2026 SGLang Team
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
"""Nemotron Shortcut vision path: Pixtral tower, then RMSNorm, 2x2 patch merge and a
two-layer tanh-GELU projector into the text stack's width."""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.managers.req_types import MultimodalDataItem
from sglang.srt.models.pixtral import PatchMerger, PixtralHFVisionModel


class NemotronShortcutMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        vision_hidden = config.vision_config.hidden_size
        text_hidden = config.text_config.hidden_size
        self.norm = RMSNorm(vision_hidden, eps=config.projector_norm_eps)
        self.patch_merger = PatchMerger(
            vision_encoder_dim=vision_hidden,
            spatial_merge_size=config.spatial_merge_size,
        )
        bias = config.multimodal_projector_bias
        self.linear_1 = nn.Linear(vision_hidden, text_hidden, bias=bias)
        self.linear_2 = nn.Linear(text_hidden, text_hidden, bias=bias)

    def forward(
        self, features: torch.Tensor, patch_grid_sizes: list[tuple[int, int]]
    ) -> torch.Tensor:
        """features: [num_patches, vision_hidden]; grid sizes are in patches."""
        x = self.patch_merger(self.norm(features), image_sizes=patch_grid_sizes)
        # `gelu_fast` in the checkpoint config is the tanh approximation.
        x = F.gelu(self.linear_1(x), approximate="tanh")
        return self.linear_2(x)


def nemotron_shortcut_image_features(
    *,
    vision_tower: PixtralHFVisionModel,
    projector: NemotronShortcutMultiModalProjector,
    items: List[MultimodalDataItem],
) -> torch.Tensor:
    """Projected features for every image, concatenated in item order.

    The tower runs one image at a time: its block mask is not honored by every
    ViT attention backend (FA4 treats the batch as one sequence), so batching
    images would let them attend to each other.
    """
    patch_size = vision_tower.patch_size
    outputs = []
    for item in items:
        pixel_values = item.feature
        sizes = item.model_specific_data["image_sizes"]
        for index in range(pixel_values.shape[0]):
            height, width = int(sizes[index][0]), int(sizes[index][1])
            features = vision_tower(
                pixel_values[index : index + 1], image_sizes=[(height, width)]
            )
            features = features.reshape(-1, features.shape[-1])
            outputs.append(
                projector(features, [(height // patch_size, width // patch_size)])
            )
    return torch.cat(outputs, dim=0)
