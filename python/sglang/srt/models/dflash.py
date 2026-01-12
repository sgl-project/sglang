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
DFlash utilities for speculative decoding.

This module provides shared utilities for DFlash:
- RMSNorm3D: RMSNorm that works with 3D tensors
- build_target_layer_ids: Compute layer IDs for multi-layer feature extraction
"""

from typing import List

import torch
from torch import nn


class RMSNorm3D(nn.Module):
    """RMSNorm that works with 3D tensors [batch, seq_len, hidden_size].

    This is needed because SGLang's RMSNorm only handles 2D tensors,
    but DFlash operates on 3D tensors throughout.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> List[int]:
    """
    Build target layer IDs for multi-layer feature extraction.

    Distributes layer selection evenly across target model depth.
    For example, with 28 target layers and 3 draft layers:
    - Returns [1, 13, 24] (start=1, end=25, spread evenly)

    Args:
        num_target_layers: Number of layers in the target model
        num_draft_layers: Number of layers in the draft model

    Returns:
        List of layer indices to extract features from
    """
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids
