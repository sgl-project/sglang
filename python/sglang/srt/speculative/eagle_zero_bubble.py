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
from __future__ import annotations

import torch


def validate_zero_bubble_config(
    *,
    enabled: bool,
    topk: int,
    num_steps: int,
    enable_multi_layer_eagle: bool,
    is_eagle3: bool,
    use_rejection_sampling: bool,
    speculative_adaptive: bool,
) -> None:
    if not enabled:
        return
    invalid = []
    if topk != 1:
        invalid.append("speculative_eagle_topk must be 1")
    if num_steps < 1:
        invalid.append("speculative_num_steps must be at least 1")
    if enable_multi_layer_eagle:
        invalid.append("multi-layer EAGLE is unsupported")
    if is_eagle3:
        invalid.append("EAGLE3 is unsupported")
    if use_rejection_sampling:
        invalid.append("rejection sampling is unsupported")
    if speculative_adaptive:
        invalid.append("adaptive speculative steps are unsupported")
    if invalid:
        raise ValueError("SGLANG_SPEC_V2_ZERO_BUBBLE: " + "; ".join(invalid))


def pad_zero_bubble_seed(
    *,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    num_steps: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_width = num_steps * topk
    if topk_p.shape != topk_index.shape:
        raise ValueError("zero-bubble topk probabilities and indices must match")
    if topk_p.shape[-1] > target_width:
        raise ValueError(
            f"zero-bubble seed width {topk_p.shape[-1]} exceeds {target_width}"
        )
    pad_width = target_width - topk_p.shape[-1]
    if pad_width == 0:
        return topk_p, topk_index
    output_shape = (*topk_p.shape[:-1], pad_width)
    return (
        torch.cat((topk_p, topk_p.new_zeros(output_shape)), dim=-1),
        torch.cat((topk_index, topk_index.new_zeros(output_shape)), dim=-1),
    )


def validate_prefetched_topk1(
    *,
    topk_index: torch.Tensor,
    batch_size: int,
    num_steps: int,
) -> None:
    expected = (batch_size, num_steps)
    if tuple(topk_index.shape) != expected:
        raise ValueError(
            f"zero-bubble prefetched token shape must be {expected}, "
            f"got {tuple(topk_index.shape)}"
        )
