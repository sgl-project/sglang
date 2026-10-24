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

"""Batch exact NPU TopK statistics across MoE layers."""

import os

import torch


class BatchedTopKStatisticsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        capacity = int(os.environ.get("SGLANG_NPU_EPLB_STATS_MAX_ROUTES", "96"))
        if not 1 <= capacity <= 65536:
            raise ValueError("Invalid batched statistics route capacity")
        # Allocate once before capture. Every captured graph writes this same
        # storage, regardless of its batch size or temporary tensor addresses.
        self._batched_routes = torch.full(
            (self._data.shape[0], capacity),
            -1,
            dtype=torch.int32,
            device=self._data.device,
        )
        self._route_offsets = [0] * self._data.shape[0]
        self._batch_collected = False

    def on_graph_capture_forward_start(self):
        # Warmups bypass ModelRunner's normal recorder reset. Only Python
        # branch state changes here; runtime reset clears the shared buffers.
        self._route_offsets = [0] * len(self._route_offsets)

    def reset(self):
        super().reset()
        self._batched_routes.fill_(-1)
        self._route_offsets = [0] * self._data.shape[0]
        self._batch_collected = False

    def on_select_experts(self, layer_idx, topk_ids):
        count = topk_ids.numel()
        start = self._route_offsets[layer_idx]
        end = start + count
        if end > self._batched_routes.shape[1]:
            # Large eager batches and repeated layer invocations retain exact
            # original counting. Stored earlier invocations are still added.
            return super().on_select_experts(layer_idx, topk_ids)
        self._batched_routes[layer_idx, start:end].copy_(topk_ids.flatten())
        self._route_offsets[layer_idx] = end

    def collect(self):
        if not self._batch_collected:
            ids = self._batched_routes
            mask = (ids >= 0) & (ids < self._data.shape[1])
            self._data.scatter_add_(1, ids.masked_fill(~mask, 0).long(), mask.int())
            self._batch_collected = True
        return super().collect()
