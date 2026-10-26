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

"""Exact Ascend route counts, batched across layers when capacity permits."""

import logging
import os
from contextlib import nullcontext

import torch

from sglang.srt.eplb import expert_distribution as distribution
from sglang.srt.hardware_backend.npu.eplb import get_eplb_device_group

logger = logging.getLogger(__name__)


class _NpuAccumulatorMixin:
    # Resolve lazily so distributed initialization and WORLD replacement remain valid.
    @property
    def _eplb_group(self):
        return get_eplb_device_group()

    @property
    def _utilization_device(self):
        return distribution.get_device_namespace().device


class NpuStatAccumulator(_NpuAccumulatorMixin, distribution._StatAccumulator):
    pass


class NpuDetailAccumulator(_NpuAccumulatorMixin, distribution._DetailAccumulator):
    pass


class NpuExpertDistributionRecorder(distribution._ExpertDistributionRecorderReal):
    """Reuse upstream recording while isolating NPU draft forward lifecycles."""

    def with_forward_pass(self, forward_pass_id, forward_batch):
        # Draft runners share target statistics. Disabling hooks alone still
        # lets the base lifecycle reset counts and append an empty draft pass.
        if self._disable_all:
            return nullcontext({})
        return super().with_forward_pass(forward_pass_id, forward_batch)


class NpuExpertDistributionGatherer(distribution._SelectExpertsSinglePassGatherer):
    """Buffer small forwards; count larger calls directly without dropping routes.

    Capture fixes the buffer slices written by each graph. Replay restores the
    graph's batching flag because Python routing hooks do not run on replay.
    The recorder owns reset/collect around each recorded forward.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        capacity_env = "SGLANG_NPU_EPLB_STATS_MAX_ROUTES"
        raw_capacity = os.environ.get(capacity_env, "96")
        try:
            capacity = int(raw_capacity)
            if not 1 <= capacity <= 65536:
                raise ValueError
        except ValueError as error:
            raise ValueError(
                f"{capacity_env} must be an integer in [1, 65536], got {raw_capacity!r}"
            ) from error
        # Capacity counts routes, not tokens: 96 holds 16 tokens at top-k 6.
        # It only selects the batching path; overflow still counts every route.
        # All captured graphs write to this shared buffer.
        self._batched_routes = torch.full(
            (self._data.shape[0], capacity),
            -1,
            dtype=torch.int32,
            device=self._data.device,
        )
        self._route_offsets = [0] * self._data.shape[0]
        self._batch_collected = False
        self._has_batched_routes = False
        logger.info("[NPU EPLB batch statistics] capacity=%d routes/layer", capacity)

    def on_graph_capture_forward_start(self) -> None:
        # Capture warmups bypass the normal reset. Reset offsets without
        # recording a buffer clear into the graph.
        self._route_offsets = [0] * len(self._route_offsets)

    def get_graph_capture_state(self) -> bool:
        return any(self._route_offsets)

    def on_graph_replay_forward_start(self, has_batched_routes: bool) -> None:
        # Graph replay does not call the Python on_select_experts hook.
        self._has_batched_routes |= has_batched_routes

    def reset(self):
        super().reset()
        if self._has_batched_routes:
            self._batched_routes.fill_(-1)
        self._route_offsets = [0] * self._data.shape[0]
        self._batch_collected = False
        self._has_batched_routes = False

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor) -> None:
        count = topk_ids.numel()
        start = self._route_offsets[layer_idx]
        end = start + count
        if end > self._batched_routes.shape[1]:
            # Count overflow directly; collect() adds any earlier buffered calls.
            return super().on_select_experts(layer_idx, topk_ids)
        self._batched_routes[layer_idx, start:end].copy_(topk_ids.flatten())
        self._route_offsets[layer_idx] = end
        if count:
            self._has_batched_routes = True

    def collect(self):
        if self._has_batched_routes and not self._batch_collected:
            ids = self._batched_routes
            mask = (ids >= 0) & (ids < self._data.shape[1])
            self._data.scatter_add_(1, ids.masked_fill(~mask, 0).long(), mask.int())
            self._batch_collected = True
        return super().collect()


def _get_npu_gatherers(recorder=None) -> dict[str, NpuExpertDistributionGatherer]:
    """Keep access to the shared recorder's internal state in one adapter."""
    if recorder is None:
        recorder = distribution.get_global_expert_distribution_recorder()
    if (
        not isinstance(recorder, distribution._ExpertDistributionRecorderReal)
        or recorder._disable_all
    ):
        return {}
    return {
        key: gatherer
        for key, gatherer in recorder._single_pass_gatherers.items()
        if isinstance(gatherer, NpuExpertDistributionGatherer)
    }


def on_graph_capture_forward_start(recorder=None):
    for gatherer in _get_npu_gatherers(recorder).values():
        gatherer.on_graph_capture_forward_start()


def get_graph_capture_state(recorder=None) -> dict[str, bool]:
    return {
        key: gatherer.get_graph_capture_state()
        for key, gatherer in _get_npu_gatherers(recorder).items()
    }


def on_graph_replay_forward_start(state, recorder=None):
    gatherers = _get_npu_gatherers(recorder)
    if not gatherers:
        return
    for key, gatherer_state in state.items():
        gatherers[key].on_graph_replay_forward_start(gatherer_state)
