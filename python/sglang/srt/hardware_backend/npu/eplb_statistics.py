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
from contextlib import contextmanager

import torch

from sglang.srt.eplb import expert_distribution as distribution
from sglang.srt.hardware_backend.npu.eplb import (
    get_eplb_device_group,
    warmup_eplb_communication,
)
from sglang.srt.hardware_backend.npu.eplb_statistics_kernel import (
    accumulate_route_counts,
)

logger = logging.getLogger(__name__)


class NpuEplbCommunication:
    """Resolve NPU statistics communication without changing accumulator classes."""

    # Resolve lazily so distributed initialization and WORLD replacement remain valid.
    @property
    def group(self) -> torch.distributed.ProcessGroup | None:
        return get_eplb_device_group()


class NpuExpertDistributionAccumulator(distribution._StatAccumulator):
    """Reduce window totals for EPLB policies that only consume summed loads."""

    def _get_physical_count_for_dump(self, output_mode):
        counts = super()._get_physical_count_for_dump(output_mode)
        if output_mode == "object" and distribution.get_exec().moe.eplb_algorithm in (
            "auto",
            "deepseek",
            "deepseek_hierarchical",
        ):
            # These policies sum the time dimension before planning. Aggregate
            # before mapping/all_reduce; keep int64 to avoid window-sum overflow.
            # File dumps and policies using per-step loads retain the full window.
            return counts.sum(dim=0, keepdim=True)
        return counts


class NpuExpertDistributionRecorder(distribution._ExpertDistributionRecorderReal):
    """Reuse upstream recording while isolating NPU draft forward lifecycles."""

    def _create_accumulator(self, expert_location_metadata, rank):
        moe = distribution.get_exec().moe
        accumulator_class = distribution._Accumulator.get_class()
        if moe.enable_eplb and not moe.elastic_ep_backend:
            # Join/scale paths need their own collective initialization order.
            warmup_eplb_communication()
            if accumulator_class is distribution._StatAccumulator:
                accumulator_class = NpuExpertDistributionAccumulator
        return accumulator_class(
            expert_location_metadata, rank, communication=NpuEplbCommunication()
        )

    def _create_gatherer(self, expert_location_metadata, rank):
        moe = distribution.get_exec().moe
        # Preserve the upstream per-token and approximate recording modes.
        if (
            moe.moe_a2a_backend == "deepep"
            and moe.expert_distribution_recorder_mode
            not in (
                "per_token",
                "stat_approx",
            )
        ):
            # Ascend receiver counts include graph padding; count valid routes.
            return NpuExpertDistributionGatherer(expert_location_metadata, rank)
        return super()._create_gatherer(expert_location_metadata, rank)

    @contextmanager
    def with_forward_pass(
        self, forward_pass_id, forward_batch, *, is_draft_worker=False
    ):
        # Draft runners share target statistics. Disabling hooks alone still
        # lets the base lifecycle reset counts and append an empty draft pass.
        if is_draft_worker:
            with self.disable_this_region():
                yield {}
        elif self._disable_all:
            yield {}
        else:
            with super().with_forward_pass(forward_pass_id, forward_batch) as outputs:
                yield outputs


class NpuExpertDistributionGatherer(distribution._SelectExpertsSinglePassGatherer):
    """Buffer small forwards; count larger calls directly without dropping routes.

    Capture fixes the buffer slices written by each graph. Replay restores the
    graph's batching flag because Python routing hooks do not run on replay.
    The recorder resets before a recorded forward and collects after all writes.
    Buffered graph replays share storage and cannot be accumulated by replaying
    multiple graphs for the same layers without an intervening collect/reset.
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
        self._direct_counting = False
        logger.info("[NPU EPLB batch statistics] capacity=%d routes/layer", capacity)

    def reset(self):
        super().reset()
        if self._has_batched_routes:
            self._batched_routes.fill_(-1)
        self._route_offsets = [0] * self._data.shape[0]
        self._batch_collected = False
        self._has_batched_routes = False

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor) -> None:
        count = topk_ids.numel()
        if count == 0:
            return
        start = self._route_offsets[layer_idx]
        end = start + count
        if self._direct_counting or end > self._batched_routes.shape[1]:
            # Count overflow directly; collect() adds any earlier buffered calls.
            return super().on_select_experts(layer_idx, topk_ids)
        self._batched_routes[layer_idx, start:end].copy_(topk_ids.flatten())
        self._route_offsets[layer_idx] = end
        self._has_batched_routes = True

    def collect(self):
        if self._has_batched_routes and not self._batch_collected:
            accumulate_route_counts(self._batched_routes, self._data)
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
    # Capture warmups bypass reset; clear offsets without capturing a buffer clear.
    for gatherer in _get_npu_gatherers(recorder).values():
        gatherer._route_offsets = [0] * len(gatherer._route_offsets)


def get_graph_capture_state(recorder=None) -> tuple[str, ...]:
    # Only buffered gatherers need replay notification; retain their names per graph.
    return tuple(
        key
        for key, gatherer in _get_npu_gatherers(recorder).items()
        if any(gatherer._route_offsets)
    )


def on_graph_replay_forward_start(state: tuple[str, ...], recorder=None):
    gatherers = _get_npu_gatherers(recorder)
    if not gatherers:
        return
    # Graph replay does not call the Python on_select_experts hook.
    for key in state:
        gatherers[key]._has_batched_routes = True


@contextmanager
def direct_route_counting():
    """Capture prefill counts directly, without replay-time Python flags."""
    gatherers = _get_npu_gatherers().values()
    previous = [(gatherer, gatherer._direct_counting) for gatherer in gatherers]
    try:
        for gatherer, _ in previous:
            gatherer._direct_counting = True
        yield
    finally:
        for gatherer, enabled in previous:
            gatherer._direct_counting = enabled
