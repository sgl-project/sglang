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

"""Ascend sender-route recording and statistics communication."""

from contextlib import contextmanager

import torch

from sglang.srt.eplb import expert_distribution as distribution
from sglang.srt.hardware_backend.npu.eplb import (
    get_eplb_device_group,
    warmup_eplb_communication,
)


class _NpuAccumulatorMixin:
    """Share the TP device group across NPU statistics modes and migration."""

    # Resolve lazily so a replaced TP group is used by subsequent dumps.
    @property
    def _communication_group(self) -> torch.distributed.ProcessGroup:
        return get_eplb_device_group()


class _NpuDetailAccumulator(_NpuAccumulatorMixin, distribution._DetailAccumulator):
    pass


class NpuExpertDistributionAccumulator(
    _NpuAccumulatorMixin, distribution._StatAccumulator
):
    """Reduce window totals for EPLB policies that only consume summed loads."""

    def _get_physical_count_for_dump(self, output_mode):
        counts = super()._get_physical_count_for_dump(output_mode)
        moe = distribution.get_exec().moe
        if (
            moe.enable_eplb
            and not moe.elastic_ep_backend
            and output_mode == "object"
            and moe.eplb_algorithm in ("auto", "deepseek", "deepseek_hierarchical")
        ):
            # These policies sum the time dimension before planning. Aggregate
            # before mapping/all_reduce; keep int64 to avoid window-sum overflow.
            # File dumps and policies using per-step loads retain the full window.
            return counts.sum(dim=0, keepdim=True)
        return counts


class NpuExpertDistributionRecorder(distribution._ExpertDistributionRecorderReal):
    """Reuse upstream recording with Ascend counters and communication."""

    def _create_accumulator(self, expert_location_metadata, rank):
        moe = distribution.get_exec().moe
        accumulator_class = {
            distribution._StatAccumulator: NpuExpertDistributionAccumulator,
            distribution._DetailAccumulator: _NpuDetailAccumulator,
        }[distribution._Accumulator.get_class()]
        if moe.enable_eplb and not moe.elastic_ep_backend:
            # Join/scale paths need their own collective initialization order.
            warmup_eplb_communication()
        return accumulator_class(expert_location_metadata, rank)

    def _create_gatherer(self, expert_location_metadata, rank):
        moe = distribution.get_exec().moe
        # Preserve the upstream per-token and approximate recording modes.
        if (
            moe.moe_a2a_backend == "deepep"
            and moe.expert_distribution_recorder_mode
            not in ("per_token", "stat_approx")
        ):
            # Ascend receiver counts include graph padding; count valid routes.
            return distribution._SelectExpertsSinglePassGatherer(
                expert_location_metadata, rank
            )
        return super()._create_gatherer(expert_location_metadata, rank)

    @contextmanager
    def with_forward_pass(self, forward_pass_id, forward_batch):
        # The NPU entry-point guard disables the whole draft lifecycle.
        if self._disable_all:
            yield {}
        else:
            with super().with_forward_pass(forward_pass_id, forward_batch) as outputs:
                yield outputs
