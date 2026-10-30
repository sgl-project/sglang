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

"""Ascend EPLB statistics, communication selection, and draft isolation."""

from contextlib import contextmanager
from functools import wraps

from torch.distributed import ProcessGroup

from sglang.srt.eplb.expert_distribution import (
    _Accumulator,
    _DetailAccumulator,
    _ExpertDistributionRecorderReal,
    _SelectExpertsSinglePassGatherer,
    _StatAccumulator,
    get_global_expert_distribution_recorder,
)
from sglang.srt.hardware_backend.npu.eplb import (
    get_eplb_device_group,
    warmup_eplb_communication,
)
from sglang.srt.runtime_context import get_exec


def isolate_draft_expert_recording(forward):
    """Disable draft hooks and let the NPU recorder skip reset/append."""

    @wraps(forward)
    def wrapped(runner, *args, **kwargs):
        if runner.is_draft_worker:
            recorder = get_global_expert_distribution_recorder()
            with recorder.disable_this_region():
                return forward(runner, *args, **kwargs)
        return forward(runner, *args, **kwargs)

    return wrapped


class _NpuEplbAccumulatorMixin:
    """Share TP communication and initialize migration channels before recording."""

    def __init__(self, *args, **kwargs):
        moe = get_exec().moe
        if moe.enable_eplb and not moe.elastic_ep_backend:
            # Join/scale paths need their own collective initialization order.
            warmup_eplb_communication()
        super().__init__(*args, **kwargs)

    @property
    def _communication_group(self) -> ProcessGroup:
        # Resolve lazily so a replaced TP group is used by subsequent dumps.
        return get_eplb_device_group()


class _NpuDetailAccumulator(_NpuEplbAccumulatorMixin, _DetailAccumulator):
    pass


class _NpuStatAccumulator(_NpuEplbAccumulatorMixin, _StatAccumulator):
    pass


class NpuExpertDistributionRecorder(_ExpertDistributionRecorderReal):
    """Reuse upstream recording with Ascend counters and communication."""

    def _create_accumulator(self, expert_location_metadata, rank):
        accumulator_class = {
            _StatAccumulator: _NpuStatAccumulator,
            _DetailAccumulator: _NpuDetailAccumulator,
        }[_Accumulator.get_class()]
        return accumulator_class(expert_location_metadata, rank)

    def _create_gatherer(self, expert_location_metadata, rank):
        moe = get_exec().moe
        # Preserve the upstream per-token and approximate recording modes.
        if (
            moe.moe_a2a_backend == "deepep"
            and moe.expert_distribution_recorder_mode
            not in ("per_token", "stat_approx")
        ):
            # Ascend receiver counts include graph padding; count valid routes.
            return _SelectExpertsSinglePassGatherer(expert_location_metadata, rank)
        return super()._create_gatherer(expert_location_metadata, rank)

    @contextmanager
    def with_forward_pass(self, forward_pass_id, forward_batch):
        # The entry guard runs before reset; model-local guards start too late.
        # Skip append as well, so draft zeros do not evict target window entries.
        if self._disable_all:
            yield {}
        else:
            with super().with_forward_pass(forward_pass_id, forward_batch) as outputs:
                yield outputs
