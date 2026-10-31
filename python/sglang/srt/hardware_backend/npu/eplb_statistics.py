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

"""Ascend expert routing statistics for EPLB."""

from sglang.srt.eplb.expert_distribution import (
    _ExpertDistributionRecorderReal,
    _SelectExpertsSinglePassGatherer,
)
from sglang.srt.runtime_context import get_exec


class NpuExpertDistributionRecorder(_ExpertDistributionRecorderReal):
    """Use valid sender routes with the upstream recorder lifecycle."""

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
