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

"""Ascend's short initial learning phase; steady scheduling stays upstream."""

import logging

from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.runtime_context import get_device, get_exec, get_parallel

logger = logging.getLogger(__name__)


class NpuEPLBManager(EPLBManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        moe = get_exec().moe
        self._initial_rebalance_iterations = (
            64
            if get_device().device == "npu"
            and get_parallel().nnodes == 1
            and moe.ep_dispatch_algorithm == "static"
            and moe.init_expert_location == "trivial"
            and moe.elastic_ep_backend is None
            and self._rebalance_num_iterations > 64
            else None
        )
        if self._initial_rebalance_iterations is not None:
            logger.info(
                "[NPU EPLB initial learning] first=64 steady=%d window=%d",
                self._rebalance_num_iterations,
                moe.expert_distribution_recorder_buffer_size,
            )

    def _entrypoint(self):
        # The first window is empty, so a shorter learning period has no stale data.
        initial = self._initial_rebalance_iterations
        if initial is not None:
            for _ in range(initial):
                yield
            # Consume the first period even if rebalance() skips this attempt.
            self._initial_rebalance_iterations = None
            logger.info("[EPLBManager] initial learning period complete")
            yield from self.rebalance()
        yield from super()._entrypoint()
