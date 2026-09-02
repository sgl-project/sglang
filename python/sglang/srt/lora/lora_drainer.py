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

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

DRAIN_SCHEDULE_TOLERANCE = 1.2


@dataclass
class AdapterStats:
    num_waiting_reqs: int = 0
    max_wait_time_secs: float = 0.0
    max_remaining_tokens: int = 0
    is_draining_for: Optional[str] = None

    def _reset_stats(self):
        self.num_waiting_reqs = 0
        self.max_wait_time_secs = 0.0
        self.max_remaining_tokens = 0

    def is_starving(self, drain_wait_threshold: float):
        return (
            self.max_wait_time_secs > drain_wait_threshold and self.num_waiting_reqs > 0
        )


class LoRADrainer:
    """
    Drainer for LoRA requests that manages draining. It tracks:
    - Number of waiting requests per adapter
    - Maximum wait time for requests needing each adapter
    - Maximum number of tokens needed for running requests for each adapter
    """

    def __init__(self, max_loras_per_batch: int, max_wait_time_secs: float = 0.0):
        self.max_loras_per_batch = max_loras_per_batch
        self.max_wait_time_secs = max_wait_time_secs
        self.adapter_to_stats: Dict[Optional[str], AdapterStats] = defaultdict(
            AdapterStats
        )

    def update_draining_state(
        self,
        waiting_queue: List[Req],
        running_reqs: List[Req],
    ) -> None:
        """
        Update LoRA drainer state based on current waiting queue and running requests.

        This method updates adapter statistics, identifies starving adapters that need
        to be scheduled, and marks adapters for draining to make room for starving ones.
        """
        self._update_adapter_stats(waiting_queue, running_reqs)
        self._update_draining_loras(running_reqs)
        self._update_fully_drained_loras(running_reqs)

    def _update_adapter_stats(
        self,
        waiting_queue: List[Req],
        running_reqs: List[Req],
    ) -> None:
        for stats in self.adapter_to_stats.values():
            stats._reset_stats()

        for req in waiting_queue:
            stats = self.adapter_to_stats[req.lora_id]

            stats.num_waiting_reqs += 1
            stats.max_wait_time_secs = max(
                stats.max_wait_time_secs,
                time.monotonic() - req.time_stats.wait_queue_entry_time,
            )

        for req in running_reqs:
            stats = self.adapter_to_stats[req.lora_id]

            stats.max_remaining_tokens = max(
                stats.max_remaining_tokens,
                req.sampling_params.max_new_tokens - len(req.output_ids),
            )

    def _update_draining_loras(self, running_reqs: List[Req]) -> None:
        """
        Select LoRA adapters to drain based on starvation detection.

        This method identifies adapters in the waiting queue that are "starving"
        (waiting too long) and marks currently running adapters as "draining"
        to make room for the starving adapters. Draining adapters will not
        accept new requests, allowing them to complete and free up slots.
        """
        running_adapter_ids = {req.lora_id for req in running_reqs}
        if len(running_adapter_ids) < self.max_loras_per_batch:
            return None

        starving_adapters = set()
        draining_for_adapters = set()
        for adapter_id, stats in self.adapter_to_stats.items():
            if stats.is_starving(self.max_wait_time_secs):
                starving_adapters.add(adapter_id)

            draining_for_adapter = stats.is_draining_for
            if draining_for_adapter is not None:
                draining_for_adapters.add(draining_for_adapter)

        new_starving_adapters = starving_adapters - draining_for_adapters
        if not new_starving_adapters:
            return None

        sorted_new_starving_adapters = sorted(
            new_starving_adapters,
            key=lambda adapter: self.adapter_to_stats[adapter].max_wait_time_secs,
            reverse=True,
        )

        eligible_to_drain_adapters = {
            adapter
            for adapter in running_adapter_ids
            if self.adapter_to_stats[adapter].is_draining_for is None
        }

        for starving_adapter in sorted_new_starving_adapters:
            if not eligible_to_drain_adapters:
                break

            min_eligible_adapter = min(
                eligible_to_drain_adapters,
                key=lambda adapter_id: self.adapter_to_stats[
                    adapter_id
                ].max_remaining_tokens,
            )

            self.adapter_to_stats[min_eligible_adapter].is_draining_for = (
                starving_adapter
            )
            logger.debug(
                f"LoRA adapter {min_eligible_adapter} is draining for {starving_adapter}"
            )

            eligible_to_drain_adapters.remove(min_eligible_adapter)

    def _update_fully_drained_loras(self, running_reqs: List[Req]) -> None:
        """
        Clear draining state for adapters that have fully drained.

        An adapter is considered fully drained when it was marked as draining
        but no longer has any running requests.
        """
        running_adapter_ids = {req.lora_id for req in running_reqs}
        for adapter_id, stats in self.adapter_to_stats.items():
            if stats.is_draining_for is None:
                continue

            if adapter_id not in running_adapter_ids:
                logger.debug(f"LoRA adapter {adapter_id} finished draining")
                stats.is_draining_for = None

    def can_schedule(self, req: Req) -> bool:
        """
        Check if a request can be scheduled based on draining state.

        If the adapter for this request is currently draining, only allow
        scheduling if the request's max_new_tokens is within tolerance of
        the max remaining tokens for the draining adapter.
        """
        stats = self.adapter_to_stats[req.lora_id]
        if not stats.is_draining_for:
            return True

        return (
            req.sampling_params.max_new_tokens
            <= stats.max_remaining_tokens * DRAIN_SCHEDULE_TOLERANCE
        )
