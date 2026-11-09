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
from typing import Dict, List, Optional

import msgspec

from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class Drain(msgspec.Struct, frozen=True):
    """
    The drained adapter leaves the batch around when `anchor_req` finishes,
    freeing its slot for `starving_adapter` (None is the base model).
    """

    starving_adapter: Optional[str]
    anchor_req: Req


def _max_rem_tokens(req: Req) -> int:
    return req.sampling_params.max_new_tokens - len(req.output_ids)


class LoRADrainer:
    """
    Frees a batch slot for LoRA adapters whose requests have waited too long.

    An adapter is starving once its oldest waiting request has waited longer than
    `max_wait_time_secs`. For each starving adapter, the drainer drains an adapter in
    the batch: the drained adapter stops taking requests that would outlast its
    running ones, so it leaves the batch when they finish and the starving adapter
    takes its slot. The adapter that has waited longest goes first, and it gets the
    adapter that will leave the batch soonest.
    """

    def __init__(self, max_loras_per_batch: int, max_wait_time_secs: float = 0.0):
        self.max_loras_per_batch = max_loras_per_batch
        self.max_wait_time_secs = max_wait_time_secs
        self.drained_adapter_to_drain: Dict[Optional[str], Drain] = {}

    def update_draining_state(
        self,
        waiting_queue: List[Req],
        running_reqs: List[Req],
        *,
        pinned_loras: Optional[set[str]] = None,
    ) -> None:
        """Ends drains that are no longer needed; starts drains for starving adapters."""
        pinned_loras = pinned_loras or set()
        running_reqs = [req for req in running_reqs if not req.finished()]
        running_adapters = {req.lora_id for req in running_reqs}
        waiting_adapters = {req.lora_id for req in waiting_queue}

        # A drain is over once the drained adapter has left the batch,
        # or once the starving adapter got a slot or has nothing waiting.
        for adapter, drain in list(self.drained_adapter_to_drain.items()):
            if (
                adapter not in running_adapters
                or drain.starving_adapter in running_adapters
                or drain.starving_adapter not in waiting_adapters
            ):
                logger.debug(f"LoRA adapter {adapter} stopped draining")
                del self.drained_adapter_to_drain[adapter]

        # Pinned adapters reserve slots even when they have no running requests.
        if len(running_adapters | pinned_loras) >= self.max_loras_per_batch:
            starving_adapters = self._starving_adapters(waiting_queue)
            self._start_drains(starving_adapters, running_reqs, pinned_loras)

    def can_schedule(self, req: Req) -> bool:
        """
        Whether `req` may join the batch.

        A drained adapter only takes requests that need at most the tokens its anchor
        request has left, so that admitting them does not delay the drain.
        """
        drain = self.drained_adapter_to_drain.get(req.lora_id)
        if drain is None:
            return True

        anchor_req = drain.anchor_req
        # A retracted or preempted anchor no longer makes progress.
        if anchor_req.finished() or anchor_req.is_retracted:
            return False

        return req.sampling_params.max_new_tokens <= _max_rem_tokens(anchor_req)

    def _starving_adapters(self, waiting_queue: List[Req]) -> List[Optional[str]]:
        """Adapters whose oldest request has waited too long, longest wait first."""
        now = time.monotonic()
        adapter_to_longest_wait: Dict[Optional[str], float] = {}
        for req in waiting_queue:
            wait = now - req.time_stats.wait_queue_entry_time
            adapter_to_longest_wait[req.lora_id] = max(
                wait, adapter_to_longest_wait.get(req.lora_id, 0.0)
            )

        starving = [
            adapter
            for adapter, wait in adapter_to_longest_wait.items()
            if wait > self.max_wait_time_secs
        ]
        return sorted(
            starving, key=lambda adapter: adapter_to_longest_wait[adapter], reverse=True
        )

    def _start_drains(
        self,
        starving_adapters: List[Optional[str]],
        running_reqs: List[Req],
        pinned_loras: set[str],
    ) -> None:
        # Each adapter in the batch leaves it once its request with the most tokens
        # left finishes.
        adapter_to_longest_req: Dict[Optional[str], Req] = {}
        for req in running_reqs:
            longest = adapter_to_longest_req.get(req.lora_id)
            if longest is None or _max_rem_tokens(req) > _max_rem_tokens(longest):
                adapter_to_longest_req[req.lora_id] = req

        running_adapters = adapter_to_longest_req.keys()
        already_drained_for = {
            drain.starving_adapter for drain in self.drained_adapter_to_drain.values()
        }
        for starving_adapter in starving_adapters:
            if (
                starving_adapter in running_adapters
                or starving_adapter in pinned_loras
                or starving_adapter in already_drained_for
            ):
                continue

            candidates = [
                adapter
                for adapter in running_adapters
                if adapter not in self.drained_adapter_to_drain
                and adapter not in pinned_loras
            ]
            if not candidates:
                return

            drained = min(
                candidates,
                key=lambda adapter: _max_rem_tokens(adapter_to_longest_req[adapter]),
            )
            self.drained_adapter_to_drain[drained] = Drain(
                starving_adapter=starving_adapter,
                anchor_req=adapter_to_longest_req[drained],
            )
            logger.debug(f"LoRA adapter {drained} is draining for {starving_adapter}")
