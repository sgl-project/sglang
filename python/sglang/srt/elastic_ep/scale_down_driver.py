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
"""Adapter between ``ScaleDownStateMachine`` and the side effects it drives.

Lives beside the state machine rather than in ``model_runner.py`` so the FSM can be
driven by a stub in tests: every transition the machine can take is one method here.
The two tails a runner owns arrive as callables, so nothing here needs a runner.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

import torch

from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPStateManager,
    _pre_nixl_retire,
    departure_announce,
    departure_cleared,
    departure_reset,
    nixl_retire_barrier_post,
    retire_barrier_check,
    retire_barrier_consume,
    retire_barrier_post,
    retiree_local_cleanup,
    try_retire_ranks,
)
from sglang.srt.elastic_ep.scale_down_state import ScaleDownStateMachine


class ScaleDownDriver:
    """FSM driver: bridges state transitions to the side effects they need."""

    def __init__(
        self,
        finalize_scale_down: Callable[..., None],
        retire_and_exit: Callable[[], None],
        is_idle: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._finalize_scale_down = finalize_scale_down
        self._retire_and_exit = retire_and_exit
        self._is_idle = is_idle

    def on_prepare(self, sm):
        departure_reset()
        ElasticEPStateManager.mark_phase("draining")

    def local_idle(self, sm) -> bool:
        """No local work left. Unwired (non-scheduler caller) reads as idle rather
        than holding the cohort on a predicate nobody can satisfy."""
        return self._is_idle is None or self._is_idle()

    def post_drain_barrier(self, sm):
        return retire_barrier_post()

    def announce_departure(self, sm):
        departure_announce()

    def departure_cleared(self, sm) -> bool:
        return departure_cleared()

    def on_depart_drain(self, sm):
        ElasticEPStateManager.mark_phase("retiring")

    def check_barrier(
        self, handle, *, block_s: Optional[float] = None, keep_serving: bool = False
    ):
        return retire_barrier_check(handle, block_s=block_s, keep_serving=keep_serving)

    def consume_barrier(self, handle):
        retire_barrier_consume(handle)

    def on_retiree_quiesce(self, sm):
        # FLIP_MASK narrows the device-side expert bound, which in-flight a2a asserts on.
        # is_fully_idle() gates the scheduler; it does not promise the device is done.
        torch.cuda.synchronize()

    def on_nixl_retire_pre(self, sm):
        # Quiesce while both peers are still connected: the barrier posts a tick early so
        # a2a keeps queueing, and disconnecting over in-flight RDMA spins forever.
        torch.cuda.synchronize()
        _pre_nixl_retire(sm.ranks_to_retire)

    def post_nixl_retire_barrier(self, sm):
        return nixl_retire_barrier_post()

    def on_flip_mask(self, sm):
        try_retire_ranks(sm.ranks_to_retire)

    def on_reconfig(self, sm):
        self._finalize_scale_down(
            ranks_to_retire=sm.ranks_to_retire,
            target_size=sm.target_size,
            effective_size=sm.effective_size,
        )

    def on_local_cleanup(self, sm):
        retiree_local_cleanup()

    def on_exit(self, sm):
        self._retire_and_exit()


def advance_scale_down(
    *,
    sm: Optional[ScaleDownStateMachine],
    driver: ScaleDownDriver,
    pending_size: int,
    effective_size: int,
    pending_since: float,
    my_global_rank: int,
    scale_timeout: float,
    fail_scale: Callable[[str, int], None],
) -> Optional[ScaleDownStateMachine]:
    """Step the shrink one tick. Returns the machine to keep, or None once terminal.

    Constructs the machine on the first pending-shrink tick, so the caller only has
    to hold whatever comes back.
    """
    if time.monotonic() - pending_since > scale_timeout:
        fail_scale(
            f"Timed out waiting for cohort at retire barrier (target={pending_size})",
            effective_size,
        )
        # Release before dropping: the machine may hold a barrier epoch this rank
        # leads, and a leaked leader wedges every later scale.
        if sm is not None:
            sm.abandon(driver)
        return None

    if sm is None:
        ranks_to_retire: List[int] = list(range(pending_size, effective_size))
        sm = ScaleDownStateMachine(
            is_retiree=my_global_rank in ranks_to_retire,
            target_size=pending_size,
            effective_size=effective_size,
            ranks_to_retire=ranks_to_retire,
            my_global_rank=my_global_rank,
        )

    sm.tick(driver)
    if sm.is_failed():
        fail_scale(sm.last_error or "unknown scale-down FSM failure", effective_size)
        # A failure can land with a barrier still posted; see abandon().
        sm.abandon(driver)
    # FAILED is terminal, so this also tears down after the failure above.
    return None if sm.is_terminal() else sm
