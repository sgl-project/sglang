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
"""Survivor and retiree tails of a Mooncake-native scale-down.

Lives beside the state machine rather than in ``model_runner.py``: this is the
ordering the shrink has to follow, not runner orchestration. It takes the few
collaborators it needs by keyword and returns what it wants written, so the
sequence can be exercised against fakes and the runner keeps its own field writes.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, NoReturn, Optional

import torch

from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPStateManager,
    await_retirees_departed,
)
from sglang.srt.elastic_ep.expert_map_repair import shrink_expert_metadata
from sglang.srt.environ import envs
from sglang.srt.eplb.eplb_manager import ExpertLayoutDivergence
from sglang.srt.runtime_context import get_exec

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.eplb.eplb_manager import EPLBManager
    from sglang.srt.managers.io_struct import ElasticScaleUpdateReq

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScaleDownFinalizeResult:
    """What the survivor tail asks ModelRunner to write once it returns."""

    reset_forward_pass_id: bool = False
    request_graph_recapture: bool = False
    scale_update: Optional[ElasticScaleUpdateReq] = None


def retire_and_exit(
    *, my_global_rank: int, park_hook: Optional[Callable[[], None]]
) -> NoReturn:
    """Retiree terminal: exit, or park for an orchestrator. FSM EXIT after cleanup."""
    if get_exec().moe.elastic_ep_retiree_lifecycle == "external":
        # Never return: the caller's event loop would collect on destroyed groups.
        logger.info(
            "[Elastic EP][retire] rank=%d parked; awaiting external termination",
            my_global_rank,
        )
        # The scheduler's park disarms its watchdog first, which this cannot reach;
        # without it the retiree SIGQUITs its parent after --watchdog-timeout. The
        # fallback covers a runner driven with no scheduler behind it.
        if park_hook is not None:
            park_hook()
        while True:
            time.sleep(5.0)
    sys.exit(0)


class ScaleDownFinalizer:
    """Survivor tail of a shrink: MoE / dp_attn / expert-location rebuild + commit."""

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        moe_ep_rank: int,
        eplb_manager: Optional[EPLBManager],
        reload_relabelled: Callable[[Dict[int, List[int]]], None],
        apply_dp_size: Callable[[int, int], None],
        scale_ready_barrier: Callable[..., None],
        init_lplb_solvers: Callable[[], None],
        reinit_recorder: Callable[[int], None],
        rearm_eplb: Callable[[], None],
    ) -> None:
        self._model_config = model_config
        self._moe_ep_rank = moe_ep_rank
        self._eplb_manager = eplb_manager
        self._reload_relabelled = reload_relabelled
        self._apply_dp_size = apply_dp_size
        self._scale_ready_barrier = scale_ready_barrier
        self._init_lplb_solvers = init_lplb_solvers
        self._reinit_recorder = reinit_recorder
        self._rearm_eplb = rearm_eplb

    def finalize(
        self,
        *,
        ranks_to_retire: List[int],
        target_size: int,
        effective_size: int,
    ) -> ScaleDownFinalizeResult:
        """Rebuild this survivor at the narrower width and commit it. No PG rebuild."""
        from sglang.srt.managers.io_struct import ElasticScaleUpdateReq

        eplb = self._eplb_manager
        await_retirees_departed(ranks_to_retire)
        ElasticEPStateManager.mark_phase("reconfiguring")

        if eplb is not None:
            eplb.reset_generator()

        # Before the truncation below, which rewrites the map the recorder converts
        # through; the reshuffle further down is what consumes these counts.
        pre_scale_logical_count = (
            eplb.snapshot_logical_count() if eplb is not None else None
        )

        # No broadcast: every survivor repairs the same pre-shrink map by the same rule,
        # so it already holds what one would send. Grow needs one; a joiner has no map.
        shrink_expert_metadata(
            model_config=self._model_config,
            from_ep_size=effective_size,
            effective_size=target_size,
            moe_ep_rank=self._moe_ep_rank,
            reload_relabelled=self._reload_relabelled,
        )

        ElasticEPStateManager.on_scale(effective_size, target_size)

        if eplb is not None:
            eplb.disable_rebalance("EPLB disabled during scale-down")

        self._apply_dp_size(target_size, self._moe_ep_rank)

        ElasticEPStateManager.mark_syncing_new_world()
        # Drain lingering GPU work still holding an RDMA slot to a retiree pre-barrier.
        torch.cuda.synchronize()
        # Commit before the barrier, so a stuck Mooncake cannot strand pre-shrink ep_size
        # against a post-shrink dp/mask.
        ElasticEPStateManager.commit_scale()
        self._scale_ready_barrier(target_size=target_size, log_tag="SURVIVOR")
        # Shrink only: on grow the rejoining rank would take experts before Mooncake
        # settled its links, and the p2p wait hangs uninterruptibly.
        if eplb is not None and envs.SGLANG_ENABLE_ELASTIC_SCALE_REBALANCE.get():
            try:
                eplb.reshuffle_for_scale(
                    target_size, logical_count=pre_scale_logical_count
                )
            except ExpertLayoutDivergence:
                # Deliberately not downgraded: a rank whose map disagrees with its
                # peers keeps serving and routes tokens to the wrong expert, so the
                # damage is wrong output nobody is told about. Propagates like the
                # orphan reload above.
                raise
            except Exception as exc:
                # Aborted with nothing installed, so the repaired mapping stands as it
                # was and EPLB retries later.
                logger.warning("[Elastic EP] scale reshuffle failed: %s", exc)

        # LPLB solvers hold the p2l / log2phy tensors and ep_size they were built with,
        # so after a shrink they are pre-shrink views. The reshuffle above rebuilds them,
        # but it is env-gated and may have failed, so rebuild unconditionally here.
        # Before the graph rebuild: a capture replays whatever the solvers hold.
        self._init_lplb_solvers()

        self._reinit_recorder(self._moe_ep_rank)
        self._rearm_eplb()

        scale_update = None
        if self._moe_ep_rank == 0:
            scale_update = ElasticScaleUpdateReq(
                success=True,
                effective_ep_size=target_size,
                slot_offset=target_size,
                slot_count=effective_size - target_size,
                direction="shrink",
            )
        # Decode graphs were captured against the pre-shrink layout, so a replay would
        # still drive the retired ranks' slots. Grow rebuilds in _finalize_scale_up;
        # shrink has to as well. Requested here, run from tick_elastic_scale: when
        # max_ep_size == tp_size this whole finalize runs at the tail of forward(),
        # and capturing there frees the graph memory pool while the output we are
        # about to return still points into it. The scheduler then decodes from
        # corrupted logits and dies indexing req_to_token out of bounds.
        return ScaleDownFinalizeResult(
            reset_forward_pass_id=True,
            request_graph_recapture=True,
            scale_update=scale_update,
        )
