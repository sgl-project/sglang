"""Survivor and retiree tails of a Mooncake-native scale-down. Collaborators arrive by
keyword and the result says what the runner should write, so fakes can stand in."""

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
        # The scheduler's hook disarms its watchdog, which would otherwise SIGQUIT
        # the parent; the loop alone covers a runner with no scheduler behind it.
        if park_hook is not None:
            park_hook()
        while True:
            time.sleep(5.0)
    sys.exit(0)


def finalize_scale_down(
    *,
    model_config: ModelConfig,
    moe_ep_rank: int,
    eplb: Optional[EPLBManager],
    reload_relabelled: Callable[[Dict[int, List[int]]], None],
    apply_dp_size: Callable[[int, int], None],
    scale_ready_barrier: Callable[..., None],
    init_lplb_solvers: Callable[[], None],
    reinit_recorder: Callable[..., None],
    rearm_eplb: Callable[[], None],
    ranks_to_retire: List[int],
    target_size: int,
    effective_size: int,
) -> ScaleDownFinalizeResult:
    """Survivor tail of a shrink: rebuild at the narrower width and commit it."""
    from sglang.srt.managers.io_struct import ElasticScaleUpdateReq

    await_retirees_departed(ranks_to_retire)
    ElasticEPStateManager.mark_phase("reconfiguring")

    logical_count = None
    if eplb is not None:
        eplb.reset_generator()
        # Before the truncation below, which rewrites the map the recorder converts
        # through; the reshuffle further down is what consumes these counts.
        logical_count = eplb.snapshot_logical_count()

    # No broadcast: every survivor repairs the same pre-shrink map by the same rule.
    shrink_expert_metadata(
        model_config=model_config,
        from_ep_size=effective_size,
        effective_size=target_size,
        moe_ep_rank=moe_ep_rank,
        reload_relabelled=reload_relabelled,
    )

    ElasticEPStateManager.on_scale(effective_size, target_size)
    if eplb is not None:
        eplb.disable_rebalance("EPLB disabled during scale-down")
    apply_dp_size(target_size, moe_ep_rank)

    ElasticEPStateManager.mark_syncing_new_world()
    # Drain lingering GPU work still holding an RDMA slot to a retiree pre-barrier.
    torch.cuda.synchronize()
    # Commit first, so a stuck barrier cannot strand old ep_size against new dp/mask.
    ElasticEPStateManager.commit_scale()
    scale_ready_barrier(target_size=target_size, log_tag="SURVIVOR")
    # Shrink only: on grow the rejoining rank would take experts before Mooncake
    # settled its links, and the p2p wait hangs uninterruptibly.
    if eplb is not None and envs.SGLANG_ENABLE_ELASTIC_SCALE_REBALANCE.get():
        try:
            eplb.reshuffle_for_scale(target_size, logical_count=logical_count)
        except ExpertLayoutDivergence:
            # Fatal: a diverged layout serves wrong experts; the FSM re-raises it.
            raise
        except Exception as exc:
            # Aborted with nothing installed: the repaired map stands, EPLB retries.
            logger.warning("[Elastic EP] scale reshuffle failed: %s", exc)

    # LPLB solvers hold pre-shrink views, and the reshuffle that rebuilds them is
    # optional. Before the graph rebuild: a capture replays whatever they hold.
    init_lplb_solvers()

    reuse_graphs = get_exec().moe.elastic_ep_reuse_cuda_graph
    # A kept decode graph still writes the gatherers it was captured against.
    reinit_recorder(keep_gatherers=reuse_graphs)
    rearm_eplb()

    scale_update = None
    if moe_ep_rank == 0:
        scale_update = ElasticScaleUpdateReq(
            success=True,
            effective_ep_size=target_size,
            slot_offset=target_size,
            slot_count=effective_size - target_size,
            direction="shrink",
        )
    # Rebuilt later from tick_elastic_scale: this may run at forward()'s tail, whose
    # output lives in the pool a capture frees. Reused graphs mask retired slots to 0.
    if reuse_graphs:
        logger.info(
            "[Elastic EP] keeping decode CUDA graphs across %d -> %d",
            effective_size,
            target_size,
        )
    return ScaleDownFinalizeResult(
        request_graph_recapture=not reuse_graphs, scale_update=scale_update
    )
