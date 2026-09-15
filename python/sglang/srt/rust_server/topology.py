"""Resolve frontend ownership from the scheduler's parallel state."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.scheduler import Scheduler


@dataclass(frozen=True, slots=True)
class FrontendTopology:
    dp_size: int
    dp_rank: int | None
    is_leader: bool

    @classmethod
    def from_parallel_state(cls, ps: ParallelState) -> FrontendTopology:
        rank = ps.dp_rank if ps.dp_size > 1 else None
        if ps.dp_size < 1 or (
            ps.dp_size > 1 and (rank is None or not 0 <= rank < ps.dp_size)
        ):
            raise ValueError(
                f"Invalid frontend DP topology: rank={rank}, size={ps.dp_size}"
            )
        return cls(
            dp_size=ps.dp_size,
            dp_rank=rank,
            is_leader=ps.pp_rank == 0 and ps.attn_tp_rank == 0 and ps.attn_cp_rank == 0,
        )


def collect_worker_infos(
    scheduler: Scheduler, *, metrics_enabled: bool = False
) -> list[dict]:
    """Share bound leader addresses, including leaders on another node."""
    import torch.distributed as dist

    from sglang.srt.rust_server.metrics import METRICS_SOURCE_ENV

    cfg = resolving_view(scheduler.server_args)
    worker = None
    if scheduler.rust_server is not None:
        host = cfg.host
        if cfg.nnodes > 1 or host in ("::", "0.0.0.0"):
            host = get_local_ip_auto()
        worker = {
            "dp_rank": scheduler.rust_server.topology.dp_rank,
            "url": NetworkAddress(host, scheduler.rust_server.http_port).to_url(),
        }
    group = scheduler.world_group.cpu_group
    local = {
        "worker": worker,
        "metrics_source": json.loads(os.environ.get(METRICS_SOURCE_ENV, "null")),
    }
    infos = [None] * dist.get_world_size(group)
    dist.all_gather_object(infos, local, group=group)
    sources = {}
    for info in infos:
        source = info["metrics_source"]
        if metrics_enabled and source is None:
            raise ValueError("A scheduler is missing its Python metrics source")
        if source is not None:
            source_id = source["source_id"]
            if source_id in sources and sources[source_id] != source:
                raise ValueError(f"Conflicting Python metrics source: {source_id}")
            sources[source_id] = source
    if metrics_enabled and len(sources) != cfg.nnodes:
        raise ValueError(
            f"Expected {cfg.nnodes} Python metrics sources, got {len(sources)}"
        )
    workers = []
    for info in infos:
        if info["worker"] is not None:
            worker = info["worker"]
            if sources:
                worker["metrics_sources"] = list(sources.values())
            workers.append(worker)
    return sorted(
        workers,
        key=lambda worker: worker["dp_rank"] or 0,
    )
