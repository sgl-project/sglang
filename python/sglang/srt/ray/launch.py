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
"""Ray-based scheduler launching and lifecycle management."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, List, Optional

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from sglang.srt.entrypoints.engine import (
    SchedulerInitResult,
    _calculate_rank_ranges,
)
from sglang.srt.ray.scheduler_actor import SchedulerActor
from sglang.srt.server_args import ZMQ_TCP_PORT_DELTA, PortArgs, ServerArgs

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RaySchedulerInitResult(SchedulerInitResult):
    _actors: Optional[List[Any]] = None
    _event_loop_refs: Optional[List[Any]] = None

    def wait_for_ready(self) -> None:
        pass

    def wait_for_completion(self) -> None:
        try:
            ray.get(self._event_loop_refs)
        except Exception as e:
            logger.error(f"Ray scheduler actor terminated with error: {e}")

    def cleanup(self) -> None:
        if self._actors is not None:
            for actor in self._actors:
                try:
                    ray.kill(actor)
                except Exception:
                    pass
            self._actors = None


def _get_rank0_node_ip(placement_group) -> str:
    """Get the IP address of the node where rank 0 will run.

    Uses a probe task to discover the IP of the placement group's first bundle node.
    This is needed because rank 0 starts the TCPStore server for torch.distributed,
    so dist_init_addr must be the IP of the node where rank 0 runs, not the driver node.
    """

    @ray.remote(num_cpus=0, num_gpus=0)
    def get_node_ip():
        return ray.util.get_node_ip_address()

    return ray.get(
        get_node_ip.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=0,
            ),
        ).remote()
    )


def launch_scheduler_ray_actors(
    server_args: ServerArgs,
    port_args: PortArgs,
) -> RaySchedulerInitResult:
    """Launch scheduler actors using Ray (unified single/multi-node).

    Auto-detects the current placement group via ``ray.util.get_current_placement_group()``.
    The caller must schedule the enclosing actor/task onto a placement group with one
    bundle per node, each bundle having ``gpus_per_node`` GPUs.  Multiple SchedulerActors
    (``num_gpus=1`` each) share a bundle's GPU pool.
    """
    if server_args.dp_size > 1:
        raise NotImplementedError(
            "Ray support for dp_size > 1 is not yet implemented. "
            "Set dp_size=1 or use_ray=False."
        )

    pg = ray.util.get_current_placement_group()
    if pg is None:
        raise RuntimeError(
            "use_ray=True requires a placement group, but none was detected. "
            "Schedule the Engine actor onto a placement group"
        )

    world_size = server_args.tp_size * server_args.pp_size
    nnodes = server_args.nnodes
    gpus_per_node = world_size // nnodes

    logger.info(
        f"Ray cluster: {nnodes} nodes, "
        f"Use {gpus_per_node} GPUs/node, world_size={world_size}"
    )

    rank0_node_ip = _get_rank0_node_ip(pg)
    dist_init_addr = f"{rank0_node_ip}:{server_args.port + ZMQ_TCP_PORT_DELTA}"
    logger.info(f"dist_init_addr: {dist_init_addr}")

    scheduler_actors = []

    for node_idx in range(nnodes):
        pp_range, tp_range, pp_per_node, tp_per_node = _calculate_rank_ranges(
            nnodes, server_args.pp_size, server_args.tp_size, node_rank=node_idx
        )
        for pp_rank in pp_range:
            for tp_rank in tp_range:
                local_gpu_idx = (pp_rank % pp_per_node) * tp_per_node + (
                    tp_rank % tp_per_node
                )

                # Compute parallelism ranks (must match mp launcher logic)
                attn_dp_size = (
                    server_args.dp_size
                    if server_args.enable_dp_attention
                    else 1
                )
                attn_tp_size = (
                    server_args.tp_size // attn_dp_size // server_args.attn_cp_size
                )
                attn_cp_rank = (
                    (tp_rank // attn_tp_size) % server_args.attn_cp_size
                )
                moe_dp_rank = tp_rank // (
                    server_args.tp_size // server_args.moe_dp_size
                )
                moe_ep_rank = (
                    tp_rank
                    % (server_args.tp_size // server_args.moe_dp_size)
                    // (
                        server_args.tp_size
                        // server_args.moe_dp_size
                        // server_args.ep_size
                    )
                )

                actor = SchedulerActor.options(
                    num_cpus=0,
                    num_gpus=1,
                    name=f"sglang_scheduler_rank0node={rank0_node_ip}_pp{pp_rank}_tp{tp_rank}",
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=node_idx,
                    ),
                ).remote(
                    server_args=server_args,
                    port_args=port_args,
                    gpu_id=local_gpu_idx,
                    tp_rank=tp_rank,
                    attn_cp_rank=attn_cp_rank,
                    moe_dp_rank=moe_dp_rank,
                    moe_ep_rank=moe_ep_rank,
                    pp_rank=pp_rank,
                    dp_rank=0,
                    dist_init_addr=dist_init_addr,
                )
                scheduler_actors.append(actor)

    try:
        scheduler_infos = ray.get(
            [actor.get_info.remote() for actor in scheduler_actors]
        )
    except ray.exceptions.RayActorError as e:
        for actor in scheduler_actors:
            try:
                ray.kill(actor)
            except Exception:
                pass
        raise RuntimeError(f"Scheduler actor failed to initialize: {e}")

    event_loop_refs = [actor.run_event_loop.remote() for actor in scheduler_actors]

    return RaySchedulerInitResult(
        scheduler_infos=scheduler_infos,
        _actors=scheduler_actors,
        _event_loop_refs=event_loop_refs,
    )
