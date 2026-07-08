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
"""RayEngine - Engine subclass that launches schedulers as Ray actors."""

from __future__ import annotations

import dataclasses
import logging
import threading
from typing import List, Optional

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from sglang.srt.entrypoints.engine import (
    Engine,
    SchedulerInitResult,
    SubprocessTarget,
    _calculate_rank_ranges,
    _compute_parallelism_ranks,
)
from sglang.srt.environ import envs
from sglang.srt.ray.scheduler_actor import SchedulerActor
from sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RaySchedulerInitResult(SchedulerInitResult):
    """SchedulerInitResult that also holds Ray actor handles for cleanup."""

    scheduler_actors: list = dataclasses.field(default_factory=list)


def _find_engine_bundle(
    placement_group: PlacementGroup, nnodes: int
) -> tuple[int, str]:
    """Find which placement group bundle is on the same node as the Engine.
    Rank0 scheduler must be co-located with the Engine. Returns (bundle_index, engine_ip).
    """
    engine_ip = ray.util.get_node_ip_address()

    @ray.remote(num_cpus=0, num_gpus=0)
    def get_node_ip():
        return ray.util.get_node_ip_address()

    bundle_ips = ray.get(
        [
            get_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_bundle_index=i,
                ),
            ).remote()
            for i in range(nnodes)
        ]
    )

    try:
        return bundle_ips.index(engine_ip), engine_ip
    except ValueError:
        raise RuntimeError(
            f"Engine node {engine_ip} not found in any placement group bundle {bundle_ips}. "
            f"Rank-0 scheduler must be co-located with the Engine."
        )


def _get_bundle_node_ip(placement_group: PlacementGroup, bundle_idx: int) -> str:
    """Get the IP address of the node where a specific bundle is located.

    Args:
        placement_group: The placement group
        bundle_idx: Bundle index to query

    Returns:
        IP address of the node where the bundle is located.
    """

    @ray.remote(num_cpus=0, num_gpus=0)
    def get_node_ip():
        return ray.util.get_node_ip_address()

    return ray.get(
        get_node_ip.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=bundle_idx,
            ),
        ).remote()
    )


def _compute_world_size(server_args: ServerArgs) -> int:
    """Compute world_size (total number of scheduler actors/GPUs needed).

    Normal: dp_size * tp_size * pp_size; DP attention: tp_size * pp_size.
    """
    if server_args.enable_dp_attention:
        return server_args.tp_size * server_args.pp_size
    return server_args.dp_size * server_args.tp_size * server_args.pp_size


def _resolve_bundle_indices(pg: PlacementGroup, world_size: int) -> List[int]:
    """Resolve bundle indices for Custom PG mode.

    Parses SGLANG_RAY_BUNDLE_INDICES env var if set; otherwise returns
    sequential indices [0, 1, ..., world_size-1].

    Args:
        pg: Placement group (used to get total_bundles count).
        world_size: Number of bundle indices expected (pre-computed via _compute_world_size).

    Returns:
        List of bundle indices of length world_size.
    """
    total_bundles = len(pg.bundle_specs)
    indices_str = envs.SGLANG_RAY_BUNDLE_INDICES.get()
    if not indices_str:
        return list(range(world_size))

    indices = list(map(int, indices_str.split(",")))

    if len(indices) != world_size:
        raise ValueError(
            f"SGLANG_RAY_BUNDLE_INDICES has {len(indices)} values, "
            f"expected {world_size}"
        )

    if len(set(indices)) != len(indices):
        raise ValueError(f"SGLANG_RAY_BUNDLE_INDICES has duplicates: {indices}")

    for idx in indices:
        if idx < 0 or idx >= total_bundles:
            raise ValueError(f"Bundle index {idx} out of range [0, {total_bundles})")

    return indices


def _validate_custom_placement_group(pg: PlacementGroup, world_size: int) -> None:
    """Validate custom placement group: 1 GPU per bundle, enough GPU bundles for world_size.

    Args:
        pg: User-provided placement group.
        world_size: Number of GPU bundles required.
    """
    bundles = pg.bundle_specs
    gpu_bundle_count = 0
    for bundle in bundles:
        gpu_count = bundle.get("GPU", 0)
        if gpu_count > 1:
            raise ValueError(
                "Custom placement group must have exactly 1 GPU per bundle. "
                f"Found bundle with {gpu_count} GPUs."
            )
        if gpu_count > 0:
            gpu_bundle_count += 1

    if gpu_bundle_count < world_size:
        raise ValueError(
            f"Custom placement group has {gpu_bundle_count} GPU bundles, "
            f"but needs {world_size} for world_size. "
            "Provide more bundles or reduce parallelism."
        )


def _create_scheduler_actor(
    pg: PlacementGroup,
    bundle_idx: int,
    gpu_id: int,
    server_args: ServerArgs,
    port_args: PortArgs,
    tp_rank: int,
    pp_rank: int,
    dp_rank: int,
    dist_init_addr: str,
    rank0_node_ip: str,
) -> SchedulerActor:
    """Create a SchedulerActor on the given placement group bundle.

    Args:
        pg: Placement group to schedule actor onto.
        bundle_idx: Bundle index within the placement group.
        gpu_id: GPU ID within the bundle (0 for custom PG, computed for auto PG).
        rank0_node_ip: IP of rank-0's node, used for NCCL rendezvous.
        dist_init_addr: Distributed init address (tcp://rank0_node_ip:nccl_port).
    """
    attn_cp_rank, moe_dp_rank, moe_ep_rank = _compute_parallelism_ranks(
        server_args, tp_rank
    )

    return SchedulerActor.options(
        num_cpus=0,
        num_gpus=1,
        name=(
            f"sglang_scheduler_node{rank0_node_ip}"
            f"_dp{dp_rank}_pp{pp_rank}_tp{tp_rank}"
            f"_pg{pg.id.hex()[:8]}_bundle{bundle_idx}"
        ),
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=bundle_idx,
        ),
    ).remote(
        server_args=server_args,
        port_args=port_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        attn_cp_rank=attn_cp_rank,
        moe_dp_rank=moe_dp_rank,
        moe_ep_rank=moe_ep_rank,
        pp_rank=pp_rank,
        dp_rank=dp_rank,
        dist_init_addr=dist_init_addr,
    )


class RayEngine(Engine):
    """Engine using Ray actors for scheduler processes."""

    def __init__(self, **kwargs):
        placement_group = kwargs.pop("placement_group", None)
        if "log_level" not in kwargs:
            kwargs["log_level"] = "error"
        server_args = ServerArgs(**kwargs)
        server_args.override("ray.placement_group", placement_group=placement_group)
        super().__init__(server_args=server_args)

    def shutdown(self):
        """Shutdown the engine — kill Ray scheduler actors then local processes."""
        for actor in self._scheduler_init_result.scheduler_actors:
            try:
                ray.kill(actor)
            except Exception:
                logger.error(f"Failed to kill Ray scheduler actor: {actor}")
        super().shutdown()

    @classmethod
    def _launch_scheduler_processes(
        cls,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: SubprocessTarget,
    ) -> tuple[SchedulerInitResult, None]:
        """Launch schedulers as Ray actors.

        Returns:
            Tuple of (RaySchedulerInitResult, None).
            scheduler_procs is None since Ray uses actors instead of mp.Process.
        """
        pg = server_args.placement_group or ray.util.get_current_placement_group()
        if pg is None:
            from ray.util.placement_group import (
                placement_group as create_placement_group,
            )

            if server_args.enable_dp_attention:
                total_gpus = server_args.tp_size * server_args.pp_size
            else:
                total_gpus = (
                    server_args.dp_size * server_args.tp_size * server_args.pp_size
                )

            nnodes = server_args.nnodes
            gpus_per_node = total_gpus // nnodes
            strategy = "STRICT_PACK" if nnodes == 1 else "SPREAD"

            logger.info(
                "No placement group detected. Auto-creating one with "
                f"{nnodes} bundle(s), {gpus_per_node} GPU(s)/bundle, "
                "placement group explicitly and schedule the Engine onto it."
            )

            pg = create_placement_group(
                [{"CPU": 1, "GPU": gpus_per_node}] * nnodes,
                strategy=strategy,
            )
            ray.get(pg.ready())

        is_custom_pg = server_args.placement_group is not None
        nnodes = server_args.nnodes
        world_size = _compute_world_size(server_args)

        if not is_custom_pg:
            engine_bundle, engine_ip = _find_engine_bundle(pg, nnodes)
            bundle_for_node = [engine_bundle] + [
                i for i in range(nnodes) if i != engine_bundle
            ]
            rank0_node_ip = engine_ip
        else:
            try:
                _validate_custom_placement_group(pg, world_size)
            except ValueError as e:
                logger.error(f"Custom placement group validation failed: {e}")
                raise RuntimeError(
                    f"Custom placement group validation failed: {e}"
                ) from e
            bundle_for_node = None
            indices_str = envs.SGLANG_RAY_BUNDLE_INDICES.get()
            rank0_bundle_idx = int(indices_str.split(",")[0]) if indices_str else 0
            rank0_node_ip = _get_bundle_node_ip(pg, rank0_bundle_idx)

        if server_args.dp_size == 1:
            dist_init_addr = f"{rank0_node_ip}:{port_args.nccl_port}"
            logger.info(f"dist_init_addr: {dist_init_addr}")

            scheduler_actors = []

            if not is_custom_pg:
                gpus_per_node = world_size // nnodes
                logger.info(
                    f"Ray cluster (auto PG): {nnodes} nodes, "
                    f"{gpus_per_node} GPUs/node, world_size={world_size}"
                )

                for node_idx in range(nnodes):
                    bundle_idx = bundle_for_node[node_idx]
                    pp_range, tp_range, pp_per_node, tp_per_node = (
                        _calculate_rank_ranges(
                            nnodes,
                            server_args.pp_size,
                            server_args.tp_size,
                            node_rank=node_idx,
                        )
                    )
                    for pp_rank in pp_range:
                        for tp_rank in tp_range:
                            local_gpu_idx = (pp_rank % pp_per_node) * tp_per_node + (
                                tp_rank % tp_per_node
                            )

                            actor = _create_scheduler_actor(
                                pg=pg,
                                bundle_idx=bundle_idx,
                                gpu_id=local_gpu_idx,
                                server_args=server_args,
                                port_args=port_args,
                                tp_rank=tp_rank,
                                pp_rank=pp_rank,
                                dp_rank=0,
                                dist_init_addr=dist_init_addr,
                                rank0_node_ip=rank0_node_ip,
                            )
                            scheduler_actors.append(actor)

            else:
                try:
                    bundle_indices = _resolve_bundle_indices(pg, world_size)
                except ValueError as e:
                    logger.error(f"Failed to resolve bundle indices: {e}")
                    raise RuntimeError(f"Failed to resolve bundle indices: {e}") from e

                logger.info(
                    f"Ray cluster (custom PG): world_size={world_size}, "
                    f"bundle_indices={bundle_indices}"
                )

                for rank in range(world_size):
                    pp_rank = rank // server_args.tp_size
                    tp_rank = rank % server_args.tp_size
                    bundle_idx = bundle_indices[rank]

                    actor = _create_scheduler_actor(
                        pg=pg,
                        bundle_idx=bundle_idx,
                        gpu_id=0,  # Each bundle has exactly 1 GPU
                        server_args=server_args,
                        port_args=port_args,
                        tp_rank=tp_rank,
                        pp_rank=pp_rank,
                        dp_rank=0,
                        dist_init_addr=dist_init_addr,
                        rank0_node_ip=rank0_node_ip,
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
                        logger.error(f"Failed to kill Ray scheduler actor: {actor}")
                raise RuntimeError(f"Scheduler actor failed to initialize: {e}")

            event_loop_refs = [
                actor.run_event_loop.remote() for actor in scheduler_actors
            ]

            def wait_for_completion():
                try:
                    ray.get(event_loop_refs)
                except Exception as e:
                    logger.error(f"Ray scheduler actor terminated with error: {e}")

            return (
                RaySchedulerInitResult(
                    scheduler_infos=scheduler_infos,
                    wait_for_completion=wait_for_completion,
                    scheduler_actors=scheduler_actors,
                ),
                None,
            )
        else:
            # Launch the data parallel controller
            return (
                cls._launch_dp_scheduler_processes(
                    server_args,
                    port_args,
                    pg,
                    bundle_for_node,
                    rank0_node_ip,
                ),
                None,
            )

    @classmethod
    def _launch_dp_scheduler_processes(
        cls,
        server_args: ServerArgs,
        port_args: PortArgs,
        pg,
        bundle_for_node: Optional[List[int]],
        rank0_node_ip: str,
    ) -> RaySchedulerInitResult:
        """Launch DP schedulers via RayDataParallelController."""
        from sglang.srt.ray.data_parallel_controller import (
            RayDataParallelController,
        )

        if server_args.enable_dp_attention:
            # DP attention folds DP into TP — total GPUs = tp_size * pp_size
            total_gpus = server_args.tp_size * server_args.pp_size
        else:
            total_gpus = server_args.dp_size * server_args.tp_size * server_args.pp_size
        gpus_per_node = total_gpus // server_args.nnodes
        logger.info(
            f"Ray DP cluster: {server_args.nnodes} nodes, "
            f"{gpus_per_node} GPUs/node, dp_size={server_args.dp_size}, "
            f"tp_size={server_args.tp_size}, pp_size={server_args.pp_size}, "
            f"enable_dp_attention={server_args.enable_dp_attention}"
        )

        # Set dist_init_addr on server_args so PortArgs.init_new() can compute
        # TCP addresses correctly (required for DP attention path).
        dp_server_args = dataclasses.replace(
            server_args,
            dist_init_addr=f"{rank0_node_ip}:{port_args.nccl_port}",
        )
        # dataclasses.replace only copies declared fields; placement_group is
        # a dynamic attribute that must be manually appended after the rebuild.
        dp_server_args.override(
            "ray.placement_group", placement_group=server_args.placement_group
        )

        # Create the DP controller in-process. This blocks until all actors
        # are initialized and their event loops have started.
        controller = RayDataParallelController(
            dp_server_args, port_args, pg, bundle_for_node, rank0_node_ip
        )

        # Start the DP controller's event loop in a daemon thread.
        # It routes requests from the tokenizer to per-DP-rank schedulers.
        dp_thread = threading.Thread(
            target=controller.event_loop, daemon=True, name="dp_controller"
        )
        dp_thread.start()

        scheduler_infos = [
            {
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
            }
        ]

        event_loop_refs = controller.event_loop_refs

        def wait_for_completion():
            try:
                ray.get(event_loop_refs)
            except Exception as e:
                logger.error(f"Ray scheduler actor terminated with error: {e}")

        return RaySchedulerInitResult(
            scheduler_infos=scheduler_infos,
            wait_for_completion=wait_for_completion,
            scheduler_actors=controller.scheduler_actors,
        )
