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
from typing import Callable

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from sglang.srt.entrypoints.engine import (
    Engine,
    SchedulerInitResult,
    _calculate_rank_ranges,
    _compute_parallelism_ranks,
)
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


class RayEngine(Engine):
    """Engine using Ray actors for scheduler processes."""

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
        run_scheduler_process_func: Callable,
    ) -> tuple[SchedulerInitResult, None]:
        """Launch schedulers as Ray actors.

        Returns:
            Tuple of (RaySchedulerInitResult, None).
            scheduler_procs is None since Ray uses actors instead of mp.Process.
        """
        pg = ray.util.get_current_placement_group()
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

        nnodes = server_args.nnodes

        # co-located with the Engine and rank0 scheduler at the same node
        engine_bundle, engine_ip = _find_engine_bundle(pg, nnodes)
        bundle_for_node = [engine_bundle] + [
            i for i in range(nnodes) if i != engine_bundle
        ]
        rank0_node_ip = engine_ip

        if server_args.dp_size == 1:
            # Launch tensor parallel scheduler actors
            world_size = server_args.tp_size * server_args.pp_size
            gpus_per_node = world_size // nnodes

            logger.info(
                f"Ray cluster: {nnodes} nodes, "
                f"Use {gpus_per_node} GPUs/node, world_size={world_size}"
            )

            dist_init_addr = f"{rank0_node_ip}:{port_args.nccl_port}"
            logger.info(f"dist_init_addr: {dist_init_addr}")

            scheduler_actors = []

            for node_idx in range(nnodes):
                bundle_idx = bundle_for_node[node_idx]
                pp_range, tp_range, pp_per_node, tp_per_node = _calculate_rank_ranges(
                    nnodes,
                    server_args.pp_size,
                    server_args.tp_size,
                    node_rank=node_idx,
                )
                for pp_rank in pp_range:
                    for tp_rank in tp_range:
                        local_gpu_idx = (pp_rank % pp_per_node) * tp_per_node + (
                            tp_rank % tp_per_node
                        )

                        attn_cp_rank, moe_dp_rank, moe_ep_rank = (
                            _compute_parallelism_ranks(server_args, tp_rank)
                        )

                        actor = SchedulerActor.options(
                            num_cpus=0,
                            num_gpus=1,
                            name=f"sglang_scheduler_node{rank0_node_ip}_pp{pp_rank}_tp{tp_rank}_pg{pg.id.hex()[:8]}_bundle{bundle_idx}",
                            scheduling_strategy=PlacementGroupSchedulingStrategy(
                                placement_group=pg,
                                placement_group_bundle_index=bundle_idx,
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
                    server_args, port_args, pg, bundle_for_node, rank0_node_ip
                ),
                None,
            )

    @classmethod
    def _launch_dp_scheduler_processes(
        cls,
        server_args: ServerArgs,
        port_args: PortArgs,
        pg,
        bundle_for_node: list,
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
