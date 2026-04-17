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
"""Ray-aware DataParallelController that launches SchedulerActors instead of mp.Process."""

from __future__ import annotations

import logging
from typing import List, Optional

import ray
import zmq
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from sglang.srt.entrypoints.engine import (
    _calculate_rank_ranges,
    _compute_parallelism_ranks,
)
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.data_parallel_controller import DataParallelController
from sglang.srt.ray.scheduler_actor import SchedulerActor
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils.network import bind_port, get_zmq_socket, get_zmq_socket_on_host

logger = logging.getLogger(__name__)


class RayDataParallelController(DataParallelController):
    """DataParallelController that uses Ray actors for scheduler processes.

    Overrides the process-spawning methods to create SchedulerActor Ray actors
    instead of mp.Process. Runs in-process (not as a separate mp.Process) and
    reuses the parent's event_loop, dispatching, and ZMQ routing.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        placement_group,
        bundle_for_node: List[int],
        rank0_node_ip: str,
    ):
        # Set Ray-specific attributes BEFORE super().__init__() because the
        # parent constructor calls launch_dp_schedulers / launch_dp_attention_schedulers
        # which we override, and those methods need these attributes.
        self.pg = placement_group
        self.bundle_for_node = bundle_for_node
        self.rank0_node_ip = rank0_node_ip
        self.scheduler_actors: List = []
        self.event_loop_refs: List = []

        # super().__init__ will call our overridden launch methods via MRO.
        # Pass run_scheduler_process_func=None since we don't spawn mp.Process.
        super().__init__(server_args, port_args, run_scheduler_process_func=None)

    def launch_dp_schedulers(self, server_args: ServerArgs, port_args: PortArgs):
        """Override: launch Ray scheduler actors per DP rank."""
        sockets = []
        dp_port_args_list = []

        for dp_rank in range(server_args.dp_size):
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name

            # Hold NCCL port so the next DP rank gets a different one
            sockets.append(bind_port(tmp_port_args.nccl_port))
            dp_port_args_list.append(tmp_port_args)

            # Create ZMQ PUSH socket for this DP rank (controller → scheduler)
            if server_args.node_rank == 0:
                self.workers[dp_rank] = get_zmq_socket(
                    self.context,
                    zmq.PUSH,
                    tmp_port_args.scheduler_input_ipc_name,
                    True,
                )

        # Release held ports before creating actors
        for sock in sockets:
            sock.close()

        # Create actors for each DP rank sequentially
        for dp_rank in range(server_args.dp_size):
            self._launch_ray_tp_group(server_args, dp_port_args_list[dp_rank], dp_rank)

    def launch_dp_attention_schedulers(
        self, server_args: ServerArgs, port_args: PortArgs
    ):
        """Override: pre-allocate ports, skip broadcast, create Ray actors."""
        # Pre-allocate worker ports on the controller node, binding to the
        # rank-0 node IP instead of tcp://* to avoid exposing unauthenticated
        # ZMQ sockets (CVE-2026-3060).
        worker_ports = []
        for dp_rank in range(server_args.dp_size):
            worker_port, worker_socket = get_zmq_socket_on_host(
                self.context, zmq.PUSH, host=self.rank0_node_ip
            )
            worker_ports.append(worker_port)
            self.workers[dp_rank] = worker_socket
            logger.debug(f"Assigned port {worker_port} to worker {dp_rank}")

        # Skip _broadcast_worker_ports — Ray creates all actors centrally,
        # so there's no need for the inter-node handshake protocol.
        self._launch_ray_tp_group(
            server_args, port_args, dp_rank=None, worker_ports=worker_ports
        )

    def _launch_ray_tp_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        dp_rank: Optional[int],
        worker_ports: Optional[List[int]] = None,
    ):
        """Create SchedulerActor Ray actors for one TP group (one DP rank).

        For DP attention, dp_rank=None and worker_ports is provided; the dp_rank
        is derived from tp_rank via compute_dp_attention_world_info.

        For regular DP, dp_rank is an integer and worker_ports is None.
        """
        nnodes = server_args.nnodes
        batch_start_idx = len(self.scheduler_actors)

        for node_idx in range(nnodes):
            bundle_idx = self.bundle_for_node[node_idx]
            pp_range, tp_range, pp_per_node, tp_per_node = _calculate_rank_ranges(
                nnodes, server_args.pp_size, server_args.tp_size, node_rank=node_idx
            )

            for pp_rank in pp_range:
                for tp_rank in tp_range:
                    rank_port_args = port_args
                    actual_dp_rank = dp_rank

                    if server_args.enable_dp_attention:
                        # DP attention: derive dp_rank from tp_rank
                        _, _, actual_dp_rank = compute_dp_attention_world_info(
                            server_args.enable_dp_attention,
                            tp_rank,
                            server_args.tp_size,
                            server_args.dp_size,
                            server_args.attn_cp_size,
                        )
                        rank_port_args = PortArgs.init_new(
                            server_args, actual_dp_rank, worker_ports
                        )
                        # All DP ranks share the same NCCL port (reuse TP group)
                        rank_port_args.nccl_port = port_args.nccl_port
                        # The detokenizer and tokenizer bind using the
                        # original port_args addresses (127.0.0.1 when
                        # dist_init_addr is unset).  Scheduler actors must
                        # connect to the same addresses.
                        rank_port_args.detokenizer_ipc_name = (
                            port_args.detokenizer_ipc_name
                        )
                        rank_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name

                    local_gpu_idx = (pp_rank % pp_per_node) * tp_per_node + (
                        tp_rank % tp_per_node
                    )

                    attn_cp_rank, moe_dp_rank, moe_ep_rank = _compute_parallelism_ranks(
                        server_args, tp_rank
                    )

                    # Each DP group needs a unique dist_init_addr for its own
                    # torch.distributed process group. Use nccl_port which is
                    # unique per DP group (regular DP) or shared (DP attention).
                    dist_init_addr = f"{self.rank0_node_ip}:{rank_port_args.nccl_port}"

                    actor = SchedulerActor.options(
                        num_cpus=0,
                        num_gpus=1,
                        name=(
                            f"sglang_scheduler_node{self.rank0_node_ip}"
                            f"_dp{actual_dp_rank}_pp{pp_rank}_tp{tp_rank}"
                            f"_pg{self.pg.id.hex()[:8]}_bundle{bundle_idx}"
                        ),
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=self.pg,
                            placement_group_bundle_index=bundle_idx,
                        ),
                    ).remote(
                        server_args=server_args,
                        port_args=rank_port_args,
                        gpu_id=local_gpu_idx,
                        tp_rank=tp_rank,
                        attn_cp_rank=attn_cp_rank,
                        moe_dp_rank=moe_dp_rank,
                        moe_ep_rank=moe_ep_rank,
                        pp_rank=pp_rank,
                        dp_rank=actual_dp_rank,
                        dist_init_addr=dist_init_addr,
                    )
                    self.scheduler_actors.append(actor)

        # Wait for all actors created in this call to initialize
        batch_actors = self.scheduler_actors[batch_start_idx:]
        try:
            scheduler_infos = ray.get(
                [actor.get_info.remote() for actor in batch_actors]
            )
        except ray.exceptions.RayActorError as e:
            for actor in self.scheduler_actors:
                try:
                    ray.kill(actor)
                except Exception:
                    logger.error(f"Failed to kill Ray scheduler actor: {actor}")
            raise RuntimeError(f"Scheduler actor failed to initialize: {e}")

        # Store init info from the first actor (same across all actors)
        if scheduler_infos:
            self.max_total_num_tokens = scheduler_infos[0]["max_total_num_tokens"]
            self.max_req_input_len = scheduler_infos[0]["max_req_input_len"]

        # Start event loops (non-blocking — runs until actor is killed)
        self.event_loop_refs.extend(
            [actor.run_event_loop.remote() for actor in batch_actors]
        )

    # Override launch_tensor_parallel_group to be a no-op since we don't use it.
    # The parent's launch_dp_schedulers/launch_dp_attention_schedulers call this,
    # but our overrides call _launch_ray_tp_group instead.
    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: Optional[int],
        worker_ports: Optional[List[int]] = None,
    ):
        raise RuntimeError(
            "RayDataParallelController should not call launch_tensor_parallel_group. "
            "Use _launch_ray_tp_group instead."
        )
