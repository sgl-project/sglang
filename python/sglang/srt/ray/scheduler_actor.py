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
"""Ray actor wrapper for SGLang Scheduler."""

from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Any, Dict, Optional

import ray

from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.runtime_context import SpawnRanks, get_device, publish, spawn_world_rank
from sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)


class CpuShareLaunch:
    """One launch's ``CpuShareBoard`` (SGLANG_SET_CPU_AFFINITY on the driver): counts
    the scheduler actors the launch places on each node and tells the board once they
    all exist. Each actor gets the board as an argument."""

    def __init__(self, pg):
        from sglang.srt.environ import envs
        from sglang.srt.utils.process import CpuShareBoard

        self.board = None
        self.node_of = {}
        if envs.SGLANG_SET_CPU_AFFINITY.get():
            self.board = ray.remote(CpuShareBoard).options(num_cpus=0).remote()
            self.node_of = ray.util.placement_group_table(pg)["bundles_to_node_id"]
        self.counts = Counter()

    def add(self, bundle_idx: int):
        if self.board is not None:
            self.counts[self.node_of[bundle_idx]] += 1

    def close(self):
        if self.board is not None:
            ray.get(self.board.expect.remote(dict(self.counts)))


def _cpu_share(board, server_args: ServerArgs, gpu_id: int) -> Optional[list[int]]:
    """This actor's CPUs, split by ``board`` with the other scheduler actors its launch
    placed on this node, or None when they cannot be gathered."""
    from sglang.srt.utils import numa_utils
    from sglang.srt.utils.process import physical_cores

    node_id = ray.get_runtime_context().get_node_id()
    # the CPUs this actor keeps once NUMA binding places it, as the launcher plans them
    cpus = numa_utils.numa_bound_cpus(server_args, gpu_id, set(os.sched_getaffinity(0)))
    return ray.get(board.register.remote(node_id, gpu_id, physical_cores(cpus)))


@ray.remote
class SchedulerActor:
    """Ray actor wrapper for SGLang Scheduler.

    Each actor manages one GPU and runs the Scheduler + TpModelWorker stack, built by
    ``start`` so a launcher can place every actor before any loads its weights.
    Ray is used for process lifecycle; ZMQ handles request/response communication.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        dist_init_addr: Optional[str] = None,
        cpu_share_board=None,
    ):
        from sglang.srt.environ import envs
        from sglang.srt.managers.scheduler import (
            configure_scheduler_process,
            resolve_spawn_dp_rank,
        )
        from sglang.srt.utils.numa_utils import (
            get_numa_node_if_available,
            numa_bind_to_node,
        )

        # Declared, not copied: Ray deserializes the argument per call, so this
        # record is the actor's own and nothing else in the process holds it.
        # The field stays the operator's input; `PortArgs.init_new` and the bags
        # this actor publishes read the decision.
        if dist_init_addr:
            declare_resolution(
                server_args, "ray.scheduler_actor", dist_init_addr=dist_init_addr
            )

        # Get actual GPU IDs from Ray runtime context
        accelerator_ids = ray.get_runtime_context().get_accelerator_ids()
        assigned_gpus = accelerator_ids.get("GPU", [])

        if assigned_gpus:
            # Ray assigned specific GPU(s), use the first one
            actual_gpu_id = int(assigned_gpus[0])
            logger.info(f"[TP{tp_rank}] Ray assigned GPU: {actual_gpu_id}")
        else:
            # Fallback to passed gpu_id
            actual_gpu_id = gpu_id
            logger.info(f"[TP{tp_rank}] Using passed gpu_id: {gpu_id}")

        dp_rank = resolve_spawn_dp_rank(dp_rank)

        # This actor takes the place of run_scheduler_process, which is where
        # a forked scheduler publishes.
        publish(
            server_args,
            role="scheduler",
            ranks=SpawnRanks(
                world_rank=spawn_world_rank(
                    server_args, tp_rank=tp_rank, pp_rank=pp_rank
                ),
                dp_rank=dp_rank,
                gpu_id=actual_gpu_id,
            ),
        )

        if cpu_share_board is not None:
            cpus = _cpu_share(cpu_share_board, server_args, actual_gpu_id)
            if cpus:
                envs.SGLANG_GPU_PROC_CPUS.set(",".join(map(str, cpus)))

        # Configure worker (logging, process title, etc.)
        configure_scheduler_process(server_args, actual_gpu_id)

        # Ray actors can't use the numactl subprocess-wrapping approach
        # (SGLANG_NUMA_BIND_V2's normal path), so bind in-process via libnuma.
        # The V1 path inside configure_scheduler_process already handles
        # SGLANG_NUMA_BIND_V2=False.
        if envs.SGLANG_NUMA_BIND_V2.get():
            numa_node = get_numa_node_if_available(server_args, actual_gpu_id)
            if numa_node is not None:
                numa_bind_to_node(numa_node)
                logger.info(
                    f"[TP{tp_rank}] Bound to NUMA node {numa_node} for GPU {actual_gpu_id}"
                )

        self._server_args = server_args
        self._port_args = port_args
        self._tp_rank = tp_rank
        self._pp_rank = pp_rank

    def start(self) -> None:
        """Create the scheduler (loads the model into GPU, initializes NCCL)."""
        from sglang.srt.managers.scheduler import Scheduler

        self.scheduler = Scheduler(
            server_args=self._server_args,
            port_args=self._port_args,
        )

    def get_info(self) -> Dict[str, Any]:
        """Return scheduler initialization info for handshake."""
        return self.scheduler.get_init_info()

    def run_event_loop(self) -> None:
        """Run the scheduler's event loop. Blocks until shutdown."""
        try:
            import torch

            # Need to set the GPU id for the event loop for nccl to work
            torch.cuda.set_device(get_device().gpu_id)
            self.scheduler.run_event_loop()
        except Exception as e:
            logger.error(f"Scheduler PP{self._pp_rank} TP{self._tp_rank} crashed: {e}")
            raise
