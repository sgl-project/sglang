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
from typing import TYPE_CHECKING, Any, Dict, Optional

import ray

if TYPE_CHECKING:
    from sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)


@ray.remote
class SchedulerActor:
    """Ray actor wrapper for SGLang Scheduler.

    Each actor manages one GPU and runs the Scheduler + TpModelWorker stack.
    Ray is used for process lifecycle; ZMQ handles request/response communication.
    """

    def __init__(
        self,
        server_args: "ServerArgs",
        port_args: "PortArgs",
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        dist_init_addr: Optional[str] = None,
    ):
        import dataclasses

        from sglang.srt.managers.scheduler import (
            Scheduler,
            configure_scheduler,
        )

        # Override dist_init_addr if provided (for multi-node)
        if dist_init_addr:
            server_args = dataclasses.replace(
                server_args, dist_init_addr=dist_init_addr
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

        # Store node info for debugging
        self._node_id = ray.get_runtime_context().get_node_id()

        # Configure worker (logging, process title, etc.)
        dp_rank = configure_scheduler(
            server_args, tp_rank, moe_ep_rank, pp_rank, dp_rank
        )

        # Create scheduler (loads model into GPU, initializes NCCL)
        self.scheduler = Scheduler(
            server_args,
            port_args,
            actual_gpu_id,  # Use discovered GPU ID
            tp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
        )

        self._tp_rank = tp_rank
        self._pp_rank = pp_rank

    def get_info(self) -> Dict[str, Any]:
        """Return scheduler initialization info for handshake."""
        return {
            "status": "ready",
            "max_total_num_tokens": self.scheduler.max_total_num_tokens,
            "max_req_input_len": self.scheduler.max_req_input_len,
        }

    def get_node_info(self) -> Dict[str, Any]:
        """Return node-specific information for coordination."""
        return {
            "node_id": ray.get_runtime_context().get_node_id(),
            "node_ip": ray.util.get_node_ip_address(),
            "tp_rank": self._tp_rank,
            "pp_rank": self._pp_rank,
            "gpu_ids": ray.get_runtime_context().get_accelerator_ids().get(
                "GPU", []
            ),
        }

    def run_event_loop(self) -> None:
        """Run the scheduler's event loop. Blocks until shutdown."""
        try:
            self.scheduler.run_event_loop()
        except Exception as e:
            logger.error(
                f"Scheduler PP{self._pp_rank} TP{self._tp_rank} crashed: {e}"
            )
            raise
