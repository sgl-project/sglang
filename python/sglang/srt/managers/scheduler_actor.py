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

if TYPE_CHECKING:
    from sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)


def create_scheduler_actor_class():
    """Factory function to create SchedulerActor class with Ray decorator.

    This factory pattern is used to avoid importing Ray at module load time,
    which allows the module to be imported even when Ray is not installed.
    """
    import ray

    @ray.remote
    class SchedulerActor:
        """
        Ray actor wrapper for SGLang Scheduler.

        Each actor manages one GPU and runs the full Scheduler + TpModelWorker stack.
        The actor preserves ZMQ for high-throughput request/response communication.
        Ray is used only for process lifecycle and resource management.
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
        ):
            import faulthandler
            import os

            import setproctitle

            from sglang.srt.disaggregation.utils import DisaggregationMode
            from sglang.srt.environ import envs
            from sglang.srt.managers.scheduler import Scheduler
            from sglang.srt.utils import configure_logger, suppress_other_loggers
            from sglang.srt.utils.common import get_bool_env_var, numa_bind_to_node
            from sglang.srt.utils.gpu_utils import set_gpu_proc_affinity

            # Generate logger prefix (same logic as run_scheduler_process)
            if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
                dp_rank = int(os.environ["SGLANG_DP_RANK"])

            prefix = ""
            if dp_rank is not None:
                prefix += f" DP{dp_rank}"
            if server_args.pp_size > 1:
                prefix += f" PP{pp_rank}"
            if server_args.tp_size > 1:
                prefix += f" TP{tp_rank}"
            if server_args.ep_size > 1:
                prefix += f" EP{moe_ep_rank}"

            # Configure process
            setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
            faulthandler.enable()

            # Configure logger
            configure_logger(server_args, prefix=prefix)
            suppress_other_loggers()

            # Set cpu affinity to this gpu process
            if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
                set_gpu_proc_affinity(
                    server_args.pp_size, server_args.tp_size, server_args.nnodes, gpu_id
                )
            if (
                numa_node := server_args.numa_node
            ) is not None and not envs.SGLANG_NUMA_BIND_V2.get():
                numa_bind_to_node(numa_node[gpu_id])

            # Create scheduler (this loads the model into GPU)
            self.scheduler = Scheduler(
                server_args,
                port_args,
                gpu_id,
                tp_rank,
                moe_ep_rank,
                pp_rank,
                dp_rank,
            )

            self._tp_rank = tp_rank
            self._pp_rank = pp_rank
            self._gpu_id = gpu_id
            self._dp_rank = dp_rank
            self._server_args = server_args
            self._disaggregation_mode = DisaggregationMode(
                server_args.disaggregation_mode
            )

        def get_info(self) -> Dict[str, Any]:
            """Return scheduler initialization info for handshake."""
            result_dict = {
                "status": "ready",
                "max_total_num_tokens": self.scheduler.max_total_num_tokens,
                "max_req_input_len": self.scheduler.max_req_input_len,
            }

            if self._server_args.remote_instance_weight_loader_use_transfer_engine():
                (
                    remote_instance_transfer_engine_session_id,
                    remote_instance_transfer_engine_weights_info_dict,
                ) = self.scheduler.get_remote_instance_transfer_engine_info()
                result_dict.update(
                    {
                        "tp_rank": self._tp_rank,
                        "remote_instance_transfer_engine_session_id": remote_instance_transfer_engine_session_id,
                        "remote_instance_transfer_engine_weights_info_dict": remote_instance_transfer_engine_weights_info_dict,
                    }
                )

            return result_dict

        def run_event_loop(self) -> None:
            """
            Run the scheduler's ZMQ event loop.

            This method blocks until shutdown. The appropriate event loop
            variant is chosen based on server_args configuration.
            """
            try:
                server_args = self._server_args
                disaggregation_mode = self._disaggregation_mode

                if disaggregation_mode == DisaggregationMode.NULL:
                    if self.scheduler.enable_pdmux:
                        self.scheduler.event_loop_pdmux()
                    elif server_args.pp_size > 1:
                        self.scheduler.event_loop_pp()
                    elif self.scheduler.enable_overlap:
                        self.scheduler.event_loop_overlap()
                    else:
                        self.scheduler.event_loop_normal()
                elif disaggregation_mode == DisaggregationMode.PREFILL:
                    if server_args.pp_size > 1:
                        self.scheduler.event_loop_pp_disagg_prefill()
                    elif self.scheduler.enable_overlap:
                        self.scheduler.event_loop_overlap_disagg_prefill()
                    else:
                        self.scheduler.event_loop_normal_disagg_prefill()
                elif disaggregation_mode == DisaggregationMode.DECODE:
                    if server_args.pp_size > 1:
                        self.scheduler.event_loop_pp_disagg_decode()
                    elif self.scheduler.enable_overlap:
                        self.scheduler.event_loop_overlap_disagg_decode()
                    else:
                        self.scheduler.event_loop_normal_disagg_decode()
                else:
                    raise ValueError(
                        f"Unknown disaggregation mode: {disaggregation_mode}"
                    )
            except Exception as e:
                logger.error(
                    f"Scheduler PP{self._pp_rank} TP{self._tp_rank} crashed: {e}"
                )
                raise

    return SchedulerActor
