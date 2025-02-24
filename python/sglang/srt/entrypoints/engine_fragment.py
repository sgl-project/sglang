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
from typing import Optional

from sglang.srt.distributed import ParallelProcessGroups
from sglang.srt.entrypoints.engine_base import EngineBase
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.orchestration.spmd.orchestrator import SpmdOrchestrator
from sglang.srt.server_args import ServerArgs


class EngineFragment(EngineBase):
    """TODO: Add docstring to describe it."""

    def __init__(
        self,
        gpu_id: int,
        nccl_port: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        parallel_process_groups: Optional[ParallelProcessGroups] = None,
        log_level: str = "error",
        *args,
        **kwargs,
    ):
        if tp_size is None:
            tp_size = parallel_process_groups.tp.device_mesh_device.size()
        server_args = ServerArgs(*args, log_level=log_level, tp_size=tp_size, **kwargs)
        self._entrypoint = SpmdOrchestrator(
            server_args=server_args,
            nccl_port=nccl_port,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            parallel_process_groups=parallel_process_groups,
        )

    def _generate_impl(self, obj: GenerateReqInput):
        return self._entrypoint.generate(obj)

    def shutdown(self):
        self._entrypoint.shutdown()
