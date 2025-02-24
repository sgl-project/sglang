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
from sglang.srt.entrypoints.engine_base import EngineBase
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.orchestration.spmd.orchestrator import SpmdOrchestrator
from sglang.srt.server_args import ServerArgs


class EngineFragment(EngineBase):
    """TODO: Add docstring to describe it."""

    def __init__(
        self,
        nccl_port: int,
        gpu_id: int,
        tp_rank: int,
        log_level: str = "error",
        *args,
        **kwargs,
    ):
        server_args = ServerArgs(*args, log_level=log_level, **kwargs)
        self._entrypoint = SpmdOrchestrator(
            server_args=server_args,
            nccl_port=nccl_port,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
        )

    def _generate_impl(self, obj: GenerateReqInput):
        return self._entrypoint.generate(obj)

    def shutdown(self):
        self._entrypoint.shutdown()
