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
from sglang import Engine
from torch.distributed.tensor import DeviceMesh


class VerlEngine:
    def __init__(
        self,
        first_rank_in_node: bool,
        device_mesh_cpu: DeviceMesh,
        **kwargs,
    ):
        self._device_mesh_cpu = device_mesh_cpu
        tp_rank = device_mesh_cpu.get_local_rank()
        tp_size = device_mesh_cpu.size()

        if first_rank_in_node:
            self._engine = Engine(**kwargs, tp_rank=tp_rank, tp_size=tp_size)
        else:
            self._engine = None
