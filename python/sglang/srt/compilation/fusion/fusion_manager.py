# Copyright 2023-2025 SGLang Team
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

import torch
from torch import fx
from torch._inductor.custom_graph_pass import CustomGraphPass

from sglang.srt.compilation.fusion.fusion_config import FusionConfig
from sglang.srt.compilation.fusion.fusion_context import (
    fusion_context,
    get_fusion_context,
)
from sglang.srt.compilation.fusion.fusion_utils import hash_dict
from sglang.srt.server_args import ServerArgs


class FusionManager(CustomGraphPass):
    def __init__(self, server_args: ServerArgs):
        self.passes: list[CustomGraphPass] = []

        self.configure(server_args)

    def __call__(self, graph: fx.Graph):
        with fusion_context():
            for pass_ in self.passes:
                pass_(graph)

            get_fusion_context().log_stats()

    def configure(self, server_args: ServerArgs):
        self.config = FusionConfig.from_server_args(server_args)

    def uuid(self):
        state = {"config": self.config.uuid(), "passes": []}
        for pass_ in self.passes:
            state["passes"].append(pass_.uuid())
        return hash_dict(state)
