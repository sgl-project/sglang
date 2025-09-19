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
from sglang.srt.compilation.fusion.passes.fused_activation import FusedActivationPass
from sglang.srt.compilation.fusion.passes.rmsnorm_quant import RMSNormQuantPass
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs


class FusionManager(CustomGraphPass):
    def __init__(self, server_args: ServerArgs, model_config: ModelConfig):
        self.passes: list[CustomGraphPass] = []

        self.configure(server_args, model_config)

    def __call__(self, graph: fx.Graph):
        with fusion_context():
            for pass_ in self.passes:
                pass_(graph)

            get_fusion_context().log_stats()

    def configure(self, server_args: ServerArgs, model_config: ModelConfig):
        self.config = FusionConfig.from_server_args_and_model_config(
            server_args, model_config
        )

        if self.config.enable_rmsnorm_quant_pass:
            self.passes.append(RMSNormQuantPass(self.config))

        if self.config.enable_fused_activation_pass:
            self.passes.append(FusedActivationPass(self.config))

    def uuid(self):
        state = {"config": self.config.uuid(), "passes": []}
        for pass_ in self.passes:
            state["passes"].append(pass_.uuid())
        return hash_dict(state)
