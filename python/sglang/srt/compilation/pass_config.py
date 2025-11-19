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

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import Optional

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class PassConfig:
    device: Optional[str]
    model_dtype: Optional[str]

    enable_fusion: bool
    disable_rmsnorm_quant_pass: bool
    disable_fused_activation_pass: bool

    rms_norm_eps: list[float]

    enable_torch_compile_graph_trace_logs: bool

    def uuid(self):
        encoded = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def from_server_args_and_model_config(
        server_args: ServerArgs, model_config: ModelConfig
    ):
        disable_rmsnorm_quant_pass = True
        rms_norm_eps = []
        if not server_args.disable_rmsnorm_quant_pass:
            disable_rmsnorm_quant_pass = False
            if model_config.hf_config.rms_norm_eps is not None:
                rms_norm_eps.append(model_config.hf_config.rms_norm_eps)
            else:
                logger.warning(
                    "RMSNorm epsilon value not found in hugging face config, "
                    "registering fusion passes for default (1e-05, 1e-06) values."
                )
                rms_norm_eps.append(1e-05)
                rms_norm_eps.append(1e-06)

        return PassConfig(
            device=server_args.device if server_args.device else None,
            model_dtype=server_args.dtype if server_args.dtype else None,
            enable_fusion=server_args.enable_torch_compile_fusion,
            disable_rmsnorm_quant_pass=disable_rmsnorm_quant_pass,
            disable_fused_activation_pass=server_args.disable_fused_activation_pass,
            enable_torch_compile_graph_trace_logs=server_args.enable_torch_compile_graph_trace_logs,
            rms_norm_eps=rms_norm_eps,
        )
