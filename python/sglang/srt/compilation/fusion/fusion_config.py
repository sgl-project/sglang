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

import logging
from dataclasses import asdict, dataclass

from sglang.srt.compilation.fusion.fusion_utils import hash_dict
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    device: str
    model_dtype: str

    enable_torch_compile_graph_trace_logs: bool

    def uuid(self):
        return hash_dict(asdict(self))

    @staticmethod
    def from_server_args(server_args: ServerArgs):
        return FusionConfig(
            device=server_args.device if server_args.device else None,
            model_dtype=server_args.dtype if server_args.dtype else None,
            enable_torch_compile_graph_trace_logs=server_args.enable_torch_compile_graph_trace_logs,
        )
