# Copyright 2026 SGLang Team
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
"""UVM-based expert weight offloading for MoE layers.

Expert weights are stored in CUDA Unified Memory (cudaMallocManaged).
Resident experts are kept in GPU VRAM (PREFER_GPU advice).
Offloaded experts live in CPU DRAM and are accessible to the GPU via PCIe
read-through (PREFER_CPU + ACCESSED_BY_GPU advice) -- no page fault overhead,
no quality loss, CUDA graph compatible.

All computation remains on GPU (no CPU inference).
Mutually exclusive with KTransformers (--kt-weight-path).

Usage
-----
Pass ``--expert-offload-num-resident N`` to enable.  Additional options:

  --expert-offload-resident-selection first_n How to choose resident experts.
  --expert-offload-resident-ids       None    Comma-separated IDs for manual selection.
"""

from sglang.srt.layers.moe.expert_offload.config import (
    ExpertOffloadConfig,
    create_expert_offload_config_from_server_args,
)
from sglang.srt.layers.moe.expert_offload.wrapper import ExpertOffloadWrapperMethod

__all__ = [
    "ExpertOffloadConfig",
    "ExpertOffloadWrapperMethod",
    "create_expert_offload_config_from_server_args",
]
